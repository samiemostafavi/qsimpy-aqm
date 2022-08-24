from __future__ import annotations

import json
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import simpy
from loguru import logger
from pr3d.common.core import ConditionalDensityEstimator
from pr3d.de import (
    ConditionalGammaEVM,
    ConditionalGammaMixtureEVM,
    ConditionalGaussianMM,
)
from pydantic import BaseModel, PrivateAttr
from qsimpy.core import Model, Task
from qsimpy.simplequeue import SimpleQueue

from qsimpy_aqm.delta import PredictorAddresses


class Horizon(BaseModel):
    """
    Optimization horizon
    """

    max_length: int = None
    min_length: int = None
    arrival_rate: float = None

    def populate(self, tasks: pd.DataFrame):
        """
        fills the tasks dataframe with new hypothetical arrivals
        works when min_length and arrival_rate are set
        """
        if (self.min_length is not None) and (self.arrival_rate is not None):
            if len(tasks) < self.min_length:
                cur_len = len(tasks)
                for _ in range(self.min_length - cur_len):
                    tasks.loc[len(tasks)] = 0
                    tasks.at[len(tasks) - 1, "delay_budget"] = (
                        tasks.at[len(tasks) - 2, "delay_budget"]
                        + 1.00 / self.arrival_rate
                    )
                    tasks.at[len(tasks) - 1, "index"] = (
                        tasks.at[len(tasks) - 2, "index"] + 1
                    )

    def haircut(self, tasks: pd.DataFrame):
        """
        To maintain low decision making duration, this function removes the last
        packets from the tasks state dataframe. For example when self.max_length
        is 10, all tasks above index 10 will be dropped from state dataframe.
        """
        if self.max_length is not None:
            if len(tasks) > self.max_length:
                logger.warning(
                    f"State vector length: {len(tasks)}, dropping to {self.max_length}"
                )
                tasks.drop(tasks.index[self.max_length :], inplace=True)


class NewDeltaQueue(SimpleQueue):
    """Models a FIFO queue with the New Delta AQM"""

    type: str = "newdeltaqueue"
    predictor_addresses: PredictorAddresses
    horizon: Horizon = None
    limit_drops: List[int] = None
    gradient_check: bool = None
    do_not_drop: bool = False
    debug_drops: bool = False
    debug_all: bool = False

    _predictor: ConditionalDensityEstimator = PrivateAttr()
    _predictor_conf: dict = PrivateAttr()
    _queue_df: pd.DataFrame = PrivateAttr(default=None)

    def prepare_for_run(self, model: Model, env: simpy.Environment, debug: bool):
        super().prepare_for_run(model, env, debug)

        with open(self.predictor_addresses.json_address) as json_file:
            self._predictor_conf = json.load(json_file)

        pred_type = self._predictor_conf["type"]
        pred_addr = self.predictor_addresses.h5_address

        # initiate the non conditional predictor
        if pred_type == "gmm":
            self._predictor = ConditionalGaussianMM(
                h5_addr=pred_addr,
            )
        elif pred_type == "gevm":
            self._predictor = ConditionalGammaEVM(
                h5_addr=pred_addr,
            )
        elif pred_type == "gmevm":
            self._predictor = ConditionalGammaMixtureEVM(
                h5_addr=pred_addr,
            )

    def calc_expected_success(
        self,
        df: pd.DataFrame,
    ):
        df["queue_length"] = np.arange(len(df))

        x = np.array(df["queue_length"].values)
        x = np.expand_dims(x, axis=1)

        y = np.array(df["delay_budget"], dtype=np.float64)
        y = y.clip(min=0.00)
        prob, logprob, cdf = self._predictor.prob_batch(x, y)
        df["success_prob"] = cdf

    def predict_success_prob(
        self,
        task_dict: dict,
        queue_lengths: List[int],
    ):
        x = np.array(queue_lengths)
        x = np.expand_dims(x, axis=1)
        y = np.array(
            np.ones(len(queue_lengths)) * task_dict["delay_budget"],
            dtype=np.float64,
        )
        y = y.clip(min=0.00)  # important
        prob, logprob, cdf = self._predictor.prob_batch(x, y)
        return cdf

    def delta_internal_limited(
        self,
        state_df: pd.DataFrame,
    ):
        results = []
        for num_drops in self.limit_drops:
            if len(state_df) <= num_drops:
                continue

            results_drop_num = []
            for drops in combinations(list(range(len(state_df))), num_drops):
                state_dropped = state_df.copy()
                if len(drops) != 0:
                    state_dropped.drop(
                        index=list(drops),
                        axis=0,
                        inplace=True,
                    )
                    state_dropped.reset_index(drop=True, inplace=True)
                self.calc_expected_success(df=state_dropped)
                res_Psi = sum(state_dropped["success_prob"])
                res_dict = {
                    idx: False if (idx in drops) else True
                    for idx in range(len(state_df))
                }
                entry = {"Psi": res_Psi, "dict": res_dict}
                results_drop_num.append(entry)

            if self.gradient_check:
                if num_drops == 0:
                    max_Psi_tmp = results_drop_num[0]["Psi"]
                    results = results + results_drop_num
                else:
                    Psis = [rdict["Psi"] for rdict in results_drop_num]
                    max_Psi = max(Psis)
                    if max_Psi <= max_Psi_tmp:
                        return results
                    else:
                        results = results + results_drop_num
                        max_Psi_tmp = max_Psi

        return results

    def delta_internal(
        self,
        state_df: pd.DataFrame,
        front_tasks_Psi: float,
        front_services_num: int,
        inp_dict: dict,
        success_probs_dict: dict,
    ):

        # everything is about the first row of df
        curr_task_dict = state_df.iloc[:1].to_dict("records")[0]

        # if len(df) is 1, means it is the end of recursive function

        if len(state_df) == 1:
            # pass the last packet always
            res_dict = {curr_task_dict["index"]: True, **inp_dict}
            res_Psi = (
                front_tasks_Psi
                + success_probs_dict[curr_task_dict["index"]][front_services_num]
            )
            # return a dict wrapped in a list to be appendable
            return [{"Psi": res_Psi, "dict": res_dict}]

        else:  # more than one row are in the states table
            # Drop branch:
            # copy df and remove the first row
            df_cp_d = state_df.copy()
            df_cp_d = df_cp_d.iloc[1:, :]
            # True: pass, False: drop
            upd_dict_d = {curr_task_dict["index"]: False, **inp_dict}
            result_d = self.delta_internal(
                state_df=df_cp_d,
                front_tasks_Psi=front_tasks_Psi,
                front_services_num=front_services_num,
                inp_dict=upd_dict_d,
                success_probs_dict=success_probs_dict,
            )

            # Pass branch:
            # copy df and remove the first row
            df_cp_p = state_df.copy()
            df_cp_p = df_cp_p.iloc[1:, :]
            # True: pass, False: drop
            upd_dict_p = {curr_task_dict["index"]: True, **inp_dict}
            # update delay_budget of the rest of the items
            result_p = self.delta_internal(
                state_df=df_cp_p,
                front_tasks_Psi=front_tasks_Psi
                + success_probs_dict[curr_task_dict["index"]][front_services_num],
                front_services_num=front_services_num + 1,
                inp_dict=upd_dict_p,
                success_probs_dict=success_probs_dict,
            )

        # return the result list
        return result_d + result_p

    def print_solution_results(self, results: dict):
        df = pd.DataFrame(columns=["idx", "Psi", "act"])
        for idx, solution in enumerate(results):
            solution_actions = dict(sorted(solution["dict"].items()))
            action_str = ""
            for a in solution_actions:
                action_str = action_str + ("p" if solution_actions[a] else "d")
            df.loc[idx] = [idx, solution["Psi"], action_str]
            # print(f"{idx}: Psi={solution['Psi']}, act={action_str}")
        df = df.sort_values(by=["Psi"], ascending=False)
        print(df.head(10))

    def prepare_delta(
        self,
        tasks: pd.DataFrame,
    ):
        state_df = pd.DataFrame()
        state_df["delay_budget"] = tasks["delay_bound"] - (
            self._env.now - tasks["start_time"]
        )
        state_df["index"] = np.arange(len(state_df))

        if self.horizon is not None:
            # populate df with horizon hypothetical tasks
            # call by ref
            self.horizon.populate(tasks=state_df)
            # remove the tail tasks with indexes larger than max_length
            # call by ref
            self.horizon.haircut(tasks=state_df)

        success_probs_dict = dict()
        if self.limit_drops is None:
            if not self.do_not_drop:
                for i, task_row in state_df.iterrows():
                    task_dict = task_row.to_dict()
                    success_probs_dict[i] = self.predict_success_prob(
                        task_dict=task_dict,
                        queue_lengths=list(range(i + 1)),
                    )
            else:
                self.calc_expected_success(df=state_df)

        return state_df, success_probs_dict

    def pop_head_queue_df(self):
        # once the decision is made, update _queue_df
        # drop last row
        self._queue_df.drop(
            index=self._queue_df.index[0],
            axis=0,
            inplace=True,
        )
        self._queue_df.reset_index(drop=True, inplace=True)  # important

    def delta_drop(self):

        if (self.horizon is None) and (len(self._queue_df) <= 1):

            if self.debug_all:
                print(
                    "DROP:False, single task queue "
                    + f"len(s):{len(self._queue_df)}, "
                    + f"task_id: {self._queue_df.at[0,'id']}"
                )

            # important, pop the head
            self.pop_head_queue_df()

            self.attributes["exp_success_rate"] = 1.00
            return False

        # prepare
        state_df, success_probs_dict = self.prepare_delta(tasks=self._queue_df)

        if not self.do_not_drop:
            # False: do not drop, True: drop
            if self.limit_drops is not None:
                results = self.delta_internal_limited(
                    state_df=state_df,
                )
            else:
                results = self.delta_internal(
                    state_df=state_df,
                    front_tasks_Psi=0,
                    front_services_num=0,
                    inp_dict={},
                    success_probs_dict=success_probs_dict,
                )

            Psis = [rdict["Psi"] for rdict in results]
            max_Psi = max(Psis)
            max_index = Psis.index(max_Psi)
            best_action = results[max_index]

            # record this decision's success_rate
            self.attributes["exp_success_rate"] = max_Psi / len(state_df)
            # return True or False for the first element (head)
            drop = not best_action["dict"][0]
        else:

            # record this decision's success_rate
            self.attributes["exp_success_rate"] = sum(state_df["success_prob"]) / len(
                state_df
            )
            # return True or False for the first element (head)
            drop = False

        if (self.debug_drops and drop) or self.debug_all:
            print(
                f"DROP: {drop} chosen action: {max_index}, task id: {self._queue_df.id[0]}"
            )
            self.print_solution_results(results)
            print("state dataframe:")
            print(f"{state_df}")

        # important, pop the head
        self.pop_head_queue_df()

        return drop

    def run(self) -> None:
        """
        serving tasks
        """
        while True:

            # pop the head
            task = yield self._store.get()
            self.attributes["queue_length"] -= 1

            # delta kicks in
            drop_task = self.delta_drop()

            # EVENT drop_decision_made
            task = self.add_records(task=task, event_name="drop_decision_made")

            # perform delta's decision
            if drop_task:
                # drop the task
                self.attributes["tasks_dropped"] += 1
                if self.drop is not None:
                    self._drop.put(task)

                # start over
                continue

            # pass the task for the service
            # EVENT service_start
            task = self.add_records(task=task, event_name="service_start")

            # get a service duration
            self.attributes["is_busy"] = True
            new_service_duration = self.service_rp.sample()
            self.attributes["last_service_duration"] = new_service_duration
            self.attributes["last_service_time"] = self._env.now

            # wait until the task is served
            yield self._env.timeout(new_service_duration)
            self.attributes["is_busy"] = False

            # EVENT service_end
            task = self.add_records(task=task, event_name="service_end")
            self.attributes["tasks_completed"] += 1

            if self._debug:
                print(task)

            # put it on the output
            if self.out is not None:
                self._out.put(task)

    def put(self, task: Task) -> None:
        """
        queuing tasks
        """

        # increase the received counter
        self.attributes["tasks_received"] += 1

        # EVENT task_reception
        task = self.add_records(task=task, event_name="task_reception")

        # check if we need to drop the task due to buffer size limit
        drop = False
        if self.queue_limit is not None:
            if self.attributes["queue_length"] + 1 >= self.queue_limit:
                drop = True

        if drop:
            # drop the task
            self.attributes["tasks_dropped"] += 1
            if self.drop is not None:
                self._drop.put(task)
        else:
            # store the task in the queue
            self.attributes["queue_length"] += 1

            # update the queue dataframe
            if self._queue_df is not None:
                self._queue_df = pd.concat(
                    [self._queue_df, pd.DataFrame([task])],
                    ignore_index=True,
                )
            else:
                self._queue_df = pd.DataFrame([task])

            # stor the task
            self._store.put(task)
