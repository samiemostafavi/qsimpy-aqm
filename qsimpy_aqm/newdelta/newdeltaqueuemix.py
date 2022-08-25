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
from pydantic import PrivateAttr
from qsimpy.core import Model, Task
from qsimpy.simplequeue import SimpleQueue

from qsimpy_aqm.delta import PredictorAddresses

from .newdeltaqueue import Horizon


class NewDeltaQueueMix(SimpleQueue):
    """
    Models a FIFO queue with the New Delta AQM
    Capable of processing mix traffic: timed packets + cross traffic
    """

    type: str = "newdeltaqueuemix"
    predictor_addresses: PredictorAddresses
    min_Psi: float
    horizon: Horizon = None
    limit_drops: List[int] = None
    limit_ct_drops: List[int] = None
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
        df.loc[df["delay_budget"] == np.inf, "success_prob"] = 0

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

        # first try dropping cross traffic
        # print(state_df)
        # print("drop cts:")
        state_df_ct = state_df[state_df["delay_budget"] == np.inf]
        ct_indices = state_df_ct["index"].to_list()
        len_ct = len(state_df_ct)
        len_timed_pkts = len(state_df) - len_ct
        zero_timed_all_ct = None

        if len_timed_pkts == 0:
            # pass all ct packets
            res_dict = {idx: True for idx in range(len(state_df))}
            res_Psi = 0.0
            entry = {"Psi": res_Psi, "dict": res_dict}
            results.append(entry)
            return results

        for num_drops in range(len(state_df_ct) + 1):

            ct_drops = ct_indices[:num_drops]
            # print(ct_drops)

            # check dropping cross traffic packets first
            state_dropped = state_df.copy()
            if len(ct_drops) != 0:
                state_dropped.drop(
                    index=ct_drops,
                    axis=0,
                    inplace=True,
                )
                state_dropped.reset_index(drop=True, inplace=True)

            self.calc_expected_success(df=state_dropped)
            # print(state_dropped)
            res_Psi = sum(state_dropped["success_prob"])
            res_dict = {
                idx: False if (idx in ct_drops) else True
                for idx in range(len(state_df))
            }
            entry = {"Psi": res_Psi, "dict": res_dict}
            results.append(entry)
            if num_drops == len(state_df_ct):
                zero_timed_all_ct = entry
            # print(f"Psi: {res_Psi/(len(state_df)-len_ct)}")
            if res_Psi / (len(state_df) - len_ct) >= self.min_Psi:
                return results

        # print("drop timed packets:")
        # if we reach here, it means all cross traffic packets must be dropped
        # or there were no cross traffic packets
        noct_state_df = state_df.drop(ct_indices, axis=0, inplace=False)
        # we do not reset indices
        # remove 0 from self.limit_drops because is has been calculated once
        # in the drop cross traffic, if it exist
        if zero_timed_all_ct is not None:
            # do not add zero
            new_limit_drops = [x for x in self.limit_drops if x != 0]
            if self.gradient_check:
                max_Psi_tmp = zero_timed_all_ct["Psi"]
        else:
            # keep zero
            new_limit_drops = self.limit_drops

        for num_drops in new_limit_drops:
            if len(noct_state_df) <= num_drops:
                continue

            results_drop_num = []
            for drops in combinations(noct_state_df["index"].to_list(), num_drops):
                # print(drops)
                state_dropped = noct_state_df.copy()
                if len(drops) != 0:
                    state_dropped.drop(
                        index=list(drops),
                        axis=0,
                        inplace=True,
                    )
                    state_dropped.reset_index(drop=True, inplace=True)
                self.calc_expected_success(df=state_dropped)
                # print(state_dropped)
                res_Psi = sum(state_dropped["success_prob"])
                res_dict = {
                    idx: False if (idx in list(drops) + ct_indices) else True
                    for idx in range(len(state_df))
                }
                entry = {"Psi": res_Psi, "dict": res_dict}
                results_drop_num.append(entry)

            Psis = [rdict["Psi"] for rdict in results_drop_num]
            max_Psi = max(Psis)

            # print(f"Max Psi: {max_Psi}, num_timed_pkts: {len_timed_pkts}, avg: {max_Psi/len_timed_pkts}_")
            if max_Psi / len_timed_pkts >= self.min_Psi:
                results = results + results_drop_num
                return results

            if self.gradient_check:
                if num_drops == 0:
                    max_Psi_tmp = results_drop_num[0]["Psi"]
                    results = results + results_drop_num
                else:
                    if max_Psi <= max_Psi_tmp:
                        return results
                    else:
                        results = results + results_drop_num
                        max_Psi_tmp = max_Psi

        # if we reach here, it means non of the solutions could reach Psi_t to
        # the required level self.min_Psi
        logger.warning(
            "Required Psi level could not be reached,"
            + " the best action will be chosen",
        )
        return results

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
            results = self.delta_internal_limited(
                state_df=state_df,
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
            tmp = pd.DataFrame([task])
            if "delay_bound" not in tmp:
                tmp["delay_bound"] = np.inf
            if self._queue_df is not None:
                self._queue_df = pd.concat(
                    [self._queue_df, tmp],
                    ignore_index=True,
                )
            else:
                self._queue_df = tmp

            # stor the task
            self._store.put(task)
