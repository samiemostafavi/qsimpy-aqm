from __future__ import annotations

import json
import logging
import traceback

import numpy as np
import pandas as pd
import simpy
from pr3d.common.core import ConditionalDensityEstimator
from pr3d.de import (
    ConditionalGammaEVM,
    ConditionalGammaMixtureEVM,
    ConditionalGaussianMM,
)
from pydantic import BaseModel, PrivateAttr
from qsimpy.core import Model, Task
from qsimpy.simplequeue import SimpleQueue


class PredictorAddresses(BaseModel):
    type: str = "predictoraddresses"
    h5_address: str
    json_address: str


class DeltaQueue(SimpleQueue):
    """Models a FIFO queue with Delta AQM"""

    type: str = "deltaqueue"
    predictor_addresses: PredictorAddresses
    debug_drops: bool = False
    debug_all: bool = False

    _predictor: ConditionalDensityEstimator = PrivateAttr()
    _predictor_conf: dict = PrivateAttr()
    _debug_json: str = PrivateAttr()
    _debug_list: list = PrivateAttr()
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

        if self.debug_drops:
            self._debug_list = []
            self._debug_json = ""

    def calc_expected_success(
        self,
        df: pd.DataFrame,
    ):
        df["delay_budget"] = df["delay_bound"] - (self._env.now - df["start_time"])
        df["queue_length"] = np.arange(len(df))
        # df["longer_delay_prob"] = np.ones(len(df)) * 1.00

        try:
            x = np.array(df["queue_length"].values)
            x = np.expand_dims(x, axis=1)

            y = np.array(df["delay_budget"], dtype=np.float64)
            y = y.clip(min=0.00)
            prob, logprob, cdf = self._predictor.prob_batch(x, y)
            df["success_prob"] = cdf
        except Exception as e:
            print(x)
            print(y)
            print(len(y))
            print(e)
            logging.error(traceback.format_exc())
        # print(df)
        return df["success_prob"].sum()

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
        if len(self._queue_df) <= 1:

            if self.debug_all:
                print(
                    "DROP:False, single task queue "
                    + f"len(s):{len(self._queue_df)}, "
                    + f"task_id: {self._queue_df.at[0,'id']}"
                )

            self.attributes["exp_success_rate"] = 1.00

            # important, pop the head
            self.pop_head_queue_df()

            return False

        # before popping the head of queue, Delta algorithm kicks in
        df_original = self._queue_df.copy()
        df_dropped = self._queue_df.copy()
        df_dropped = df_dropped.iloc[1:, :]
        s1 = self.calc_expected_success(df_original)
        s2 = self.calc_expected_success(df_dropped)
        delta = s2 - s1
        drop = delta > 0
        if drop:
            self.attributes["exp_success_rate"] = s2
        else:
            self.attributes["exp_success_rate"] = s1

        if (self.debug_drops and drop) or self.debug_all:
            print(
                f"DROP:{drop}, delta:{delta}, s_dropped: {s2}, "
                + f"s_original:{s1}, len(s):{len(df_original)}"
            )
            print(df_original)
            dict_orig = df_original[
                ["queue_length", "delay_budget", "success_prob"]
            ].to_dict()
            print(df_dropped)
            dict_drop = df_dropped[
                ["queue_length", "delay_budget", "success_prob"]
            ].to_dict()
            dict_both = {"orig": dict_orig, "dropped": dict_drop}
            self._debug_list.append(dict_both)
            self._debug_json = json.dumps(self._debug_list, indent=2)

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
