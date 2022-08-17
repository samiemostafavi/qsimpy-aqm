from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import simpy
from loguru import logger
from pydantic import PrivateAttr
from pyspark.sql import SparkSession
from qsimpy.core import Model
from qsimpy.simplequeue import SimpleQueue

warnings.filterwarnings("ignore")


def init_spark():

    # "spark.driver.memory" must not exceed the total memory of the device: SWAP + RAM
    # "spark.sql.execution.arrow.pyspark.enabled" is for faster conversion of Pandas dataframe to spark

    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )

    sc = spark.sparkContext
    return spark, sc


# init Spark
spark, sc = init_spark()


class EmpiricalPredictor:
    df_arr: dict

    def __init__(self, records_path, conditions):
        self.df_arr = {}
        for results_dir in conditions:
            q_len = conditions[results_dir]
            cond_records_path = records_path + "/" + results_dir
            all_files = os.listdir(cond_records_path)
            files = []
            for f in all_files:
                if f.endswith(".parquet"):
                    files.append(cond_records_path + "/" + f)

            df = spark.read.parquet(*files)
            logger.info(f"Parquet files in {cond_records_path} are loaded.")
            logger.info(
                f"Total number of samples in this empirical dataset: {df.count()}"
            )
            self.df_arr[str(q_len)] = df

    def get_prob(self, queue_length, delay_budget):

        cond_df = self.df_arr[str(int(queue_length))].alias("cond_df")
        total_count = cond_df.count()
        cond_df = cond_df.where(cond_df["end2end_delay"] <= delay_budget)
        success_count = cond_df.count()
        return success_count / total_count


class DataQueue(SimpleQueue):
    """Models a FIFO queue with Delta AQM"""

    type: str = "deltaqueue"
    debug_drops: bool = False
    records_path: str
    conditions: dict

    _debug_json: str = PrivateAttr()
    _debug_list: list = PrivateAttr()
    _predictor: EmpiricalPredictor

    def prepare_for_run(self, model: Model, env: simpy.Environment, debug: bool):
        super().prepare_for_run(model, env, debug)

        self._predictor = EmpiricalPredictor(
            records_path=self.records_path,
            conditions=self.conditions,
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

        # calculate the prediction
        queue_lengths_arr = df["queue_length"].values
        delay_budgets_arr = np.array(df["delay_budget"], dtype=np.float64)
        cdf = []
        for queue_length, delay_budget in zip(queue_lengths_arr, delay_budgets_arr):
            prob = self._predictor.get_prob(
                queue_length=queue_length,
                delay_budget=delay_budget,
            )
            cdf.append(prob)
        df["success_prob"] = cdf

        return df["success_prob"].sum()

    def run(self) -> None:
        """
        serving tasks
        """
        while True:
            # before popping the head of queue, Delta algorithm kicks in
            df_original = pd.DataFrame(self._store.items)
            if len(df_original) > 1:
                df_dropped = df_original.copy()
                df_dropped = df_dropped.iloc[1:, :]
                s1 = self.calc_expected_success(df_original)
                s2 = self.calc_expected_success(df_dropped)
                delta = s2 - s1
                if delta > 0:
                    if self.debug_drops:
                        print(
                            f"DROP: delta:{delta}, s_dropped: {s2}, s_original:{s1}, len(s):{len(df_original)}"
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

                    d_task = yield self._store.get()
                    # drop the task
                    self.attributes["tasks_dropped"] += 1
                    if self.drop is not None:
                        self._drop.put(d_task)

                    # start over
                    continue

            # server takes the head task from the queue
            task = yield self._store.get()
            self.attributes["queue_length"] -= 1

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
