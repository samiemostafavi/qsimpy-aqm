from __future__ import annotations

from dataclasses import field, make_dataclass

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from qsimpy.core import Task
from qsimpy.simplequeue import SimpleQueue

from qsimpy_aqm.newdelta import Horizon


class OfflineOptimumQueue(SimpleQueue):
    """Models a FIFO queue with offline optimum AQM"""

    type: str = "offlineoptimumqueue"
    _queue_df: pd.DataFrame = PrivateAttr(default=None)
    horizon: Horizon = None
    debug_drops: bool = False
    debug_all: bool = False

    def oo_internal(self, df: pd.DataFrame, inp_dict: dict):

        # everything is about the first row of df
        item = df.iloc[:1].to_dict("records")[0]

        # if len(df) is 1, means it is the end of recursive function
        if len(df) == 1:
            # count the number of passes
            values = np.array(list(inp_dict.values()))
            count = np.count_nonzero(values)
            # check if this row is gonna pass itself and update the dict
            if item["delay_budget"] < item["oo_service_delay"]:
                res_dict = {item["index"]: False, **inp_dict}
            else:
                res_dict = {item["index"]: True, **inp_dict}
                count = count + 1

            # return a dict wrapped in a list to be appendable
            return [{"Psi": count, "dict": res_dict}]

        # more than one row are in the df
        if item["delay_budget"] < item["oo_service_delay"]:

            # copy df and remove the first row
            df_cp = df.copy()
            df_cp = df_cp.iloc[1:, :]
            # Drop only:
            # True: pass, False: drop
            upd_dict_d = {item["index"]: False, **inp_dict}
            return self.oo_internal(df_cp, upd_dict_d)

        else:

            # Drop branch:
            # copy df and remove the first row
            df_cp_d = df.copy()
            df_cp_d = df_cp_d.iloc[1:, :]
            # True: pass, False: drop
            upd_dict_d = {item["index"]: False, **inp_dict}
            result_d = self.oo_internal(df_cp_d, upd_dict_d)

            # Pass branch:
            # copy df and remove the first row
            df_cp_p = df.copy()
            df_cp_p = df_cp_p.iloc[1:, :]
            # True: pass, False: drop
            upd_dict_p = {item["index"]: True, **inp_dict}
            # update delay_budget of the rest of the items
            df_cp_p["delay_budget"] = df_cp_p["delay_budget"] - item["oo_service_delay"]
            result_p = self.oo_internal(df_cp_p, upd_dict_p)

            # return the result list
            return result_d + result_p

    def prepare_oo(
        self,
        tasks: pd.DataFrame,
    ):
        state_df = pd.DataFrame()
        # obtain all tasks delay budgets
        state_df["delay_budget"] = tasks["delay_bound"] - (
            self._env.now - tasks["start_time"]
        )
        state_df["index"] = np.arange(len(state_df))
        state_df["oo_service_delay"] = tasks["oo_service_delay"]

        if self.horizon is not None:
            # populate df with horizon hypothetical tasks
            # call by ref
            self.horizon.populate(tasks=state_df)
            # remove the tail tasks with indexes larger than max_length
            # call by ref
            self.horizon.haircut(tasks=state_df)

        return state_df

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

    def pop_head_queue_df(self):
        # once the decision is made, update _queue_df
        # drop last row
        self._queue_df.drop(
            index=self._queue_df.index[0],
            axis=0,
            inplace=True,
        )
        self._queue_df.reset_index(drop=True, inplace=True)  # important

    def oo_drop(self):

        # prepare
        state_df = self.prepare_oo(tasks=self._queue_df)

        # False: do not drop, True: drop
        results = self.oo_internal(df=state_df, inp_dict={})
        Psis = [rdict["Psi"] for rdict in results]
        max_Psi = max(Psis)
        max_index = Psis.index(max_Psi)
        chosen_one = results[max_index]

        # return True or False for the first element (head)
        drop = not chosen_one["dict"][0]

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

            # server takes the head task from the queue
            task = yield self._store.get()
            self.attributes["queue_length"] -= 1

            # OO algorithm kicks in
            drop_task = self.oo_drop()

            # EVENT drop_decision_made
            task = self.add_records(task=task, event_name="drop_decision_made")

            # perform offline optimum's decision
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
            new_service_duration = task.oo_service_delay
            self.attributes["last_service_duration"] = task.oo_service_delay
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
            # sample the task's service duration and add to the fields
            task.__class__ = make_dataclass(
                "OOGeneratedTimedTask",
                fields=[("oo_service_delay", float, field(default=-1))],
                bases=(task.__class__,),
            )
            task.oo_service_delay = self.service_rp.sample()

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

            self._store.put(task)
