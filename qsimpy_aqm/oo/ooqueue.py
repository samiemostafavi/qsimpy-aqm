from __future__ import annotations

from dataclasses import field, make_dataclass

import numpy as np
import pandas as pd
from qsimpy import SimpleQueue
from qsimpy.core import Task


class OfflineOptimumQueue(SimpleQueue):
    """Models a FIFO queue with offline optimum AQM"""

    type: str = "offlineoptimumqueue"

    def dequeue_oo_internal(self, df: pd.DataFrame, inp_dict: dict):

        # everything is about the first row of df
        item = df.iloc[:1].to_dict("records")[0]

        # if len(df) is 1, means it is the end of recursive function
        if len(df) == 1:
            # count the number of passes
            values = np.array(list(inp_dict.values()))
            count = np.count_nonzero(values)
            # check if this row is gonna pass itself and update the dict
            if item["delay_budget"] < item["oo_service_delay"]:
                res_dict = {item["id"]: False, **inp_dict}
            else:
                res_dict = {item["id"]: True, **inp_dict}
                count = count + 1

            # return a dict wrapped in a list to be appendable
            return [{"score": count, "dict": res_dict}]

        # more than one row are in the df
        if item["delay_budget"] < item["oo_service_delay"]:

            # copy df and remove the first row
            df_cp = df.copy()
            df_cp = df_cp.iloc[1:, :]
            # Drop only:
            # True: pass, False: drop
            upd_dict_d = {item["id"]: False, **inp_dict}
            return self.dequeue_oo_internal(df_cp, upd_dict_d)

        else:

            # Drop branch:
            # copy df and remove the first row
            df_cp_d = df.copy()
            df_cp_d = df_cp_d.iloc[1:, :]
            # True: pass, False: drop
            upd_dict_d = {item["id"]: False, **inp_dict}
            result_d = self.dequeue_oo_internal(df_cp_d, upd_dict_d)

            # Pass branch:
            # copy df and remove the first row
            df_cp_p = df.copy()
            df_cp_p = df_cp_p.iloc[1:, :]
            # True: pass, False: drop
            upd_dict_p = {item["id"]: True, **inp_dict}
            # update delay_budget of the rest of the items
            df_cp_p["delay_budget"] = df_cp_p["delay_budget"] - item["oo_service_delay"]
            result_p = self.dequeue_oo_internal(df_cp_p, upd_dict_p)

            # return the result list
            return result_d + result_p

    def dequeue_oo(self, df) -> bool:

        # gonna decide whether to drop the first item or not
        first_item = df.iloc[:1].to_dict("records")[0]

        # False: do not drop, True: drop
        results = self.dequeue_oo_internal(df=df, inp_dict={})
        scores = [rdict["score"] for rdict in results]
        max_score = max(scores)
        max_index = scores.index(max_score)
        chosen_one = results[max_index]
        return not chosen_one["dict"][first_item["id"]]

    def run(self) -> None:
        """
        serving tasks
        """
        while True:

            # before popping the head of queue, OO algorithm kicks in
            # create the dataframe
            df_original = pd.DataFrame(self._store.items)
            if len(df_original) > 0:
                # obtain all tasks delay budgets
                df_original["delay_budget"] = df_original["delay_bound"] - (
                    self._env.now - df_original["start_time"]
                )
                if self.dequeue_oo(df_original):
                    if self._debug:
                        # print(f"DROP: delta:{delta}, s_dropped: {s2}, s_original:{s1}, len(s):{len(df_original)}")
                        print(df_original)
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
            self._store.put(task)
