from __future__ import annotations

import math

import numpy as np
import simpy
from pydantic import PrivateAttr
from qsimpy.core import Model, Task
from qsimpy.simplequeue import SimpleQueue


class CodelQueue(SimpleQueue):
    """Models a FIFO queue with Codel AQM"""

    type: str = "codelqueue"
    interval: np.float64
    target: np.float64

    _dropping: bool = PrivateAttr()
    _next_drop_time: np.float64 = PrivateAttr()
    _count: int = PrivateAttr()
    _lastcount: int = PrivateAttr()
    _latest_queue_delay: np.float64 = PrivateAttr()
    _first_above_time: np.float64 = PrivateAttr()
    _entrance_timestamps: list = PrivateAttr()

    def prepare_for_run(self, model: Model, env: simpy.Environment, debug: bool):
        super().prepare_for_run(model, env, debug)

        self._dropping = False
        self._next_drop_time = 0
        self._count = 0
        self._lastcount = 0
        self._latest_queue_delay = 0
        self._first_above_time = 0
        self._entrance_timestamps = []

    def control_law_codel(self, time, count):
        next_drop_time = time + self.interval / math.sqrt(count)
        return next_drop_time

    def dequeue_codel_internal(self):
        # here sojourn_time means the latest queueing delay
        sojourn_time = self._latest_queue_delay
        ok_to_drop = False

        # interval is finished
        if sojourn_time < self.target:
            # went below - stay below for at least self.interval duration
            self._first_above_time = 0
        else:
            if self._first_above_time == 0:
                # just went above self.target from below.
                # if after self.interval time it was still above self.target, will say it's ok to drop.
                self._first_above_time = self._env.now + self.interval
            elif self._env.now >= self._first_above_time:
                ok_to_drop = True

        return ok_to_drop

    def dequeue_codel(self):

        # False: do not drop, True: drop
        drop = False
        ok_to_drop = self.dequeue_codel_internal()

        if self._dropping:
            # if we are in the dropping state
            if not ok_to_drop:
                # sojourn_time below self.target - leave the drop state
                self._dropping = False
            else:
                # Time for the next drop.  Drop current packet and dequeue
                # next.  If the dequeue doesn't take us out of dropping
                # state, schedule the next drop.
                if (self._env.now >= self._next_drop_time) and (self._dropping):
                    drop = True  # drop
                    self._count = self._count + 1
                    # schedule the next drop
                    self._next_drop_time = self.control_law_codel(
                        self._next_drop_time, self._count
                    )

        elif ok_to_drop:
            # If we get here, we were not in the drop state. But we are entering it
            # The 'ok_to_drop' means that the sojourn_time got higher than 'self.target'
            # for at least 'self.interval' duration, so we enter dropping state.
            drop = True  # drop
            self._dropping = True

            # If min went above TARGET close to when it last went
            # below, assume that the drop rate that controlled the
            # queue on the last cycle is a good starting point to
            # control it now.  ('drop_next' will be at most 'INTERVAL'
            # later than the time of the last drop, so 'now - drop_next'
            # is a good approximation of the time from the last drop
            # until now.) Implementations vary slightly here; this is
            # the Linux version, which is more widely deployed and
            # tested.
            delta = self._count - self._lastcount
            self._count = 1
            if (delta > 1.00) and (
                (self._env.now - self._next_drop_time) < (16.00 * self.interval)
            ):
                self._count = delta

            self._next_drop_time = self.control_law_codel(self._env.now, self._count)
            self._lastcount = self._count

        return drop

    def run(self) -> None:
        """
        serving tasks
        """
        while True:

            # CoDel dequeue procedure
            drop = self.dequeue_codel()
            if drop:
                d_task = yield self._store.get()
                # drop the task
                self.attributes["tasks_dropped"] += 1

                # pop the oldest timestamp because that corresponds to the head task
                self._latest_queue_delay = (
                    self._env.now - self._entrance_timestamps.pop(0)
                )

                if self.drop is not None:
                    self._drop.put(d_task)

                # start over
                continue

            # server takes the head task from the queue
            task = yield self._store.get()
            self.attributes["queue_length"] -= 1

            # calculate the latest queuing_delay
            # pop the oldest timestamp because that corresponds to the head task
            self._latest_queue_delay = self._env.now - self._entrance_timestamps.pop(0)

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
            # records the timestamp for calculating the queuing delay
            self._entrance_timestamps.append(self._env.now)
            self.attributes["queue_length"] += 1
            self._store.put(task)
