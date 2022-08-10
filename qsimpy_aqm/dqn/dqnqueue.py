from __future__ import annotations

from statistics import mean
from typing import Any, Dict, FrozenSet

import gym
import numpy as np
import simpy
import simpy.events
from gym import spaces
from pydantic import PrivateAttr
from qsimpy import Entity, SimpleQueue
from qsimpy.core import Model, Task
from qsimpy.random import Deterministic, RandomProcess
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy


class Timer(Entity):
    """Executes certain function with given inter-arrival time distribution.
    Set "run_fn" method to the desired functionality

    Parameters
    ----------
    env : Environment
        the QSimPy simulation environment
    arrival_rp : RandomProcess
        A RandomProcess object that its sample_n fn returns the successive inter-arrival times of the function
    initial_delay : number
        Starts task generation after an initial delay. Default = 0
    finish_time : number
        Stops generation at the finish time. Default is infinite
    """

    type: str = "timer"
    events: FrozenSet[str] = {"tick"}
    attributes: Dict[str, Any] = {
        "ticks_passed": 0,
    }
    initial_delay: float = 0
    finish_time: float = None

    # Arrival random process
    arrival_rp: RandomProcess

    def __init__(self, **data):
        if isinstance(data["arrival_rp"], RandomProcess):
            data["arrival_rp"] = data["arrival_rp"].dict()
        super().__init__(**data)

    def prepare_for_run(self, model: Model, env: simpy.Environment, debug: bool):
        self._model = model
        self._env = env
        self._debug = debug

        if self.out is not None:
            self._out = model.entities[self.out]
        if self.drop is not None:
            self._drop = model.entities[self.drop]

        self.arrival_rp.prepare_for_run()
        self._action = model._env.process(
            self.run()
        )  # starts the run() method as a SimPy process

    def clean_attributes(self):
        for att in self.attributes:
            self.attributes[att] = 0

    def run_fn(self):
        pass

    def run(self):
        """The generator function used in simulations."""
        yield self._env.timeout(self.initial_delay)
        if self.finish_time is None:
            _finish_time = float("inf")
        while self._env.now < _finish_time:
            # wait for next transmission
            yield self._env.timeout(self.arrival_rp.sample())
            self.run_fn()
            self.attributes["ticks_passed"] += 1


class DQNTimer(Timer):
    _dqn_queue: DQNQueue = PrivateAttr()
    _enqueue_count: int = PrivateAttr()
    _drop_count: int = PrivateAttr()
    _drop_prob: float = PrivateAttr()
    _queue_length: int = PrivateAttr()
    _service_delay_mean: float = PrivateAttr()

    def prepare_for_run(
        self, model: Model, env: simpy.Environment, debug: bool, dqn_queue: DQNQueue
    ):
        super().prepare_for_run(model, env, debug)
        self._dqn_queue = dqn_queue
        self._enqueue_count = 0
        self._drop_count = 0
        self._drop_prob = 0
        self._queue_length = 0
        self._service_delay_mean = mean(
            self._dqn_queue.service_rp._pregenerated_samples
        )

    def run_fn(self):

        # read the last drop_prob and queue length
        self._drop_prob = float(self._dqn_queue._aqm_drop)
        queue_length = len(self._dqn_queue._store.items)

        """print(
            "time: {}, queue_length: {}, tasks_recv: {}, tasks_dropped: {}, drop_prob: {}".format(
                self._dqn_queue._env.now,
                queue_length,
                self._dqn_queue.attributes['tasks_received'],
                self._dqn_queue.attributes['tasks_dropped'],
                self._drop_prob,
            )
        )"""

        # Calculating dequeue_rate
        if self._dqn_queue.attributes["last_service_duration"] > 0:
            dequeue_rate = 1.00 / self._dqn_queue.attributes["last_service_duration"]
        else:
            dequeue_rate = 1.00 / self._service_delay_mean

        # Calculating enqueue_rate
        n_enq = self._dqn_queue.attributes["tasks_received"] - self._enqueue_count
        n_drop = self._dqn_queue.attributes["tasks_dropped"] - self._drop_count
        if n_enq + n_drop > 0:
            enqueue_rate = n_enq / (n_enq + n_drop)
        else:
            enqueue_rate = 0
        self._enqueue_count = self._dqn_queue.attributes["tasks_received"]
        self._drop_count = self._dqn_queue.attributes["tasks_dropped"]

        # Calculating min_delay
        min_delay = queue_length * self._service_delay_mean

        # Calculating enqueue_reward
        enqueue_reward = (
            (1 - self._dqn_queue.delta)
            * (min_delay - self._dqn_queue.delay_ref)
            * enqueue_rate
        )
        if self._drop_prob and (min_delay < self._dqn_queue.delay_ref):
            enqueue_reward = 0

        # Calculating delay_reward
        delay_reward = self._dqn_queue.delta * (
            self._dqn_queue.delay_ref - self._dqn_queue.attributes["last_queue_delay"]
        )

        # Calculating the reward
        if (self._queue_length == 0) and (self._drop_prob == 1):
            reward = -1
        elif (queue_length == 0) and (self._drop_prob == 0) and (enqueue_rate == 0):
            reward = -1
        else:
            reward = delay_reward + enqueue_reward

        """print(
            "reward: {}, delay_reward: {}, enqueue_reward: {}, old_queue_length: {}, old_drop_prob: {}, enqueue_rate: {}".format(
                reward,
                delay_reward,
                enqueue_reward,
                self._queue_length,
                self._drop_prob,
                enqueue_rate,
            )
        )"""

        # update queue_length
        self._queue_length = queue_length

        # Clip the reward
        if reward > 1:
            reward = 1
        elif reward < -1:
            reward = -1

        # update observations
        self._dqn_queue._observations = (
            self._queue_length,
            dequeue_rate,
            self._dqn_queue.attributes["last_queue_delay"],
            reward,
        )

        # use RL model and update drop action
        if not self._dqn_queue._training:
            # _states are only useful when using LSTM policies
            action, _states = self._dqn_queue._rl_model.predict(
                np.array(
                    [
                        queue_length,
                        dequeue_rate,
                        self._dqn_queue.attributes["last_queue_delay"],
                    ]
                ).astype(np.float32)
            )
            self._dqn_queue.set_action(action)


class DQNQueue(SimpleQueue):
    """
    Models a FIFO queue with Deep Reinforcement Learning (DRL) AQM
    It is a custom OpenAI gym environment that follows gym interface.
    This specific method implements "Deep Reinforcement Learning Based Active Queue
    Management for IoT Networks" by Kim et al.
    """

    type: str = "dqnqueue"
    rl_model_address: str = ""
    t_interval: float
    delta: float
    delay_ref: float

    _latest_queue_delay: np.float64 = PrivateAttr()
    _entrance_timestamps: list = PrivateAttr()
    _aqm_drop: bool = PrivateAttr()
    _observations: tuple = PrivateAttr()
    _gym_env: QueueGymEnv = PrivateAttr()
    _rl_model = PrivateAttr()
    _training: bool = PrivateAttr()
    _timer: Timer = PrivateAttr()

    def prepare_for_run(self, model: Model, env: simpy.Environment, debug: bool):
        super().prepare_for_run(model, env, debug)

        self._entrance_timestamps = []
        self._latest_queue_delay = 0
        self._aqm_drop = False
        self._observations = ()
        self._training = False
        self._observations = tuple(np.zeros(4))

        self.attributes = {
            **self.attributes,
            "last_queue_delay": 0,
            "last_service_duration": 0,
        }

        # create the timer
        arrival = Deterministic(
            rate=1.00 / self.t_interval,
            seed=0,
            dtype="float64",
        )
        self._timer = DQNTimer(
            name="dqn-timer",
            arrival_rp=arrival,
        )
        self._timer.prepare_for_run(model, env, debug, self)

        # create Gym environment
        self._gym_env = QueueGymEnv(self)
        if self.rl_model_address == "":
            self._rl_model = PPO(MlpPolicy, self._gym_env, verbose=0)
        else:
            self._rl_model = PPO.load(self.rl_model_address)

    def learn(self, total_timesteps):
        self._training = True  # turn on training
        self._rl_model.learn(total_timesteps=total_timesteps)
        self._training = False  # turn off training

    def save(self, address) -> None:
        self._rl_model.save(address)

    def run(self) -> None:
        """
        serving tasks
        """
        while True:
            if self._aqm_drop:
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
            self.attributes[
                "last_queue_delay"
            ] = self._env.now - self._entrance_timestamps.pop(0)

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

    def set_action(self, action):
        if action == 1:
            self._aqm_drop = True
        elif action == 0:
            self._aqm_drop = False
        else:
            raise Exception(f"wrong action set {action}")

    def get_observations(self):
        return self._observations


class QueueGymEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where connects to MATLAB through a shared memory interface.
    It is an implementation of this paper "Deep Reinforcement Learning Based Active Queue
    Management for IoT Networks" by Kim et al.
    """

    # Define constants for clearer code
    PASS = 0
    DROP = 1

    def __init__(self, queue: DQNQueue):
        super(QueueGymEnv, self).__init__()

        # set the QSimpy queue
        self._drl_queue = queue

        # Define action and observation space
        # They must be gym.spaces objects
        # Using discrete actions, we have two: { drop , pass }
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # The observation will be the { number of packets L, dequeue rate R_deq, and queuing delay d }
        # this can be described both by Discrete and Box space
        # number of packets L, dequeue rate R_deq, queuing delay d
        self.observation_space = spaces.Box(
            low=0, high=np.inf, dtype=np.float32, shape=(3,)
        )

    def reset(self):
        """
        Important: the observation must be a Tuple of numpy arrays
        each box represents a numpy array
        :return: Tuple(np.array)
        """
        # Initialize the agent
        self.backlog = 0
        self.dequeue_rate = 0
        self.queue_delay = 0

        # here we convert each number to its corresponding type to make it more general
        # (in case we want to use continuous actions)
        return np.array([self.backlog, self.dequeue_rate, self.queue_delay]).astype(
            np.float32
        )

    def step(self, action):
        if (action != self.DROP) and (action != self.PASS):
            raise ValueError(
                "Received invalid action={} which is not part of the action space".format(
                    action
                )
            )

        # The experiment finishes
        done = False

        # send the action
        self._drl_queue.set_action(action)
        ticks_passed_old = self._drl_queue._timer.attributes["ticks_passed"]

        while True:
            # wait until one task get processed
            try:
                self._drl_queue._env.step()
                if (
                    self._drl_queue._timer.attributes["ticks_passed"]
                    == ticks_passed_old + 1
                ):
                    # prepare observations and read the reward
                    (
                        self.backlog,
                        self.dequeue_rate,
                        self.queue_delay,
                        reward,
                    ) = self._drl_queue.get_observations()
                    """print(
                        (
                            self.backlog,
                            self.dequeue_rate,
                            self.queue_delay,
                            reward,
                        )
                    )"""
                    break
            except Exception as ex:
                # means there is no more events to process
                print(ex)
                done = True
                break

        # Account for the boundaries of the grid
        self.backlog = np.clip(self.backlog, 0, np.inf)
        self.dequeue_rate = np.clip(self.dequeue_rate, 0, np.inf)
        self.queue_delay = np.clip(self.queue_delay, 0, np.inf)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array([self.backlog, self.dequeue_rate, self.queue_delay]).astype(
                np.float32
            ),
            reward,
            done,
            info,
        )

    def close(self):
        pass
