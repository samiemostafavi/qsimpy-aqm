from __future__ import annotations

import gym
import numpy as np
import simpy
import simpy.events
from gym import spaces
from pydantic import PrivateAttr
from qsimpy import SimpleQueue
from qsimpy.core import Model, Task
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy


class DRLQueue(SimpleQueue):
    """
    Models a FIFO queue with Deep Reinforcement Learning (DRL) AQM
    It is a custom OpenAI gym environment that follows gym interface.
    """

    type: str = "drlqueue"
    rl_model_address: str = ""

    _entrance_timestamps: list = PrivateAttr()
    _latest_queue_delay: np.float64 = PrivateAttr()
    _aqm_drop: bool = PrivateAttr()
    _observations: tuple = PrivateAttr()
    _gym_env: QueueGymEnv = PrivateAttr()
    _rl_model = PrivateAttr()
    _training: bool = PrivateAttr()

    def prepare_for_run(self, model: Model, env: simpy.Environment, debug: bool):
        super().prepare_for_run(model, env, debug)

        self._entrance_timestamps = []
        self._latest_queue_delay = 0
        self._aqm_drop = False
        self._observations = ()
        self._training = False
        self._observations = tuple(np.random.randint(256, size=4))

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

            # DRL dequeue procedure
            if not self._training:
                # _states are only useful when using LSTM policies
                action, _states = self._rl_model.predict(self._observations)
                self.set_action(action)

            drop = self._action
            if drop:
                d_task = yield self._store.get()
                # drop the task
                self.attributes["tasks_dropped"] += 1

                # pop the oldest timestamp because that corresponds to the head task
                self._latest_queue_delay = (
                    self._env.now - self._entrance_timestamps.pop(0)
                )
                # record RL observations
                self._observations = tuple(np.random.randint(256, size=4))

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
            # record RL observations
            self._observations = tuple(np.random.randint(256, size=4))
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

    def __init__(self, queue: DRLQueue):
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
        tasks_completed_old = self._drl_queue.attributes["tasks_completed"]
        tasks_dropped_old = self._drl_queue.attributes["tasks_dropped"]

        while True:
            # wait until one task get processed
            try:
                self._drl_queue._env.step()
                if (
                    self._drl_queue.attributes["tasks_completed"]
                    == tasks_completed_old + 1
                ) or (
                    self._drl_queue.attributes["tasks_dropped"] == tasks_dropped_old + 1
                ):
                    # prepare observations and read the reward
                    (
                        self.backlog,
                        self.dequeue_rate,
                        self.queue_delay,
                        reward,
                    ) = self._drl_queue.get_observations()
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
