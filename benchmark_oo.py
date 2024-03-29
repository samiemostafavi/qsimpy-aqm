import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

# https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
# https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
# The problem is, that Tensorflow is greedy in allocating all available VRAM. That causes issues when multi processes start using CUDA
# import tensorflow as tf
from loguru import logger

from qsimpy_aqm.newdelta.newdeltaqueue import Horizon

# To make tensorflow and CUDA work with multiprocessing, this article really helped:
# https://sefiks.com/2019/03/20/tips-and-tricks-for-gpu-and-multiprocessing-in-tensorflow/

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# If got any errors, try wiping CUDA cache: sudo rm -rf .nv/


def create_run_graph(params):

    # Must move all tf context initializations inside the child process
    from qsimpy.core import Model, TimedSource
    from qsimpy.polar import PolarSink
    from qsimpy.random import Deterministic

    from qsimpy_aqm.oo import OfflineOptimumQueue
    from qsimpy_aqm.random import HeavyTailGamma

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Offline Optimum AQM benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate=0.095,
        seed=params["arrival_seed"],
        dtype="float64",
    )
    source = TimedSource(
        name="start-node",
        arrival_rp=arrival,
        task_type="0",
        delay_bound=params["target_delay"],
    )
    model.add_entity(source)

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=params["service_seed"],
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=0.1,
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=params["arrivals_number"],
    )
    queue = OfflineOptimumQueue(
        name="queue",
        service_rp=service,
        horizon=Horizon(
            max_length=15,
            min_length=None,
            arrival_rate=None,
        ),
        debug_all=False,
        debug_drops=False,
    )
    model.add_entity(queue)

    # Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
    sink = PolarSink(
        name="sink",
        batch_size=10000,
    )
    # define postprocess function: the name must be 'user_fn'

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        df["service_delay"] = df["end_time"] - df["service_time"]
        df["queue_delay"] = df["service_time"] - df["queue_time"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    # Wire start-node, queue, end-node, and sink together
    source.out = queue.name
    queue.out = sink.name
    queue.drop = sink.name

    # Setup task records
    model.set_task_records(
        {
            "timestamps": {
                source.name: {
                    "task_generation": "start_time",
                },
                queue.name: {
                    "task_reception": "queue_time",
                    "service_start": "service_time",
                    "service_end": "end_time",
                },
            },
            "attributes": {
                source.name: {
                    "task_generation": {
                        queue.name: {
                            "queue_length": "queue_length",
                        },
                    },
                },
            },
        }
    )

    modeljson = model.json()
    with open(
        params["records_path"] + f"{params['run_number']}_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        logger.info(
            f"{params['run_number']}: Simulation progress {100.0*float(model.env.now)/float(params['until'])}% done"
        )

    for step in np.arange(
        0, params["until"], params["until"] * params["report_state"], dtype=int
    ):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params["until"])
    end = time.time()
    logger.info(
        "{0}: Run finished in {1} seconds".format(params["run_number"], end - start)
    )

    logger.info(
        "{0}: Source generated {1} tasks".format(
            params["run_number"], source.get_attribute("tasks_generated")
        )
    )
    logger.info(
        "{0}: Queue completed {1}, dropped {2}".format(
            params["run_number"],
            queue.get_attribute("tasks_completed"),
            queue.get_attribute("tasks_dropped"),
        )
    )
    logger.info(
        "{0}: Sink received {1} tasks".format(
            params["run_number"], sink.get_attribute("tasks_received")
        )
    )

    start = time.time()

    # Process the collected data
    df = sink.received_tasks
    # print(df)

    end = time.time()

    df.write_parquet(
        file=params["records_path"] + f"{params['run_number']}_records.parquet",
        compression="snappy",
    )

    logger.info(
        "{0}: Data processing finished in {1} seconds".format(
            params["run_number"], end - start
        )
    )


if __name__ == "__main__":

    # project folder setting
    p = Path(__file__).parents[0]
    project_path = str(p) + "/projects/oo_benchmark_highutil/"
    os.makedirs(project_path, exist_ok=True)

    # simulation parameters
    # quantile values of no-aqm model with p1 as gpd_concentration
    """
    bench_params = {  # target_delay
        "p999": 119.36120,
        "p99": 82.02233,
        "p9": 43.50905,
        "p8": 31.81568,
    }
    """
    # 0.095 arrival rate quantiles:
    bench_params = {  # target_delay
        "p999": 293.10694,
        "p99": 186.76862,
        "p9": 96.69882,
        "p8": 69.02151,
    }

    # another important
    mp.set_start_method("spawn", force=True)

    # 4 x 4, until 1000000 took 7 hours
    sequential_runs = 1  # 1
    parallel_runs = 16  # 16
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):

            # parameter figure out
            keys = list(bench_params.keys())

            # LIMIT
            if i % len(keys) != 0:
                continue

            key_this_run = keys[i % len(keys)]

            # create and prepare the results directory
            results_path = project_path + key_this_run + "_results/"
            records_path = results_path + "records_oo/"
            os.makedirs(records_path, exist_ok=True)

            params = {
                "records_path": records_path,
                "arrivals_number": 1000000,  # 5M #1.5M
                "run_number": j * parallel_runs + i,
                "arrival_seed": 100234 + i * 100101 + j * 10223,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "target_delay": bench_params[key_this_run],  # tail decays
                "until": int(
                    1000000
                ),  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # 0.05 # report when 10%, 20%, etc progress reaches
            }

            p = mp.Process(target=create_run_graph, args=(params,))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
                p.join()
                exit(0)
