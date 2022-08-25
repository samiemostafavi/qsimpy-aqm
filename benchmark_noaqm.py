import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger


def create_run_graph(params):

    # Must move all tf context initializations inside the child process
    from qsimpy.core import Model, Source
    from qsimpy.polar import PolarSink
    from qsimpy.random import Deterministic
    from qsimpy.simplequeue import SimpleQueue

    from qsimpy_aqm.random import HeavyTailGamma

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"New Delta AQM benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate=0.095,
        seed=params["arrival_seed"],
        dtype="float64",
    )
    source = Source(
        name="start-node",
        arrival_rp=arrival,
        task_type="0",
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
    queue = SimpleQueue(
        name="queue",
        service_rp=service,
        # queue_limit=10, #None
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

    quant_labels = [0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    res = np.quantile(
        a=np.array(queue.service_rp._pregenerated_samples),
        q=quant_labels,
    )
    quants = {label: res[idx] for idx, label in enumerate(quant_labels)}
    service_mean = np.mean(queue.service_rp._pregenerated_samples)
    logger.info(
        f"Service mean: {service_mean},\
    arrival mean: {1.00/arrival.rate}, utilization: {service_mean*arrival.rate}\n \
    quantiles:{quants}"
    )

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
    project_path = str(p) + "/projects/no_aqm/"

    # another important
    mp.set_start_method("spawn", force=True)

    # 4 x 4, until 1000000 took 7 hours
    sequential_runs = 1  # 2  # 2  # 4
    parallel_runs = 16  # 8  # 8  # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):

            # create and prepare the results directory
            records_path = project_path + "records_new/"
            os.makedirs(records_path, exist_ok=True)

            params = {
                "records_path": records_path,
                "arrivals_number": 1000000,  # 5M #1.5M
                "run_number": j * parallel_runs + i,
                "arrival_seed": 100234 + i * 100101 + j * 10223,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "until": int(
                    1000000  # 00
                ),  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.01,  # 0.05 # report when 10%, 20%, etc progress reaches
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
