import os
import time
from typing import List

import numpy as np
import polars as pl

# import seaborn as sns
from loguru import logger
from qsimpy.core import Model, TimedSource
from qsimpy.polar import PolarSink
from qsimpy.random import Deterministic
from qsimpy.simplequeue import SimpleQueue

from qsimpy_aqm.random import HeavyTailGamma

# Create the QSimPy environment
# a class for keeping all of the entities and accessing their attributes
model = Model(name="test multihop aqm")

# Create a source
# arrival process deterministic
arrival = Deterministic(
    rate=0.095,
    seed=100234,
    dtype="float64",
)
source = TimedSource(
    name="start-node",
    arrival_rp=arrival,
    task_type="0",
    delay_bound=202.55575775238685,  # 131.0544, 107.70, 73.76106050610542 # 265.52116995349246,
)
model.add_entity(source)

N_HOPS = 3
queues: List[SimpleQueue] = []
services: List[HeavyTailGamma] = []
for hop in range(N_HOPS):
    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=120034 + hop * 23400,
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=0.3,  # p9
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=100000,
        be_quiet=True,
    )
    queue = SimpleQueue(
        name=f"queue_{hop}",
        service_rp=service,
    )
    model.add_entity(queue)
    queues.append(queue)
    services.append(service)

# Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
sink = PolarSink(
    name="sink",
    batch_size=10000,
)
# define postprocess function: the name must be 'user_fn'


def user_fn(df):
    # df is pandas dataframe in batch_size
    df["end2end_delay"] = df["end_time"] - df["start_time"]

    # process service delay, queue delay
    for h in range(N_HOPS):
        df[f"service_delay_{h}"] = df[f"end_time_{h}"] - df[f"service_time_{h}"]
        df[f"queue_delay_{h}"] = df[f"service_time_{h}"] - df[f"queue_time_{h}"]

    # process time-in-service
    # p is predictor num, h is hop num
    for p in range(N_HOPS):
        for j in range(N_HOPS - p):
            h = j + p

            # process time in service
            df[f"time_in_service_p{p}_h{h}"] = df.apply(
                lambda row: (
                    row[f"queue_time_{p}"] - row[f"last_service_time_p{p}_h{h}"]
                )
                if row[f"queue_is_busy_p{p}_h{h}"]
                else None,
                axis=1,
            ).astype("float64")

            # process longer_delay_prob here for benchmark purposes
            df[f"longer_delay_prob_p{p}_h{h}"] = np.float64(1.00) - services[h].cdf(
                y=df[f"time_in_service_p{p}_h{h}"].to_numpy(),
            )
            df[f"longer_delay_prob_p{p}_h{h}"] = df[
                f"longer_delay_prob_p{p}_h{h}"
            ].fillna(np.float64(0.00))
            del df[f"last_service_time_p{p}_h{h}"], df[f"queue_is_busy_p{p}_h{h}"]

    # delete remaining items
    for h in range(N_HOPS):
        del df[f"end_time_{h}"], df[f"service_time_{h}"], df[f"queue_time_{h}"]

    return df


sink._post_process_fn = user_fn
model.add_entity(sink)

# Wire start-node, queue, end-node, and sink together
source.out = queues[0].name
for idx, queue in enumerate(queues):
    if idx == N_HOPS - 1:
        queue.out = sink.name
        queue.drop = sink.name
    else:
        queue.out = queues[idx + 1].name
        queue.drop = sink.name


# Setup task records
timestamps = {
    source.name: {
        "task_generation": "start_time",
    },
    sink.name: {
        "task_reception": "end_time",
    },
}
for hop, queue in enumerate(queues):
    timestamps = {
        **timestamps,
        queue.name: {
            "task_reception": f"queue_time_{hop}",
            "service_start": f"service_time_{hop}",
            "service_end": f"end_time_{hop}",
        },
    }
attributes_base = {
    source.name: {
        "task_generation": {
            queue.name: {
                "queue_length": f"queue_length_p0_h{hop}",
                "last_service_time": f"last_service_time_p0_h{hop}",
                "is_busy": f"queue_is_busy_p0_h{hop}",
            }
            for hop, queue in enumerate(queues)
        },
    }
}
attributes_rest = {
    queue_i.name: {
        "service_end": {
            queue_j.name: {
                "queue_length": f"queue_length_p{i+1}_h{j+i+1}",
                "last_service_time": f"last_service_time_p{i+1}_h{j+i+1}",
                "is_busy": f"queue_is_busy_p{i+1}_h{j+i+1}",
            }
            for j, queue_j in enumerate(queues[i + 1 :])
        },
    }
    for i, queue_i in enumerate(queues[:-1])
}
model.set_task_records(
    {
        "timestamps": timestamps,
        "attributes": {
            **attributes_base,
            **attributes_rest,
        },
    }
)


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

# run configuration
until = 1000  # 100000
report_state_frac = 0.01  # every 1% report


# report timesteps
def report_state(time_step):
    yield model.env.timeout(time_step)
    logger.info(f"Simulation progress {100.0*float(model.env.now)/float(until)}% done")


for step in np.arange(0, until, float(until) * float(report_state_frac), dtype=int):
    model.env.process(report_state(step))

project_path = "projects/newdelta_limited_test"
os.makedirs(project_path, exist_ok=True)

modeljson = model.json()
with open(
    project_path + "/model.json",
    "w",
    encoding="utf-8",
) as f:
    f.write(modeljson)

# Run!
start = time.time()
model.env.run(until=until)
end = time.time()
print("Run finished in {0} seconds".format(end - start))

print("Source generated {0} tasks".format(source.get_attribute("tasks_generated")))
print(
    "Queue completed {0}, dropped {1}".format(
        queue.get_attribute("tasks_completed"),
        queue.get_attribute("tasks_dropped"),
    )
)
print(" Sink received {0} tasks".format(sink.get_attribute("tasks_received")))

start = time.time()

# Process the collected data
df = sink.received_tasks
print(df)

df.write_parquet(
    file=project_path + "/records.parquet",
    compression="snappy",
)

df_dropped = df.filter(pl.col("end_time") == -1)
df_finished = df.filter(pl.col("end_time") >= 0)
# df = df_finished

res = []
pd_df = df.to_pandas()
all_missed = len(
    pd_df[(pd_df["end2end_delay"] > source.delay_bound) | (pd_df["end_time"] == -1)]
)
print(f"{all_missed/len(pd_df)} fraction of tasks failed.")
