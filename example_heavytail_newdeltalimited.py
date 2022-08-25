import os
import time

import numpy as np
import polars as pl

# import seaborn as sns
from loguru import logger
from qsimpy.core import Model, TimedSource
from qsimpy.polar import PolarSink
from qsimpy.random import Deterministic

from qsimpy_aqm.delta import PredictorAddresses
from qsimpy_aqm.newdelta import NewDeltaQueue
from qsimpy_aqm.random import HeavyTailGamma

# from qsimpy_aqm.newdelta import Horizon

# Create the QSimPy environment
# a class for keeping all of the entities and accessing their attributes
model = Model(name="test delta aqm")

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

# Queue and Server
# service process a HeavyTailGamma
service = HeavyTailGamma(
    seed=120034,
    gamma_concentration=5,
    gamma_rate=0.5,
    gpd_concentration=0.1,  # p9
    threshold_qnt=0.8,
    dtype="float64",
    batch_size=10000000,
)

queue = NewDeltaQueue(
    name="queue",
    service_rp=service,
    predictor_addresses=PredictorAddresses(
        h5_address="predictors/gmevm_model.h5",
        json_address="predictors/gmevm_model.json",
    ),
    # horizon=Horizon(
    #    max_length=None,
    #    min_length=10,
    #    arrival_rate=arrival.rate,
    # ),
    limit_drops=[0, 1, 2, 3],
    gradient_check=True,
    debug_drops=True,
    do_not_drop=False,
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
            queue.name: {
                "drop_decision_made": {
                    queue.name: {
                        "exp_success_rate": "exp_success_rate",
                    },
                },
            },
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
until = 1000000  # 100000
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
