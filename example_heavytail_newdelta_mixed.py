import time

import numpy as np
import polars as pl

# import seaborn as sns
from loguru import logger
from qsimpy.core import Model, Source, TimedSource
from qsimpy.polar import PolarSink
from qsimpy.random import Deterministic, Exponential

from arrivals import HeavyTailGamma
from qsimpy_aqm.delta import PredictorAddresses
from qsimpy_aqm.newdelta import NewDeltaQueueMix

# Create the QSimPy environment
# a class for keeping all of the entities and accessing their attributes
model = Model(name="test delta aqm")

# Create a source
# arrival process deterministic
arrival = Deterministic(
    rate=0.08,
    seed=100234,
    dtype="float64",
)
source = TimedSource(
    name="start-node",
    arrival_rp=arrival,
    task_type="0",
    delay_bound=31.81568,
)
model.add_entity(source)

# Create cross traffic source
# arrival process exponential
ct_arrival = Exponential(
    rate=0.01,
    seed=900234,
    dtype="float64",
)
ct_source = Source(
    name="ct-start-node",
    arrival_rp=ct_arrival,
    task_type="1",
)
model.add_entity(ct_source)

# Queue and Server
# service process a HeavyTailGamma
service = HeavyTailGamma(
    seed=120034,
    gamma_concentration=5,
    gamma_rate=0.5,
    gpd_concentration=0.1,  # p9
    threshold_qnt=0.8,
    dtype="float64",
    batch_size=1000000,
)

queue = NewDeltaQueueMix(
    name="queue",
    service_rp=service,
    min_Psi=0.8,
    predictor_addresses=PredictorAddresses(
        h5_address="predictors/gmevm_model.h5",
        json_address="predictors/gmevm_model.json",
    ),
    limit_drops=[0, 1, 2, 3],
    gradient_check=True,
    debug_drops=False,
    debug_all=False,
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
ct_source.out = queue.name
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

# run configuration
until = 100000
report_state_frac = 0.1  # every 1% report


# report timesteps
def report_state(time_step):
    yield model.env.timeout(time_step)
    logger.info(f"Simulation progress {100.0*float(model.env.now)/float(until)}% done")


for step in np.arange(0, until, float(until) * float(report_state_frac), dtype=int):
    model.env.process(report_state(step))

# Run!
start = time.time()
model.env.run(until=until)
end = time.time()
print("Run finished in {0} seconds".format(end - start))

print("Source generated {0} tasks".format(source.get_attribute("tasks_generated")))
print(
    "Cross traffic source generated {0} tasks".format(
        ct_source.get_attribute("tasks_generated")
    )
)
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

df_cross_traffic = df.filter(pl.col("delay_bound").is_null())
df_timed_pkts = df.filter(pl.col("delay_bound").is_not_null())

df_timed_pkts_dropped = df_timed_pkts.filter(pl.col("end_time") == -1).to_pandas()
df_timed_pkts_passed = df_timed_pkts.filter(pl.col("end_time") >= 0).to_pandas()
df_timed_pkts_failed = df_timed_pkts_passed[
    (df_timed_pkts_passed["end2end_delay"] > source.delay_bound)
]

logger.info(
    "Timed packets total: "
    + f"{len(df_timed_pkts.to_pandas())}, "
    + f"dropped: {len(df_timed_pkts_dropped)}, "
    + f"passed:{len(df_timed_pkts_passed)}, delayed:{len(df_timed_pkts_failed)}"
)

df_ct_pkts_dropped = df_cross_traffic.filter(pl.col("end_time") == -1).to_pandas()
df_ct_pkts_passed = df_cross_traffic.filter(pl.col("end_time") >= 0).to_pandas()

logger.info(
    f"Cross traffic packets total: {len(df_cross_traffic.to_pandas())}, "
    + f"dropped: {len(df_ct_pkts_dropped)}, passed:{len(df_ct_pkts_passed)}"
)
