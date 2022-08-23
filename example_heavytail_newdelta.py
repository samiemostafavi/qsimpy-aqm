import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# import seaborn as sns
from loguru import logger
from qsimpy.core import Model, TimedSource
from qsimpy.polar import PolarSink
from qsimpy.random import Deterministic

from arrivals import HeavyTailGamma
from qsimpy_aqm.delta import DeltaQueue, PredictorAddresses

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
    delay_bound=164.569,  # 131.0544, 107.70, 73.76106050610542 # 265.52116995349246,
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
    batch_size=1000000,
)

queue = DeltaQueue(
    name="queue",
    service_rp=service,
    predictor_addresses=PredictorAddresses(
        h5_address="predictors/gmevm_model.h5",
        json_address="predictors/gmevm_model.json",
    ),
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
            queue.name: {
                "drop_decision_made": {
                    queue.name: {
                        "exp_success_rate": "exp_success_rate",
                        "cur_success_rate": "success_rate",
                    },
                },
            },
        },
    }
)

# prepare for run
model.prepare_for_run(debug=False)

# run configuration
until = 1000000  # 100000
report_state_frac = 0.01  # every 1% report


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
    "Queue completed {0}, dropped {1}".format(
        queue.get_attribute("tasks_completed"),
        queue.get_attribute("tasks_dropped"),
    )
)
print(" Sink received {0} tasks".format(sink.get_attribute("tasks_received")))

start = time.time()

# Process the collected data
df = sink.received_tasks
df_dropped = df.filter(pl.col("end_time") == -1)
df_finished = df.filter(pl.col("end_time") >= 0)
# df = df_finished

res = []
pd_df = df.to_pandas()
all_missed = len(
    pd_df[(pd_df["end2end_delay"] > source.delay_bound) | (pd_df["end_time"] == -1)]
)
print(f"{all_missed/len(pd_df)} fraction of tasks failed.")

print(f"0.999 quantile: {pd_df.end2end_delay.quantile(0.999)}")


while len(pd_df) > queue.horizon.length:
    head_n = pd_df.head(queue.horizon.length)
    count = len(
        head_n[
            (head_n["end2end_delay"] > source.delay_bound) | (head_n["end_time"] == -1)
        ]
    )
    res.append(1.00 - (count / queue.horizon.length))
    # drop first row
    pd_df.drop(
        index=pd_df.index[0],
        axis=0,
        inplace=True,
    )
    pd_df.reset_index(drop=True, inplace=True)  # important

fig, ax = plt.subplots()
ax.plot(df["exp_success_rate"])
plt.savefig("noaqm_predicted_success_rate_p999_h10.png")

fig, ax = plt.subplots()
ax.plot(res)
plt.savefig("noaqm_measured_success_rate_p999_h10.png")

# plot end-to-end delay profile
# sns.set_style("darkgrid")
# sns.displot(df["end2end_delay"], kde=True)
# plt.savefig("end2end_aqm.png")

# sns.displot(df["service_delay"], kde=True)
# plt.savefig("service_delay_aqm.png")

# sns.displot(df["queue_delay"], kde=True)
# plt.savefig("queue_delay_aqm.png")
