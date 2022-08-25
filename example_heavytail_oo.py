import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import qsimpy
import seaborn as sns
from qsimpy.random import Deterministic

from qsimpy_aqm.oo import OfflineOptimumQueue
from qsimpy_aqm.random import HeavyTailGamma

# Create the QSimPy environment
# a class for keeping all of the entities and accessing their attributes
model = qsimpy.Model(name="test delta aqm")

# Create a source
# arrival process deterministic
arrival = Deterministic(
    rate=0.095,
    seed=100234,
    dtype="float64",
)
source = qsimpy.TimedSource(
    name="start-node",
    arrival_rp=arrival,
    task_type="0",
    delay_bound=131.0544,  # 57.15 p8 #131.0544 p999
)
model.add_entity(source)

# Queue and Server
# service process a HeavyTailGamma
service = HeavyTailGamma(
    seed=120034,
    gamma_concentration=5,
    gamma_rate=0.5,
    gpd_concentration=0.1,
    threshold_qnt=0.8,
    dtype="float64",
    batch_size=1000000,
)

queue = OfflineOptimumQueue(
    name="queue",
    service_rp=service,
)
model.add_entity(queue)

# Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
sink = qsimpy.PolarSink(
    name="sink",
    batch_size=10000,
)
# define postprocess function: the name must be 'user_fn'


def user_fn(df):
    # df is pandas dataframe in batch_size
    df["end2end_delay"] = df["end_time"] - df["start_time"]
    df["service_delay"] = df["end_time"] - df["service_time"]
    df["queue_delay"] = df["service_time"] - df["queue_time"]
    # process time in service
    df["time_in_service"] = df.apply(
        lambda row: (row.start_time - row.last_service_time)
        if row.queue_is_busy
        else None,
        axis=1,
    ).astype("float64")
    # process longer_delay_prob here for benchmark purposes
    df["longer_delay_prob"] = np.float64(1.00) - service.cdf(
        y=df["time_in_service"].to_numpy(),
    )
    df["longer_delay_prob"] = df["longer_delay_prob"].fillna(np.float64(0.00))
    del df["last_service_time"], df["queue_is_busy"]
    return df


# convert it to string and pass it to the sink function
# user_fn_str = importable(user_fn, source=True)
# sink.set_post_process_fn(fn_str=user_fn_str)
# sink.set_post_process_fn

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
                        "last_service_time": "last_service_time",
                        "is_busy": "queue_is_busy",
                    },
                },
            },
        },
    }
)

# prepare for run
model.prepare_for_run(debug=False)

# Run!
start = time.time()
model.env.run(until=10000)
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
df = df_finished

# plot end-to-end delay profile
sns.set_style("darkgrid")
sns.displot(df["end2end_delay"], kde=True)
plt.savefig("end2end_ooaqm.png")

sns.displot(df["service_delay"], kde=True)
plt.savefig("service_delay_ooaqm.png")

sns.displot(df["queue_delay"], kde=True)
plt.savefig("queue_delay_ooaqm.png")
