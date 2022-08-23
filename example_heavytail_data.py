import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from qsimpy.core import Model, TimedSource
from qsimpy.polar import PolarSink
from qsimpy.random import Deterministic

from arrivals import HeavyTailGamma
from qsimpy_aqm.data import DataQueue

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
    delay_bound=31.81568,
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

# queue = qsimpy.SimpleQueue(
#    name='queue',
#    service_rp= service,
#    #queue_limit=10, #None
# )

queue = DataQueue(
    name="queue",
    service_rp=service,
    debug_drops=True,
    records_path="predictors/records",
    conditions={
        "0_results": 0,
        "1_results": 1,
        "2_results": 2,
        "3_results": 3,
        "4_results": 4,
        "5_results": 5,
        "6_results": 6,
        "7_results": 7,
        "8_results": 8,
        "9_results": 9,
        "10_results": 10,
        "11_results": 11,
        "12_results": 12,
        "13_results": 13,
        "14_results": 14,
    },
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

# run configuration
until = 100000  # 100000
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

# delta troubleshoot
if queue.debug_drops:
    with open("delta_debug_big.json", "w") as json_file:
        json_file.write(queue._debug_json)

# Process the collected data
df = sink.received_tasks
df_dropped = df.filter(pl.col("end_time") == -1)
df_finished = df.filter(pl.col("end_time") >= 0)
df = df_finished

# plot end-to-end delay profile
sns.set_style("darkgrid")
sns.displot(df["end2end_delay"], kde=True)
plt.savefig("end2end_aqm.png")

sns.displot(df["service_delay"], kde=True)
plt.savefig("service_delay_aqm.png")

sns.displot(df["queue_delay"], kde=True)
plt.savefig("queue_delay_aqm.png")
