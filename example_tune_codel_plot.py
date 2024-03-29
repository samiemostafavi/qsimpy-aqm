import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession


def init_spark():

    # "spark.driver.memory" must not exceed the total memory of the device: SWAP + RAM

    spark = (
        SparkSession.builder.appName("LoadParquets")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )

    sc = spark.sparkContext
    return spark, sc


# init Spark
spark, sc = init_spark()

# open the dataframe from parquet files
project_folder = "projects/codel_tuning/"
project_paths = [
    project_folder + name
    for name in os.listdir(project_folder)
    if os.path.isdir(os.path.join(project_folder, name))
]

# limit
# project_paths = ['projects/delta_benchmark/p8_results']
logger.info(f"All project folders: {project_paths}")

bench_params = {  # target_delay
    "m1_results": 1,
    "m2_results": 2,
    "m3_results": 3,
    "m4_results": 4,
    "m5_results": 5,
    "m6_results": 6,
    "m7_results": 7,
    "m8_results": 8,
    "m9_results": 9,
}

results = pd.DataFrame(columns=["parameter set", "result"])
for folder_name in bench_params.keys():
    project_path = [s for s in project_paths if folder_name in s]
    project_path = project_path[0]
    # print(project_path)

    logger.info(f"Starting importing parquet files in: {project_path}")

    records_path = project_path + "/records_codel/"
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + f)

    # limit
    # files = [files[0]]

    df = spark.read.parquet(*files)

    total_tasks = df.count()
    logger.info(f"Number of imported samples: {total_tasks}")
    dropped_tasks = df.where(df.end_time == -1).count()
    logger.info(f"Number of dropped tasks: {dropped_tasks}")
    delayed_tasks = df.where(df.end2end_delay > df.delay_bound).count()
    logger.info(f"Number of delayed tasks: {delayed_tasks}")

    results.loc[len(results)] = [
        str(bench_params[folder_name]),
        (dropped_tasks + delayed_tasks) / total_tasks,
    ]

ax = results.plot(x="parameter set", y=["result"], kind="bar")
# ax.set_yscale('log')
# ax.set_yticks(1.00 - np.array(list(bench_params.values())))
ax.set_xlabel("Target delay")
ax.set_ylabel("Failed tasks ratio")
# draw the legend
ax.legend()
ax.grid()
plt.tight_layout()
plt.savefig("codel_tuning.png")


exit(0)
bars = [str(par) for par in bench_params.values()]


y_pos = np.arange(len(bars))
fig, ax = plt.subplots()
ax.bar(
    y_pos,
    results,
    label="delta",
)
ax.bar(
    y_pos,
    1.00 - np.array(list(bench_params.values())),
    label="no-aqm",
)
# fix x axis
# ax.set_xticks(range(math.ceil(minx),math.floor(maxx),100))
plt.xticks(y_pos, bars)
plt.yticks(y_pos, list(bench_params.values()))
ax.set_yscale("log")
ax.set_xlabel("Target delay")
ax.set_ylabel("Failed tasks ratio")

# draw the legend
ax.legend()
ax.grid()

fig.savefig("result.png")
