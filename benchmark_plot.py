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
project_folder = "projects/aqm_benchmark_highutil/"
project_paths = [
    project_folder + name
    for name in os.listdir(project_folder)
    if os.path.isdir(os.path.join(project_folder, name))
]

# limit
# project_paths = ['projects/delta_benchmark/p8_results']
logger.info(f"All project folders: {project_paths}")

bench_params = {  # target_delay
    "p8_results": 0.8,
    "p9_results": 0.9,
    "p99_results": 0.99,
    "p999_results": 0.999,
}

records_paths = {
    # "records_codel": "codel",
    "newdelta": "newdelta",
    # "delta": "delta",
    "oo": "offline-optimum",
}

results = pd.DataFrame(
    columns=["delay target", "no-aqm", *list(records_paths.values())]
)

for folder_name in bench_params.keys():
    project_path = [s for s in project_paths if folder_name in s]
    project_path = project_path[0]
    # print(project_path)

    logger.info(f"Starting importing parquet files in: {project_path}")

    res_arr = []
    for key in records_paths.keys():
        logger.info(f"AQM method: {key}")

        records_path = project_path + "/" + key + "/"
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

        res_arr.append((dropped_tasks + delayed_tasks) / total_tasks)

    results.loc[len(results)] = [
        str(bench_params[folder_name]),
        1.00 - bench_params[folder_name],
        *res_arr,
    ]

ax = results.plot(
    x="delay target", y=["no-aqm", *list(records_paths.values())], kind="bar"
)
ax.set_yscale("log")
ax.set_yticks(1.00 - np.array(list(bench_params.values())))
ax.set_xlabel("Target delay")
ax.set_ylabel("Failed tasks ratio")
# draw the legend
ax.legend()
ax.grid()
plt.tight_layout()
plt.savefig(project_folder + "result_newdelta.png")


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
