from pyspark.sql import SparkSession
import os


records_path = 'projects/no_aqm/records1M/'
qrange_list = [0.8,0.9,0.99,0.999]

def init_spark():

    # "spark.driver.memory" must not exceed the total memory of the device: SWAP + RAM

    spark = SparkSession.builder \
        .appName("LoadParquets") \
        .config("spark.executor.memory","6g") \
        .config("spark.driver.memory", "70g") \
        .config("spark.driver.maxResultSize",0) \
        .getOrCreate()

    sc = spark.sparkContext
    return spark,sc


# init Spark
spark,sc = init_spark()

all_files = os.listdir(records_path)
files = []
for f in all_files:
    if f.endswith(".parquet"):
        files.append(records_path + f)
        small_df=spark.read.parquet(records_path + f)
        res = small_df.approxQuantile('end2end_delay',qrange_list,0)
        res_dict = {
            str(ql):qv for ql,qv in zip(qrange_list,res)
        }
        print(f"file {f} - {res_dict}")

df=spark.read.parquet(*files)
res = df.approxQuantile('end2end_delay',qrange_list,0)

res_dict = {
    str(ql):qv for ql,qv in zip(qrange_list,res)
}
print(f"Overall - {res_dict}")
