# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: Ingest Apple Health XML → Delta

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "health_dashboard")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_path = f"/Volumes/{catalog}/{schema}/raw_exports"

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_exports")

# COMMAND ----------

import zipfile
import xml.etree.ElementTree as ET
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import re
from datetime import datetime

# COMMAND ----------

# Locate the most recent export zip in the volume
zip_files = dbutils.fs.ls(volume_path)
zip_files = [f for f in zip_files if f.name.endswith(".zip")]
assert zip_files, f"No export.zip found in {volume_path} — upload your Apple Health export first"
latest_zip = sorted(zip_files, key=lambda f: f.modificationTime, reverse=True)[0].path
print(f"Processing: {latest_zip}")

# COMMAND ----------

local_zip = "/tmp/apple_health_export.zip"
dbutils.fs.cp(latest_zip, f"file:{local_zip}")

records = []
with zipfile.ZipFile(local_zip, "r") as zf:
    with zf.open("apple_health_export/export.xml") as f:
        for event, elem in ET.iterparse(f, events=["end"]):
            if elem.tag == "Record":
                records.append({
                    "type": elem.get("type"),
                    "source_name": elem.get("sourceName"),
                    "source_version": elem.get("sourceVersion"),
                    "unit": elem.get("unit"),
                    "value": elem.get("value"),
                    "start_date": elem.get("startDate"),
                    "end_date": elem.get("endDate"),
                    "creation_date": elem.get("creationDate"),
                })
                elem.clear()

print(f"Parsed {len(records):,} records")

# COMMAND ----------

df = spark.createDataFrame(records)
(df.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{catalog}.{schema}.bronze_health_records"))

print(f"Written to {catalog}.{schema}.bronze_health_records")
display(df.groupBy("type").count().orderBy("count", ascending=False))

# COMMAND ----------

# Parse Workout records separately
workouts = []
with zipfile.ZipFile(local_zip, "r") as zf:
    with zf.open("apple_health_export/export.xml") as f:
        for event, elem in ET.iterparse(f, events=["end"]):
            if elem.tag == "Workout":
                workouts.append({
                    "activity_type": elem.get("workoutActivityType"),
                    "duration": elem.get("duration"),
                    "duration_unit": elem.get("durationUnit"),
                    "total_distance": elem.get("totalDistance"),
                    "total_energy_burned": elem.get("totalEnergyBurned"),
                    "source_name": elem.get("sourceName"),
                    "start_date": elem.get("startDate"),
                    "end_date": elem.get("endDate"),
                    "creation_date": elem.get("creationDate"),
                })
                elem.clear()

if workouts:
    df_workouts = spark.createDataFrame(workouts)
    (df_workouts.write
       .format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(f"{catalog}.{schema}.bronze_workouts"))
    print(f"Written {len(workouts):,} workouts to {catalog}.{schema}.bronze_workouts")
