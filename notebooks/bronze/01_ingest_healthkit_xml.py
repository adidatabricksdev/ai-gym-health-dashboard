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

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_exports")

# COMMAND ----------

import zipfile
import xml.etree.ElementTree as ET

# Copy zip from volume to local driver
zip_files = [f for f in dbutils.fs.ls(volume_path) if f.name.endswith(".zip")]
assert zip_files, f"No export.zip found in {volume_path} — upload your Apple Health export first"
latest_zip = sorted(zip_files, key=lambda f: f.modificationTime, reverse=True)[0].path

local_zip = "/tmp/apple_health_export.zip"
dbutils.fs.cp(latest_zip, f"file:{local_zip}")
print(f"Processing: {latest_zip}")

# COMMAND ----------

# Parse all Record, Workout, and ActivitySummary elements in a single pass
records = []
workouts = []
activity_summaries = []

with zipfile.ZipFile(local_zip, "r") as zf:
    with zf.open("apple_health_export/export.xml") as f:
        for event, elem in ET.iterparse(f, events=["end"]):

            if elem.tag == "Record":
                records.append({
                    "type":          elem.get("type"),
                    "source_name":   elem.get("sourceName"),
                    "unit":          elem.get("unit"),
                    "value":         elem.get("value"),
                    "start_date":    elem.get("startDate"),
                    "end_date":      elem.get("endDate"),
                    "creation_date": elem.get("creationDate"),
                })

            elif elem.tag == "Workout":
                # WorkoutStatistics are child elements
                stats = {s.get("type"): s.get("sum") or s.get("average")
                         for s in elem.findall("WorkoutStatistics")}
                workouts.append({
                    "activity_type":        elem.get("workoutActivityType"),
                    "duration_mins":        elem.get("duration"),
                    "source_name":          elem.get("sourceName"),
                    "start_date":           elem.get("startDate"),
                    "end_date":             elem.get("endDate"),
                    "creation_date":        elem.get("creationDate"),
                    "total_energy_burned":  stats.get("HKQuantityTypeIdentifierActiveEnergyBurned"),
                    "total_distance":       stats.get("HKQuantityTypeIdentifierDistanceWalkingRunning"),
                    "avg_heart_rate":       stats.get("HKQuantityTypeIdentifierHeartRate"),
                })

            elif elem.tag == "ActivitySummary":
                activity_summaries.append({
                    "date":                        elem.get("dateComponents"),
                    "active_energy_burned":        elem.get("activeEnergyBurned"),
                    "active_energy_burned_goal":   elem.get("activeEnergyBurnedGoal"),
                    "exercise_time_mins":          elem.get("appleExerciseTime"),
                    "exercise_time_goal_mins":     elem.get("appleExerciseTimeGoal"),
                    "stand_hours":                 elem.get("appleStandHours"),
                    "stand_hours_goal":            elem.get("appleStandHoursGoal"),
                })

            elem.clear()

print(f"Records:            {len(records):,}")
print(f"Workouts:           {len(workouts):,}")
print(f"Activity summaries: {len(activity_summaries):,}")

# COMMAND ----------

df_records = spark.createDataFrame(records)
(df_records.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.bronze_health_records"))

df_workouts = spark.createDataFrame(workouts)
(df_workouts.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.bronze_workouts"))

df_activity = spark.createDataFrame(activity_summaries)
(df_activity.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.bronze_activity_summary"))

print("Bronze tables written successfully")
display(df_records.groupBy("type").count().orderBy("count", ascending=False))
