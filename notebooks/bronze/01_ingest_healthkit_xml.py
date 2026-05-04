# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: Ingest Apple Health XML → Delta

# COMMAND ----------

dbutils.widgets.text("catalog", "ai_gym")
dbutils.widgets.text("schema", "bronze")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_path = f"/Volumes/{catalog}/{schema}/raw_exports"

# COMMAND ----------

import zipfile
import xml.etree.ElementTree as ET

zip_files = [f for f in dbutils.fs.ls(volume_path) if f.name.endswith(".zip")]
assert zip_files, f"No export.zip found in {volume_path} — upload your Apple Health export first"
latest_zip_info = sorted(zip_files, key=lambda f: f.modificationTime, reverse=True)[0]
# Unity Catalog volumes are directly accessible via POSIX path — no copy to /tmp needed
posix_path = latest_zip_info.path.replace("dbfs:", "")
print(f"Processing: {posix_path}")

# COMMAND ----------

records = []
workouts = []
activity_summaries = []

with zipfile.ZipFile(posix_path, "r") as zf:
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
                stats = {s.get("type"): s.get("sum") or s.get("average")
                         for s in elem.findall("WorkoutStatistics")}
                workouts.append({
                    "activity_type":       elem.get("workoutActivityType"),
                    "duration_mins":       elem.get("duration"),
                    "source_name":         elem.get("sourceName"),
                    "start_date":          elem.get("startDate"),
                    "end_date":            elem.get("endDate"),
                    "creation_date":       elem.get("creationDate"),
                    "total_energy_burned": stats.get("HKQuantityTypeIdentifierActiveEnergyBurned"),
                    "total_distance":      stats.get("HKQuantityTypeIdentifierDistanceWalkingRunning"),
                    "avg_heart_rate":      stats.get("HKQuantityTypeIdentifierHeartRate"),
                })

            elif elem.tag == "ActivitySummary":
                activity_summaries.append({
                    "date":                      elem.get("dateComponents"),
                    "active_energy_burned":      elem.get("activeEnergyBurned"),
                    "active_energy_burned_goal": elem.get("activeEnergyBurnedGoal"),
                    "exercise_time_mins":        elem.get("appleExerciseTime"),
                    "exercise_time_goal_mins":   elem.get("appleExerciseTimeGoal"),
                    "stand_hours":               elem.get("appleStandHours"),
                    "stand_hours_goal":          elem.get("appleStandHoursGoal"),
                })

            elem.clear()

print(f"Records:            {len(records):,}")
print(f"Workouts:           {len(workouts):,}")
print(f"Activity summaries: {len(activity_summaries):,}")

# COMMAND ----------

# Coerce None to empty string so Spark can infer types on sparse fields
def nulls_to_empty(rows):
    return [{k: (v if v is not None else "") for k, v in row.items()} for row in rows]

spark.createDataFrame(nulls_to_empty(records)).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.health_records")
spark.createDataFrame(nulls_to_empty(workouts)).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.workouts")
spark.createDataFrame(nulls_to_empty(activity_summaries)).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.activity_summary")

print(f"Written: {catalog}.{schema}.health_records / workouts / activity_summary")
