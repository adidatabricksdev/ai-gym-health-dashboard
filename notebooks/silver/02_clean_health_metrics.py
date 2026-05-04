# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Clean & Type Health Metrics

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "health_dashboard")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

df = spark.table(f"{catalog}.{schema}.bronze_health_records")

# Common cleaning: parse timestamps, cast value to double, dedup
df_clean = (df
    .withColumn("value_num",  F.col("value").cast(DoubleType()))
    .withColumn("start_ts",   F.to_timestamp("start_date",   "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("end_ts",     F.to_timestamp("end_date",     "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("date",       F.to_date("start_ts"))
    .filter(F.col("start_ts").isNotNull())
    .dropDuplicates(["type", "start_date", "end_date", "value", "source_name"])
)

def save(df_in, record_type, cols, table):
    (df_in.filter(F.col("type") == record_type)
          .select(*cols)
          .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable(f"{catalog}.{schema}.{table}"))
    print(f"  {table}")

print("Writing silver tables:")

# Heart rate (bpm)
save(df_clean, "HKQuantityTypeIdentifierHeartRate",
     ["date", "start_ts", "end_ts", "value_num", "source_name"],
     "silver_heart_rate")

# Resting heart rate (bpm)
save(df_clean, "HKQuantityTypeIdentifierRestingHeartRate",
     ["date", "start_ts", "end_ts", "value_num", "source_name"],
     "silver_resting_heart_rate")

# HRV (ms)
save(df_clean, "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
     ["date", "start_ts", "end_ts", "value_num", "source_name"],
     "silver_hrv")

# VO2 Max (mL/min·kg)
save(df_clean, "HKQuantityTypeIdentifierVO2Max",
     ["date", "start_ts", "value_num", "source_name"],
     "silver_vo2_max")

# Active energy (kcal)
save(df_clean, "HKQuantityTypeIdentifierActiveEnergyBurned",
     ["date", "start_ts", "end_ts", "value_num", "source_name"],
     "silver_active_energy")

# Exercise time (mins)
save(df_clean, "HKQuantityTypeIdentifierAppleExerciseTime",
     ["date", "start_ts", "end_ts", "value_num", "source_name"],
     "silver_exercise_time")

# Step count
save(df_clean, "HKQuantityTypeIdentifierStepCount",
     ["date", "start_ts", "end_ts", "value_num", "source_name"],
     "silver_steps")

# Walking/running distance (km)
save(df_clean, "HKQuantityTypeIdentifierDistanceWalkingRunning",
     ["date", "start_ts", "end_ts", "value_num", "source_name"],
     "silver_distance")

# Weight (kg)
save(df_clean, "HKQuantityTypeIdentifierBodyMass",
     ["date", "start_ts", "value_num", "source_name"],
     "silver_weight")

# Body fat %
save(df_clean, "HKQuantityTypeIdentifierBodyFatPercentage",
     ["date", "start_ts", "value_num", "source_name"],
     "silver_body_fat")

# BMI
save(df_clean, "HKQuantityTypeIdentifierBodyMassIndex",
     ["date", "start_ts", "value_num", "source_name"],
     "silver_bmi")

# COMMAND ----------

# Sleep — value is a string category, not a number
# Map Apple's sleep stage codes to readable labels
sleep_stage_map = {
    "HKCategoryValueSleepAnalysisInBed":       "In Bed",
    "HKCategoryValueSleepAnalysisAsleep":       "Asleep",
    "HKCategoryValueSleepAnalysisAwake":        "Awake",
    "HKCategoryValueSleepAnalysisAsleepCore":   "Core",
    "HKCategoryValueSleepAnalysisAsleepDeep":   "Deep",
    "HKCategoryValueSleepAnalysisAsleepREM":    "REM",
}
stage_map_expr = F.create_map([F.lit(x) for pair in sleep_stage_map.items() for x in pair])

df_sleep = (df_clean
    .filter(F.col("type") == "HKCategoryTypeIdentifierSleepAnalysis")
    .withColumn("sleep_stage", stage_map_expr[F.col("value")])
    .withColumn("duration_hrs",
        F.round((F.unix_timestamp("end_ts") - F.unix_timestamp("start_ts")) / 3600, 3))
    .select("date", "start_ts", "end_ts", "sleep_stage", "duration_hrs", "source_name")
)
(df_sleep.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.silver_sleep"))
print("  silver_sleep")

# COMMAND ----------

# Workouts
df_workouts = (spark.table(f"{catalog}.{schema}.bronze_workouts")
    .withColumn("start_ts",      F.to_timestamp("start_date",   "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("end_ts",        F.to_timestamp("end_date",     "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("date",          F.to_date("start_ts"))
    .withColumn("duration_mins", F.col("duration_mins").cast(DoubleType()))
    .withColumn("calories",      F.col("total_energy_burned").cast(DoubleType()))
    .withColumn("distance_km",   F.col("total_distance").cast(DoubleType()))
    .withColumn("avg_hr",        F.col("avg_heart_rate").cast(DoubleType()))
    .withColumn("activity",      F.regexp_replace("activity_type", "HKWorkoutActivityType", ""))
    .dropDuplicates(["start_ts", "end_ts", "activity"])
    .select("date", "start_ts", "end_ts", "activity", "duration_mins",
            "calories", "distance_km", "avg_hr", "source_name")
)
(df_workouts.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.silver_workouts"))
print("  silver_workouts")

# COMMAND ----------

# Activity summary (rings) — already one row per day
df_activity = (spark.table(f"{catalog}.{schema}.bronze_activity_summary")
    .withColumn("date",                      F.to_date("date"))
    .withColumn("active_energy_burned",      F.col("active_energy_burned").cast(DoubleType()))
    .withColumn("active_energy_burned_goal", F.col("active_energy_burned_goal").cast(DoubleType()))
    .withColumn("exercise_time_mins",        F.col("exercise_time_mins").cast(DoubleType()))
    .withColumn("exercise_time_goal_mins",   F.col("exercise_time_goal_mins").cast(DoubleType()))
    .withColumn("stand_hours",               F.col("stand_hours").cast(DoubleType()))
    .withColumn("stand_hours_goal",          F.col("stand_hours_goal").cast(DoubleType()))
    .withColumn("move_ring_pct",
        F.round(F.col("active_energy_burned") / F.col("active_energy_burned_goal") * 100, 1))
    .withColumn("exercise_ring_pct",
        F.round(F.col("exercise_time_mins") / F.col("exercise_time_goal_mins") * 100, 1))
    .withColumn("stand_ring_pct",
        F.round(F.col("stand_hours") / F.col("stand_hours_goal") * 100, 1))
)
(df_activity.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.silver_activity_rings"))
print("  silver_activity_rings")

print("\nAll silver tables written successfully")
