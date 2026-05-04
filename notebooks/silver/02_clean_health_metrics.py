# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Clean & Type Health Metrics

# COMMAND ----------

dbutils.widgets.text("catalog", "ai_gym")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("schema", "silver")

catalog       = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
schema        = dbutils.widgets.get("schema")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Use TRY_CAST so empty strings and non-numeric values become null instead of erroring
def safe_double(col_name):
    return F.expr(f"TRY_CAST(`{col_name}` AS DOUBLE)")

df = spark.table(f"{catalog}.{bronze_schema}.health_records")

df_clean = (df
    .withColumn("value_num", safe_double("value"))
    .withColumn("start_ts",  F.to_timestamp("start_date", "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("end_ts",    F.to_timestamp("end_date",   "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("date",      F.to_date("start_ts"))
    .filter(F.col("start_ts").isNotNull())
    .dropDuplicates(["type", "start_date", "end_date", "value", "source_name"])
)

def save(df_in, record_type, cols, table):
    (df_in.filter(F.col("type") == record_type)
          .select(*cols)
          .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable(f"{catalog}.{schema}.{table}"))
    print(f"  {catalog}.{schema}.{table}")

print("Writing silver tables:")
save(df_clean, "HKQuantityTypeIdentifierHeartRate",               ["date", "start_ts", "end_ts", "value_num", "source_name"], "heart_rate")
save(df_clean, "HKQuantityTypeIdentifierRestingHeartRate",        ["date", "start_ts", "end_ts", "value_num", "source_name"], "resting_heart_rate")
save(df_clean, "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",["date", "start_ts", "end_ts", "value_num", "source_name"], "hrv")
save(df_clean, "HKQuantityTypeIdentifierVO2Max",                  ["date", "start_ts", "value_num", "source_name"],           "vo2_max")
save(df_clean, "HKQuantityTypeIdentifierActiveEnergyBurned",      ["date", "start_ts", "end_ts", "value_num", "source_name"], "active_energy")
save(df_clean, "HKQuantityTypeIdentifierAppleExerciseTime",       ["date", "start_ts", "end_ts", "value_num", "source_name"], "exercise_time")
save(df_clean, "HKQuantityTypeIdentifierStepCount",               ["date", "start_ts", "end_ts", "value_num", "source_name"], "steps")
save(df_clean, "HKQuantityTypeIdentifierDistanceWalkingRunning",  ["date", "start_ts", "end_ts", "value_num", "source_name"], "distance")
save(df_clean, "HKQuantityTypeIdentifierBodyMass",                ["date", "start_ts", "value_num", "source_name"],           "weight")
save(df_clean, "HKQuantityTypeIdentifierBodyFatPercentage",       ["date", "start_ts", "value_num", "source_name"],           "body_fat")
save(df_clean, "HKQuantityTypeIdentifierBodyMassIndex",           ["date", "start_ts", "value_num", "source_name"],           "bmi")

# COMMAND ----------

sleep_stage_map = {
    "HKCategoryValueSleepAnalysisInBed":     "In Bed",
    "HKCategoryValueSleepAnalysisAsleep":    "Asleep",
    "HKCategoryValueSleepAnalysisAwake":     "Awake",
    "HKCategoryValueSleepAnalysisAsleepCore":"Core",
    "HKCategoryValueSleepAnalysisAsleepDeep":"Deep",
    "HKCategoryValueSleepAnalysisAsleepREM": "REM",
}
stage_map_expr = F.create_map([F.lit(x) for pair in sleep_stage_map.items() for x in pair])

(df_clean
    .filter(F.col("type") == "HKCategoryTypeIdentifierSleepAnalysis")
    .withColumn("sleep_stage",  stage_map_expr[F.col("value")])
    .withColumn("duration_hrs", F.round((F.unix_timestamp("end_ts") - F.unix_timestamp("start_ts")) / 3600, 3))
    .select("date", "start_ts", "end_ts", "sleep_stage", "duration_hrs", "source_name")
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.sleep"))
print(f"  {catalog}.{schema}.sleep")

# COMMAND ----------

(spark.table(f"{catalog}.{bronze_schema}.workouts")
    .withColumn("start_ts",      F.to_timestamp("start_date", "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("end_ts",        F.to_timestamp("end_date",   "yyyy-MM-dd HH:mm:ss Z"))
    .withColumn("date",          F.to_date("start_ts"))
    .withColumn("duration_mins", safe_double("duration_mins"))
    .withColumn("calories",      safe_double("total_energy_burned"))
    .withColumn("distance_km",   safe_double("total_distance"))
    .withColumn("avg_hr",        safe_double("avg_heart_rate"))
    .withColumn("activity",      F.regexp_replace("activity_type", "HKWorkoutActivityType", ""))
    .dropDuplicates(["start_ts", "end_ts", "activity"])
    .select("date", "start_ts", "end_ts", "activity", "duration_mins", "calories", "distance_km", "avg_hr", "source_name")
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.workouts"))
print(f"  {catalog}.{schema}.workouts")

# COMMAND ----------

(spark.table(f"{catalog}.{bronze_schema}.activity_summary")
    .withColumn("date",                      F.to_date("date"))
    .withColumn("active_energy_burned",      safe_double("active_energy_burned"))
    .withColumn("active_energy_burned_goal", safe_double("active_energy_burned_goal"))
    .withColumn("exercise_time_mins",        safe_double("exercise_time_mins"))
    .withColumn("exercise_time_goal_mins",   safe_double("exercise_time_goal_mins"))
    .withColumn("stand_hours",               safe_double("stand_hours"))
    .withColumn("stand_hours_goal",          safe_double("stand_hours_goal"))
    .withColumn("move_ring_pct",     F.round(F.expr("try_divide(active_energy_burned, active_energy_burned_goal)") * 100, 1))
    .withColumn("exercise_ring_pct", F.round(F.expr("try_divide(exercise_time_mins, exercise_time_goal_mins)")   * 100, 1))
    .withColumn("stand_ring_pct",    F.round(F.expr("try_divide(stand_hours, stand_hours_goal)")                 * 100, 1))
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.activity_rings"))
print(f"  {catalog}.{schema}.activity_rings")

print("\nAll silver tables written successfully")
