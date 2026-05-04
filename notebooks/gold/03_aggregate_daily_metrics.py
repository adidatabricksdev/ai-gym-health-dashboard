# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Daily Aggregated Health Metrics

# COMMAND ----------

dbutils.widgets.text("catalog", "ai_gym")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("schema", "gold")

catalog       = dbutils.widgets.get("catalog")
silver_schema = dbutils.widgets.get("silver_schema")
schema        = dbutils.widgets.get("schema")

# COMMAND ----------

from pyspark.sql import functions as F

dates = spark.sql(f"""
    SELECT DISTINCT date FROM {catalog}.{silver_schema}.heart_rate
    UNION SELECT DISTINCT date FROM {catalog}.{silver_schema}.steps
    UNION SELECT DISTINCT date FROM {catalog}.{silver_schema}.activity_rings
""")

# COMMAND ----------

hr_daily       = spark.table(f"{catalog}.{silver_schema}.heart_rate").groupBy("date").agg(
                     F.round(F.avg("value_num"), 1).alias("avg_hr"),
                     F.min("value_num").alias("min_hr"),
                     F.max("value_num").alias("max_hr"))
rhr_daily      = spark.table(f"{catalog}.{silver_schema}.resting_heart_rate").groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("resting_hr"))
hrv_daily      = spark.table(f"{catalog}.{silver_schema}.hrv").groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("hrv_ms"))
vo2_daily      = spark.table(f"{catalog}.{silver_schema}.vo2_max").groupBy("date").agg(F.round(F.avg("value_num"), 2).alias("vo2_max"))
steps_daily    = spark.table(f"{catalog}.{silver_schema}.steps").groupBy("date").agg(F.round(F.sum("value_num"), 0).alias("steps"))
distance_daily = spark.table(f"{catalog}.{silver_schema}.distance").groupBy("date").agg(F.round(F.sum("value_num"), 2).alias("distance_km"))
energy_daily   = spark.table(f"{catalog}.{silver_schema}.active_energy").groupBy("date").agg(F.round(F.sum("value_num"), 1).alias("active_energy_kcal"))
weight_daily   = spark.table(f"{catalog}.{silver_schema}.weight").groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("weight_kg"))
body_fat_daily = spark.table(f"{catalog}.{silver_schema}.body_fat").groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("body_fat_pct"))

rings = spark.table(f"{catalog}.{silver_schema}.activity_rings").select(
    "date", "active_energy_burned", "active_energy_burned_goal",
    "exercise_time_mins", "exercise_time_goal_mins",
    "stand_hours", "stand_hours_goal",
    "move_ring_pct", "exercise_ring_pct", "stand_ring_pct")

sleep_daily = (spark.table(f"{catalog}.{silver_schema}.sleep")
    .filter(F.col("sleep_stage") != "In Bed")
    .groupBy("date").agg(
        F.round(F.sum("duration_hrs"), 2).alias("total_sleep_hrs"),
        F.round(F.sum(F.when(F.col("sleep_stage") == "Deep", F.col("duration_hrs")).otherwise(0)), 2).alias("deep_sleep_hrs"),
        F.round(F.sum(F.when(F.col("sleep_stage") == "REM",  F.col("duration_hrs")).otherwise(0)), 2).alias("rem_sleep_hrs"),
        F.round(F.sum(F.when(F.col("sleep_stage") == "Core", F.col("duration_hrs")).otherwise(0)), 2).alias("core_sleep_hrs")))

# COMMAND ----------

gold = (dates
    .join(hr_daily,       "date", "left")
    .join(rhr_daily,      "date", "left")
    .join(hrv_daily,      "date", "left")
    .join(vo2_daily,      "date", "left")
    .join(steps_daily,    "date", "left")
    .join(distance_daily, "date", "left")
    .join(energy_daily,   "date", "left")
    .join(rings,          "date", "left")
    .join(sleep_daily,    "date", "left")
    .join(weight_daily,   "date", "left")
    .join(body_fat_daily, "date", "left")
    .orderBy("date"))

(gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.daily_health"))

print(f"Written {catalog}.{schema}.daily_health: {gold.count()} days")
display(gold.orderBy(F.col("date").desc()).limit(14))
