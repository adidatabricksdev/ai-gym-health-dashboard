# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Daily Aggregated Health Metrics

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "health_dashboard")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from pyspark.sql import functions as F

# Build a spine of all dates with data
dates = spark.sql(f"""
    SELECT DISTINCT date FROM {catalog}.{schema}.silver_heart_rate
    UNION
    SELECT DISTINCT date FROM {catalog}.{schema}.silver_steps
    UNION
    SELECT DISTINCT date FROM {catalog}.{schema}.silver_activity_rings
""")

# COMMAND ----------

# Heart rate: daily avg, min, max
hr_daily = (spark.table(f"{catalog}.{schema}.silver_heart_rate")
    .groupBy("date").agg(
        F.round(F.avg("value_num"), 1).alias("avg_hr"),
        F.min("value_num").alias("min_hr"),
        F.max("value_num").alias("max_hr"),
    ))

# Resting HR: take last reading of the day (Apple Watch updates it once daily)
rhr_daily = (spark.table(f"{catalog}.{schema}.silver_resting_heart_rate")
    .groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("resting_hr")))

# HRV: daily avg
hrv_daily = (spark.table(f"{catalog}.{schema}.silver_hrv")
    .groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("hrv_ms")))

# VO2 Max: last reading per day (infrequent)
vo2_daily = (spark.table(f"{catalog}.{schema}.silver_vo2_max")
    .groupBy("date").agg(F.round(F.avg("value_num"), 2).alias("vo2_max")))

# Steps: sum
steps_daily = (spark.table(f"{catalog}.{schema}.silver_steps")
    .groupBy("date").agg(F.round(F.sum("value_num"), 0).alias("steps")))

# Distance: sum (km)
distance_daily = (spark.table(f"{catalog}.{schema}.silver_distance")
    .groupBy("date").agg(F.round(F.sum("value_num"), 2).alias("distance_km")))

# Active energy: sum (kcal)
energy_daily = (spark.table(f"{catalog}.{schema}.silver_active_energy")
    .groupBy("date").agg(F.round(F.sum("value_num"), 1).alias("active_energy_kcal")))

# Activity rings (already daily)
rings = spark.table(f"{catalog}.{schema}.silver_activity_rings").select(
    "date", "active_energy_burned", "active_energy_burned_goal",
    "exercise_time_mins", "exercise_time_goal_mins",
    "stand_hours", "stand_hours_goal",
    "move_ring_pct", "exercise_ring_pct", "stand_ring_pct"
)

# Sleep: total sleep per night (exclude "In Bed"), primary sleep window = longest block
sleep_daily = (spark.table(f"{catalog}.{schema}.silver_sleep")
    .filter(F.col("sleep_stage") != "In Bed")
    .groupBy("date").agg(
        F.round(F.sum("duration_hrs"), 2).alias("total_sleep_hrs"),
        F.round(F.sum(F.when(F.col("sleep_stage") == "Deep",  F.col("duration_hrs")).otherwise(0)), 2).alias("deep_sleep_hrs"),
        F.round(F.sum(F.when(F.col("sleep_stage") == "REM",   F.col("duration_hrs")).otherwise(0)), 2).alias("rem_sleep_hrs"),
        F.round(F.sum(F.when(F.col("sleep_stage") == "Core",  F.col("duration_hrs")).otherwise(0)), 2).alias("core_sleep_hrs"),
    ))

# Weight: last reading of the day
weight_daily = (spark.table(f"{catalog}.{schema}.silver_weight")
    .groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("weight_kg")))

body_fat_daily = (spark.table(f"{catalog}.{schema}.silver_body_fat")
    .groupBy("date").agg(F.round(F.avg("value_num"), 1).alias("body_fat_pct")))

# COMMAND ----------

# Join everything onto the date spine
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
    .orderBy("date")
)

(gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.gold_daily_health"))

print(f"Written gold_daily_health: {gold.count()} days")
display(gold.orderBy(F.col("date").desc()).limit(14))
