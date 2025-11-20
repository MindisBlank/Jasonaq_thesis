# src/phasebalance/smartmeter_fetch.py

from __future__ import annotations
from typing import Iterable, List  # if you want type hints
from datetime import datetime
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pathlib import Path

import pandas as pd

# Result types we treat as phases A/B/C
PHASES = [
    "LPQ_Current_L1_AVG",
    "LPQ_Current_L2_AVG",
    "LPQ_Current_L3_AVG",
]

TIME_RES = "15 minutes"  # aggregation resolution

def _parse_substation_name(dnr_str: str) -> int | None:
    """
    Convert strings like 'D1070', 'D0411' to integer substation IDs 1070, 411.

    Rules:
      - must start with 'D'
      - strip 'D' and leading zeros
      - return None if it can't be parsed cleanly
    """
    if not isinstance(dnr_str, str):
        return None

    dnr_str = dnr_str.strip()
    if not dnr_str or not dnr_str.startswith("D"):
        return None

    # Take everything after the 'D'
    digits = dnr_str[1:]

    # Strip leading zeros: '0411' -> '411', '0000' -> ''
    digits = digits.lstrip("0")
    if digits == "":
        return None

    try:
        return int(digits)
    except ValueError:
        return None

def load_substations_from_devices(devices_csv_path: str | Path) -> list[int]:
    """
    Read devices.csv and return a sorted list of distinct substation ids
    inferred from the 'name' column (e.g. 'D0411' -> 411).
    """
    devices_csv_path = Path(devices_csv_path)
    df = pd.read_csv(devices_csv_path)

    if "dnr_str" not in df.columns:
        raise ValueError(
            f"'dnr_str' column not found in {devices_csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df["substation_id"] = df["dnr_str"].map(_parse_substation_name)

    # Drop rows that couldn't be parsed to a valid int
    subs = (
        df["substation_id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return subs

def get_spark():
    """
    Create a Databricks-connected Spark session.
    Relies on Databricks Connect config set up by the VS Code extension.
    """
    return DatabricksSession.builder.getOrCreate()


def _build_meteringpoint_frames(spark, substation_ids):
    """
    Returns:
      mp_completed: metering points (only Completed) with mp_id + substation_id
      mp_status: per-substation per_completed + counts
    """
    mp = spark.read.table("veiturdata_enriched_prd.utilities.metering_points")

    # Base filter: current + non-transport + in our substation list
    mp_base = (
        mp.filter(
            (F.col("dreifistodvanumer").isin(substation_ids))
            & (F.col("lh_is_current") == F.lit(True))
            & (F.col("er_flutningur") == F.lit("N"))
        )
    )

    # Per-substation fraction of Completed installations
    mp_status = (
        mp_base.groupBy("dreifistodvanumer")
        .agg(
            F.sum(
                F.when(F.col("uppsetningar_stada") == "Completed", 1).otherwise(0)
            ).alias("n_completed"),
            F.count("*").alias("n_total_mps"),
        )
        .withColumn(
            "per_completed",
            F.when(
                F.col("n_total_mps") > 0,
                F.col("n_completed") / F.col("n_total_mps"),
            ).otherwise(F.lit(None)),
        )
        .select(
            F.col("dreifistodvanumer").alias("substation_id"),
            "per_completed",
            "n_completed",
            "n_total_mps",
        )
    )

    # Only metering points with uppsetningar_stada == 'Completed'
    mp_completed = (
        mp_base.filter(F.col("uppsetningar_stada") == "Completed")
        .select(
            F.col("husveita_fastanumer").cast("string").alias("mp_id"),
            F.col("dreifistodvanumer").alias("substation_id"),
        )
    )

    return mp_completed, mp_status


def fetch_and_save_smartmeter(
    substation_ids,
    start_date: str,
    end_date: str,
    output_path: str,
    time_res: str = TIME_RES,
    phases: list[str] = PHASES,
):
    """
    Fetch smartmeter data for given substations and time window, aggregate to
    `time_res`, and save to a local file.

    Output columns:
      substation_id, ts, I_a, I_b, I_c, I_total, n_mps, per_completed
    """
    spark = get_spark()

    # --- Metering point info (mapping + per_completed) ---
    mp_completed, mp_status = _build_meteringpoint_frames(spark, substation_ids)

    # --- Smartmeter raw table ---
    sm_ts = spark.read.table("veiturdata_base_prd.ami.smartmeter")

    # 1) Join smartmeter with Completed metering points
    joined = (
        sm_ts.withColumn("meteringpointid", F.col("meteringpointid").cast("string"))
        .join(mp_completed, F.col("meteringpointid") == F.col("mp_id"), "inner")
        .filter(
            (F.col("timestamp") >= F.lit(start_date))
            & (F.col("timestamp") <= F.lit(end_date))
            & (F.col("validity") == F.lit("Valid"))
            & (F.col("resulttype").isin(*phases))
        )
        .select(
            F.to_timestamp("timestamp").alias("ts"),
            "resulttype",
            F.col("value").cast("double").alias("value_A"),
            "meteringpointid",
            "substation_id",
        )
    )

    # 2) Deduplicate per (substation, meter, phase, ts)
    w = Window.partitionBy(
        "substation_id", "meteringpointid", "resulttype", "ts"
    ).orderBy(F.col("ts"))

    joined_dedup = (
        joined.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # 3) Sum current per (substation, ts, phase) at native resolution
    sum_per_phase = (
        joined_dedup.groupBy("substation_id", "ts", "resulttype")
        .agg(F.sum("value_A").alias("sum_A"))
    )

    # 3b) Aggregate currents to time_res (e.g. 15 minutes) using avg over window
    sum_per_phase_res = (
        sum_per_phase.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "ts_res", "resulttype")
        .agg(F.avg("sum_A").alias("sum_A_res"))
    )

    # 3c) n_mps at aggregated resolution: distinct meters per (substation, ts_res)
    n_mps_res = (
        joined_dedup.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "ts_res")
        .agg(F.countDistinct("meteringpointid").alias("n_mps"))
        .withColumnRenamed("ts_res", "ts")  # <- add this
    )

    # 4) Pivot phases → I_a / I_b / I_c
    substation_currents_res = (
        sum_per_phase_res.groupBy("substation_id", "ts_res")
        .pivot("resulttype", phases)
        .agg(F.first("sum_A_res"))
        .withColumnRenamed("ts_res", "ts")
        .orderBy("substation_id", "ts")
    )

    # Rename phase columns to I_a / I_b / I_c
    result = (
        substation_currents_res
        .withColumnRenamed(phases[0], "I_a")
        .withColumnRenamed(phases[1], "I_b")
        .withColumnRenamed(phases[2], "I_c")
    )

    # 5) Join n_mps and per_completed
    result = result.join(n_mps_res, on=["substation_id", "ts"], how="left")
    result = result.join(mp_status.select("substation_id", "per_completed"), on="substation_id", how="left")

    # 6) Total current
    result = result.withColumn(
        "I_total",
        F.coalesce(F.col("I_a"), F.lit(0.0))
        + F.coalesce(F.col("I_b"), F.lit(0.0))
        + F.coalesce(F.col("I_c"), F.lit(0.0)),
    )

    # 7) Final column order
    result = (
        result.select(
            "substation_id",
            "ts",
            "I_a",
            "I_b",
            "I_c",
            "I_total",
            "n_mps",
            "per_completed",
        )
        .orderBy("substation_id", "ts")
    )

    # 8) Save to disk (Parquet if extension is .parquet, else CSV)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = result.toPandas()

    #Workaround: Spark Connect adds non-JSON-serializable objects into attrs
    pdf.attrs.clear()

    if output_path.suffix.lower() == ".parquet":
        pdf.to_parquet(output_path, index=False)
    else:
        pdf.to_csv(output_path, index=False)

    print(f"Saved {len(pdf)} rows to {output_path}")

    return pdf


if __name__ == "__main__":
    # Path to devices.csv – adjust to wherever it lives in your repo
    devices_csv_path = Path("metadata/devices.csv")  # or Path("data/devices.csv")

    # 1) Build list of distinct substation IDs from devices.csv
    substation_ids = load_substations_from_devices(devices_csv_path)

    #substation_ids = substation_ids[:2]  # TEMP: only first 5 for testing

    print(f"Found {len(substation_ids)} substations from devices.csv")
    print("First few:", substation_ids[:10])


    # 2) Define time window
    start = "2025-11-01 12:00:00"
    end   = "2025-11-03 12:00:00"

    #Turn start/end into compact tags for the filename
    fmt_in = "%Y-%m-%d %H:%M:%S"
    start_dt = datetime.strptime(start, fmt_in)
    end_dt   = datetime.strptime(end,   fmt_in)
    start_tag = start_dt.strftime("%Y%m%d")
    end_tag   = end_dt.strftime("%Y%m%d")


    # 3) Build output path dynamically
    out_path = Path("data") / f"smartmeter_15min_all_from_devices_{start_tag}_{end_tag}.parquet"

    fetch_and_save_smartmeter(
        substation_ids=substation_ids,
        start_date=start,
        end_date=end,
        output_path=str(out_path),
    )
