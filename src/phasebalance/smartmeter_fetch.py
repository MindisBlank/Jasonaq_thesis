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

POWER_PHASES = ["AMP_Power_L1_AVG_T0","AMP_Power_L2_AVG_T0","AMP_Power_L3_AVG_T0",] # time resolution is 10 min here
TIME_RES = "15 minutes"  # aggregation resolution


Voltage_PHASES = [
    "LPQ_Voltage_L1_AVG",
    "LPQ_Voltage_L2_AVG",
    "LPQ_Voltage_L3_AVG",
]

def load_substation_transformers_from_parquet(parquet_path: str | Path) -> list[tuple[int, str]]:
    """
    Read a parquet file and return a sorted list of (substation_id, transformer)
    pairs inferred from columns:
      - dnr_str (e.g. 'D0411' -> 411)
      - transformer (e.g. 'sp1'/'sp2' or 'all')

    Returns only sp1/sp2 pairs (drops 'all' and nulls).
    """
    parquet_path = Path(parquet_path)

    try:
        df = pd.read_parquet(parquet_path, columns=["dnr_str", "transformer"])
    except Exception:
        df = pd.read_parquet(parquet_path)

    if "dnr_str" not in df.columns:
        raise ValueError(
            f"'dnr_str' column not found in {parquet_path}. "
            f"Available columns: {list(df.columns)}"
        )
    if "transformer" not in df.columns:
        raise ValueError(
            f"'transformer' column not found in {parquet_path}. "
            f"Available columns: {list(df.columns)}"
        )

    d = df[["dnr_str", "transformer"]].dropna().copy()
    d["dnr_str"] = d["dnr_str"].astype(str).str.strip()
    d["transformer"] = d["transformer"].astype(str).str.strip().str.lower()

    # Keep only sp1/sp2 (drop 'all' or anything else)
    d = d[d["transformer"].isin(["sp1", "sp2"])]

    d["substation_id"] = d["dnr_str"].map(_parse_substation_name)
    d = d.dropna(subset=["substation_id"])
    d["substation_id"] = d["substation_id"].astype(int)

    pairs = (
        d[["substation_id", "transformer"]]
        .drop_duplicates()
        .sort_values(["substation_id", "transformer"])
        .apply(lambda r: (int(r["substation_id"]), str(r["transformer"])), axis=1)
        .tolist()
    )

    print(f"Found {len(pairs)} (substation_id, transformer) pairs in parquet: {parquet_path}")
    print("First few:", pairs[:10])
    return pairs

def fetch_and_save_smartmeter_voltage(
    substation_ids,
    start_date: str,
    end_date: str,
    output_path: str,
    time_res: str = TIME_RES,
    substation_transformers: list[tuple[int, str]] | None = None,  # 👈 NEW
):
    """
    Fetch smartmeter voltages for given substations and time window, aggregate to
    `time_res`, and save to a local file.

    Output columns:
      substation_id, transformer, ts, V_a, V_b, V_c, per_completed

    NOTE:
      This expects your _build_meteringpoint_frames(...) to return mp_completed/mp_status
      that include a 'transformer' column derived from metering_points.dspennir (1->sp1, 2->sp2).
    """
    spark = get_spark()

    # Re-use mapping + per_completed from your existing helper (now transformer-aware)
    mp_completed, mp_status = _build_meteringpoint_frames(spark, substation_ids,substation_transformers=substation_transformers,)

    sm_ts = spark.read.table("veiturdata_base_prd.ami.smartmeter")

    # 1) Join smartmeter with Completed metering points
    joined = (
        sm_ts.withColumn("meteringpointid", F.col("meteringpointid").cast("string"))
        .join(mp_completed, F.col("meteringpointid") == F.col("mp_id"), "inner")
        .filter(
            (F.col("timestamp") >= F.lit(start_date))
            & (F.col("timestamp") <= F.lit(end_date))
            & (F.col("validity") == F.lit("Valid"))
            & (F.col("resulttype").isin(*Voltage_PHASES))
        )
        .select(
            F.to_timestamp("timestamp").alias("ts"),
            "resulttype",
            F.col("value").cast("double").alias("value_V"),
            "meteringpointid",
            "substation_id",
            "transformer",  # 👈 NEW
        )
    )

    # 2) Deduplicate per (substation, transformer, meter, phase, ts)
    w = Window.partitionBy(
        "substation_id", "transformer", "meteringpointid", "resulttype", "ts"
    ).orderBy(F.col("ts"))

    joined_dedup = (
        joined.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # 3) Aggregate voltages to time_res using average per (substation, transformer) & phase
    v_per_phase_res = (
        joined_dedup
        .withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res", "resulttype")
        .agg(F.avg("value_V").alias("V_res"))
    )

    # 4) Pivot phases → V_a / V_b / V_c
    v_pivot = (
        v_per_phase_res.groupBy("substation_id", "transformer", "ts_res")
        .pivot("resulttype", Voltage_PHASES)
        .agg(F.first("V_res"))
        .withColumnRenamed("ts_res", "ts")
        .orderBy("substation_id", "transformer", "ts")
    )

    result = (
        v_pivot
        .withColumnRenamed(Voltage_PHASES[0], "V_a")
        .withColumnRenamed(Voltage_PHASES[1], "V_b")
        .withColumnRenamed(Voltage_PHASES[2], "V_c")
    )

    # 5) Attach per_completed per (substation, transformer)
    result = result.join(
        mp_status.select("substation_id", "transformer", "per_completed"),
        on=["substation_id", "transformer"],
        how="left",
    )

    # 6) Final column order
    result = (
        result.select(
            "substation_id",
            "transformer",
            "ts",
            "V_a",
            "V_b",
            "V_c",
            "per_completed",
        )
        .orderBy("substation_id", "transformer", "ts")
    )

    # 7) Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = result.toPandas()
    pdf.attrs.clear()

    if output_path.suffix.lower() == ".parquet":
        pdf.to_parquet(output_path, index=False)
    else:
        pdf.to_csv(output_path, index=False)

    print(f"Saved {len(pdf)} rows to {output_path}")
    return pdf

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

def load_substations_from_parquet(parquet_path: str | Path) -> list[int]:
    """
    Read a parquet file and return a sorted list of distinct substation ids
    inferred from the 'dnr_str' column (e.g. 'D0411' -> 411).

    Also prints how many unique dnr_str are present (useful sanity check).
    """
    parquet_path = Path(parquet_path)

    # Read only the dnr_str column if possible (faster on big files)
    try:
        df = pd.read_parquet(parquet_path, columns=["dnr_str"])
    except Exception:
        df = pd.read_parquet(parquet_path)

    if "dnr_str" not in df.columns:
        raise ValueError(
            f"'dnr_str' column not found in {parquet_path}. "
            f"Available columns: {list(df.columns)}"
        )

    dnr_clean = (
        df["dnr_str"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    unique_dnr = sorted(dnr_clean.drop_duplicates().tolist())
    print(f"Found {len(unique_dnr)} unique dnr_str in parquet: {parquet_path}")
    print("First few dnr_str:", unique_dnr[:10])

    # Convert Dxxxx -> int substation_id using your existing parser
    substation_ids = (
        pd.Series(unique_dnr)
        .map(_parse_substation_name)
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return substation_ids

def _build_meteringpoint_frames(spark, substation_ids, substation_transformers: list[tuple[int, str]] | None = None):
    """
    Returns:
      mp_completed: metering points (only Completed/non-planned) with mp_id + substation_id + transformer
      mp_status: per-(substation, transformer) per_completed + counts

    If substation_transformers is provided ([(substation_id,'sp1'),...]),
    we filter metering points down to ONLY those (substation, transformer) pairs.
    """
    mp = spark.read.table("veiturdata_enriched_prd.utilities.metering_points")

    mp_base = (
        mp.filter(
            (F.col("dreifistodvanumer").isin(substation_ids))
            & (F.col("lh_is_current") == F.lit(True))
            & (F.col("er_flutningur") == F.lit("N"))
        )
        .withColumn(
            "transformer",
            F.when(F.col("dspennir") == F.lit(1), F.lit("sp1"))
             .when(F.col("dspennir") == F.lit(2), F.lit("sp2"))
             .otherwise(F.lit(None))
        )
    )

    # 👇 NEW: hard filter to only pairs you actually have Janitza for
    if substation_transformers:
        allowed_df = (
            spark.createDataFrame(substation_transformers, ["substation_id", "transformer"])
            .dropDuplicates()
            .withColumnRenamed("transformer", "transformer_allowed")
        )

        mp_base = mp_base.join(
            allowed_df,
            (mp_base["dreifistodvanumer"] == allowed_df["substation_id"])
            & (mp_base["transformer"] == allowed_df["transformer_allowed"]),
            "inner",
        ).drop("substation_id", "transformer_allowed")


    mp_status = (
        mp_base.groupBy("dreifistodvanumer", "transformer")
        .agg(
            F.sum(F.when(F.col("uppsetningar_stada") == "Completed", 1).otherwise(0)).alias("n_completed"),
            F.count("*").alias("n_total_mps"),
        )
        .withColumn(
            "per_completed",
            F.when(F.col("n_total_mps") > 0, F.col("n_completed") / F.col("n_total_mps"))
             .otherwise(F.lit(None)),
        )
        .select(
            F.col("dreifistodvanumer").alias("substation_id"),
            "transformer",
            "per_completed",
            "n_completed",
            "n_total_mps",
        )
    )

    mp_completed = (
        mp_base.filter(F.col("uppsetningar_stada") != F.lit("planned"))
        .select(
            F.col("husveita_fastanumer").cast("string").alias("mp_id"),
            F.col("dreifistodvanumer").alias("substation_id"),
            "transformer",
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
    substation_transformers: list[tuple[int, str]] | None = None,
):
    """
    Fetch smartmeter data for given substations and time window, aggregate to
    `time_res`, and save to a local file.

    Output columns:
      substation_id, ts, I_a, I_b, I_c, I_total, n_mps, per_completed
    """
    spark = get_spark()

    # --- Metering point info (mapping + per_completed) ---
    mp_completed, mp_status = _build_meteringpoint_frames(spark, substation_ids, substation_transformers=substation_transformers)

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
            "transformer",   # 👈 NEW
        )
    )

    # 2) Deduplicate per (substation, meter, phase, ts)
    w = Window.partitionBy(
    "substation_id", "transformer", "meteringpointid", "resulttype", "ts"
    ).orderBy(F.col("ts"))


    joined_dedup = (
        joined.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # 3) Sum current per (substation, ts, phase) at native resolution
    sum_per_phase = (
    joined_dedup.groupBy("substation_id", "transformer", "ts", "resulttype")
    .agg(F.sum("value_A").alias("sum_A"))
)


    # 3b) Aggregate currents to time_res (e.g. 15 minutes) using avg over window
    sum_per_phase_res = (
        sum_per_phase.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res", "resulttype")
        .agg(F.avg("sum_A").alias("sum_A_res"))
    )

    # 3c) n_mps at aggregated resolution: distinct meters per (substation, ts_res)
    n_mps_res = (
        joined_dedup.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res")
        .agg(F.countDistinct("meteringpointid").alias("n_mps"))
        .withColumnRenamed("ts_res", "ts")  # <- add this
    )

    # 4) Pivot phases → I_a / I_b / I_c
    substation_currents_res = (
        sum_per_phase_res.groupBy("substation_id", "transformer", "ts_res")
        .pivot("resulttype", phases)
        .agg(F.first("sum_A_res"))
        .withColumnRenamed("ts_res", "ts")
        .orderBy("substation_id", "transformer", "ts")
    )

    # Rename phase columns to I_a / I_b / I_c
    result = (
        substation_currents_res
        .withColumnRenamed(phases[0], "I_a")
        .withColumnRenamed(phases[1], "I_b")
        .withColumnRenamed(phases[2], "I_c")
    )

    # 5) Join n_mps and per_completed
    result = result.join(n_mps_res, on=["substation_id", "transformer", "ts"], how="left")
    result = result.join(
        mp_status.select("substation_id", "transformer", "per_completed"),
        on=["substation_id", "transformer"],
        how="left",
    )


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
            "transformer",   # 👈 NEW
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

    janitza_parquet_path = Path("data/janitza_3phase_I_by_transformer_20250801_0000_20251101_0000.parquet")

    # 1) Build list of distinct substation IDs from devices.csv
    pairs = load_substation_transformers_from_parquet(janitza_parquet_path)
    substation_ids = sorted({sid for sid, _ in pairs})

    print(f"Found {len(substation_ids)} substations from parquet pairs.")
    print("First few substations:", substation_ids[:10])
    print("First few (substation, transformer) pairs:", pairs[:10])

    # define the profile
    Profile = PHASES

    # 2) Define time window
    start = "2025-08-01 00:00:00"
    end   = "2025-11-01 00:00:00"

    #Turn start/end into compact tags for the filename
    fmt_in = "%Y-%m-%d %H:%M:%S"
    start_dt = datetime.strptime(start, fmt_in)
    end_dt   = datetime.strptime(end,   fmt_in)
    start_tag = start_dt.strftime("%Y%m%d")
    end_tag   = end_dt.strftime("%Y%m%d")


    # 3) Build output path dynamically
    out_path = Path("data") / f"smartmeter_15min_all_by_transformer_{start_tag}_{end_tag}_{Profile}.parquet"

    fetch_and_save_smartmeter(
        substation_ids=substation_ids,
        start_date=start,
        end_date=end,
        output_path=str(out_path),
        time_res="15 minutes",
        phases=Profile,
        substation_transformers=pairs,
    )

    # fetch_and_save_smartmeter_voltage(
    #     substation_ids=substation_ids,
    #     start_date=start,
    #     end_date=end,
    #     output_path=str(out_path),
    #     time_res="10 minutes",
    #     substation_transformers=pairs,
    # )
