# src/phasebalance/smartmeter_fetch.py

from __future__ import annotations
from typing import Iterable, List  # if you want type hints
from datetime import datetime
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pathlib import Path
from typing import Optional, Sequence,Iterable
import pandas as pd

# Result types we treat as phases A/B/C
PHASES = [
    "LPQ_Current_L1_AVG",
    "LPQ_Current_L2_AVG",
    "LPQ_Current_L3_AVG",
]

POWER_PHASES = ["AMP_Power_L1_AVG_T0","AMP_Power_L2_AVG_T0","AMP_Power_L3_AVG_T0",] # time resolution is 10 min here
TIME_RES = "15 minutes"  # aggregation resolution

POWER_PHS = ["AMP_Power_L1_AVG_T0","AMP_Power_L2_AVG_T0","AMP_Power_L3_AVG_T0"]
E_Q_PLUS  = "LP1_ReactiveEnergyPlus_CUM_T0"
E_Q_MINUS = "LP1_ReactiveEnergyMinus_CUM_T0"
E_P_PLUS  = "LP1_ActiveEnergyPlus_CUM_T0"
E_P_MINUS = "LP1_ActiveEnergyMinus_CUM_T0"
V_PHS     = ["LPQ_Voltage_L1_AVG","LPQ_Voltage_L2_AVG","LPQ_Voltage_L3_AVG"]


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
            # join key used everywhere else
            F.col("husveita_fastanumer").cast("string").alias("mp_id"),

            # existing fields used by your current logic
            F.col("dreifistodvanumer").alias("substation_id"),
            "transformer",

            # NEW: requested metadata from metering_points
            F.col("tegund").cast("string").alias("tegund"),
            F.col("husveita_fastanumer").cast("string").alias("husveita_fastanumer"),
            F.col("kennitalamaelistadar").cast("string").alias("kennitalamaelistadar"),
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
    *,
    power_phases: list[str] = POWER_PHS,
    voltage_phases: list[str] = V_PHS,
    e_q_plus: str = E_Q_PLUS,
    e_q_minus: str = E_Q_MINUS,
    assume_power_kw: bool = False,   # if True: kW/kvar -> kVA; convert to W with *1000 when computing I
):
    """
    Fetch smartmeter data for given substations and time window, aggregate to `time_res`,
    and save to disk.

    Existing outputs:
      I_a, I_b, I_c, I_total (sum of RMS magnitudes), n_mps, per_completed,
      n_eligible, n_total_mps

    New outputs:
      P_a, P_b, P_c, P_total (from AMP_Power_*),
      Q_total (from delta of LP1_ReactiveEnergy +/- cumulative),
      V_ph_avg (avg of phase-to-neutral voltages),
      S_total, I_est (power-based current estimate)

    Notes:
      - I_total is still your "sum of RMS magnitudes" aggregation.
      - I_est is an apples-to-apples-ish feeder current estimate derived from aggregate S/V.
    """
    spark = get_spark()

    mp_completed, mp_status = _build_meteringpoint_frames(
        spark, substation_ids, substation_transformers=substation_transformers
    )

    sm_ts = spark.read.table("veiturdata_base_prd.ami.smartmeter")

    wanted_resulttypes = list(phases) + list(power_phases) + list(voltage_phases) + [e_q_plus, e_q_minus]

    joined = (
        sm_ts.withColumn("meteringpointid", F.col("meteringpointid").cast("string"))
        .join(mp_completed, F.col("meteringpointid") == F.col("mp_id"), "inner")
        .filter(
            (F.col("timestamp") >= F.lit(start_date))
            & (F.col("timestamp") <= F.lit(end_date))
            & (F.col("validity") == F.lit("Valid"))
            & (F.col("resulttype").isin(*wanted_resulttypes))
        )
        .select(
            F.to_timestamp("timestamp").alias("ts"),
            "resulttype",
            F.col("value").cast("double").alias("val"),
            F.col("meteringpointid").alias("meteringpointid"),
            "substation_id",
            "transformer",
        )
    )

    # Deduplicate per (substation, transformer, meter, resulttype, ts)
    w_dedup = Window.partitionBy(
        "substation_id", "transformer", "meteringpointid", "resulttype", "ts"
    ).orderBy(F.col("ts"))

    joined_dedup = (
        joined.withColumn("rn", F.row_number().over(w_dedup))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # ------------------------------------------------------------------
    # A) CURRENT aggregation (your existing method)
    # ------------------------------------------------------------------
    cur = joined_dedup.filter(F.col("resulttype").isin(*phases))

    # n_mps = number of meters that reported a CURRENT record in that bin
    n_mps_res = (
        cur.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res")
        .agg(F.countDistinct("meteringpointid").alias("n_mps"))
        .withColumnRenamed("ts_res", "ts")
    )

    # sum across meters at native ts then average inside 15-min bin (same as your pipeline)
    sum_per_phase = (
        cur.groupBy("substation_id", "transformer", "ts", "resulttype")
        .agg(F.sum("val").alias("sum_I"))
    )

    sum_per_phase_res = (
        sum_per_phase.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res", "resulttype")
        .agg(F.avg("sum_I").alias("I_res"))
    )

    cur_pivot = (
        sum_per_phase_res.groupBy("substation_id", "transformer", "ts_res")
        .pivot("resulttype", phases)
        .agg(F.first("I_res"))
        .withColumnRenamed("ts_res", "ts")
    )

    cur_pivot = (
        cur_pivot
        .withColumnRenamed(phases[0], "I_a")
        .withColumnRenamed(phases[1], "I_b")
        .withColumnRenamed(phases[2], "I_c")
    )

    cur_pivot = cur_pivot.withColumn(
        "I_total",
        F.coalesce(F.col("I_a"), F.lit(0.0))
        + F.coalesce(F.col("I_b"), F.lit(0.0))
        + F.coalesce(F.col("I_c"), F.lit(0.0)),
    )

    # ------------------------------------------------------------------
    # B) POWER aggregation (P_total from AMP_Power_*), aggregated like currents
    # ------------------------------------------------------------------
    p = joined_dedup.filter(F.col("resulttype").isin(*power_phases))

    sumP_per_phase = (
        p.groupBy("substation_id", "transformer", "ts", "resulttype")
        .agg(F.sum("val").alias("sum_P"))
    )

    sumP_per_phase_res = (
        sumP_per_phase.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res", "resulttype")
        .agg(F.avg("sum_P").alias("P_res"))
    )

    p_pivot = (
        sumP_per_phase_res.groupBy("substation_id", "transformer", "ts_res")
        .pivot("resulttype", power_phases)
        .agg(F.first("P_res"))
        .withColumnRenamed("ts_res", "ts")
    )

    p_pivot = (
        p_pivot
        .withColumnRenamed(power_phases[0], "P_a")
        .withColumnRenamed(power_phases[1], "P_b")
        .withColumnRenamed(power_phases[2], "P_c")
        .withColumn(
            "P_total",
            F.coalesce(F.col("P_a"), F.lit(0.0))
            + F.coalesce(F.col("P_b"), F.lit(0.0))
            + F.coalesce(F.col("P_c"), F.lit(0.0)),
        )
    )

    # ------------------------------------------------------------------
    # C) VOLTAGE aggregation (V_ph_avg)
    # ------------------------------------------------------------------
    v = joined_dedup.filter(F.col("resulttype").isin(*voltage_phases))

    # average voltage across meters at native ts, then average inside bin
    v_per_phase = (
        v.groupBy("substation_id", "transformer", "ts", "resulttype")
        .agg(F.avg("val").alias("avg_V"))
    )

    v_per_phase_res = (
        v_per_phase.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res", "resulttype")
        .agg(F.avg("avg_V").alias("V_res"))
    )

    v_pivot = (
        v_per_phase_res.groupBy("substation_id", "transformer", "ts_res")
        .pivot("resulttype", voltage_phases)
        .agg(F.first("V_res"))
        .withColumnRenamed("ts_res", "ts")
    )

    v_pivot = (
    v_pivot
    .withColumnRenamed(voltage_phases[0], "V_a")
    .withColumnRenamed(voltage_phases[1], "V_b")
    .withColumnRenamed(voltage_phases[2], "V_c")
    )

    # null-aware mean of available phase voltages (don’t treat missing as 0 V)
    v_sum = (
        F.coalesce(F.col("V_a"), F.lit(0.0)) +
        F.coalesce(F.col("V_b"), F.lit(0.0)) +
        F.coalesce(F.col("V_c"), F.lit(0.0))
    )
    v_cnt = (
        F.when(F.col("V_a").isNotNull(), F.lit(1)).otherwise(F.lit(0)) +
        F.when(F.col("V_b").isNotNull(), F.lit(1)).otherwise(F.lit(0)) +
        F.when(F.col("V_c").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )

    v_pivot = v_pivot.withColumn(
        "V_ph_avg",
        F.when(v_cnt > 0, v_sum / v_cnt).otherwise(F.lit(None))
    )




    # ------------------------------------------------------------------
    # D) Q_total from reactive energy cumulative deltas (kvarh -> kvar)
    # ------------------------------------------------------------------
    q = joined_dedup.filter(F.col("resulttype").isin(e_q_plus, e_q_minus))

    # pivot only the two energy channels per meter+ts (cheap)
    q_wide = (
        q.groupBy("substation_id", "transformer", "meteringpointid", "ts")
        .pivot("resulttype", [e_q_plus, e_q_minus])
        .agg(F.first("val"))
        .withColumn(
            "E_Q_net",
            F.coalesce(F.col(e_q_plus), F.lit(0.0)) - F.coalesce(F.col(e_q_minus), F.lit(0.0))
        )
    )

    w_q = Window.partitionBy("substation_id", "transformer", "meteringpointid").orderBy("ts")
    q_wide = q_wide.withColumn("E_Q_net_prev", F.lag("E_Q_net").over(w_q))

    q_wide = q_wide.withColumn(
        "dE_Q_kvarh",
        F.when(F.col("E_Q_net_prev").isNull(), F.lit(None))
         .otherwise(F.col("E_Q_net") - F.col("E_Q_net_prev"))
    )

    # Guard: drop negative deltas (meter rollover/reset) rather than polluting sums
    q_wide = q_wide.withColumn(
        "dE_Q_kvarh",
        F.when(F.col("dE_Q_kvarh") < F.lit(0.0), F.lit(None)).otherwise(F.col("dE_Q_kvarh"))
    )

    bin_hours = float(_parse_minutes(time_res)) / 60.0

    q_bin = (
        q_wide.withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy("substation_id", "transformer", "ts_res")
        .agg(F.sum("dE_Q_kvarh").alias("E_Q_bin_kvarh"))
        .withColumn("Q_total", F.col("E_Q_bin_kvarh") / F.lit(bin_hours))
        .drop("E_Q_bin_kvarh")
        .withColumnRenamed("ts_res", "ts")
    )

    # ------------------------------------------------------------------
    # E) Join everything + compute I_est
    # ------------------------------------------------------------------
    result = cur_pivot

    result = result.join(n_mps_res, on=["substation_id", "transformer", "ts"], how="left")
    result = result.join(p_pivot,    on=["substation_id", "transformer", "ts"], how="left")
    result = result.join(v_pivot,    on=["substation_id", "transformer", "ts"], how="left")
    result = result.join(q_bin,      on=["substation_id", "transformer", "ts"], how="left")

    # per_completed snapshot
    result = result.join(
        mp_status.select("substation_id", "transformer", "per_completed"),
        on=["substation_id", "transformer"],
        how="left",
    )

    # rollout counts (time-varying n_eligible + n_total_mps)
    rollout_ts = build_rollout_timeseries(
        spark=spark,
        substation_ids=substation_ids,
        start_date=start_date,
        end_date=end_date,
        time_res=time_res,
        substation_transformers=substation_transformers,
    )

    result = result.join(rollout_ts, on=["substation_id", "transformer", "ts"], how="left")

    # Apparent power and current estimate:
    # If P_total in kW and Q_total in kvar -> S_total in kVA.
    result = result.withColumn(
        "S_total",
        F.sqrt(
            F.coalesce(F.col("P_total"), F.lit(0.0)) * F.coalesce(F.col("P_total"), F.lit(0.0))
            + F.coalesce(F.col("Q_total"), F.lit(0.0)) * F.coalesce(F.col("Q_total"), F.lit(0.0))
        )
    )

    # I_est ≈ S / (3*V_ph_avg), with kVA->VA conversion if assume_power_kw=True
    k_factor = F.lit(1000.0) if assume_power_kw else F.lit(1.0)

    den = F.lit(3.0) * F.col("V_ph_avg")
    num = F.coalesce(F.col("S_total"), F.lit(0.0)) * k_factor
    # equivalent "sum of phase currents" (matches your I_total definition)
    result = result.withColumn(
        "I_sum_equiv",
        F.try_divide(num, F.col("V_ph_avg"))
    )

    # equivalent "balanced line current"
    result = result.withColumn(
        "I_line_equiv",
        F.try_divide(num, F.lit(3.0) * F.col("V_ph_avg"))
    )


    # ------------------------------------------------------------------
    # Final selection + write
    # ------------------------------------------------------------------
    result = (
        result.select(
            "substation_id",
            "transformer",
            "ts",

            # currents (naive aggregation)
            "I_a", "I_b", "I_c", "I_total",

            # power-based estimate
            "P_a", "P_b", "P_c", "P_total",
            "Q_total",
            "V_a", "V_b", "V_c", "V_ph_avg",
            "S_total",
            "I_sum_equiv",
            "I_line_equiv",

            # counts
            "n_mps",
            "n_eligible",
            "n_total_mps",

            # snapshot ratio (keep if you like)
            "per_completed",
        )
        .orderBy("substation_id", "transformer", "ts")
    )

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

def fetch_and_save_smartmeter_all_meters_one_parquet(
    *,
    substation_ids: list[int],
    start_date: str,
    end_date: str,
    output_path: str,
    time_res: str = TIME_RES,
    substation_transformers: list[tuple[int, str]] | None = None,
):
    spark_df = fetch_smartmeter_all_profiles_per_meter_spark(
        substation_ids=substation_ids,
        start_date=start_date,
        end_date=end_date,
        time_res=time_res,
        substation_transformers=substation_transformers,
    )

    pdf = spark_df.toPandas()
    pdf.attrs.clear()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(output_path, index=False)
    print(f"✅ Wrote ONE parquet with {len(pdf)} rows to {output_path}")
    return pdf

def fetch_smartmeter_all_profiles_per_meter_spark(
    *,
    substation_ids: list[int],
    start_date: str,
    end_date: str,
    time_res: str = TIME_RES,
    current_phases: Sequence[str] = PHASES,
    voltage_phases: Sequence[str] = Voltage_PHASES,
    power_phases: Sequence[str] = POWER_PHASES,
    substation_transformers: list[tuple[int, str]] | None = None,
):
    """
    Returns a Spark DataFrame with ONE row per (meteringpointid, ts_res)
    containing:
      - I_a,I_b,I_c (+ I_total)
      - V_a,V_b,V_c
      - P_a,P_b,P_c (+ P_total)
      - mp metadata: tegund, husveita_fastanumer, kennitalamaelistadar
      - substation/transformer
      - per_completed + n_completed + n_total_mps (per substation/transformer)

    This DOES NOT sum across meters. It keeps each meter separate.
    """
    spark = get_spark()

    mp_completed, mp_status = _build_meteringpoint_frames(
        spark, substation_ids, substation_transformers=substation_transformers
    )

    sm_ts = spark.read.table("veiturdata_base_prd.ami.smartmeter")

    wanted_resulttypes = list(current_phases) + list(voltage_phases) + list(power_phases)

    joined = (
        sm_ts.withColumn("meteringpointid", F.col("meteringpointid").cast("string"))
        .join(mp_completed, F.col("meteringpointid") == F.col("mp_id"), "inner")
        .filter(
            (F.col("timestamp") >= F.lit(start_date))
            & (F.col("timestamp") <= F.lit(end_date))
            & (F.col("validity") == F.lit("Valid"))
            & (F.col("resulttype").isin(*wanted_resulttypes))
        )
        .select(
            F.to_timestamp("timestamp").alias("ts"),
            "resulttype",
            F.col("value").cast("double").alias("val"),
            F.col("meteringpointid").alias("meteringpointid"),
            "substation_id",
            "transformer",
            # metadata (from metering_points via mp_completed)
            "tegund",
            "husveita_fastanumer",
            "kennitalamaelistadar",
        )
    )

    # Deduplicate per (substation, transformer, meter, resulttype, ts)
    w = Window.partitionBy(
        "substation_id", "transformer", "meteringpointid", "resulttype", "ts"
    ).orderBy(F.col("ts"))

    joined_dedup = (
        joined.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # Resample to time_res per meter + resulttype
    res = (
        joined_dedup
        .withColumn("ts_res", F.window("ts", time_res).start)
        .groupBy(
            "substation_id", "transformer", "meteringpointid",
            "tegund", "husveita_fastanumer", "kennitalamaelistadar",
            "ts_res", "resulttype"
        )
        .agg(F.avg("val").alias("val_res"))
    )

    # Pivot ALL resulttypes to wide columns
    wide = (
        res.groupBy(
            "substation_id", "transformer", "meteringpointid",
            "tegund", "husveita_fastanumer", "kennitalamaelistadar",
            "ts_res"
        )
        .pivot("resulttype", wanted_resulttypes)
        .agg(F.first("val_res"))
        .withColumnRenamed("ts_res", "ts")
    )

    # Rename to nice canonical columns (currents)
    wide = (
        wide
        .withColumnRenamed(current_phases[0], "I_a")
        .withColumnRenamed(current_phases[1], "I_b")
        .withColumnRenamed(current_phases[2], "I_c")
        .withColumnRenamed(voltage_phases[0], "V_a")
        .withColumnRenamed(voltage_phases[1], "V_b")
        .withColumnRenamed(voltage_phases[2], "V_c")
        .withColumnRenamed(power_phases[0], "P_a")
        .withColumnRenamed(power_phases[1], "P_b")
        .withColumnRenamed(power_phases[2], "P_c")
    )

    # Totals (per meter)
    wide = wide.withColumn(
        "I_total",
        F.coalesce(F.col("I_a"), F.lit(0.0)) +
        F.coalesce(F.col("I_b"), F.lit(0.0)) +
        F.coalesce(F.col("I_c"), F.lit(0.0))
    )

    wide = wide.withColumn(
        "P_total",
        F.coalesce(F.col("P_a"), F.lit(0.0)) +
        F.coalesce(F.col("P_b"), F.lit(0.0)) +
        F.coalesce(F.col("P_c"), F.lit(0.0))
    )

    # Attach per_completed + counts per (substation, transformer)
    wide = wide.join(
        mp_status.select("substation_id", "transformer", "per_completed", "n_completed", "n_total_mps"),
        on=["substation_id", "transformer"],
        how="left",
    )

    return wide.orderBy("substation_id", "transformer", "meteringpointid", "ts")

def fetch_and_save_smartmeter_per_meter_parquets(
    *,
    substation_ids: list[int],
    start_date: str,
    end_date: str,
    output_dir: str,
    time_res: str = TIME_RES,
    substation_transformers: list[tuple[int, str]] | None = None,
):
    """
    ONE Spark job -> ONE Pandas DF -> write one local parquet per meter.
    Much faster than running a Spark job per meter.
    """
    spark_df = fetch_smartmeter_all_profiles_per_meter_spark(
        substation_ids=substation_ids,
        start_date=start_date,
        end_date=end_date,
        time_res=time_res,
        substation_transformers=substation_transformers,
    )

    # Single action: compute everything once
    pdf_all = spark_df.toPandas()
    pdf_all.attrs.clear()

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    n_written = 0
    # Split locally (fast)
    for (sid, tr, mpid), g in pdf_all.groupby(["substation_id", "transformer", "meteringpointid"], sort=False):
        sid = int(sid)
        tr = str(tr)
        mpid = str(mpid)

        out_path = out_base / f"D{sid:04d}" / tr
        out_path.mkdir(parents=True, exist_ok=True)

        fpath = out_path / f"mp_{mpid}.parquet"
        g.sort_values("ts").to_parquet(fpath, index=False)
        n_written += 1

    print(f"✅ Wrote {n_written} per-meter parquet files under: {out_base}")
    return n_written

def filter_pairs_for_feeder(
    pairs: list[tuple[int, str]],
    substation_id: int,
    transformer: str,
) -> list[tuple[int, str]]:
    """
    Keep only (substation_id, transformer) from an existing pairs list.
    transformer expected 'sp1' or 'sp2'.
    """
    tr = str(transformer).strip().lower()
    if tr not in {"sp1", "sp2"}:
        raise ValueError(f"transformer must be 'sp1' or 'sp2', got: {transformer}")

    filtered = [(sid, trf) for (sid, trf) in pairs if int(sid) == int(substation_id) and str(trf).lower() == tr]
    if not filtered:
        # Helpful debug: show available transformers for that substation
        avail = sorted({str(t).lower() for (sid, t) in pairs if int(sid) == int(substation_id)})
        raise ValueError(
            f"No pair found for substation_id={substation_id}, transformer={tr}. "
            f"Available transformers for {substation_id}: {avail}"
        )
    return filtered

def export_single_feeder_per_meter(
    *,
    janitza_pairs: list[tuple[int, str]],
    substation_id: int,
    transformer: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    time_res: str = "15 minutes",
    max_meters: int | None = None,
):
    """
    Convenience wrapper:
      - filters janitza_pairs down to a single (substation, transformer)
      - runs per-meter export (one parquet per meter)
    """
    feeder_pairs = filter_pairs_for_feeder(janitza_pairs, substation_id, transformer)

    fetch_and_save_smartmeter_per_meter_parquets(
        substation_ids=[int(substation_id)],
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        time_res=time_res,
        substation_transformers=feeder_pairs,  # <-- same filtering logic in _build_meteringpoint_frames
    )


#new #

def _parse_minutes(time_res: str) -> int:
    """
    Expect formats like '15 minutes' or '10 minute(s)'.
    Returns integer minutes.
    """
    parts = str(time_res).strip().split()
    if not parts:
        raise ValueError(f"Bad time_res: {time_res}")
    try:
        return int(parts[0])
    except Exception as e:
        raise ValueError(f"Could not parse minutes from time_res={time_res!r}") from e

def build_rollout_timeseries(
    *,
    spark,
    substation_ids: list[int],
    start_date: str,
    end_date: str,
    time_res: str,
    substation_transformers: list[tuple[int, str]] | None = None,
):
    """
    Returns a Spark DF with:
      substation_id, transformer, ts, n_total_mps, n_eligible

    Definitions:
      n_total_mps   = total MPs in the CURRENT mapping snapshot (lh_is_current==True)
      n_eligible(t) = cumulative # of MPs that have ever reached 'Completed' by time t
                      (based on historical rows using lh_valid_from)

    This is computed with:
      - one read of metering_points (current snapshot) for mapping + n_total
      - one read of metering_points (history) for completed_from
      - one generated timestamp grid (start..end at time_res)
      - one window cumulative sum (no range joins)
    """
    mp = spark.read.table("veiturdata_enriched_prd.utilities.metering_points")

    # --- CURRENT snapshot mapping universe (consistent with your join logic) ---
    mp_current = (
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
        .select(
            F.col("husveita_fastanumer").cast("string").alias("mp_id"),
            F.col("dreifistodvanumer").alias("substation_id"),
            "transformer",
        )
        .dropna(subset=["substation_id", "transformer", "mp_id"])
        .dropDuplicates(["mp_id"])
    )

    # optional: restrict to Janitza-available (substation, transformer) pairs
    if substation_transformers:
        allowed_df = (
            spark.createDataFrame(substation_transformers, ["substation_id", "transformer"])
            .dropDuplicates()
            .withColumnRenamed("transformer", "transformer_allowed")
        )
        mp_current = (
            mp_current.join(
                allowed_df,
                (mp_current["substation_id"] == allowed_df["substation_id"])
                & (mp_current["transformer"] == allowed_df["transformer_allowed"]),
                "inner",
            )
            .drop(allowed_df["substation_id"])
            .drop("transformer_allowed")
        )

    # N_total: count MPs in the mapping universe (per feeder/transformer)
    denom = (
        mp_current.groupBy("substation_id", "transformer")
        .agg(F.countDistinct("mp_id").alias("n_total_mps"))
    )

    # --- HISTORY: when did each MP first become Completed? ---
    # Keep this as minimal as possible to reduce scan cost.
    mp_hist = (
        mp.filter(
            (F.col("dreifistodvanumer").isin(substation_ids))
            & (F.col("er_flutningur") == F.lit("N"))
        )
        .select(
            F.col("husveita_fastanumer").cast("string").alias("mp_id"),
            F.col("uppsetningar_stada").alias("status"),
            F.to_timestamp("lh_valid_from").alias("valid_from"),
        )
        .dropna(subset=["mp_id", "status", "valid_from"])
    )

    completed_from = (
        mp_hist.filter(F.col("status") == F.lit("Completed"))
        .groupBy("mp_id")
        .agg(F.min("valid_from").alias("completed_from"))
        .filter(F.col("completed_from") <= F.to_timestamp(F.lit(end_date)))
    )

    # Assign feeder mapping using CURRENT mapping (consistent with smartmeter join)
    completed_from_mapped = (
        completed_from.join(mp_current, on="mp_id", how="inner")
        .select("substation_id", "transformer", "mp_id", "completed_from")
    )

    # Event count at the bin when the MP first became completed
    events = (
        completed_from_mapped
        .withColumn("ts", F.window("completed_from", time_res).start)
        .groupBy("substation_id", "transformer", "ts")
        .agg(F.countDistinct("mp_id").alias("n_new_completed"))
    )

    # --- Build ts grid: start..end at resolution ---
    step_min = _parse_minutes(time_res)
    grid = spark.sql(
        f"""
        SELECT explode(sequence(
            to_timestamp('{start_date}'),
            to_timestamp('{end_date}'),
            interval {step_min} minutes
        )) AS ts
        """
    )

    feeders = denom.select("substation_id", "transformer").dropDuplicates()

    timeline = (
        feeders.crossJoin(grid)
        .join(events, on=["substation_id", "transformer", "ts"], how="left")
        .withColumn("n_new_completed", F.coalesce(F.col("n_new_completed"), F.lit(0)))
    )

    w = (
        Window.partitionBy("substation_id", "transformer")
        .orderBy("ts")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    timeline = (
        timeline.withColumn("n_eligible", F.sum("n_new_completed").over(w))
        .join(denom, on=["substation_id", "transformer"], how="left")
        .select("substation_id", "transformer", "ts", "n_total_mps", "n_eligible")
    )

    return timeline

if __name__ == "__main__":

    janitza_parquet_path = Path("data/janitza_3phase_I_by_transformer_20250801_0000_20251101_0000.parquet")

    # 1) Build list of distinct substation IDs from devices.csv
    pairs = load_substation_transformers_from_parquet(janitza_parquet_path)
    substation_ids = sorted({sid for sid, _ in pairs})

    print(f"Found {len(substation_ids)} substations from parquet pairs.")
    print("First few substations:", substation_ids[:10])
    print("First few (substation, transformer) pairs:", pairs[:10])

    SID = 1364
    TR  = "sp1"

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
    out_path = Path("data") / f"smartmeter_15min_all_by_transformer_{start_tag}_{end_tag}_{Profile}_NEWMETH.parquet"

        # Per-meter export (one parquet per meteringpointid)
    #out_path = Path("data") / f"smartmeter_per_meter_ALL_D{SID:04d}_{TR}.parquet"

    # fetch_and_save_smartmeter_all_meters_one_parquet(
    # substation_ids=[SID],
    # start_date=start,
    # end_date=end,
    # output_path=str(out_path),
    # time_res="15 minutes",
    # substation_transformers=filter_pairs_for_feeder(pairs, SID, TR),
    # )

    fetch_and_save_smartmeter(
        substation_ids=substation_ids,
        start_date=start,
        end_date=end,
        output_path=str(out_path),
        time_res="15 minutes",
        phases=Profile,
        substation_transformers=pairs,
    )
