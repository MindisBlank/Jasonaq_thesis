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

TIME_RES = "15 minutes"  # aggregation resolution

POWER_PHS = ["AMP_Power_L1_AVG_T0","AMP_Power_L2_AVG_T0","AMP_Power_L3_AVG_T0"] # 10 min averages
E_Q_PLUS  = "LP1_ReactiveEnergyPlus_CUM_T0"
E_Q_MINUS = "LP1_ReactiveEnergyMinus_CUM_T0"
E_P_PLUS  = "LP1_ActiveEnergyPlus_CUM_T0"
E_P_MINUS = "LP1_ActiveEnergyMinus_CUM_T0"
V_PHS     = ["LPQ_Voltage_L1_AVG","LPQ_Voltage_L2_AVG","LPQ_Voltage_L3_AVG"]
PF_RT = "LP1_PowerFactor_LASTAVG"


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

def get_spark():
    """
    Create a Databricks-connected Spark session.
    Relies on Databricks Connect config set up by the VS Code extension.
    """
    return DatabricksSession.builder.getOrCreate()

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

    wanted_resulttypes = list(phases) + list(power_phases) + list(voltage_phases) + [PF_RT]

    joined = (
        sm_ts.withColumn("meteringpointid", F.col("meteringpointid").cast("string"))
        .join(mp_completed, F.col("meteringpointid") == F.col("mp_id"), "inner")
        .filter(
            (F.col("timestamp") >= F.lit(start_date))
            & (F.col("timestamp") <= F.lit(end_date))
            & (F.col("resulttype").isin(*wanted_resulttypes))
            & (
                F.col("validity").isin("Valid", "Validated")
                | (F.col("resulttype") == F.lit(PF_RT))
            )
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
    # D) Power factor: compute PF_used (scalar) + PF_sm (per-bin average)
    # ------------------------------------------------------------------
    pf_rows = joined_dedup.filter(F.col("resulttype") == F.lit(PF_RT))
    p_rows  = joined_dedup.filter(F.col("resulttype").isin(*power_phases))

    # Pivot PF and P_a/P_b/P_c at meter+ts to apply filtering
    pfp = (
        pf_rows.select("substation_id","transformer","meteringpointid","ts", F.col("val").alias("pf"))
        .join(
            p_rows.groupBy("substation_id","transformer","meteringpointid","ts","resulttype")
                .agg(F.first("val").alias("pval")),
            on=["substation_id","transformer","meteringpointid","ts"],
            how="inner",
        )
    )

    pfp_wide = (
        pfp.groupBy("substation_id","transformer","meteringpointid","ts")
        .pivot("resulttype", power_phases)
        .agg(F.first("pval"))
        .withColumn(
            "P_total_meter",
            F.coalesce(F.col(power_phases[0]), F.lit(0.0))
            + F.coalesce(F.col(power_phases[1]), F.lit(0.0))
            + F.coalesce(F.col(power_phases[2]), F.lit(0.0))
        )
    )

    # Apply your robust filters (PF>=0.3 and P_total>=200)
    pfp_wide = (
    pfp.groupBy("substation_id","transformer","meteringpointid","ts","pf")
       .pivot("resulttype", power_phases)
       .agg(F.first("pval"))
       .withColumn(
           "P_total_meter",
           F.coalesce(F.col(power_phases[0]), F.lit(0.0))
         + F.coalesce(F.col(power_phases[1]), F.lit(0.0))
         + F.coalesce(F.col(power_phases[2]), F.lit(0.0))
       )
    )

    pfp_filt = pfp_wide.filter(
        (F.col("pf").isNotNull())
        & (F.col("pf") >= F.lit(0.3))
        & (F.col("pf") <= F.lit(1.0))
        & (F.col("P_total_meter") >= F.lit(200.0))
    )

    print("PF/P joined rows:", pfp.count())
    print("PF/P filtered rows:", pfp_filt.count())


    # Compute scalar PF_used per (substation, transformer) over the whole window
    pf_used_df = (
        pfp_filt.groupBy("substation_id","transformer")
                .agg(F.avg("pf").alias("PF_used"))
    )

    # Attach PF_used to the per-bin table later (after we have ts bins)
    # Also compute PF_sm = average PF across meters per bin (useful for debugging)
    pf_sm_bin = (
        pfp_filt.withColumn("ts_res", F.window("ts", time_res).start)
                .groupBy("substation_id","transformer","ts_res")
                .agg(F.avg("pf").alias("PF_sm"))
                .withColumnRenamed("ts_res","ts")
    )



    # ------------------------------------------------------------------
    # E) Join everything + compute I_est
    # ------------------------------------------------------------------
    result = cur_pivot

    result = result.join(n_mps_res, on=["substation_id", "transformer", "ts"], how="left")
    result = result.join(p_pivot,    on=["substation_id", "transformer", "ts"], how="left")
    result = result.join(v_pivot,    on=["substation_id", "transformer", "ts"], how="left")
    # Join PF_used (feeder scalar) + PF_sm (per-bin average)
    result = result.join(pf_used_df, on=["substation_id","transformer"], how="left")
    result = result.join(pf_sm_bin,  on=["substation_id","transformer","ts"], how="left")

    # Fallback if PF_used missing (safety)
    result = result.withColumn("PF_used", F.coalesce(F.col("PF_used"), F.lit(0.9)))
    # Compute per-phase reactive power using PF_used
    tanphi = F.tan(F.acos(F.col("PF_used")))

    result = (
    result
    .withColumn("Q_a", F.coalesce(F.col("P_a"), F.lit(0.0)) * tanphi)
    .withColumn("Q_b", F.coalesce(F.col("P_b"), F.lit(0.0)) * tanphi)
    .withColumn("Q_c", F.coalesce(F.col("P_c"), F.lit(0.0)) * tanphi)
    .withColumn("Q_total", F.col("Q_a") + F.col("Q_b") + F.col("Q_c"))
    )

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
            "PF_used", "PF_sm",
            "Q_a", "Q_b", "Q_c", "Q_total",
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

def fetch_meter_data_for_PF(
    *,
    substation_id: int,
    transformer: str,
    start_date: str,
    end_date: str,
    output_path: str,
    time_res: str = TIME_RES,  # e.g. "15 minutes"
    phases: list[str] = PHASES,
    power_phases: list[str] = POWER_PHS,
    voltage_phases: list[str] = V_PHS,
    e_q_plus: str = E_Q_PLUS,
    e_q_minus: str = E_Q_MINUS,
) -> pd.DataFrame:
    """
    PF export (single parquet) at METER resolution:
      - rows: (substation_id, transformer, mp_id, ts)
      - columns: per-meter currents/power/voltages + per-meter Q estimate + meter metadata
      - designed for joining to ArcGIS using mp_id / husveita_fastanumer / etc.

    Writes ONE file to `output_path`.
    """

    tr = str(transformer).strip().lower()
    if tr not in {"sp1", "sp2"}:
        raise ValueError(f"transformer must be 'sp1' or 'sp2', got: {transformer!r}")

    spark = get_spark()

    # ------------------------------
    # 1) Metering point mapping + metadata (per meter, not aggregated)
    # ------------------------------
    mp = spark.read.table("veiturdata_enriched_prd.utilities.metering_points")

    mp_base = (
        mp.filter(
            (F.col("dreifistodvanumer") == F.lit(int(substation_id)))
            & (F.col("lh_is_current") == F.lit(True))
            & (F.col("er_flutningur") == F.lit("N"))
        )
        .withColumn(
            "transformer",
            F.when(F.col("dspennir") == F.lit(1), F.lit("sp1"))
             .when(F.col("dspennir") == F.lit(2), F.lit("sp2"))
             .otherwise(F.lit(None))
        )
        .filter(F.col("transformer") == F.lit(tr))
    )

    mp_completed = (
        mp_base.filter(F.col("uppsetningar_stada") != F.lit("planned"))
        .select(
            # join key to smartmeter (internal)
            F.col("husveita_fastanumer").cast("string").alias("mp_id"),

            # feeder identifiers
            F.col("dreifistodvanumer").cast("int").alias("substation_id"),
            "transformer",

            # meter identifiers for ArcGIS joins
            F.col("husveita_fastanumer").cast("string").alias("husveita_fastanumer"),
            F.col("kennitalamaelistadar").cast("string").alias("kennitalamaelistadar"),
            F.col("numer_heimlagnar").cast("string").alias("numer_heimlagnar"),  # <-- NEW

            # requested metadata (NO renames)
            F.col("dspennir").cast("int").alias("dspennir"),
            F.col("tengiskapur").cast("string").alias("tengiskapur"),
            F.col("submeter_pairing_group").cast("string").alias("submeter_pairing_group"),
            F.col("latitude").cast("double").alias("latitude"),
            F.col("longitude").cast("double").alias("longitude"),
            F.col("lysingnotkunarflokks").cast("string").alias("lysingnotkunarflokks"),
            F.col("notkunarstadur").cast("string").alias("notkunarstadur"),
        )
        .dropDuplicates(["mp_id"])
    )

    # Optional: add per_completed snapshot (feeder-level) as a scalar column if you still want it
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
            "n_total_mps",
        )
    )

    # ------------------------------
    # 2) Fetch smartmeter rows for those meters
    # ------------------------------
    sm_ts = spark.read.table("veiturdata_base_prd.ami.smartmeter")

    wanted_resulttypes = list(phases) + list(power_phases) + list(voltage_phases) + [PF_RT]

    raw = (
        sm_ts.withColumn("meteringpointid", F.col("meteringpointid").cast("string"))
        .join(mp_completed.select("mp_id"), F.col("meteringpointid") == F.col("mp_id"), "inner")
        .filter(
            (F.col("timestamp") >= F.lit(start_date))
            & (F.col("timestamp") <= F.lit(end_date))
            & (F.col("resulttype").isin(*wanted_resulttypes))
            & (
                F.col("validity").isin("Valid", "Validated")
                | (F.col("resulttype") == F.lit(PF_RT))
            )
        )
        .select(
            F.to_timestamp("timestamp").alias("ts"),
            F.col("meteringpointid").alias("mp_id"),
            F.col("resulttype").alias("resulttype"),
            F.col("value").cast("double").alias("val"),
        )
    )

    # Deduplicate per (mp_id, resulttype, ts)
    w_dedup = Window.partitionBy("mp_id", "resulttype", "ts").orderBy(F.col("ts"))
    raw = raw.withColumn("rn", F.row_number().over(w_dedup)).filter(F.col("rn") == 1).drop("rn")

    # Bin timestamp to time_res
    raw = raw.withColumn("ts_res", F.window("ts", time_res).start)

    # ------------------------------
    # 3) Pivot currents / powers / voltages per meter per bin
    # ------------------------------
    def pivot_avg(resulttypes: list[str], rename_map: dict[str, str]):
        df = raw.filter(F.col("resulttype").isin(*resulttypes))
        # average inside bin (per mp)
        b = (
            df.groupBy("mp_id", "ts_res", "resulttype")
            .agg(F.avg("val").alias("v"))
        )
        wide = (
            b.groupBy("mp_id", "ts_res")
            .pivot("resulttype", resulttypes)
            .agg(F.first("v"))
        )
        for k, v in rename_map.items():
            wide = wide.withColumnRenamed(k, v)
        return wide

    cur_wide = pivot_avg(
        phases,
        {phases[0]: "I_a", phases[1]: "I_b", phases[2]: "I_c"},
    ).withColumn(
        "I_total",
        F.coalesce(F.col("I_a"), F.lit(0.0))
        + F.coalesce(F.col("I_b"), F.lit(0.0))
        + F.coalesce(F.col("I_c"), F.lit(0.0)),
    )

    p_wide = pivot_avg(
        power_phases,
        {power_phases[0]: "P_a", power_phases[1]: "P_b", power_phases[2]: "P_c"},
    ).withColumn(
        "P_total",
        F.coalesce(F.col("P_a"), F.lit(0.0))
        + F.coalesce(F.col("P_b"), F.lit(0.0))
        + F.coalesce(F.col("P_c"), F.lit(0.0)),
    )

    v_wide = pivot_avg(
        voltage_phases,
        {voltage_phases[0]: "V_a", voltage_phases[1]: "V_b", voltage_phases[2]: "V_c"},
    )
    v_sum = F.coalesce(F.col("V_a"), F.lit(0.0)) + F.coalesce(F.col("V_b"), F.lit(0.0)) + F.coalesce(F.col("V_c"), F.lit(0.0))
    v_cnt = (
        F.when(F.col("V_a").isNotNull(), 1).otherwise(0)
        + F.when(F.col("V_b").isNotNull(), 1).otherwise(0)
        + F.when(F.col("V_c").isNotNull(), 1).otherwise(0)
    )
    v_wide = v_wide.withColumn("V_ph_avg", F.when(v_cnt > 0, v_sum / v_cnt).otherwise(F.lit(None)))

    pf_wide = pivot_avg(
        [PF_RT],
        {PF_RT: "PF_sm"}
    )

    # ------------------------------
    # 4) Compute feeder PF_used and per-meter Q from PF_used
    # ------------------------------

    # Build a table with P_total and PF_sm to compute PF_used
    pf_for_used = (
        p_wide.join(pf_wide, on=["mp_id","ts_res"], how="inner")
            .filter(
                (F.col("PF_sm").isNotNull())
                & (F.col("PF_sm") >= F.lit(0.3))
                & (F.col("PF_sm") <= F.lit(1.0))
                & (F.col("P_total") >= F.lit(200.0))
            )
    )

    # Collect scalar PF_used for this (substation, transformer, period)
    pf_used = pf_for_used.agg(F.avg("PF_sm").alias("PF_used")).collect()[0]["PF_used"]
    if pf_used is None:
        pf_used = 0.9  # safety fallback

    tanphi_lit = F.tan(F.acos(F.lit(float(pf_used))))

    # Compute Q per phase from P per phase (per meter)
    q_from_pf = (
        p_wide
        .select("mp_id", "ts_res", "P_a", "P_b", "P_c")  # only what we need to compute Q
        .withColumn("PF_used", F.lit(float(pf_used)))
        .withColumn("Q_a", F.coalesce(F.col("P_a"), F.lit(0.0)) * tanphi_lit)
        .withColumn("Q_b", F.coalesce(F.col("P_b"), F.lit(0.0)) * tanphi_lit)
        .withColumn("Q_c", F.coalesce(F.col("P_c"), F.lit(0.0)) * tanphi_lit)
        .withColumn("Q_total", F.col("Q_a") + F.col("Q_b") + F.col("Q_c"))
        .select("mp_id", "ts_res", "PF_used", "Q_a", "Q_b", "Q_c", "Q_total")
    )


    # ------------------------------
    # 5) Join per-meter signals + attach meter metadata (so ArcGIS join is trivial)
    # ------------------------------
    # Start from a base timeline = union of keys present in any wide table
    keys = (
        cur_wide.select("mp_id", "ts_res")
        .union(p_wide.select("mp_id", "ts_res"))
        .union(v_wide.select("mp_id", "ts_res"))
        .union(pf_wide.select("mp_id", "ts_res"))
        .dropDuplicates()
    )


    out = (
        keys.join(cur_wide,   on=["mp_id", "ts_res"], how="left")
            .join(p_wide,     on=["mp_id", "ts_res"], how="left")
            .join(v_wide,     on=["mp_id", "ts_res"], how="left")
            .join(pf_wide,    on=["mp_id", "ts_res"], how="left")     # PF_sm per meter/bin
            .join(q_from_pf,  on=["mp_id", "ts_res"], how="left")     # Q_* from PF_used
            .join(mp_completed, on="mp_id", how="left")
            .join(mp_status, on=["substation_id", "transformer"], how="left")
            .withColumnRenamed("ts_res", "ts")
    )


    # Optional: if you want a per-meter apparent power too
    out = out.withColumn(
        "S_total",
        F.sqrt(
            F.coalesce(F.col("P_total"), F.lit(0.0)) * F.coalesce(F.col("P_total"), F.lit(0.0))
            + F.coalesce(F.col("Q_total"), F.lit(0.0)) * F.coalesce(F.col("Q_total"), F.lit(0.0))
        )
    )

    out = (
        out.select(
            "substation_id", "transformer",

            # mp_id dropped (same as husveita_fastanumer)
            "husveita_fastanumer",
            "kennitalamaelistadar",
            "numer_heimlagnar",          
            "ts",

            # currents
            "I_a", "I_b", "I_c", "I_total",

            "P_a", "P_b", "P_c", "P_total",
            "PF_used", "PF_sm",
            "Q_a", "Q_b", "Q_c", "Q_total",
            "V_a", "V_b", "V_c", "V_ph_avg",
            "S_total",

            # meter metadata
            "dspennir",
            "latitude",
            "longitude",
            "tengiskapur",
            "notkunarstadur",
            "lysingnotkunarflokks",
            "submeter_pairing_group",

            # feeder snapshot info (optional)
            "per_completed",
            "n_total_mps",
        )
        .orderBy("husveita_fastanumer", "ts")
    )

    # ------------------------------
    # 6) Write ONE output file
    # ------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = out.toPandas()
    pdf.attrs.clear()

    if out_path.suffix.lower() == ".parquet":
        pdf.to_parquet(out_path, index=False)
    else:
        pdf.to_csv(out_path, index=False)

    print(f"Saved {len(pdf)} rows to {out_path} (per-meter PF table)")
    return pdf


if __name__ == "__main__":

    SID = 670
    TR  = "sp1"

    choice = 2 # 1=all substations, 2=single substation

    # 2) Define time window
    start = "2025-08-01 00:00:00"
    end   = "2025-09-01 00:00:00"

    #Turn start/end into compact tags for the filename
    fmt_in = "%Y-%m-%d %H:%M:%S"
    start_dt = datetime.strptime(start, fmt_in)
    end_dt   = datetime.strptime(end,   fmt_in)
    start_tag = start_dt.strftime("%Y%m%d")
    end_tag   = end_dt.strftime("%Y%m%d")

    

    if choice == 1:

        print("Fetching data for ALL substations' transformers from Janitza parquet")

        janitza_parquet_path = Path("data/all_substations_current_202508-202511.parquet")

        # 1) Build list of distinct substation IDs from devices.csv
        pairs = load_substation_transformers_from_parquet(janitza_parquet_path)
        substation_ids = sorted({sid for sid, _ in pairs})

        print(f"Found {len(substation_ids)} substations from parquet pairs.")
        print("First few (substation, transformer) pairs:", pairs[:10])

        # define the profile
        Profile = PHASES
        out_path = Path("data") / f"smartmeter_15min_all_by_transformer_{start_tag}_{end_tag}_NEWMETH.parquet"

        fetch_and_save_smartmeter(
        substation_ids=substation_ids,
        start_date=start,
        end_date=end,
        output_path=str(out_path),
        time_res="15 minutes",
        phases=Profile,
        substation_transformers=pairs,
        )

    elif choice == 2:
        print("Fetching data for substation", SID, "transformer", TR)

        out_path = Path("data") /"smartmeter" / f"smartmeter_15min__substation_{SID:04d}_transformer_{TR}_{start_tag}_{end_tag}.parquet"

        df = fetch_meter_data_for_PF(
        substation_id=SID,
        transformer=TR,
        start_date="2025-09-01 00:00:00",
        end_date="2025-10-01 00:00:00",
        output_path=str(out_path),
        time_res="15 minutes",
        )




#dspennir, tengiskapur,submeter_pairing_group,submeter_pairing_group, latitude, longitude, lysingnotkunarflokks,kennitalamaelistadar, notkunarstadur, husveita_fastanumer, dreifistodvanumer