# src/phasebalance/janitza_scrapper.py
import os
import time

import pandas as pd

from janitza_fetch import fetch_hist_json
import phase_unbalance_utils as m


def _safe_fetch(device_id, variable_backend, phase_backend, timebase, start, end):
    """
    Fetch and return None if response is missing/empty.
    If variable_backend is a list/tuple, try in order until one returns values.
    """
    backends = variable_backend if isinstance(variable_backend, (list, tuple)) else [variable_backend]

    for vb in backends:
        js = fetch_hist_json(
            device_id=device_id,
            variable_backend=vb,
            phase_backend=phase_backend,
            timebase=timebase,
            start=start,
            end=end,
        )
        if m._has_values(js):
            return js

    return None

def _extract_avg_df(json_obj, base_name: str) -> pd.DataFrame:
    """
    Build a DataFrame with one column: {base}_avg indexed by timestamp (startTime).
    """
    col = f"{base_name}_avg"
    if json_obj is None or not m._has_values(json_obj):
        return pd.DataFrame(columns=[col])

    s_avg = m._series_from_values(json_obj, "avg")
    s_avg.name = col
    return s_avg.to_frame()

def _extract_triplet_df(json_obj, base_name: str) -> pd.DataFrame:
    """
    Build a small DataFrame with columns: {base}_avg, {base}_min, {base}_max
    indexed by timestamp (startTime).
    """
    cols = [f"{base_name}_avg", f"{base_name}_min", f"{base_name}_max"]
    if json_obj is None or not m._has_values(json_obj):
        return pd.DataFrame(columns=cols)

    s_avg = m._series_from_values(json_obj, "avg")
    s_min = m._series_from_values(json_obj, "min")
    s_max = m._series_from_values(json_obj, "max")

    s_avg.name = f"{base_name}_avg"
    s_min.name = f"{base_name}_min"
    s_max.name = f"{base_name}_max"

    return pd.concat([s_avg, s_min, s_max], axis=1)

def compute_metrics_from_json_on_the_fly(
    json_in01=None, json_in02=None, json_in03=None,
    json_va=None, json_vb=None, json_vc=None,
) -> pd.DataFrame:
    """
    Compute *per timestamp* metrics using the *avg* series of each input.
    Returns a DataFrame indexed by timestamp with:
      - vuf_magnitude
      - cur_ratio
      - cur_dev_ratio
    """
    # avg series (named so downstream is clean)
    s_Ia = m._series_from_values(json_in01, "Ia") if json_in01 is not None else pd.Series(dtype=float, name="Ia")
    s_Ib = m._series_from_values(json_in02, "Ib") if json_in02 is not None else pd.Series(dtype=float, name="Ib")
    s_Ic = m._series_from_values(json_in03, "Ic") if json_in03 is not None else pd.Series(dtype=float, name="Ic")

    s_Va = m._series_from_values(json_va, "Va") if json_va is not None else pd.Series(dtype=float, name="Va")
    s_Vb = m._series_from_values(json_vb, "Vb") if json_vb is not None else pd.Series(dtype=float, name="Vb")
    s_Vc = m._series_from_values(json_vc, "Vc") if json_vc is not None else pd.Series(dtype=float, name="Vc")

    df = pd.concat([s_Ia, s_Ib, s_Ic, s_Va, s_Vb, s_Vc], axis=1)
    if df.empty:
        return pd.DataFrame(columns=["vuf_magnitude", "cur_ratio", "cur_dev_ratio"])

    df = df.sort_index()

    out_rows = []
    for ts, row in df.iterrows():
        Ia = None if pd.isna(row.get("Ia")) else float(row.get("Ia"))
        Ib = None if pd.isna(row.get("Ib")) else float(row.get("Ib"))
        Ic = None if pd.isna(row.get("Ic")) else float(row.get("Ic"))

        Va = None if pd.isna(row.get("Va")) else float(row.get("Va"))
        Vb = None if pd.isna(row.get("Vb")) else float(row.get("Vb"))
        Vc = None if pd.isna(row.get("Vc")) else float(row.get("Vc"))

        metrics_t = m.compute_meter_metrics(
            Ia=Ia, Ib=Ib, Ic=Ic,
            Va_mag=Va, Vb_mag=Vb, Vc_mag=Vc,
        )

        out_rows.append(
            {
                "ts": ts,
                "vuf_magnitude": metrics_t.get("vuf_magnitude"),
                "cur_ratio": metrics_t.get("cur_ratio"),
                "cur_dev_ratio": metrics_t.get("cur_dev_ratio"),
            }
        )

    return pd.DataFrame(out_rows).set_index("ts").sort_index()

def _fetch_3phase_df(
    did: str,
    plan: dict,
    keys: tuple[str, str, str],          # ("PA","PB","PC") etc
    base_names: tuple[str, str, str],    # ("Pa","Pb","Pc") etc
    variable_backend,
    timebase: str,
    start: str,
    end: str,
    extractor,                           # _extract_triplet_df or _extract_avg_df
) -> list[pd.DataFrame]:
    """
    Always returns a list of 3 DataFrames (may be empty) for A/B/C.
    If the plan is missing a key, its df is empty.
    """
    out = []
    for k, bn in zip(keys, base_names):
        if plan and k in plan:
            js = _safe_fetch(did, variable_backend, plan[k]["type_backend"], timebase, start, end)
        else:
            js = None
        out.append(extractor(js, bn))
    return out

def _fetch_total_df(
    did: str,
    plan: dict,
    key: str,
    base_name: str,
    variable_backend,
    timebase: str,
    start: str,
    end: str,
    extractor,
) -> pd.DataFrame:
    if plan and key in plan:
        js = _safe_fetch(
            did,
            variable_backend,                 
            plan[key]["type_backend"],
            timebase,
            start,
            end,
        )
        return extractor(js, base_name)
    return extractor(None, base_name)



def main():
    # --- config ---
    OUT_DIR = "data"
    os.makedirs(OUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"janitza_timeseries_{timestamp}_All_mes.parquet")

    devices = pd.read_csv("metadata/devices.csv")
    caps = pd.read_csv("metadata/capabilities.csv")

    sample_time = "15m" #20250801_20251101
    start = "2025-08-01 00:00"
    end = "2025-11-01 00:00"

    devices["device_id"] = devices["device_id"].astype(str)
    caps["device_id"] = caps["device_id"].astype(str)

    # device_type_name lookup (from capabilities.csv)
    if "device_type_name" in caps.columns:
        device_type_lookup = ( 
            caps.groupby("device_id")["device_type_name"]
            .apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else None)
            .to_dict()
        )
    else:
        device_type_lookup = {}

    # Channel specs (NO sequences)
    CHANNEL_SPEC_I = {
        "IA": {"value_backend": "I_Effective", "type_candidates": ["Input01", "L1"]},
        "IB": {"value_backend": "I_Effective", "type_candidates": ["Input02", "L2"]},
        "IC": {"value_backend": "I_Effective", "type_candidates": ["Input03", "L3"]},
    }

    CHANNEL_SPEC_V = {
        "VA": {"value_backend": "U_Effective", "type_candidates": ["Input01", "L1"]},
        "VB": {"value_backend": "U_Effective", "type_candidates": ["Input02", "L2"]},
        "VC": {"value_backend": "U_Effective", "type_candidates": ["Input03", "L3"]},
    }

    CHANNEL_SPEC_P = {
        "PA": {"value_backend": "PowerActive", "type_candidates": ["Input01", "L1"]},
        "PB": {"value_backend": "PowerActive", "type_candidates": ["Input02", "L2"]},
        "PC": {"value_backend": "PowerActive", "type_candidates": ["Input03", "L3"]},
    }

    CHANNEL_SPEC_S = {
        "SA": {"value_backend": "PowerApparent", "type_candidates": ["Input01", "L1"]},
        "SB": {"value_backend": "PowerApparent", "type_candidates": ["Input02", "L2"]},
        "SC": {"value_backend": "PowerApparent", "type_candidates": ["Input03", "L3"]},
    }

    CHANNEL_SPEC_Q = {
        "QA": {"value_backend": "PowerReactivefund", "type_candidates": ["Input01", "L1"]},
        "QB": {"value_backend": "PowerReactivefund", "type_candidates": ["Input02", "L2"]},
        "QC": {"value_backend": "PowerReactivefund", "type_candidates": ["Input03", "L3"]},
    }

    CHANNEL_SPEC_P_TOTAL = {
        "P_TOTAL": {"value_backend": "ActiveEnergy", "type_candidates": ["SUM13"]}, #3600 sample time Seems like the 801 meters are a hasle again they sample at 60 and 900 not always the same need to make sure my fallback code works for this 
    }

    CHANNEL_SPEC_Q_TOTAL = {
        "Q_TOTAL": {"value_backend": ["ReactiveEnergy","ReactiveEnergyInd"], "type_candidates": ["SUM13"]}, # Ditto above.
    }

    # Filter for devices that have all three resolved channels
    matches = []
    for _, row in devices.iterrows():
        did = row["device_id"]
        name = row.get("name", "")
        planI = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_I, require_all=True)
        planV = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_V, require_all=True)
        if planI and planV:
            matches.append((did, name, planI, planV))

    print(f"\nTotal devices with 3-phase currents AND 3-phase voltages: {len(matches)}")

    all_dfs = []

    for did, name, planI, planV in matches:
        pa, pb, pc = planI["IA"]["type_backend"], planI["IB"]["type_backend"], planI["IC"]["type_backend"]
        pva, pvb, pvc = planV["VA"]["type_backend"], planV["VB"]["type_backend"], planV["VC"]["type_backend"]
        device_type_name = device_type_lookup.get(str(did), None)

        print(
            f"📡 Fetching {did} ({name}) | type={device_type_name} | "
            f"I: A={pa},B={pb},C={pc} | V: A={pva},B={pvb},C={pvc}"
        )

        # optional measurements to fetch if available
        plan_P = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_P, require_all=False)
        plan_S = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_S, require_all=False)
        plan_Q = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_Q, require_all=False)
        plan_P_total = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_P_TOTAL, require_all=False)
        plan_Q_total = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_Q_TOTAL, require_all=False)

        optional_dfs = []

        # 3-phase POWER (triplets)
        optional_dfs += _fetch_3phase_df(
            did, plan_P,
            keys=("PA","PB","PC"),
            base_names=("Pa","Pb","Pc"),
            variable_backend="PowerActive",
            timebase=sample_time,
            start=start, end=end,
            extractor=_extract_triplet_df
        )

        optional_dfs += _fetch_3phase_df(
            did, plan_S,
            keys=("SA","SB","SC"),
            base_names=("Sa","Sb","Sc"),
            variable_backend="PowerApparent",
            timebase=sample_time,
            start=start, end=end,
            extractor=_extract_triplet_df
        )

        optional_dfs += _fetch_3phase_df(
            did, plan_Q,
            keys=("QA","QB","QC"),
            base_names=("Qa","Qb","Qc"),
            variable_backend="PowerReactivefund",
            timebase=sample_time,
            start=start, end=end,
            extractor=_extract_triplet_df
        )

        # TOTAL ENERGY (avg-only) — only if you still want it; otherwise remove these two lines entirely
        optional_dfs.append(
            _fetch_total_df(did, plan_P_total, "P_TOTAL", "P_total",
                            variable_backend="ActiveEnergy", timebase="1 hour",
                            start=start, end=end, extractor=_extract_avg_df)
        )
        optional_dfs.append(
            _fetch_total_df(did, plan_Q_total, "Q_TOTAL", "Q_total",
                            variable_backend=["ReactiveEnergy","ReactiveEnergyInd"], timebase="1 hour",
                            start=start, end=end, extractor=_extract_avg_df)
        )


        # Fetch required I,V
        data_Ia = _safe_fetch(did, planI["IA"]["value_backend"], pa, sample_time, start, end)
        data_Ib = _safe_fetch(did, planI["IB"]["value_backend"], pb, sample_time, start, end)
        data_Ic = _safe_fetch(did, planI["IC"]["value_backend"], pc, sample_time, start, end)

        data_Va = _safe_fetch(did, planV["VA"]["value_backend"], pva, sample_time, start, end)
        data_Vb = _safe_fetch(did, planV["VB"]["value_backend"], pvb, sample_time, start, end)
        data_Vc = _safe_fetch(did, planV["VC"]["value_backend"], pvc, sample_time, start, end)

        if (data_Ia is None or data_Ib is None or data_Ic is None or
            data_Va is None or data_Vb is None or data_Vc is None):
            print(f"⚠️ Skipping {did} ({name}) — missing I or V data.")
            continue

        nI = len(data_Ia.get("values", []))
        nV = len(data_Va.get("values", []))
        if nI <= 100 or nV <= 100:
            print(f"⚠️ Skipping {did} ({name}) — insufficient data points (I: {nI}, V: {nV}).")
            continue
        

        # measurements avg/min/max per timestamp
        df_meas = pd.concat(
            [
                _extract_triplet_df(data_Ia, "Ia"),
                _extract_triplet_df(data_Ib, "Ib"),
                _extract_triplet_df(data_Ic, "Ic"),
                _extract_triplet_df(data_Va, "Va"),
                _extract_triplet_df(data_Vb, "Vb"),
                _extract_triplet_df(data_Vc, "Vc"),
                *optional_dfs,
            ],
            axis=1,
        ).sort_index()

        # metrics per timestamp
        df_metrics = compute_metrics_from_json_on_the_fly(
            json_in01=data_Ia, json_in02=data_Ib, json_in03=data_Ic,
            json_va=data_Va, json_vb=data_Vb, json_vc=data_Vc,
        )

        df_full = pd.concat([df_meas, df_metrics], axis=1).sort_index()



        # attach metadata
        df_full["device_id"] = did
        df_full["name"] = name
        df_full["device_type_name"] = device_type_name
        df_full["window_start"] = start
        df_full["window_end"] = end
        df_full["sample_time"] = sample_time

        df_full["Ia_label"] = pa
        df_full["Ib_label"] = pb
        df_full["Ic_label"] = pc
        df_full["Va_label"] = pva
        df_full["Vb_label"] = pvb
        df_full["Vc_label"] = pvc

        df_full = df_full.reset_index(names="ts")
        all_dfs.append(df_full)

        print(f"✅ {did} done — rows: {len(df_full)} (I pts: {nI} | V pts: {nV})")
        time.sleep(0.1)

    if not all_dfs:
        print("\nNo results to save.")
        return

    df_out = pd.concat(all_dfs, ignore_index=True)

    # column order: identifiers first
    front_cols = [
        "device_id", "name", "device_type_name", "ts",
        "window_start", "window_end", "sample_time",
    ]

    # Put ALL label columns next (Ia_label, Pa_label, etc.) without hardcoding them
    label_cols = sorted([c for c in df_out.columns if c.endswith("_label")])

    front_cols = [c for c in front_cols if c in df_out.columns] + label_cols
    front_cols = list(dict.fromkeys(front_cols))  # dedupe, preserve order

    other_cols = [c for c in df_out.columns if c not in front_cols]
    df_out = df_out[front_cols + sorted(other_cols)]

    # write parquet (fallback to csv if parquet engine missing)
    try:
        df_out.to_parquet(out_path, index=False)
        print(f"\n📦 Parquet written to {out_path}")
    except Exception as e:
        fallback = out_path.replace(".parquet", ".csv")
        df_out.to_csv(fallback, index=False)
        print(f"\n⚠️ Parquet write failed ({e}). Wrote CSV instead: {fallback}")



if __name__ == "__main__":
    main()
