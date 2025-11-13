#src/phasebalance/janitza_scrapper.py
import pandas as pd
from janitza_fetch import fetch_hist_json
import time
import math
import phase_unbalance_utils as m
import os
from datetime import datetime


def _avg_from_json(obj):
    """Return the average of the 'avg' column over the whole window."""
    s = m._series_from_values(obj,"avg")
    return float(s.mean()) if not s.empty else None

def _flatten_metrics(metrics_dict, prefix=""):
    """
    Flatten nested dicts (one level) into a single dict with 'prefix.key' names.
    e.g. {"phase_unbalance_metrics": {"UC": 1.2}} -> {"phase_unbalance_metrics.UC": 1.2}
    """
    flat = {}
    for k, v in metrics_dict.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{key}.{sk}"] = sv
        else:
            flat[key] = v
    return flat

def _compute_metrics_from_json(json_in01=None, json_in02=None, json_in03=None,
                               json_va=None, json_vb=None, json_vc=None,
                               json_I0=None, json_I1=None, json_I2=None,
                               json_V0=None, json_V1=None, json_V2=None):
    """
    Average each series over the window and feed to compute_meter_metrics.
    Any missing input becomes None, and compute_meter_metrics will handle it.
    """
    # Current magnitudes (A)
    ia = _avg_from_json(json_in01) if json_in01 is not None else None
    ib = _avg_from_json(json_in02) if json_in02 is not None else None
    ic = _avg_from_json(json_in03) if json_in03 is not None else None

    # Voltage magnitudes (V) — optional if you decide to fetch later
    va = _avg_from_json(json_va) if json_va is not None else None
    vb = _avg_from_json(json_vb) if json_vb is not None else None
    vc = _avg_from_json(json_vc) if json_vc is not None else None

    # Sequence magnitudes — optional if you decide to fetch later
    I0_mag = _avg_from_json(json_I0) if json_I0 is not None else None
    I1_mag = _avg_from_json(json_I1) if json_I1 is not None else None
    I2_mag = _avg_from_json(json_I2) if json_I2 is not None else None
    V0_mag = _avg_from_json(json_V0) if json_V0 is not None else None
    V1_mag = _avg_from_json(json_V1) if json_V1 is not None else None
    V2_mag = _avg_from_json(json_V2) if json_V2 is not None else None

    
    # Compute metrics (robust to Nones)
    metrics = m.compute_meter_metrics(
        Ia=ia, Ib=ib, Ic=ic,
        Va_mag=va, Vb_mag=vb, Vc_mag=vc,
        I0_mag=I0_mag, I1_mag=I1_mag, I2_mag=I2_mag,
        V0_mag=V0_mag, V1_mag=V1_mag, V2_mag=V2_mag,
    )
    return metrics

def has_capability(cap_df, device_id, **criteria):
    """
    Return True/False if the given device_id has at least one row
    in cap_df matching all non-None criteria.
    Example:
        has_capability(cap, 301, value_backend="I_Effective", type_backend="Input01")
    """
    #This is really slow condsider making it better later
    # Normalize device_id column to string for safe comparison
    cap_df = cap_df.copy()
    cap_df["device_id"] = cap_df["device_id"].astype(str)
    device_id = str(device_id)

    df = cap_df[cap_df["device_id"] == device_id]
    if df.empty:
        return False

    # Apply all criteria (ignore None)
    for col, val in criteria.items():
        if val is None or col not in df.columns:
            continue
        df = df[df[col].astype(str).str.contains(str(val), regex=True, na=False)]
        if df.empty:
            return False

    return not df.empty

def _safe_fetch(device_id, variable_backend, phase_backend, timebase, start, end):
    """Fetch and return None if response is missing/empty."""
    js = fetch_hist_json(
        device_id=device_id,
        variable_backend=variable_backend,
        phase_backend=phase_backend,
        timebase=timebase,
        start=start, end=end,
    )
    return js if m._has_values(js) else None

def compute_metrics_from_json_on_the_fly(
    json_in01=None, json_in02=None, json_in03=None,
    json_va=None, json_vb=None, json_vc=None,
    json_I0=None, json_I1=None, json_I2=None,
    json_V0=None, json_V1=None, json_V2=None,
):
    """
    Compute metrics 'per instant' (per timestamp) using scalar inputs, then
    average the resulting metrics over time. Returns a nested dict with the
    SAME structure as m.compute_meter_metrics, but whose values are the
    time-averaged metrics.

    Notes
    -----
    - Missing inputs at a given timestamp are passed as None.
    - Only numeric metric values are averaged; non-numeric are ignored.
    - If a metric is never numeric across the window, it returns None.
    """

    # 1) Build series for each json (indexed by startTime ns)
    s_Ia = m._series_from_values(json_in01, "Ia")
    s_Ib = m._series_from_values(json_in02, "Ib")
    s_Ic = m._series_from_values(json_in03, "Ic")

    s_Va = m._series_from_values(json_va, "Va")
    s_Vb = m._series_from_values(json_vb, "Vb")
    s_Vc = m._series_from_values(json_vc, "Vc")

    s_I0 = m._series_from_values(json_I0, "I0")
    s_I1 = m._series_from_values(json_I1, "I1")
    s_I2 = m._series_from_values(json_I2, "I2")

    s_V0 = m._series_from_values(json_V0, "V0")
    s_V1 = m._series_from_values(json_V1, "V1")
    s_V2 = m._series_from_values(json_V2, "V2")

    # 2) Align on the union of timestamps (outer join), keep as DataFrame
    df = pd.concat(
        [
            s_Ia, s_Ib, s_Ic,
            s_Va, s_Vb, s_Vc,
            s_I0, s_I1, s_I2,
            s_V0, s_V1, s_V2,
        ],
        axis=1,
    )

    if df.empty:
        # Nothing to compute -> ask compute_meter_metrics with all None (lets it decide),
        # or just return an empty dict. Using all-None keeps structure stable.
        return m.compute_meter_metrics(
            Ia=None, Ib=None, Ic=None,
            Va_mag=None, Vb_mag=None, Vc_mag=None,
            I0_mag=None, I1_mag=None, I2_mag=None,
            V0_mag=None, V1_mag=None, V2_mag=None,
        )

    df = df.sort_index()

    # 3) Helper: recursively average a list of metric dicts into one dict (same shape)
    def _numeric_mean(vals):
        import math
        nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        return float(sum(nums) / len(nums)) if nums else None

    def _merge_avg(dicts):
        """
        dicts: list of nested dicts with identical key shape (keys may be
        missing in some entries). Returns a single dict with numeric means.
        """
        out = {}
        # Collect all keys present anywhere
        keys = set().union(*[d.keys() for d in dicts if isinstance(d, dict)])
        for k in keys:
            # Gather all values for this key
            vs = [d.get(k, None) for d in dicts if isinstance(d, dict)]
            # If these are dict-like, recurse
            if any(isinstance(v, dict) for v in vs):
                child_dicts = [v for v in vs if isinstance(v, dict)]
                out[k] = _merge_avg(child_dicts) if child_dicts else {}
            else:
                # Average numeric values; ignore None/non-numeric
                out[k] = _numeric_mean(vs)
        return out

    # 4) Iterate per timestamp; feed scalars to compute_meter_metrics
    metric_dicts = []
    for _, row in df.iterrows():
        metrics_t = m.compute_meter_metrics(
            Ia=(None if pd.isna(row.get("Ia")) else float(row.get("Ia"))),
            Ib=(None if pd.isna(row.get("Ib")) else float(row.get("Ib"))),
            Ic=(None if pd.isna(row.get("Ic")) else float(row.get("Ic"))),
            Va_mag=(None if pd.isna(row.get("Va")) else float(row.get("Va"))),
            Vb_mag=(None if pd.isna(row.get("Vb")) else float(row.get("Vb"))),
            Vc_mag=(None if pd.isna(row.get("Vc")) else float(row.get("Vc"))),
            I0_mag=(None if pd.isna(row.get("I0")) else float(row.get("I0"))),
            I1_mag=(None if pd.isna(row.get("I1")) else float(row.get("I1"))),
            I2_mag=(None if pd.isna(row.get("I2")) else float(row.get("I2"))),
            V0_mag=(None if pd.isna(row.get("V0")) else float(row.get("V0"))),
            V1_mag=(None if pd.isna(row.get("V1")) else float(row.get("V1"))),
            V2_mag=(None if pd.isna(row.get("V2")) else float(row.get("V2"))),
        )
        # Keep only dicts (some implementations could return None on fully-missing)
        if isinstance(metrics_t, dict):
            metric_dicts.append(metrics_t)

    if not metric_dicts:
        # Nothing computed; return all-None structure
        return m.compute_meter_metrics(
            Ia=None, Ib=None, Ic=None,
            Va_mag=None, Vb_mag=None, Vc_mag=None,
            I0_mag=None, I1_mag=None, I2_mag=None,
            V0_mag=None, V1_mag=None, V2_mag=None,
        )

    # 5) Average across time, preserving nested structure
    return _merge_avg(metric_dicts)

def main():
    # --- config ---
    OUT_DIR = "results"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"metrics_results_{timestamp}.csv"  # Construct filename with timestamp
    os.makedirs(OUT_DIR, exist_ok=True)

    devices = pd.read_csv("metadata/devices.csv")
    caps = pd.read_csv("metadata/capabilities.csv")

    sample_time = "15m"
    start = "2025-09-01 12:00"
    end = "2025-09-02 12:00"

    # Normalize types
    devices["device_id"] = devices["device_id"].astype(str)
    caps["device_id"] = caps["device_id"].astype(str)


    # What we want to fetch: 3-phase currents with naming fallbacks
    CHANNEL_SPEC_I = {
        "IA": {"value_backend": "I_Effective", "type_candidates": ["Input01","Input05", "L1"]},
        "IB": {"value_backend": "I_Effective", "type_candidates": ["Input02","Input06", "L2"]},
        "IC": {"value_backend": "I_Effective", "type_candidates": ["Input03","Input07", "L3",]},
    }

    CHANNEL_SPEC_V = {
        "VA": {"value_backend": "U_Effective", "type_candidates": ["Input01","Input05","L1"]},
        "VB": {"value_backend": "U_Effective", "type_candidates": ["Input02","Input06","L2"]},
        "VC": {"value_backend": "U_Effective", "type_candidates": ["Input03","Input07","L3"]},
    }

    CHANNEL_SPEC_I4 = {"IA": {"value_backend": "I_Effective", "type_candidates": ["Input04","Input08","L4"]}}

    CHANNEL_SPEC_SEQ_V = {
        "V0": {"value_backend": "ZeroPhaseSeq", "type_candidates": ["Overall"]},
        "V1": {"value_backend": "PositivePhaseSeq", "type_candidates": ["Overall"]},
        "V2": {"value_backend": "NegativePhaseSeq", "type_candidates": ["Overall"]},
    }

    CHANNEL_SPEC_SEQ_I = {
        "I0": {"value_backend": "ZeroPhaseSeq_I", "type_candidates":["Overall","SUM13",]},
        "I1": {"value_backend": "PositivePhaseSeq_I", "type_candidates": ["Overall","SUM13",]},
        "I2": {"value_backend": "NegativePhaseSeq_I", "type_candidates": ["Overall","SUM13",]},
    }
    

    # Filter for devices that have all three resolved channels
    matches = [] # (did, name, planI, planV)
    for _, row in devices.iterrows():
        did  = row["device_id"]
        name = row.get("name", "")
        planI = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_I, require_all=True)
        planV = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_V, require_all=True)
        if planI and planV:
            matches.append((did, name, planI, planV))


    print(f"\nTotal devices with 3-phase currents AND 3-phase voltages: {len(matches)}")


    all_rows = []
    for did, name, planI, planV in matches:

        # Current channels
        pa = planI["IA"]["type_backend"]
        pb = planI["IB"]["type_backend"]
        pc = planI["IC"]["type_backend"]
        # Voltage channels
        pva = planV["VA"]["type_backend"]
        pvb = planV["VB"]["type_backend"]
        pvc = planV["VC"]["type_backend"]

        print(f"📡 Fetching device {did} ({name}) | I phases: A={pa},B={pb},C={pc} | V phases: A={pva},B={pvb},C={pvc}")
        
        planI4 = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_I4, require_all=True)
        planSeqI = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_SEQ_I, require_all=True)
        planSeqV  = m.resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_SEQ_V, require_all=True)

        data_I_eff_in04 = None
        data_I0 = data_I1 = data_I2 = None
        data_V0 = data_V1 = data_V2 = None

        if planI4:
            data_I_eff_in04 = _safe_fetch(did, "I_Effective", planI4["IA"]["type_backend"],sample_time, start, end)
        if planSeqI :
            data_I0 = _safe_fetch(did, "ZeroPhaseSeq_I",     planSeqI["I0"]["type_backend"], sample_time, start, end)
            data_I1 = _safe_fetch(did, "PositivePhaseSeq_I", planSeqI["I1"]["type_backend"], sample_time, start, end)
            data_I2 = _safe_fetch(did, "NegativePhaseSeq_I", planSeqI["I2"]["type_backend"], sample_time, start, end)

        if planSeqV :
            data_V0 = _safe_fetch(did, "ZeroPhaseSeq",       planSeqV["V0"]["type_backend"], sample_time, start, end)
            data_V1 = _safe_fetch(did, "PositivePhaseSeq",   planSeqV["V1"]["type_backend"], sample_time, start, end)
            data_V2 = _safe_fetch(did, "NegativePhaseSeq",   planSeqV["V2"]["type_backend"], sample_time, start, end)
         

        data_I_eff_in01 = fetch_hist_json(device_id=did,variable_backend=planI["IA"]["value_backend"],phase_backend=pa,timebase=sample_time,start=start,end=end,)
        data_I_eff_in02 = fetch_hist_json(device_id=did,variable_backend=planI["IB"]["value_backend"],phase_backend=pb,timebase=sample_time,start=start,end=end,)
        data_I_eff_in03 = fetch_hist_json(device_id=did,variable_backend=planI["IC"]["value_backend"],phase_backend=pc,timebase=sample_time,start=start,end=end,)

        data_U_eff_va = fetch_hist_json(device_id=did,variable_backend=planV["VA"]["value_backend"],phase_backend=pva,timebase=sample_time,start=start, end=end,)
        data_U_eff_vb = fetch_hist_json(device_id=did,variable_backend=planV["VB"]["value_backend"],phase_backend=pvb,timebase=sample_time,start=start,end=end,)
        data_U_eff_vc = fetch_hist_json(device_id=did,variable_backend=planV["VC"]["value_backend"],phase_backend=pvc,timebase=sample_time,start=start, end=end,)


        # --- skip if any I or V missing ---
        if (data_I_eff_in01 is None or data_I_eff_in02 is None or data_I_eff_in03 is None or
            data_U_eff_va is None or data_U_eff_vb is None or data_U_eff_vc is None):
            print(f"⚠️ Skipping device {did} ({name}) — missing I or V data.")
            continue

        nI1 = len(data_I_eff_in01.get("values", []))
        nV1 = len(data_U_eff_va.get("values", []))

        # Make sure the device has sufficient data points sometime the device might only have one point
        if nI1 <= 10 or nV1 <= 10:
            print(f"⚠️ Skipping device {did} ({name}) — insufficient data points (I: {nI1}, V: {nV1}).")
            continue

        print(f" CALCULATING METRICS NOW ")
        # --- compute metrics from window averages (I & V) ---
        metrics = _compute_metrics_from_json(
            json_in01=data_I_eff_in01, json_in02=data_I_eff_in02, json_in03=data_I_eff_in03,
            json_va=data_U_eff_va, json_vb=data_U_eff_vb, json_vc=data_U_eff_vc,
            json_I0=data_I0, json_I1=data_I1, json_I2=data_I2,
            json_V0=data_V0, json_V1=data_V1, json_V2=data_V2,
        )


         # Averages for traceability
        ia_avg = _avg_from_json(data_I_eff_in01)
        ib_avg = _avg_from_json(data_I_eff_in02)
        ic_avg = _avg_from_json(data_I_eff_in03)
        va_avg = _avg_from_json(data_U_eff_va)
        vb_avg = _avg_from_json(data_U_eff_vb)
        vc_avg = _avg_from_json(data_U_eff_vc)


        flat = _flatten_metrics(metrics)
        flat["device_id"]    = did
        flat["name"]         = name
        flat["window_start"] = start
        flat["window_end"]   = end
        flat["sample_time"]  = sample_time
        flat["Ia_avg"]       = ia_avg
        flat["Ib_avg"]       = ib_avg
        flat["Ic_avg"]       = ic_avg
        flat["Va_avg"]       = va_avg
        flat["Vb_avg"]       = vb_avg
        flat["Vc_avg"]       = vc_avg

        # record which concrete labels were used
        flat["Ia_label"] = pa;  flat["Ib_label"] = pb;  flat["Ic_label"] = pc
        flat["Va_label"] = pva; flat["Vb_label"] = pvb; flat["Vc_label"] = pvc


            # optional fields for diagnostics (only if fetched & non-empty)
        if data_I_eff_in04 is not None:
            flat["I4_avg"]   = _avg_from_json(data_I_eff_in04)
            flat["I4_label"] = planI4["IA"]["type_backend"]


        all_rows.append(flat)

        print(f"✅ Points fetched — I: {nI1}| V: {nV1}")

        time.sleep(0.1)  # be nice to the API

    if not all_rows:
        print("\nNo results to save.")
        return


    df_out = pd.DataFrame(all_rows)

    # stable column order (device info first)
    front_cols = [
        "device_id","name","window_start","window_end","sample_time",
        "Ia_avg","Ib_avg","Ic_avg","Va_avg","Vb_avg","Vc_avg"
    ]
    other_cols = [c for c in df_out.columns if c not in front_cols]
    df_out = df_out[front_cols + sorted(other_cols)]

    out_path = os.path.join(OUT_DIR, filename)
    df_out.to_csv(out_path, index=False)
    print(f"\n📁 Results written to {out_path}")

if __name__ == "__main__":
    main()
