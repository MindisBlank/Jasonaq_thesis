import pandas as pd
from janitza_fetch import fetch_hist_json
import time
import math
import phase_unbalance_utils as m
import os
from datetime import datetime

def _series_from_values(obj, label):
    """
    Turn {"values":[{"startTime":ns, "avg":x}, ...]} into a pandas Series
    indexed by startTime (ns) with the 'avg' value. label is only for naming.
    """
    vals = obj.get("values", []) if isinstance(obj, dict) else []
    data = {}
    for v in vals:
        st = v.get("startTime")
        avg = v.get("avg")
        if st is None or avg is None:
            continue
        try:
            data[int(st)] = float(avg)
        except (TypeError, ValueError):
            continue
    if not data:
        return pd.Series(dtype=float, name=label)
    s = pd.Series(data).sort_index()
    s.name = label
    return s

def _avg_from_json(obj):
    """Return the average of the 'avg' column over the whole window."""
    s = _series_from_values(obj, label="tmp")
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

def _compute_stats(json_in1, json_in4): # Remove this 
    """
    Align Input01 & Input04 series by startTime and compute:
      - mean absolute difference (MAD) of avg values over the overlap
      - ratio = sum(avg_in1) / sum(avg_in4) over the overlap
    Returns (mad, ratio, n_points) or (None, None, 0) if insufficient overlap.
    """
    s1 = _series_from_values(json_in1, "in1")
    s4 = _series_from_values(json_in4, "in4")

    if s1.empty or s4.empty:
        return (None, None, 0)

    joined = pd.concat([s1, s4], axis=1, join="inner").dropna()
    if joined.empty:
        return (None, None, 0)

    mad = (joined["in1"] - joined["in4"]).abs().mean()

    denom = joined["in4"].sum()
    ratio = (joined["in1"].sum() / denom) if abs(denom) > 1e-12 else None

    return (float(mad), (float(ratio) if ratio is not None else None), int(len(joined)))

# -------------------- core compute --------------------
def _compute_metrics_from_json(json_in01=None, json_in02=None, json_in03=None,
                               json_va=None, json_vb=None, json_vc=None,
                               json_pf_a=None, json_pf_b=None, json_pf_c=None):
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

    # Power factors — optional if you decide to fetch later
    pf_a = _avg_from_json(json_pf_a) if json_pf_a is not None else None
    pf_b = _avg_from_json(json_pf_b) if json_pf_b is not None else None
    pf_c = _avg_from_json(json_pf_c) if json_pf_c is not None else None

    # Compute metrics (robust to Nones)
    metrics = m.compute_meter_metrics(
        Ia=ia, Ib=ib, Ic=ic,
        Va_mag=va, Vb_mag=vb, Vc_mag=vc,
        pfA=pf_a, pfB=pf_b, pfC=pf_c,
        # If you someday add phasors, plug them here as *_phasor
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

def resolve_channels(cap_df: pd.DataFrame,
                     device_id: str,
                     channels: dict,
                     require_all: bool = True) -> dict:
    """
    Resolve the (value_backend, type_backend) to use for each logical channel.

    channels: dict of
      channel_key -> {
         "value_backend": "<exact string>",          # e.g., "I_Effective"
         "type_candidates": [list of exact strings], # priority order, e.g., ["Input01","L1"]
      }

    Returns:
      {
        channel_key: {"value_backend": "...", "type_backend": "<chosen candidate>"},
        ...
      }
      If require_all=True and any channel can't be resolved, returns {}.
    """
    cap = cap_df.copy()
    cap["device_id"] = cap["device_id"].astype(str)
    did = str(device_id)

    # Get only rows for this device
    dev_rows = cap[cap["device_id"] == did]
    if dev_rows.empty:
        return {}

    out = {}
    for key, spec in channels.items():
        value_backend = str(spec["value_backend"])
        candidates = [str(c) for c in spec.get("type_candidates", [])]

        # Filter to rows with matching value_backend
        pool = dev_rows[dev_rows["value_backend"].astype(str) == value_backend]
        if pool.empty:
            if require_all:
                return {}
            continue

        # Pick first matching candidate
        chosen = None
        for cand in candidates:
            if not pool[pool["type_backend"].astype(str) == cand].empty:
                chosen = cand
                break

        if chosen is None:
            if require_all:
                return {}
            continue

        out[key] = {"value_backend": value_backend, "type_backend": chosen}

    return out




def main():
    # --- config ---

    OUT_DIR = "results"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    # Construct filename with timestamp
    filename = f"metrics_results_{timestamp}.csv"
    os.makedirs(OUT_DIR, exist_ok=True)

    devices = pd.read_csv("metadata/devices.csv")
    caps = pd.read_csv("metadata/capabilities.csv")

    sample_time = "15m"
    start = "2025-09-01 12:00"
    end = "2025-10-01 12:00"

    # Normalize types
    devices["device_id"] = devices["device_id"].astype(str)
    caps["device_id"] = caps["device_id"].astype(str)


    # What we want to fetch: 3-phase currents with naming fallbacks
    CHANNEL_SPEC_I = {
        "IA": {"value_backend": "I_Effective", "type_candidates": ["Input01", "L1"]},
        "IB": {"value_backend": "I_Effective", "type_candidates": ["Input02", "L2"]},
        "IC": {"value_backend": "I_Effective", "type_candidates": ["Input03", "L3"]},
    }

    # Filter for devices that have all three resolved channels
    matches = []  # (did, name, plan)
    for _, row in devices.iterrows():
        did  = row["device_id"]
        name = row.get("name", "")
        plan = resolve_channels(caps, device_id=did, channels=CHANNEL_SPEC_I, require_all=True)
        if plan:  # only keep full 3-phase devices
            matches.append((did, name, plan))

    print(f"\nTotal devices with 3-phase currents: {len(matches)}")

    all_rows = []
    for did, name, plan in matches:
        # plan looks like:
        # {"IA":{"value_backend":"I_Effective","type_backend":"Input01" or "L1"}, ...}

        pa = plan["IA"]["type_backend"]
        pb = plan["IB"]["type_backend"]
        pc = plan["IC"]["type_backend"]

        print(f"📡 Fetching data for device {did} ({name}) using phases: "
              f"A={pa}, B={pb}, C={pc}...")

        # --- fetch 3-phase current histories over the window ---
        data_in01 = fetch_hist_json(
            device_id=did,
            variable_backend=plan["IA"]["value_backend"],   # "I_Effective"
            phase_backend=pa,                                # resolved "Input01" or "L1"
            timebase=sample_time,
            start=start,
            end=end,
        )
        data_in02 = fetch_hist_json(
            device_id=did,
            variable_backend=plan["IB"]["value_backend"],
            phase_backend=pb,
            timebase=sample_time,
            start=start,
            end=end,
        )
        data_in03 = fetch_hist_json(
            device_id=did,
            variable_backend=plan["IC"]["value_backend"],
            phase_backend=pc,
            timebase=sample_time,
            start=start,
            end=end,
        )
        # --- skip if any of the phases returned no data ---
        if data_in01 is None or data_in02 is None or data_in03 is None:
            print(f"⚠️ Skipping device {did} ({name}) — missing data in one or more phases.")
            continue

        # --- compute metrics from monthly averages ---
        metrics = _compute_metrics_from_json(
            json_in01=data_in01, json_in02=data_in02, json_in03=data_in03,
        )

        # Also store the averaged input currents for traceability
        ia_avg = _avg_from_json(data_in01)
        ib_avg = _avg_from_json(data_in02)
        ic_avg = _avg_from_json(data_in03)

        flat = _flatten_metrics(metrics)
        flat["device_id"]    = did
        flat["name"]         = name
        flat["window_start"] = start
        flat["window_end"]   = end
        flat["sample_time"]  = sample_time
        flat["Ia_avg"]       = ia_avg
        flat["Ib_avg"]       = ib_avg
        flat["Ic_avg"]       = ic_avg
        # record which concrete labels were used
        flat["Ia_label"]     = pa
        flat["Ib_label"]     = pb
        flat["Ic_label"]     = pc

        all_rows.append(flat)

        n1 = len(data_in01.get("values", []))
        n2 = len(data_in02.get("values", []))
        n3 = len(data_in03.get("values", []))
        print(f"✅ Points fetched: {pa}={n1}, {pb}={n2}, {pc}={n3}")

        # polite pause to avoid rate limits
        time.sleep(10)

    # --- save results ---
    if not all_rows:
        print("\nNo results to save.")
        return

    df_out = pd.DataFrame(all_rows)

    # stable column order (device info first)
    front_cols = ["device_id", "name", "window_start", "window_end", "sample_time", "Ia_avg", "Ib_avg", "Ic_avg"]
    other_cols = [c for c in df_out.columns if c not in front_cols]
    df_out = df_out[front_cols + sorted(other_cols)]

    out_path = os.path.join(OUT_DIR, filename)
    df_out.to_csv(out_path, index=False)
    print(f"\n📁 Results written to {out_path}")

if __name__ == "__main__":
    main()
