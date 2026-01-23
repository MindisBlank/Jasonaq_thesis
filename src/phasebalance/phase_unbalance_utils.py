"""
phase_unbalance_utils.py
-----------
Contains all calculation functions for electrical imbalance and power quality
metrics derived from smart meter or substation data.

This module provides current, voltage, and sequence-based imbalance metrics,
neutral current estimations, and helper functions to compute robustly even when
some inputs are missing.
"""
import cmath
import math
from typing import Optional, Dict, Any
import pandas as pd

# ---------- Helper functions----------
def _series_from_values(obj, label_or_which="avg", *, convert_ts=True):
    """
    Backwards-compatible:
    - If label_or_which is one of ("avg","min","max"), treat it as the field to extract.
    - Otherwise treat it as a label (old behaviour) and extract "avg".
    """

    # Detect old vs new usage
    if label_or_which in ("avg", "min", "max"):
        which = label_or_which
        label = label_or_which
    else:
        which = "avg"
        label = label_or_which

    vals = obj.get("values", []) if isinstance(obj, dict) else []
    data = {}

    for v in vals:
        st = v.get("startTime")
        val = v.get(which)
        if st is None or val is None:
            continue
        try:
            data[int(st)] = float(val)
        except (TypeError, ValueError):
            continue

    if not data:
        return pd.Series(dtype=float, name=label)

    s = pd.Series(data).sort_index()

    # Optional conversion
    if convert_ts:
        s.index = pd.to_datetime(s.index, unit="ns", utc=True)

    s.name = label
    return s

def _has_values(obj: object) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("values"), list)
        and len(obj["values"]) > 0
    )

def resolve_channels(
    cap_df: pd.DataFrame,
    device_id: str,
    channels: dict,
    require_all: bool = True,
    return_all: bool = False,
) -> dict:
    """
    Resolve the (value_backend, type_backend) to use for each logical channel.

    Supports:
      - value_backend as a single string (old behavior)
      - value_backend as a list/tuple/set of strings (new behavior; priority order)

    Parameters
    ----------
    cap_df : DataFrame
        Capabilities table with at least columns: device_id, value_backend, type_backend.
    device_id : str
        The device_id whose channels we are resolving.
    channels : dict
        Mapping:
          channel_key -> {
             "value_backend": "<exact string>" OR ["cand1","cand2",...],
             "type_candidates": [list of exact strings], # priority order, e.g., ["Input01","L1"]
          }
    require_all : bool, default True
        If True, and any channel can't be resolved, returns {}.
    return_all : bool, default False
        If False (default): for each key picks a single 'best' type_backend, the first
        candidate that exists, preserving the old behaviour.

        If True: for each key returns all matching candidates in priority order in
        a list under "type_backends", and also sets "type_backend" to the first one
        for backward compatibility.

    Returns
    -------
    dict
        {
          channel_key: {
              "value_backend": "<chosen value_backend>",
              "type_backend": "<chosen type_backend>",
              # if return_all=True:
              # "type_backends": ["cand1", "cand2", ...],
          },
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

    out: dict[str, dict] = {}
    for key, spec in channels.items():
        vb_raw: Any = spec.get("value_backend")
        if isinstance(vb_raw, (list, tuple, set)):
            value_backends = [str(v) for v in vb_raw]
        else:
            value_backends = [str(vb_raw)]

        candidates = [str(c) for c in spec.get("type_candidates", [])]

        # Filter to rows with matching value_backend (any candidate)
        pool = dev_rows[dev_rows["value_backend"].astype(str).isin(value_backends)]
        if pool.empty:
            if require_all:
                return {}
            continue

        # Choose which value_backend to use (first that actually exists, in priority order)
        chosen_vb = next(
            (vb for vb in value_backends
             if not pool[pool["value_backend"].astype(str) == vb].empty),
            None
        )
        if chosen_vb is None:
            if require_all:
                return {}
            continue

        # Now restrict pool to the chosen value_backend
        pool_vb = pool[pool["value_backend"].astype(str) == chosen_vb]
        if pool_vb.empty:
            if require_all:
                return {}
            continue


        # Collect all matching candidates, in priority order
        matched: list[str] = []
        for cand in candidates:
            if not pool[pool["type_backend"].astype(str) == cand].empty:
                matched.append(cand)

        if not matched:
            if require_all:
                return {}
            continue

        # Backwards-compatible: always keep a single 'type_backend' (first match)
        entry = {
            "value_backend": chosen_vb,
            "type_backend": matched[0],
        }

        # Optionally include ALL matches
        if return_all:
            entry["type_backends"] = matched

        out[key] = entry

    return out

def parse_dnr(d):
    """
    Convert DNR-style IDs to integers.

    Examples:
        'D0411' -> 411
        'D1354' -> 1354
    """
    s = str(d).strip()
    if not s or not s.startswith("D"):
        return None

    digits = s[1:].lstrip("0")  # remove 'D' and leading zeros
    if not digits:
        return None

    return int(digits)

def make_expected_grid(df, freq=pd.Timedelta(minutes=15)):
    """Build full expected 15-min grid per (substation_id, transformer) over its observed span."""
    frames = []
    grouped = df.groupby(["substation_id", "transformer"], sort=False)

    for (substation_id, transformer), group in grouped:
        t0 = group["ts"].min()
        t1 = group["ts"].max()
        grid = pd.date_range(t0, t1, freq=freq)

        frames.append(
            pd.DataFrame(
                {
                    "substation_id": substation_id,
                    "transformer": transformer,
                    "ts": grid,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)

def filter_devices_by_name_pattern(
    df_jn: pd.DataFrame,
    df_devices: pd.DataFrame,
    *,
    name_col: str = "name",
    device_id_col: str = "device_id",
    pattern: str = (
        r"Measurement Group 2|Measurement Group 3|Mod 1|Mod1|Mod2|Mod3|Mod\. 1|Mod\. 2|LR|241|"
        r"Device-4|Device-233|Device-162"
    ),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows from df_jn whose device_id matches devices in df_devices whose `name_col`
    matches `pattern`.

    Returns:
        (df_jn_filtered, devices_to_remove) where devices_to_remove is Int64 dtype.
    """
    devices_to_remove = pd.to_numeric(
        df_devices.loc[
            df_devices[name_col].astype(str).str.contains(pattern, na=False, regex=True),
            device_id_col,
        ],
        errors="coerce",
    ).dropna().astype("Int64")

    out = df_jn.copy()
    out[device_id_col] = pd.to_numeric(out[device_id_col], errors="coerce").astype("Int64")

    out = out[~out[device_id_col].isin(set(devices_to_remove.tolist()))].copy()
    return out, devices_to_remove

def filter_low_load_samples(
    df: pd.DataFrame,
    *,
    x_mean_phase_A: float = 50.0,     # keep sample if mean(Ia,Ib,Ic) >= this
    use_isum: bool = False,           # if True: use I_sum threshold instead
    isum_min_A: float = 150.0,        # keep sample if (Ia+Ib+Ic) >= this
    min_valid_points_per_device: int = 0,  # e.g. 2000 to drop dead devices entirely (optional)
) -> pd.DataFrame:
    """
    filter *samples* (rows), not devices.
    Keeps only timestamps where the feeder load is meaningful.

    Requirements for a row to be kept:
      - Ia_avg, Ib_avg, Ic_avg must be numeric
      - load condition passes:
          * mean_phase >= x_mean_phase_A
        OR (if use_isum=True)
          * I_sum >= isum_min_A

    Optional:
      - drop devices that have too few remaining valid points (useful if some meters were offline).
    """
    out = df.copy()

    # Ensure numeric device_id + currents
    out["device_id"] = pd.to_numeric(out["device_id"], errors="coerce").astype("Int64")
    Ia = pd.to_numeric(out["Ia_avg"], errors="coerce")
    Ib = pd.to_numeric(out["Ib_avg"], errors="coerce")
    Ic = pd.to_numeric(out["Ic_avg"], errors="coerce")

    # Compute load measures per row
    mean_phase = (Ia + Ib + Ic) / 3.0
    isum = Ia + Ib + Ic

    # Keep rows with valid current measurements
    valid_curr = mean_phase.notna()

    # Apply load threshold
    if use_isum:
        keep = valid_curr & (isum >= float(isum_min_A))
    else:
        keep = valid_curr & (mean_phase >= float(x_mean_phase_A))

    out = out.loc[keep].copy()

    # Optional: drop devices with too few surviving points
    if min_valid_points_per_device and min_valid_points_per_device > 0:
        counts = out.groupby("device_id").size()
        good_devices = counts[counts >= int(min_valid_points_per_device)].index
        out = out[out["device_id"].isin(good_devices)].copy()

    return out
# ---------- NaN constants ----------
NAN = float("nan")
CNAN = complex(float("nan"), float("nan"))

# ---------- metrics ----------
_SEQ_NANS = {
    "M2_mag": NAN,
    "M0_mag": NAN,
}

def dib(Ia: float, Ib: float, Ic: float) -> float:
    """
    Calculate the Distribution Imbalance Index (DIB).
    Formula:
        DIB = (max(Ia, Ib, Ic) - I_avg) / (Ia + Ib + Ic)
    where:
        Ia, Ib, Ic : float
            Phase currents (in amperes)
        I_avg : float
            Average of the three phase currents
    Returns
    -------
    float
        dib_value : Distribution Imbalance Index (unitless)
    """
    I_avg = (Ia + Ib + Ic) / 3
    dib_value = (max(Ia, Ib, Ic) - I_avg) / (Ia + Ib + Ic)
    return dib_value

def vuf_magnitude(Va: float, Vb: float, Vc: float) -> float:
    """
    Calculate the Voltage Unbalance Factor (VUF) based on voltage magnitudes.
    Formula:
        VUF = (max(Va, Vb, Vc) - V_avg) / V_avg
    where:
        Va, Vb, Vc : float
            Phase voltages (in volts)
        V_avg : float
            Average of the three phase voltages
    Returns
    -------
    float
        Voltage Unbalance Factor (unitless)
    """
    V_avg = (Va + Vb + Vc) / 3
    vuf_value = ((max(Va, Vb, Vc) - V_avg) / V_avg)*100.0
    return vuf_value

def vuf_symmetrical(V0: float, V1: float, V2: float) -> float:
    """
    Calculate the Voltage Unbalance Factor (VUF) directly from sequence component magnitudes.

    Formula:
        VUF = |V_neg| / |V_pos|*100

    Parameters
    ----------
    V0, V1, V2 : float
        Zero-, positive-, and negative-sequence voltage magnitudes (V)

    Returns
    -------
    float
        Voltage Unbalance Factor (percentage)
    """
    if V1 < 1e-9:  # avoid divide-by-zero or noise
        return float("nan")
    return (V2 / V1) * 100.0

def sequence_unbalance_factors(I0: float, I1: float, I2: float) -> dict:
    """
    Compute magnitude-based unbalance factors given the sequence current magnitudes.
    Formulas:
        M2 = |I2| / |I1|
        M0 = |I0| / |I1|
    Parameters
    ----------
    I0, I1, I2 : float
        Zero-, positive-, and negative-sequence current magnitudes (A).
    Returns
    -------
    dict
        {
            "M2_mag": float,  # Negative-sequence unbalance factor
            "M0_mag": float,  # Zero-sequence unbalance factor
        }
    """
    if I1 < 1e-9:
        return {
            "M2_mag": float('nan'),
            "M0_mag": float('nan'),
        }

    M2_mag = I2 / I1
    M0_mag = I0 / I1
    return {
        "M2_mag": M2_mag,
        "M0_mag": M0_mag,
    }

def cur_ratio(ia: float, ib: float, ic: float) -> float:
    """
    Calculate the Current Unbalance Ratio (CUR) in percent.

    Formula:
        CUR[%] = max(|Ia - Ib|, |Ia - Ic|, |Ib - Ic|) /
                 (|Ia| + |Ib| + |Ic|) * 100

    Parameters
    ----------
    ia, ib, ic : float
        Phase currents (TRMS scalar values, can be signed if direction known)

    Returns
    -------
    float
        Current Unbalance Ratio in percent.
    """
    denom = abs(ia) + abs(ib) + abs(ic)
    if denom == 0:
        return 0.0

    numerator = max(abs(ia - ib), abs(ia - ic), abs(ib - ic))
    return (numerator / denom) * 100.0

def cur_dev_ratio(ia: float, ib: float, ic: float) -> float:
    """
    Calculate the deviation-based Current Unbalance Ratio (CUR_dev) in percent.

    Formula:
        CUR_dev[%] = max(|Ia - I_avg|, |Ib - I_avg|, |Ic - I_avg|) /
                     (|Ia| + |Ib| + |Ic|) * 100

    Parameters
    ----------
    ia, ib, ic : float
        Phase currents (TRMS scalar values, can be signed if direction known)

    Returns
    -------
    float
        Deviation-based Current Unbalance Ratio in percent.
    """
    denom = abs(ia) + abs(ib) + abs(ic)
    if denom == 0:
        return 0.0

    i_avg = (ia + ib + ic) / 3.0
    numerator = max(abs(ia - i_avg), abs(ib - i_avg), abs(ic - i_avg))
    return (numerator / denom) * 100.0

def neutral_from_trms_120deg(IA: float, IB: float, IC: float) -> float:
    """
    Estimate neutral current using only TRMS magnitudes,
    assuming 120° separation and similar PF angles.
    """
    return math.sqrt(
        IA*IA + IB*IB + IC*IC - IA*IB - IB*IC - IC*IA
    )

def _safe_call(cond, fn, default, *args, **kwargs):
    """
    Safely call a function if a condition is met.

    Parameters
    ----------
    cond : bool
        Condition determining whether to call the function.
    fn : callable
        Function to call if the condition is True.
    default : any
        Default value to return if the condition is False or an error occurs.
    *args, **kwargs
        Arguments and keyword arguments for the function.

    Returns
    -------
    any
        The result of fn(*args, **kwargs) if successful; otherwise, the default value.
    """
    if not cond:
        return default

    try:
        return fn(*args, **kwargs)
    except Exception:
        return default

def compute_meter_metrics(
    Ia: Optional[float] = None,
    Ib: Optional[float] = None,
    Ic: Optional[float] = None,
    Va_mag: Optional[float] = None,
    Vb_mag: Optional[float] = None,
    Vc_mag: Optional[float] = None,
    I0_mag: Optional[float] = None,
    I1_mag: Optional[float] = None,
    I2_mag: Optional[float] = None,
    V0_mag: Optional[float] = None,
    V1_mag: Optional[float] = None,
    V2_mag: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Collects all imbalance metrics robustly.
    Only computes metrics when required inputs are present.
    Missing inputs return NaN or NaN-filled dicts.
    """
    results = {
    #    "dib": NAN,
        "vuf_magnitude": NAN,
    #    "vuf_symmetrical": NAN,
    #    "sequence_unbalance_factors": _SEQ_NANS.copy(),
        "cur_ratio": NAN,
        "cur_dev_ratio": NAN,
    #    "neutral_from_trms_120deg": NAN,
    }

    has_I = Ia is not None and Ib is not None and Ic is not None
    has_Vmag = Va_mag is not None and Vb_mag is not None and Vc_mag is not None
    #has_seq = (I0_mag is not None) and (I1_mag is not None) and (I2_mag is not None)
    #has_Vseq = (V0_mag is not None) and (V1_mag is not None) and (V2_mag is not None)

    # --- Current metrics ---
    #results["dib"] = _safe_call(has_I, dib, NAN, Ia, Ib, Ic)
    results["cur_ratio"] = _safe_call(has_I, cur_ratio, NAN, Ia, Ib, Ic)
    results["cur_dev_ratio"] = _safe_call(has_I, cur_dev_ratio, NAN, Ia, Ib, Ic)
    #results["neutral_from_trms_120deg"] = _safe_call(has_I, neutral_from_trms_120deg, NAN, Ia, Ib, Ic)

    # --- Voltage metrics ---
    results["vuf_magnitude"] = _safe_call(has_Vmag, vuf_magnitude, NAN, Va_mag, Vb_mag, Vc_mag)
    #results["vuf_symmetrical"] = _safe_call(has_Vseq, vuf_symmetrical, NAN, V0_mag, V1_mag, V2_mag)

    # --- Sequence metrics ---
    #results["sequence_unbalance_factors"] = _safe_call(
    #    has_seq, sequence_unbalance_factors, _SEQ_NANS.copy(), I0_mag, I1_mag, I2_mag
    #)

    return results

# -------------------------- Example usage CLI --------------------------
if __name__ == "__main__":
    """
    Run this module directly to see example calls and outputs.

    $ python phase_unbalance_utils.py
    """
    from pprint import pprint

    print("\nExample 1 — Only TRMS currents (Ia, Ib, Ic):")
    res1 = compute_meter_metrics(Ia=420.0, Ib=280.0, Ic=275.0)
    pprint(res1)


    print("\nExample 3 — Voltage magnitudes (VUF magnitude):")
    res3 = compute_meter_metrics(Va_mag=230.0, Vb_mag=226.0, Vc_mag=233.0)
    pprint(res3)

    print("\nExample 4 — Current and voltage phasors (sequence + VUF symmetrical):")
    # Example phasors: magnitude ∠ angle(deg)
    def phasor(mag, deg):
        return cmath.rect(mag, math.radians(deg))

    print("\nDone.")