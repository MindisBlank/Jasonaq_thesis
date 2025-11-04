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

# ---------- NaN constants ----------
NAN = float("nan")
CNAN = complex(float("nan"), float("nan"))

# ---------- Default NaN dictionaries ----------
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
    vuf_value = (max(Va, Vb, Vc) - V_avg) / V_avg
    return vuf_value

def vuf_symmetrical(V0: float, V1: float, V2: float) -> float:
    """
    Calculate the Voltage Unbalance Factor (VUF) directly from sequence component magnitudes.

    Formula:
        VUF = |V_neg| / |V_pos|

    Parameters
    ----------
    V0, V1, V2 : float
        Zero-, positive-, and negative-sequence voltage magnitudes (V)

    Returns
    -------
    float
        Voltage Unbalance Factor (unitless)
    """
    if V1 < 1e-9:  # avoid divide-by-zero or noise
        return float("nan")
    return V2 / V1

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
        "dib": NAN,
        "vuf_magnitude": NAN,
        "vuf_symmetrical": NAN,
        "sequence_unbalance_factors": _SEQ_NANS.copy(),
        "cur_ratio": NAN,
        "cur_dev_ratio": NAN,
        "neutral_from_trms_120deg": NAN,
    }

    has_I = Ia is not None and Ib is not None and Ic is not None
    has_Vmag = Va_mag is not None and Vb_mag is not None and Vc_mag is not None
    has_seq = (I0_mag is not None) and (I1_mag is not None) and (I2_mag is not None)
    has_Vseq = (V0_mag is not None) and (V1_mag is not None) and (V2_mag is not None)

    # --- Current metrics ---
    results["dib"] = _safe_call(has_I, dib, NAN, Ia, Ib, Ic)
    results["cur_ratio"] = _safe_call(has_I, cur_ratio, NAN, Ia, Ib, Ic)
    results["cur_dev_ratio"] = _safe_call(has_I, cur_dev_ratio, NAN, Ia, Ib, Ic)
    results["neutral_from_trms_120deg"] = _safe_call(has_I, neutral_from_trms_120deg, NAN, Ia, Ib, Ic)

    # --- Voltage metrics ---
    results["vuf_magnitude"] = _safe_call(has_Vmag, vuf_magnitude, NAN, Va_mag, Vb_mag, Vc_mag)
    results["vuf_symmetrical"] = _safe_call(has_Vseq, vuf_symmetrical, NAN, V0_mag, V1_mag, V2_mag)

    # --- Sequence metrics ---
    results["sequence_unbalance_factors"] = _safe_call(
        has_seq, sequence_unbalance_factors, _SEQ_NANS.copy(), I0_mag, I1_mag, I2_mag
    )

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