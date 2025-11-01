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
_PHASE_NANS = {
    "UC": NAN,
    "Unbalance_a": NAN,
    "Unbalance_b": NAN,
    "Unbalance_c": NAN,
}
_SEQ_NANS = {
    "I0": CNAN,
    "I1": CNAN,
    "I2": CNAN,
    "M2": CNAN,
    "M0": CNAN,
    "M2_mag": NAN,
    "M2_angle_deg": NAN,
    "M0_mag": NAN,
    "M0_angle_deg": NAN,
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

def vuf_symmetrical(Va: complex, Vb: complex, Vc: complex) -> float: # This might need to be change to have the sequeneces as an input
    """
    Calculate the Voltage Unbalance Factor (VUF) using symmetrical components.
    Formula:
        VUF = |V_neg| / |V_pos|
    where:
        Va, Vb, Vc : complex
            Phase voltages as complex phasors (magnitude and phase)
        a : complex
            Operator a = e^(j*120°) = -0.5 + j*(sqrt(3)/2)

    Returns
    -------
    float
        Voltage Unbalance Factor (unitless)
    """
    a = cmath.exp(1j * 2 * cmath.pi / 3)
    V_pos = (Va + a * Vb + a**2 * Vc) / 3
    V_neg = (Va + a**2 * Vb + a * Vc) / 3

    if abs(V_pos) == 0:
        return float("nan")  # Avoid divide-by-zero errors

    vuf_value = abs(V_neg) / abs(V_pos)
    return vuf_value

def current_unbalance_rms(Ia: float, Ib: float, Ic: float) -> float:
    """
    Calculate the instantaneous RMS-based current unbalance.

    Formula:
        I_unbalance = sqrt(Ia^2 + Ib^2 + Ic^2 - 3 * I_avg^2)

    where:
        Ia, Ib, Ic : float
            Phase currents (in amperes)
        I_avg : float
            Average of the three phase currents

    Returns
    -------
    float
        Instantaneous RMS-based current unbalance (in amperes)
    """
    I_avg = (Ia + Ib + Ic) / 3
    I_unbalance = math.sqrt(Ia**2 + Ib**2 + Ic**2 - 3 * (I_avg**2))
    return I_unbalance

def phase_unbalance_metrics(Ia: float, Ib: float, Ic: float) -> dict:
    """
    Calculate multiple phase unbalance metrics.

    Includes:
        - UC  : Combined unbalance coefficient
                UC = (1/3) * ((Ia/I_avg)^2 + (Ib/I_avg)^2 + (Ic/I_avg)^2)
        - Unbalance_a : Ia / I_avg
        - Unbalance_b : Ib / I_avg
        - Unbalance_c : Ic / I_avg

    where:
        Ia, Ib, Ic : float
            Phase currents (in amperes)
        I_avg : float
            Average of the three phase currents

    Returns
    -------
    dict
        Dictionary containing:
            {
                "UC": float,
                "Unbalance_a": float,
                "Unbalance_b": float,
                "Unbalance_c": float
            }
    """
    I_avg = (Ia + Ib + Ic) / 3

    if I_avg == 0:
        raise ValueError("Average current (I_avg) is zero; cannot compute ratios.")

    unbalance_a = Ia / I_avg
    unbalance_b = Ib / I_avg
    unbalance_c = Ic / I_avg
    uc = (1 / 3) * (unbalance_a**2 + unbalance_b**2 + unbalance_c**2)

    return {
        "UC": uc,
        "Unbalance_a": unbalance_a,
        "Unbalance_b": unbalance_b,
        "Unbalance_c": unbalance_c,
    }

def sequence_unbalance_factors(Ia: complex, Ib: complex, Ic: complex) -> dict: # Same with VUF symetrical I might need to adjust the inputs here we assume a perfect 120 deg shift but ofcourse in the real wrold that is not the case
    """
    Compute the complex unbalance factors for sequence currents.

    Formulas:
        a = e^(j*120°)
        I0 = (Ia + Ib + Ic) / 3
        I1 = (Ia + a * Ib + a**2 * Ic) / 3
        I2 = (Ia + a**2 * Ib + a * Ic) / 3

        M2 = I2 / I1 = |I2| / |I1| ∠(δ2 - δ1)
        M0 = I0 / I1 = |I0| / |I1| ∠(δ0 - δ1)

    Parameters
    ----------
    Ia, Ib, Ic : complex
        Phase currents as complex phasors (magnitude ∠ angle in radians).

    Returns
    -------
    dict
        {
            "I0": complex,  # Zero-sequence current
            "I1": complex,  # Positive-sequence current
            "I2": complex,  # Negative-sequence current
            "M2": complex,  # Complex unbalance factor for negative sequence
            "M0": complex,  # Complex unbalance factor for zero sequence
            "M2_mag": float,
            "M2_angle_deg": float,
            "M0_mag": float,
            "M0_angle_deg": float
        }
    """
    # Operator a = e^(j*120°)
    a = cmath.exp(1j * 2 * cmath.pi / 3)

    # Sequence components
    I0 = (Ia + Ib + Ic) / 3
    I1 = (Ia + a * Ib + a**2 * Ic) / 3
    I2 = (Ia + a**2 * Ib + a * Ic) / 3

    if abs(I1) == 0:
        raise ValueError("Positive-sequence current I1 is zero; cannot compute ratios.") # might want to remove the raise here and just return nans
 
    if abs(I1) < 1e-6:
        M2 = complex(float('nan'))
        M0 = complex(float('nan'))
    else:
        # Complex unbalance factors
        M2 = I2 / I1
        M0 = I0 / I1

    # Magnitude and angle (degrees)
    M2_mag = abs(M2)
    M2_angle_deg = math.degrees(cmath.phase(M2))
    M0_mag = abs(M0)
    M0_angle_deg = math.degrees(cmath.phase(M0))

    return {
        "I0": I0,
        "I1": I1,
        "I2": I2,
        "M2": M2,
        "M0": M0,
        "M2_mag": M2_mag,
        "M2_angle_deg": M2_angle_deg,
        "M0_mag": M0_mag,
        "M0_angle_deg": M0_angle_deg,
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

def neutral_from_trms_and_pf(IA: float, IB: float, IC: float,
                             pfA: float, pfB: float, pfC: float,
                             lagging: bool = True) -> float:
    """
    Estimate neutral current using TRMS magnitudes + per-phase PF.
    Assumes each phase current lags its own phase voltage by acos(pf).
    Set lagging=False if you know currents lead (capacitive).
    """
    # base phase angles in degrees
    base = [0.0, -120.0, 120.0]
    # per-phase PF angles in radians (sign: +lag, -lead)
    sign = 1.0 if lagging else -1.0
    thetas = [sign*math.acos(max(min(pf, 1.0), -1.0)) for pf in (pfA, pfB, pfC)]
    mags = [IA, IB, IC]

    # build phasors and sum
    Rx = 0.0
    Ry = 0.0
    for M, base_deg, th in zip(mags, base, thetas):
        ang = math.radians(base_deg) + th
        Rx += M * math.cos(ang)
        Ry += M * math.sin(ang)
    return math.hypot(Rx, Ry)

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
    Ia_phasor: Optional[complex] = None,
    Ib_phasor: Optional[complex] = None,
    Ic_phasor: Optional[complex] = None,
    Va_phasor: Optional[complex] = None,
    Vb_phasor: Optional[complex] = None,
    Vc_phasor: Optional[complex] = None,
    pfA: Optional[float] = None,
    pfB: Optional[float] = None,
    pfC: Optional[float] = None,
    lagging: bool = True,
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
        "current_unbalance_rms": NAN,
        "phase_unbalance_metrics": _PHASE_NANS.copy(),
        "sequence_unbalance_factors": _SEQ_NANS.copy(),
        "cur_ratio": NAN,
        "cur_dev_ratio": NAN,
        "neutral_from_trms_120deg": NAN,
        "neutral_from_trms_and_pf": NAN,
    }

    has_I = Ia is not None and Ib is not None and Ic is not None
    has_Vmag = Va_mag is not None and Vb_mag is not None and Vc_mag is not None
    has_Iph = Ia_phasor is not None and Ib_phasor is not None and Ic_phasor is not None
    has_Vph = Va_phasor is not None and Vb_phasor is not None and Vc_phasor is not None
    has_pf = all(p is not None for p in (pfA, pfB, pfC))

    # --- Current metrics ---
    results["dib"] = _safe_call(has_I, dib, NAN, Ia, Ib, Ic)
    results["current_unbalance_rms"] = _safe_call(has_I, current_unbalance_rms, NAN, Ia, Ib, Ic)
    results["phase_unbalance_metrics"] = _safe_call(has_I, phase_unbalance_metrics, _PHASE_NANS.copy(), Ia, Ib, Ic)
    results["cur_ratio"] = _safe_call(has_I, cur_ratio, NAN, Ia, Ib, Ic)
    results["cur_dev_ratio"] = _safe_call(has_I, cur_dev_ratio, NAN, Ia, Ib, Ic)
    results["neutral_from_trms_120deg"] = _safe_call(has_I, neutral_from_trms_120deg, NAN, Ia, Ib, Ic)

    # --- Neutral from PF ---
    results["neutral_from_trms_and_pf"] = _safe_call(
        has_I and has_pf, neutral_from_trms_and_pf, NAN, Ia, Ib, Ic, pfA, pfB, pfC, lagging=lagging
    )

    # --- Voltage metrics ---
    results["vuf_magnitude"] = _safe_call(has_Vmag, vuf_magnitude, NAN, Va_mag, Vb_mag, Vc_mag)
    results["vuf_symmetrical"] = _safe_call(has_Vph, vuf_symmetrical, NAN, Va_phasor, Vb_phasor, Vc_phasor)

    # --- Sequence metrics ---
    results["sequence_unbalance_factors"] = _safe_call(
        has_Iph, sequence_unbalance_factors, _SEQ_NANS.copy(), Ia_phasor, Ib_phasor, Ic_phasor
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

    print("\nExample 2 — Currents + per-phase PF (neutral estimate with PF):")
    res2 = compute_meter_metrics(
        Ia=420.0, Ib=280.0, Ic=275.0,
        pfA=0.99, pfB=0.98, pfC=0.97,  # set lagging=False if capacitive
    )
    pprint(res2)

    print("\nExample 3 — Voltage magnitudes (VUF magnitude):")
    res3 = compute_meter_metrics(Va_mag=230.0, Vb_mag=226.0, Vc_mag=233.0)
    pprint(res3)

    print("\nExample 4 — Current and voltage phasors (sequence + VUF symmetrical):")
    # Example phasors: magnitude ∠ angle(deg)
    def phasor(mag, deg):
        return cmath.rect(mag, math.radians(deg))

    res4 = compute_meter_metrics(
        Ia_phasor=phasor(420, 0),
        Ib_phasor=phasor(280, -120),
        Ic_phasor=phasor(275, 120),
        Va_phasor=phasor(230, 0),
        Vb_phasor=phasor(230, -120),
        Vc_phasor=phasor(230, 120),
    )
    pprint(res4)

    print("\nDone.")