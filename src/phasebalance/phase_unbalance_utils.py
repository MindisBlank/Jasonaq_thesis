import cmath
import math

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
