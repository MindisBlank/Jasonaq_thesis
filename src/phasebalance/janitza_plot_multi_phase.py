#!/usr/bin/env python3
# janitza_plot_multi_phase.py

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
try:
    from .janitza_fetch import fetch_hist_json
    from .phase_unbalance_utils import _series_from_values, resolve_channels
except ImportError:  # pragma: no cover - for running as a script
    from janitza_fetch import fetch_hist_json
    from phase_unbalance_utils import _series_from_values, resolve_channels





CHANNEL_SPEC_I = {
    "IA": {"value_backend": "I_Effective", "type_candidates": ["Input01","Input05", "L1","L5"]},
    "IB": {"value_backend": "I_Effective", "type_candidates": ["Input02","Input06", "L2","L6"]},
    "IC": {"value_backend": "I_Effective", "type_candidates": ["Input03","Input07", "L3","L7"]},
}

CHANNEL_SPEC_I4 = {
    "IA": {"value_backend": "I_Effective", "type_candidates": ["Input04","Input08","L4"]},
}

CHANNEL_SPEC_SEQ_I = {
    "I0": {"value_backend": "ZeroPhaseSeq_I", "type_candidates":["Overall","SUM13"]},
    "I1": {"value_backend": "PositivePhaseSeq_I", "type_candidates": ["Overall","SUM13"]},
    "I2": {"value_backend": "NegativePhaseSeq_I", "type_candidates": ["Overall","SUM13"]},
}


def fetch_phase_series(
    *,
    device_id: int,
    variable_backend: str,
    phase_backend: str,
    timebase: int | str,
    start: str,
    end: str,
    auth_token: Optional[str] = None,
    which: str = "avg",
) -> Tuple[pd.Series, Dict]:
    """
    Fetch one phase and return (Series, raw_payload).
    Series is indexed by UTC timestamps. If fetch fails or no points, returns empty Series.
    """
    payload = fetch_hist_json(
            device_id=device_id,
            variable_backend=variable_backend,
            phase_backend=phase_backend,
            timebase=timebase,
            start=start,
            end=end,
            auth_token=auth_token,
            dry_run=False,
        )

    s = _series_from_values(payload, which)
    if s.empty:
        print(f"⚠️  device {device_id} phase {phase_backend}: no data in window.")
    return s.rename(phase_backend), payload


def plot_multi_phase(
    *,
    device_id: int ,
    name: Optional[str] = None,
    variable_backend: str = "I_Effective",
    phases: Iterable[str] = ("L1", "L2", "L3"),
    timebase: int | str = "15m",
    start: str = "2025-10-01 00:00",
    end: str = "2025-10-02 00:00",
    which: str = "avg",  # "avg" | "min" | "max"
    auth_token: Optional[str] = None,
    show: bool = True,
):
    """
    Fetch multiple phases for one device and plot them on a single chart.
    Returns (fig, ax, frame) where frame is a tidy DataFrame with one column per phase.
    """
    series_list = []
    payloads = {}

    for ph in phases:
        s, p = fetch_phase_series(
            device_id=device_id,
            variable_backend=variable_backend,
            phase_backend=ph,
            timebase=timebase,
            start=start,
            end=end,
            auth_token=auth_token,
            which=which,
        )
        payloads[ph] = p
        if not s.empty:
            series_list.append(s)

    if not series_list:
        print(f"❌ device {device_id}: no data for any requested phases {list(phases)}.")
        return None, None, pd.DataFrame()

    # Align all series by timestamp (inner join to keep only common buckets)
    df = pd.concat(series_list, axis=1, join="inner").sort_index()
    if df.empty:
        print(f"⚠️  device {device_id}: no overlapping timestamps across requested phases.")
        return None, None, pd.DataFrame()

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(ax=ax)  # let matplotlib pick distinct colors

    title_name = name or f"Device {device_id}"
    ax.set_title(f"{title_name} : {variable_backend} ({which})  [{start} → {end}]  TB={timebase}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(payloads[next(iter(payloads))].get("valueType", {}).get("unit", ""))  # try unit, else blank
    ax.grid(True, alpha=0.3)
    ax.legend(title="Phase/Channel", ncols=min(df.shape[1], 3))

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax, df


def _combine_phase_series(phase_to_series_list: dict, how: str = "sum") -> pd.DataFrame:
    """
    phase_to_series_list: {"Input01": [Series(dev1), Series(dev2), ...], ...}
    Returns a DataFrame with one column per phase where each column is the
    sum/mean across devices, aligned by timestamp.
    """
    cols = {}
    for phase, series_list in phase_to_series_list.items():
        if not series_list:
            continue
        # align all device series for this phase (outer → keep all timestamps)
        mat = pd.concat(series_list, axis=1, join="outer").sort_index()
        if how == "mean":
            col = mat.mean(axis=1, skipna=True)
        else:  # default sum
            col = mat.sum(axis=1, skipna=True)
        col.name = phase
        cols[phase] = col
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols.values(), axis=1, join="outer").sort_index()
    # Drop rows where all phases are NaN
    df = df.dropna(how="all")
    return df


def plot_multi_phase_multi_device(
    *,
    device_ids: Iterable[int | str],
    name: Optional[str] = None,
    variable_backend: str = "I_Effective",
    phases: Iterable[str] = ("Input01", "Input02", "Input03"),
    timebase: int | str = "15m",
    start: str = "2025-10-01 00:00",
    end: str = "2025-10-02 00:00",
    which: str = "avg",           # "avg" | "min" | "max"
    auth_token: Optional[str] = None,
    combine: str = "sum",         # "sum" or "mean" across device_ids
    show: bool = True,
):
    """
    Fetch the same set of phases from multiple devices (e.g., two Janitza meters
    at one substation), combine per phase across devices (sum or mean), and plot
    one line per phase. Legend shows only the phase names.
    """
    # Gather per-phase series from all devices
    phase_to_series_list: dict[str, list[pd.Series]] = {ph: [] for ph in phases}
    unit_hint = ""

    for did in device_ids:
        for ph in phases:
            s, payload = fetch_phase_series(
                device_id=did,
                variable_backend=variable_backend,
                phase_backend=ph,
                timebase=timebase,
                start=start,
                end=end,
                auth_token=auth_token,
                which=which,
            )
            if not s.empty:
                phase_to_series_list[ph].append(s)
                if not unit_hint:
                    unit_hint = (payload or {}).get("valueType", {}).get("unit", "")

    # Build combined per-phase frame
    df = _combine_phase_series(phase_to_series_list, how=combine)
    if df.empty:
        print(f"❌ No data to plot for {list(device_ids)} phases {list(phases)} in window.")
        return None, None, pd.DataFrame()

    # Optional: keep only timestamps present in ALL phases (tighter comparison)
    # df = df.dropna(how="any")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(ax=ax)  # one line per phase

    title_name = name or f"Devices {','.join(map(str, device_ids))}"
    ax.set_title(f"{title_name} : {variable_backend} ({which})  [{start} → {end}]  TB={timebase}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(unit_hint)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Phase", ncols=min(df.shape[1], 3))

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax, df


def _load_metadata(devices_csv: Path, capabilities_csv: Optional[Path]) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load devices and optional capabilities metadata.

    Parameters
    ----------
    devices_csv
        Path to `devices.csv` metadata file.
    capabilities_csv
        Path to `capabilities.csv` metadata file. If ``None`` or missing, only the
        devices table is returned.
    """

    devices = pd.read_csv(devices_csv)
    caps: Optional[pd.DataFrame] = None
    if capabilities_csv is not None and Path(capabilities_csv).exists():
        caps = pd.read_csv(capabilities_csv)
    return devices, caps


def _devices_for_dnr(devices: pd.DataFrame, dnr_str: str) -> pd.DataFrame:
    """Filter the devices table for a specific ``dnr_str``."""

    if "dnr_str" not in devices.columns:
        raise KeyError("devices.csv must contain a 'dnr_str' column")
    mask = devices["dnr_str"].astype(str).str.fullmatch(str(dnr_str))
    return devices.loc[mask].copy()


def _phases_for_measurement(
    capabilities: Optional[pd.DataFrame],
    device_ids: Sequence[int | str],
    value_backend: str,
) -> list[str]:
    """Infer the list of phase/type_backend channels for the given measurement."""

    default_currents = ["L1", "L2", "L3"]
    default_inputs = ["Input01", "Input02", "Input03"]

    if capabilities is None or capabilities.empty:
        return default_currents if "I_" in value_backend else default_inputs

    caps = capabilities.copy()
    if "device_id" not in caps.columns or "value_backend" not in caps.columns:
        return default_currents if "I_" in value_backend else default_inputs

    caps["device_id"] = caps["device_id"].astype(str)
    device_ids_str = {str(did) for did in device_ids}

    phase_candidates: dict[str, dict[str, str]] = {}
    caps["value_backend"] = caps["value_backend"].astype(str)

    for did in device_ids_str:
        subset = caps[
            (caps["device_id"] == did)
            & (caps["value_backend"] == str(value_backend))
        ]
        type_candidates = subset["type_backend"].dropna().astype(str).tolist()
        spec = {
            f"{value_backend}_{idx}": {
                "value_backend": value_backend,
                "type_candidates": [candidate],
            }
            for idx, candidate in enumerate(type_candidates)
        }
        resolved = resolve_channels(caps, device_id=did, channels=spec, require_all=False)
        for plan in resolved.values():
            phase = plan.get("type_backend")
            if phase:
                phase_candidates.setdefault(phase, plan)

    if not phase_candidates:
        return default_currents if "I_" in value_backend else default_inputs

    return sorted(phase_candidates.keys())


def plot_substation_measurement(
    *,
    dnr_str: str,
    value_backend: str,
    devices_csv: Path = Path("metadata/devices.csv"),
    capabilities_csv: Optional[Path] = Path("metadata/capabilities.csv"),
    timebase: int | str = "15m",
    start: str = "2025-10-01 00:00",
    end: str = "2025-10-02 00:00",
    which: str = "avg",
    auth_token: Optional[str] = None,
    combine: str = "sum",
    show: bool = True,
    profile: str = "3phase_I",
):
    """
    Plot a measurement for all devices that belong to a substation.

    Profiles
    --------
    profile = "3phase_I"
        Uses CHANNEL_SPEC_I and plots three lines: IA, IB, IC.
        Each line is the sum/mean of the corresponding phase across all devices.

    profile = "neutral_I"
        Uses CHANNEL_SPEC_I4 and plots a single line: IA (neutral total).

    profile = "sum13_Iseq"
        Uses CHANNEL_SPEC_SEQ_I and plots I0, I1, I2 (sequence currents)
        aggregated across devices.

    Any other profile (or if you later extend this) falls back to the old
    generic behaviour using _phases_for_measurement + plot_multi_phase_multi_device.
    """

    # --- Load metadata and select devices for this substation ---
    devices, caps = _load_metadata(devices_csv, capabilities_csv)
    substation_devices = _devices_for_dnr(devices, dnr_str)
    if substation_devices.empty:
        raise ValueError(f"No devices found for dnr_str '{dnr_str}'.")

    device_ids = substation_devices["device_id"].astype(str).tolist()

    # If we don't have capabilities, we can't do the fancy profile mapping,
    # so fall back to the old generic behaviour.
    if caps is None or caps.empty:
        phases = _phases_for_measurement(None, device_ids, value_backend)
        title = f"Substation {dnr_str}"
        return plot_multi_phase_multi_device(
            device_ids=device_ids,
            name=title,
            variable_backend=value_backend,
            phases=phases,
            timebase=timebase,
            start=start,
            end=end,
            which=which,
            auth_token=auth_token,
            combine=combine,
            show=show,
        )

    caps = caps.copy()
    caps["device_id"] = caps["device_id"].astype(str)

    # --- Choose which spec to use based on profile ---
    if profile == "3phase_I":
        channel_spec = CHANNEL_SPEC_I
    elif profile == "neutral_I":
        channel_spec = CHANNEL_SPEC_I4
    elif profile == "sum13_Iseq":
        channel_spec = CHANNEL_SPEC_SEQ_I
    else:
        # Unknown profile → fall back to generic behaviour
        phases = _phases_for_measurement(caps, device_ids, value_backend)
        title = f"Substation {dnr_str} (profile={profile})"
        return plot_multi_phase_multi_device(
            device_ids=device_ids,
            name=title,
            variable_backend=value_backend,
            phases=phases,
            timebase=timebase,
            start=start,
            end=end,
            which=which,
            auth_token=auth_token,
            combine=combine,
            show=show,
        )

    # --- Gather series per canonical phase key (IA, IB, IC, Neutral, I0, I1, I2, ...) ---
    phase_to_series_list: dict[str, list[pd.Series]] = {
        key: [] for key in channel_spec.keys()
    }
    unit_hint = ""

    for did in device_ids:
        # Ask resolve_channels to return *all* matches
        plans = resolve_channels(
            caps,
            device_id=str(did),
            channels=channel_spec,
            require_all=False,   # some devices may be missing some channels
            return_all=True,     # 👈 NEW: get type_backends list
        )
        if not plans:
            print(f"⚠️  device {did}: no channels resolved for profile={profile}")
            continue
        for phase_key, plan in plans.items():
            vb = plan.get("value_backend", value_backend)

            # Use all type_backends if present, otherwise fall back to single one
            tb_list = plan.get("type_backends")
            if not tb_list:
                tb = plan.get("type_backend")
                if not tb:
                    continue
                tb_list = [tb]
                # 🔍 Debug print: which physical channels are used for this logical phase
            print(f"Substation {dnr_str} | device {did} | phase {phase_key} "
                    f"→ type_backends: {', '.join(map(str, tb_list))}")
            for type_backend in tb_list:
                s, payload = fetch_phase_series(
                    device_id=int(did),
                    variable_backend=vb,
                    phase_backend=type_backend,
                    timebase=timebase,
                    start=start,
                    end=end,
                    auth_token=auth_token,
                    which=which,
                )
                if not s.empty:
                    phase_to_series_list[phase_key].append(s)
                    if not unit_hint:
                        unit_hint = (payload or {}).get("valueType", {}).get("unit", "")

    # --- Combine across devices (sum or mean per phase) ---
    df = _combine_phase_series(phase_to_series_list, how=combine)
    if df.empty:
        print(f"❌ No data to plot for substation {dnr_str} (profile={profile}).")
        return None, None, pd.DataFrame()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(ax=ax)

    title = f"Substation {dnr_str} ({profile})"
    ax.set_title(f"{title} : {value_backend} ({which})  [{start} → {end}]  TB={timebase}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(unit_hint)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Phase / Channel", ncols=min(df.shape[1], 3))

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Janitza measurements for an entire substation.")
    parser.add_argument("--dnr_str",default="D0579", help="Substation identifier (dnr_str) as listed in devices.csv")
    parser.add_argument("--value_backend",default="I_Effective", help="Measurement backend variable, e.g. I_Effective")
    parser.add_argument("--profile",default="3phase_I",choices=["3phase_I", "neutral_I", "sum13_Iseq"],help="Which channel profile to plot: " "3phase_I = IA/IB/IC total, neutral_I = neutral current, " "sum13_Iseq = sequence currents (I0/I1/I2 from SUM13/Overall)",)
    parser.add_argument("--devices-csv", default="metadata/devices.csv", help="Path to devices metadata CSV")
    parser.add_argument("--capabilities-csv",default="metadata/capabilities.csv",help="Path to capabilities metadata CSV (optional)",)
    parser.add_argument("--timebase", default="15m", help="Time bucket size for the fetch requests")
    parser.add_argument("--start", default="2025-09-01 00:00", help="Start timestamp (UTC) for the window")
    parser.add_argument("--end", default="2025-10-02 00:00", help="End timestamp (UTC) for the window")
    parser.add_argument("--which", default="avg", choices=["avg", "min", "max"], help="Statistic to plot from the API")
    parser.add_argument("--combine", default="sum", choices=["sum", "mean"], help="How to combine multiple devices per phase")
    parser.add_argument("--auth-token", default=None, help="API auth token (overrides environment)")
    parser.add_argument("--no-show", action="store_true", help="Fetch data without displaying the plot")

    args = parser.parse_args()

    plot_substation_measurement(
        dnr_str=args.dnr_str,
        value_backend=args.value_backend,
        devices_csv=Path(args.devices_csv),
        capabilities_csv=Path(args.capabilities_csv) if args.capabilities_csv else None,
        timebase=args.timebase,
        start=args.start,
        end=args.end,
        which=args.which,
        auth_token=args.auth_token,
        combine=args.combine,
        show=not args.no_show,
        profile=args.profile, 
    )