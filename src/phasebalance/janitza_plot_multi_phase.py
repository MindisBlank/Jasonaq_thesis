#!/usr/bin/env python3
# janitza_plot_multi_phase.py

import json
from typing import Iterable, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from janitza_fetch import fetch_hist_json


def _series_from_values(payload: Dict, which: str = "avg") -> pd.Series:
    """
    Convert {"values":[{"startTime":ns, "avg":x, ...}, ...]} to a pandas Series
    indexed by UTC timestamps with the chosen statistic (avg|min|max).
    """
    vals = payload.get("values", []) if isinstance(payload, dict) else []
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
        return pd.Series(dtype=float, name=which)

    s = pd.Series(data).sort_index()
    # Convert Unix ns → UTC datetime index
    s.index = pd.to_datetime(s.index, unit="ns", utc=True)
    s.name = which
    return s


def fetch_phase_series(
    *,
    device_id: int | str,
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

    s = _series_from_values(payload, which=which)
    if s.empty:
        print(f"⚠️  device {device_id} phase {phase_backend}: no data in window.")
    return s.rename(phase_backend), payload


def plot_multi_phase(
    *,
    device_id: int | str,
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


if __name__ == "__main__":
    # Example: two Janitza devices at one substation, aggregate per phase
    # plot_multi_phase_multi_device(
    #     device_ids=(301, 309),         # ← put your two device_ids here
    #     name="D0613 total",
    #     variable_backend="I_Effective",
    #     phases=("Input01", "Input02", "Input03"),  # or ("L1","L2","L3")
    #     timebase="15m",
    #     start="2025-10-21 12:00",
    #     end="2025-10-27 12:00",
    #     which="avg",
    #     auth_token=None,
    #     combine="sum",                  # "sum" or "mean"
    #     show=True,
    # )

    plot_multi_phase(
        device_id=369,
        name="D1390",
        variable_backend="I_Effective",
        phases=("L1", "L2", "L3"),   # ← change to ("Input01","Input02","Input03") if your currents are on inputs
        timebase="10m",
        start="2025-10-21 12:00",
        end="2025-10-27 12:00",
        which="avg",
        auth_token=None,
        show=True,
    )