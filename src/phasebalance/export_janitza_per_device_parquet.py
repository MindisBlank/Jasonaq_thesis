#!/usr/bin/env python3
# export_janitza_per_device_parquet.py

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd

# Reuse existing code (no re-implementations)
try:
    # if running as part of a package
    from .janitza_plot_multi_phase import (
        fetch_phase_series,
        _load_metadata,
        _devices_for_dnr,
        CHANNEL_SPEC_I,
        CHANNEL_SPEC_P,
        CHANNEL_SPEC_V,
        CHANNEL_SPEC_I4,
        CHANNEL_SPEC_SEQ_I,
    )
    from .phase_unbalance_utils import resolve_channels
except ImportError:  # pragma: no cover
    # if running as a plain script
    from janitza_plot_multi_phase import (
        fetch_phase_series,
        _load_metadata,
        _devices_for_dnr,
        CHANNEL_SPEC_I,
        CHANNEL_SPEC_P,
        CHANNEL_SPEC_V,
        CHANNEL_SPEC_I4,
        CHANNEL_SPEC_SEQ_I,
    )
    from phase_unbalance_utils import resolve_channels


PROFILE_TO_SPEC = {
    "3phase_I": CHANNEL_SPEC_I,       # IA, IB, IC
    "3phase_P": CHANNEL_SPEC_P,       # PA, PB, PC
    "3phase_V": CHANNEL_SPEC_V,       # VA, VB, VC
    "neutral_I": CHANNEL_SPEC_I4,     # IA (neutral channel)
    "sum13_Iseq": CHANNEL_SPEC_SEQ_I, # I0, I1, I2
}


def fetch_substation_timeseries_per_device(
    *,
    dnr_str: str,
    profile: str,
    devices_csv: Path,
    capabilities_csv: Path,
    timebase: str,
    start: str,
    end: str,
    which: str = "avg",
    auth_token: str | None = None,
) -> pd.DataFrame:
    """
    Fetch Janitza time series per device_id for one substation.
    Returns a tidy DataFrame with columns like:
      ts, dnr_str, device_id, name, <channels...>, (optional totals), plus *_type_backend trace columns
    """
    if profile not in PROFILE_TO_SPEC:
        raise ValueError(f"Unknown profile='{profile}'. Choose from: {sorted(PROFILE_TO_SPEC)}")

    devices, caps = _load_metadata(devices_csv, capabilities_csv)
    if caps is None or caps.empty:
        raise ValueError(f"capabilities_csv is required and must be non-empty: {capabilities_csv}")

    sub_devs = _devices_for_dnr(devices, dnr_str)
    if sub_devs.empty:
        raise ValueError(f"No devices found for dnr_str='{dnr_str}' in {devices_csv}")

    caps = caps.copy()
    caps["device_id"] = caps["device_id"].astype(str)

    channel_spec = PROFILE_TO_SPEC[profile]

    frames: list[pd.DataFrame] = []

    for _, dev in sub_devs.iterrows():
        did = int(dev["device_id"])
        name = str(dev.get("name", "") or "")

        plan = resolve_channels(
            caps,
            device_id=str(did),
            channels=channel_spec,
            require_all=False,   # don’t fail the whole device if one channel missing
            return_all=False,    # pick best candidate only (Input01 vs L1, etc.)
        )

        if not plan:
            print(f"⚠️  {dnr_str} | device {did}: could not resolve channels for profile={profile}")
            continue

        series_list: list[pd.Series] = []
        trace_cols: dict[str, str] = {}
        unit_hint = ""

        for logical_key, info in plan.items():
            vb = info["value_backend"]
            tb = info["type_backend"]
            trace_cols[f"{logical_key}_type_backend"] = str(tb)

            s, payload = fetch_phase_series(
                device_id=did,
                variable_backend=vb,
                phase_backend=tb,
                timebase=timebase,
                start=start,
                end=end,
                auth_token=auth_token,
                which=which,
            )
            if s.empty:
                continue

            # Detect “all zeros” series (useful for debugging dead channels)
            if s.fillna(0).eq(0).all():
                print(f"⚠️  {dnr_str} | device {did} | {logical_key} ({tb}): all zeros in window.")

            series_list.append(s.rename(logical_key))

            if not unit_hint:
                unit_hint = (payload or {}).get("valueType", {}).get("unit", "") or ""

        if not series_list:
            print(f"❌ {dnr_str} | device {did}: no data for any resolved channels in window.")
            continue

        df_dev = pd.concat(series_list, axis=1, join="outer").sort_index()
        if df_dev.empty:
            continue

        df_dev.index.name = "ts"
        df_dev = df_dev.reset_index()

        # Add identifiers
        df_dev["dnr_str"] = dnr_str
        df_dev["device_id"] = did
        df_dev["name"] = name
        df_dev["unit"] = unit_hint
        df_dev["profile"] = profile
        df_dev["timebase"] = timebase
        df_dev["window_start"] = start
        df_dev["window_end"] = end

        # Add traceability: which concrete channels were used
        for k, v in trace_cols.items():
            df_dev[k] = v

        # Optional totals (per device, NOT across devices)
        if profile == "3phase_I" and all(c in df_dev.columns for c in ("IA", "IB", "IC")):
            df_dev["I_total"] = df_dev[["IA", "IB", "IC"]].sum(axis=1, skipna=True)
        if profile == "3phase_P" and all(c in df_dev.columns for c in ("PA", "PB", "PC")):
            df_dev["P_total"] = df_dev[["PA", "PB", "PC"]].sum(axis=1, skipna=True)

        frames.append(df_dev)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["device_id", "ts"])
    return out


if __name__ == "__main__":
    # ------------------- EDIT ONLY THIS BLOCK -------------------
    DNR_STR   = "D1496"
    PROFILE   = "3phase_I"        # "3phase_I", "3phase_P", "3phase_V", "neutral_I", "sum13_Iseq"
    TIMEBASE  = "15m"
    START     = "2025-11-01 12:00"
    END       = "2025-11-03 12:00"
    WHICH     = "avg"             # "avg" | "min" | "max"

    DEVICES_CSV = Path("metadata/devices.csv")
    CAPS_CSV    = Path("metadata/capabilities.csv")

    OUT_DIR = Path("data")
    # ------------------------------------------------------------

    df = fetch_substation_timeseries_per_device(
        dnr_str=DNR_STR,
        profile=PROFILE,
        devices_csv=DEVICES_CSV,
        capabilities_csv=CAPS_CSV,
        timebase=TIMEBASE,
        start=START,
        end=END,
        which=WHICH,
        auth_token=None,
    )

    if df.empty:
        print("No data fetched — nothing to save.")
        raise SystemExit(0)

    fmt = "%Y-%m-%d %H:%M"
    start_tag = datetime.strptime(START, fmt).strftime("%Y%m%d_%H%M")
    end_tag   = datetime.strptime(END, fmt).strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"janitza_{DNR_STR}_{PROFILE}_per_device_{start_tag}_{end_tag}.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    print(df.head(3).to_string(index=False))
