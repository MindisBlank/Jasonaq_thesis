"""Fetch and export Janitza data for every device in one substation.

Usage is intentionally minimal: provide the substation number and the variable
backend name, e.g. ``python scripts/run_substation_janitza.py 1353 I_Effective``.
All devices with a matching ``dnr_str`` in ``metadata/devices.csv`` will be
exported to ``results/sub<substation>_<window>/`` with one Parquet per device.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from phasebalance.plot_janitza import fetch_device_variable


DEFAULT_DEVICES = Path("metadata/devices.csv")
DEFAULT_CAPABILITIES = Path("metadata/capabilities.csv")
DEFAULT_TIMEBASE = "15m"
DEFAULT_START = "-7d"
DEFAULT_END = "now"
DEFAULT_WHICH = "avg"
RESULTS_ROOT = Path("results")
substation_slug = ""


def _slugify_timeframe(start: str, end: str) -> str:
    window = f"{start}_to_{end}"
    return re.sub(r"[^0-9A-Za-z]+", "_", window).strip("_") or "window"


def _load_substation_devices(substation: str) -> pd.Series:
    df = pd.read_csv(DEFAULT_DEVICES)
    if "dnr_str" not in df.columns or "device_id" not in df.columns:
        raise ValueError("metadata/devices.csv must contain dnr_str and device_id columns")

    mask = df["dnr_str"].astype(str).str.lower() == substation.lower()
    subset = df.loc[mask, "device_id"].dropna().unique()
    if subset.size == 0:
        raise ValueError(f"No devices found for substation '{substation}' in {DEFAULT_DEVICES}")

    return subset


def _export_device(device_id: int, variable: str, start: str, end: str) -> None:
    df = fetch_device_variable(
        device_id=device_id,
        variable_backend=variable,
        timebase=DEFAULT_TIMEBASE,
        start=start,
        end=end,
        capabilities_csv=DEFAULT_CAPABILITIES,
        phases=None,
        which=DEFAULT_WHICH,
        auth_token=None,
    )

    if df.empty:
        print(f"⚠️  No data fetched for device {device_id}; skipping export.")
        return

    folder = RESULTS_ROOT / f"sub{substation_slug}_{_slugify_timeframe(start, end)}"
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"device_{device_id}_{variable}.parquet"
    df.to_parquet(out_path)
    print(f"✅ Saved {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch one Janitza variable for every device in a substation and export Parquet files",
        add_help=True,
    )
    parser.add_argument("substation", help="Substation number (dnr_str)")
    parser.add_argument("variable", help="Variable backend name, e.g. I_Effective")
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Window start (default: -7d)",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help="Window end (default: now)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    global substation_slug
    substation_slug = str(args.substation)

    devices = _load_substation_devices(substation_slug)

    for device_id in devices:
        try:
            device_int = int(device_id)
        except (TypeError, ValueError):
            print(f"⚠️  Invalid device_id {device_id}; skipping.")
            continue

        _export_device(device_int, args.variable, args.start, args.end)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
