"""Tools for fetching and plotting a Janitza variable for multiple devices.

This module reads the capabilities table to discover which phases/channels
devices expose for a given variable. It fetches historical values for all 
requested devices, combines them into a single DataFrame, and plots them
individually and as a grand total.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .janitza_fetch import fetch_hist_json
    from .phase_unbalance_utils import _series_from_values
except ImportError:
    from janitza_fetch import fetch_hist_json
    from phase_unbalance_utils import _series_from_values


def _load_channels(
    *,
    capabilities_csv: Path,
    device_id: int,
    variable_backend: str,
    phases: Optional[Iterable[str]] = None,
) -> list[str]:
    """Return the list of phase/type backends for this device+variable."""

    df = pd.read_csv(capabilities_csv)
    df["device_id"] = df["device_id"].astype(str)
    df["value_backend"] = df["value_backend"].astype(str)

    subset = df[
        (df["device_id"] == str(device_id))
        & (df["value_backend"].str.lower() == variable_backend.lower())
    ]

    if phases:
        phase_set = {p.lower() for p in phases}
        subset = subset[subset["type_backend"].str.lower().isin(phase_set)]

    channels = (
        subset["type_backend"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if not channels:
        # We return empty list instead of raising error to allow partial success
        # when fetching multiple devices
        return []

    return channels


def fetch_multidevice_variable(
    *,
    device_ids: list[int],
    variable_backend: str,
    timebase: str | int,
    start: str,
    end: str,
    capabilities_csv: Path = Path("metadata/capabilities.csv"),
    phases: Optional[Iterable[str]] = None,
    which: str = "avg",
    auth_token: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch variable for multiple devices and combine into one DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns are named 'D{device_id}:{channel}'.
    """

    all_series = {}

    for dev_id in set(device_ids):
        # 1. Discover channels for this specific device
        channels = _load_channels(
            capabilities_csv=capabilities_csv,
            device_id=dev_id,
            variable_backend=variable_backend,
            phases=phases,
        )

        if not channels:
            print(f"⚠️  Skipping Device {dev_id}: No channels found for {variable_backend}.")
            continue

        # 2. Fetch data for each channel
        for phase_backend in channels:
            payload = fetch_hist_json(
                device_id=dev_id,
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
                print(f"⚠️  No data for Device {dev_id} - {phase_backend} in window.")
            else:
                # 3. Rename series to include Device ID so columns don't collide
                unique_col_name = f"D{dev_id}:{phase_backend}"
                all_series[unique_col_name] = s.rename(unique_col_name)

    if not all_series:
        return pd.DataFrame()

    # Combine all devices and phases into one big DataFrame
    df = pd.concat(all_series.values(), axis=1, join="outer").sort_index()
    return df


def plot_multidevice(
    df: pd.DataFrame,
    *,
    device_ids: list[int],
    variable_backend: str,
    which: str,
    start: str,
    end: str,
    show: bool = True,
):
    """Render two subplots: individual channels (all devices) and a grand total."""

    if df.empty:
        print("❌ Nothing to plot: empty DataFrame.")
        return None, None

    # Create two subplots sharing the X-axis
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # --- Subplot 1: All Individual Channels ---
    df.plot(ax=ax1)
    
    dev_str = ", ".join(map(str, device_ids))
    if len(device_ids) > 5:
        dev_str = f"{len(device_ids)} Devices"

    ax1.set_title(
        f"Devices [{dev_str}] – {variable_backend} ({which}) – Individual Channels"
    )
    ax1.set_ylabel(variable_backend)
    ax1.grid(True, alpha=0.3)
    
    # Place legend outside if there are too many items
    if len(df.columns) > 10:
        ax1.legend(title="Device:Channel", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    else:
        ax1.legend(title="Device:Channel", loc="best")

    # --- Subplot 2: Grand Total (Sum) ---
    # Sum across all columns (all devices, all phases)
    combined_series = df.sum(axis=1)
    combined_series.plot(ax=ax2, color="black", linestyle="-", linewidth=2)
    
    ax2.set_title("Grand Total (Sum of all devices & phases)")
    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel(f"Total {variable_backend}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(["Grand Total"], loc="upper right")

    # Main title
    fig.suptitle(f"Multi-Device Analysis: {start} → {end}", fontsize=12)

    plt.tight_layout()
    
    if show:
        plt.show()

    return fig, (ax1, ax2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=( "Fetch channels for multiple Janitza devices and plot them individually and combined." ) )
    parser.add_argument( "--device-id",type=int,nargs='+',required=True, help="List of Janitza device IDs (e.g. 207 208 209)")    # Changed to nargs='+' to accept multiple IDs
    parser.add_argument("--variable",  default="I_Effective", help="Variable backend name (e.g. I_Effective, U_Effective)",)
    parser.add_argument("--phases",nargs="+",help="Optional list of phase/type backends to include (e.g. Input01 Input02)",)
    parser.add_argument("--timebase", default="15m", help="Timebase like 15m/900/1h")
    parser.add_argument("--start", default="-7d", help="Window start (e.g. -7d, 2025-01-01 00:00)")
    parser.add_argument("--end", default="now", help="Window end (e.g. now, 2025-01-02 00:00)")
    parser.add_argument("--which",choices=["avg", "min", "max"],default="avg",help="Which field to extract from samples", )
    parser.add_argument( "--capabilities-csv",type=Path,default=Path("metadata/capabilities.csv"),help="Path to capabilities.csv generated by janitza_value_fetch",)
    parser.add_argument("--auth-token", help="Optional Bearer token for GridVis", default=None)
    parser.add_argument("--parquet",type=Path,help="Save the fetched DataFrame to this Parquet path", )
    parser.add_argument( "--no-plot",action="store_true", help="Skip plotting", )
    return parser


def main(argv: Optional[list[str]] = None):
    args = _build_arg_parser().parse_args(argv)

    df = fetch_multidevice_variable(
        device_ids=args.device_id,
        variable_backend=args.variable,
        timebase=args.timebase,
        start=args.start,
        end=args.end,
        capabilities_csv=args.capabilities_csv,
        phases=args.phases,
        which=args.which,
        auth_token=args.auth_token,
    )

    if args.parquet:
        if df.empty:
            print("⚠️  Skipping Parquet export: no data fetched.")
        else:
            args.parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(args.parquet)
            print(f"✅ Saved Parquet to {args.parquet}")

    if not args.no_plot:
        plot_multidevice(
            df,
            device_ids=args.device_id,
            variable_backend=args.variable,
            which=args.which,
            start=args.start,
            end=args.end,
            show=True,
        )


if __name__ == "__main__":
    main()