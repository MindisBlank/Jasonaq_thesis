"""Tools for fetching and plotting a single Janitza variable for one device.

This module reads the capabilities table to discover which phases/channels a
device exposes for a given variable (e.g., ``I_Effective``). It then calls the
existing :func:`fetch_hist_json` helper to pull historical values for each
channel over a configurable window, returning a tidy ``DataFrame`` indexed by
UTC timestamps. Results can be visualised with ``matplotlib`` or saved as a
Parquet file.
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
        raise ValueError(
            f"No channels found for device {device_id} and variable {variable_backend} "
            f"in {capabilities_csv}."
        )

    return channels


def fetch_device_variable(
    *,
    device_id: int,
    variable_backend: str,
    timebase: str | int,
    start: str,
    end: str,
    capabilities_csv: Path = Path("metadata/capabilities.csv"),
    phases: Optional[Iterable[str]] = None,
    which: str = "avg",
    auth_token: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch one variable for a device across all available channels.

    Parameters
    ----------
    device_id
        Janitza device identifier.
    variable_backend
        Backend name such as ``"I_Effective"`` or ``"U_Effective"``.
    timebase
        Cadence accepted by :func:`fetch_hist_json`, e.g., ``"15m"`` or ``900``.
    start, end
        Time window (e.g., ``"-7d"`` to ``"now"``).
    capabilities_csv
        Path to the capabilities table produced by ``janitza_value_fetch.py``.
    phases
        Optional iterable of phase/type backends to restrict to (e.g.,
        ``["Input01", "Input02", "Input03"]``). When omitted, all available
        channels for the variable are pulled.
    which
        Which field to extract from each sample: ``"avg"`` (default), ``"min"``,
        or ``"max"``.
    auth_token
        Optional bearer token for GridVis.

    Returns
    -------
    pandas.DataFrame
        Columns correspond to phase/type backends, indexed by UTC timestamps.
    """

    channels = _load_channels(
        capabilities_csv=capabilities_csv,
        device_id=device_id,
        variable_backend=variable_backend,
        phases=phases,
    )

    series = {}
    for phase_backend in channels:
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
            print(f"⚠️  No data for {phase_backend} in window {start} → {end}.")
        series[phase_backend] = s.rename(phase_backend)

    if not series:
        return pd.DataFrame()

    df = pd.concat(series.values(), axis=1, join="outer").sort_index()
    return df


def plot_variable(
    df: pd.DataFrame,
    *,
    device_id: int,
    variable_backend: str,
    which: str,
    start: str,
    end: str,
    show: bool = True,
):
    """Render a simple line plot for the fetched variable."""

    if df.empty:
        print("❌ Nothing to plot: empty DataFrame.")
        return None, None

    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(ax=ax)

    ax.set_title(
        f"Device {device_id} – {variable_backend} ({which})  [{start} → {end}]"
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(variable_backend)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Phase/Channel", ncols=min(df.shape[1], 3))

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch all channels for a single Janitza variable on one device and "
            "either plot or save as Parquet."
        )
    )

    parser.add_argument("--device-id", type=int, required=True, help="Janitza device ID")
    parser.add_argument(
        "--variable",
        default="I_Effective",
        help="Variable backend name (e.g. I_Effective, U_Effective)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        help="Optional list of phase/type backends to include (e.g. Input01 Input02)",
    )
    parser.add_argument("--timebase", default="15m", help="Timebase like 15m/900/1h")
    parser.add_argument("--start", default="-7d", help="Window start (e.g. -7d, 2025-01-01 00:00)")
    parser.add_argument("--end", default="now", help="Window end (e.g. now, 2025-01-02 00:00)")
    parser.add_argument(
        "--which",
        choices=["avg", "min", "max"],
        default="avg",
        help="Which field to extract from samples",
    )
    parser.add_argument(
        "--capabilities-csv",
        type=Path,
        default=Path("metadata/capabilities.csv"),
        help="Path to capabilities.csv generated by janitza_value_fetch",
    )
    parser.add_argument("--auth-token", help="Optional Bearer token for GridVis", default=None)
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Save the fetched DataFrame to this Parquet path",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (useful when only exporting Parquet)",
    )

    return parser


def main(argv: Optional[list[str]] = None):
    args = _build_arg_parser().parse_args(argv)

    df = fetch_device_variable(
        device_id=args.device_id,
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
        plot_variable(
            df,
            device_id=args.device_id,
            variable_backend=args.variable,
            which=args.which,
            start=args.start,
            end=args.end,
            show=True,
        )


if __name__ == "__main__":
    main()
