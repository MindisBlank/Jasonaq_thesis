#src/phasebalance/janitza_report_device.py
"""Generate an imbalance report for a single Janitza device.

This script fetches three-phase current and voltage time series for a device
from the Janitza REST API, computes per-timestep imbalance metrics, detects
events, summarises their characteristics, and produces a set of diagnostic
plots along with CSV outputs describing the metrics and detected events.

The implementation mirrors the scraper's helper utilities as requested in the
user instructions.
"""

from __future__ import annotations
from typing import Any
import argparse
import datetime as dt
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import mpltern
from janitza_fetch import fetch_hist_json
from janitza_scrapper import resolve_channels
from phase_unbalance_utils import cur_ratio,cur_dev_ratio

CHANNEL_SPEC_I = {
        "IA": {"value_backend": "I_Effective", "type_candidates": ["Input01","Input05", "L1"]},
        "IB": {"value_backend": "I_Effective", "type_candidates": ["Input02","Input06", "L2"]},
        "IC": {"value_backend": "I_Effective", "type_candidates": ["Input03","Input07", "L3",]},
    }

CHANNEL_SPEC_V = {
        "VA": {"value_backend": "U_Effective", "type_candidates": ["Input01","Input05","L1"]},
        "VB": {"value_backend": "U_Effective", "type_candidates": ["Input02","Input06","L2"]},
        "VC": {"value_backend": "U_Effective", "type_candidates": ["Input03","Input07","L3"]},
    }

PAIR_COLORS = {"Ia-Ib": "#1b9e77", "Ib-Ic": "#d95f02", "Ic-Ia": "#7570b3"}
SEVERITY_COLORS = {
    "mild": "#ccebc5",
    "moderate": "#fdb462",
    "severe": "#fb8072",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

def _has_values(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    vals = obj.get("values")
    return isinstance(vals, list) and len(vals) > 0

def _series_from_values(obj: object, label: str) -> pd.Series:
    vals = obj.get("values", []) if isinstance(obj, dict) else []
    data: Dict[int, float] = {}
    for entry in vals:
        if not isinstance(entry, dict):
            continue
        st = entry.get("startTime")
        avg = entry.get("avg")
        if st is None or avg is None:
            continue
        try:
            data[int(st)] = float(avg)
        except (TypeError, ValueError):
            continue

    if not data:
        return pd.Series(dtype=float, name=label)

    series = pd.Series(data).sort_index()
    series.name = label
    series.index = pd.to_datetime(series.index, unit="ns")
    return series

def _vec_metric(fn, a, b, c):
    f = np.vectorize(lambda x,y,z: float(fn(float(x), float(y), float(z))))
    return f(a, b, c)

# culprit in 2 lines
_CANON = {"IaIb":"Ia-Ib","IbIa":"Ia-Ib","IbIc":"Ib-Ic","IcIb":"Ib-Ic","IcIa":"Ic-Ia","IaIc":"Ic-Ia"}
def label_culprit_pair(row: pd.Series) -> str:
    hi, lo = row[["Ia","Ib","Ic"]].idxmax(), row[["Ia","Ib","Ic"]].idxmin()
    hi_str, lo_str = str(hi), str(lo)
    return _CANON.get(hi_str.replace("a","A")+lo_str.replace("a","A"), "Ia-Ib")


def _contiguous_indices(mask: pd.Series) -> List[pd.Index]:
    if mask.empty:
        return []

    mask_values = mask.to_numpy()
    true_indices = np.where(mask_values)[0]
    if true_indices.size == 0:
        return []

    splits = np.where(np.diff(true_indices) > 1)[0] + 1
    blocks = np.split(true_indices, splits)
    indices: List[pd.Index] = []
    index_array = mask.index
    for block in blocks:
        indices.append(index_array[block])
    return indices


def detect_events(
    series: pd.Series,
    tau_on: float,
    tau_off: Optional[float] = None,
    gap_merge: int = 1,
    min_len: int = 1,
) -> List[Tuple[int, int]]:
    if series.empty:
        return []

    values = series.to_numpy()
    tau_off_val = tau_off if tau_off is not None else tau_on
    events: List[Tuple[int, int]] = []

    active = False
    start_idx: Optional[int] = None
    last_above: Optional[int] = None

    for idx, val in enumerate(values):
        if not active:
            if val >= tau_on:
                active = True
                start_idx = idx
                last_above = idx
        else:
            if val >= tau_off_val:
                last_above = idx
            else:
                if start_idx is not None and last_above is not None:
                    events.append((start_idx, last_above))
                active = False
                start_idx = None
                last_above = None

    if active and start_idx is not None and last_above is not None:
        events.append((start_idx, last_above))

    if not events:
        return []

    filtered_events = [
        (s, e) for (s, e) in events if (e - s + 1) >= max(1, min_len)
    ]

    if not filtered_events:
        return []

    merged_events: List[Tuple[int, int]] = []
    current_start, current_end = filtered_events[0]
    for start, end in filtered_events[1:]:
        gap = start - current_end - 1
        if gap <= max(0, gap_merge):
            current_end = max(current_end, end)
        else:
            merged_events.append((current_start, current_end))
            current_start, current_end = start, end
    merged_events.append((current_start, current_end))

    return merged_events

@dataclass
class EventRecord:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_h: float
    peak: float
    culprit_pair: str
    max_delta_I_at_peak: Any
    ia_peak: Any
    ib_peak: Any
    ic_peak: Any
    i_sum_peak: Any


def _compute_sample_delta(timebase: str, index: pd.Index) -> Optional[pd.Timedelta]:
    try:
        delta = pd.to_timedelta(timebase)
        if delta <= pd.Timedelta(0):
            raise ValueError
        return delta
    except Exception:
        pass

    if len(index) > 1:
        diffs = index.to_series().diff().dropna()
        if not diffs.empty:
            # Convert median float (nanoseconds) back to Timedelta
            return pd.to_timedelta(diffs.median(), unit="ns")

    return None


def summarise_events(
    df: pd.DataFrame,
    events: List[Tuple[int, int]],
    sample_delta: Optional[pd.Timedelta],
) -> List[EventRecord]:
    if not events:
        return []

    records: List[EventRecord] = []
    add_delta = sample_delta if sample_delta is not None else pd.Timedelta(0)

    for start_pos, end_pos in events:
        segment = df.iloc[start_pos : end_pos + 1]
        if segment.empty:
            continue
        start_time = segment.index[0]
        end_time = segment.index[-1]
        if add_delta > pd.Timedelta(0):
            duration = (end_time - start_time + add_delta).total_seconds() / 3600.0
        else:
            duration = max((end_time - start_time).total_seconds() / 3600.0, 0.0)

        peak_idx = segment["imbalance"].idxmax()
        peak_row = df.loc[peak_idx]
        records.append(
            EventRecord(
                start_time=start_time,
                end_time=end_time,
                duration_h=duration,
                peak=float(segment["imbalance"].max()),
                culprit_pair=str(peak_row["culprit_pair"]),
                max_delta_I_at_peak=peak_row["I_max"] - peak_row["I_min"],
                ia_peak=peak_row.get("Ia", np.nan),
                ib_peak=peak_row.get("Ib", np.nan),
                ic_peak=peak_row.get("Ic", np.nan),
                i_sum_peak=peak_row.get("I_sum", np.nan),
            )
        )

    return records


def summarise_bands(
    df: pd.DataFrame,
    series: pd.Series,
    events_per_band: Dict[str, List[EventRecord]],
    thresholds: Dict[str, float],
    sample_delta: Optional[pd.Timedelta],
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    sample_hours = (
        sample_delta.total_seconds() / 3600.0 if sample_delta is not None else np.nan
    )

    total_samples = float(len(series)) if len(series) else np.nan

    for band, threshold in thresholds.items():
        mask = series >= threshold
        count_above = int(mask.sum())
        taat_h = count_above * sample_hours if not math.isnan(sample_hours) else np.nan
        duty = (count_above / total_samples) if total_samples else np.nan
        csi = float(((series - threshold).clip(lower=0.0)).sum())

        events = events_per_band.get(band, [])
        event_count = len(events)
        durations = [ev.duration_h for ev in events if ev.duration_h is not None]
        mean_duration = float(np.mean(durations)) if durations else np.nan
        median_duration = float(np.median(durations)) if durations else np.nan
        lce = float(np.max(durations)) if durations else 0.0
        peaks = [ev.peak for ev in events]
        peak_p95 = float(np.percentile(peaks, 95)) if len(peaks) >= 1 else np.nan

        summary[f"{band}_taat_h"] = taat_h
        summary[f"{band}_duty"] = duty
        summary[f"{band}_csi"] = csi
        summary[f"{band}_event_count"] = float(event_count)
        summary[f"{band}_mean_duration_h"] = mean_duration
        summary[f"{band}_median_duration_h"] = median_duration
        summary[f"{band}_lce_h"] = lce
        summary[f"{band}_event_peak_p95"] = peak_p95

    return summary


def plot_colored_cur(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    device_id: int,
    metric_name: str,
    start: str,
    end: str,
    dpi: int,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax.plot(df.index, df["imbalance"], color="#333333", linewidth=1.2, label=metric_name)

    for pair, color in PAIR_COLORS.items():
        mask = df["culprit_pair"] == pair
        for block in _contiguous_indices(mask):
            if len(block) < 2:
                continue  # need at least two timestamps to draw a segment

            # Ensure DatetimeIndex -> Python datetimes -> matplotlib dates
            dt_block = pd.DatetimeIndex(block).to_pydatetime()
            x = mdates.date2num(dt_block)
            y = df.loc[block, "imbalance"].to_numpy(dtype=float)

            # Build segments as a list of 2-point tuples (type-checker friendly)
            segments = [((x[i], y[i]), (x[i+1], y[i+1])) for i in range(len(x) - 1)]

            lc = LineCollection(segments, colors=color, linewidths=2.0, alpha=0.9)
            ax.add_collection(lc)


    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_title(
        f"Device {device_id} - {metric_name} {start} → {end}",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ymin, ymax = ax.get_ylim()
    prev = 0.0
    order = ["mild", "moderate", "severe"]
    for band in order:
        threshold = thresholds.get(band)
        if threshold is None:
            continue
        lower = prev
        upper = threshold
        ax.axhspan(lower, upper, color=SEVERITY_COLORS[band], alpha=0.08)
        prev = threshold
    ax.set_ylim(ymin, max(ymax, prev * 1.2))

    handles = [Line2D([0], [0], color=color, lw=2) for color in PAIR_COLORS.values()]
    ax.legend(handles, PAIR_COLORS.keys(), title="Culprit pair", loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_event_stripes(
    events: Sequence[EventRecord],
    device_id: int,
    metric_name: str,
    start: str,
    end: str,
    dpi: int,
    output_path: str,
    sample_delta: Optional[pd.Timedelta],
) -> None:
    if not events:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
        ax.text(
            0.5,
            0.5,
            "No events detected",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        fig.suptitle(
            f"Device {device_id} - Events ({metric_name}) {start} → {end}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(events) * 0.3)), dpi=dpi)

    y_positions = np.arange(len(events))
    height = 0.8
    for idx, event in enumerate(events):
        start_num = float(mdates.date2num(event.start_time.to_pydatetime()))
        end_num = float(mdates.date2num(event.end_time.to_pydatetime()))
        if sample_delta is not None:
            end_num += float(sample_delta.total_seconds() / (24 * 3600))
        width = float(end_num - start_num)
        ax.broken_barh(
            [(start_num, width)],
            (float(y_positions[idx] - height / 2.0), float(height)),
            facecolors=PAIR_COLORS.get(event.culprit_pair, "#999999"),
            alpha=0.8,
        )
        ax.text(
            start_num,
            float(y_positions[idx]),
            f"{event.peak:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Evt {i+1}" for i in range(len(events))])
    ax.set_xlabel("Time")
    ax.set_title(
        f"Device {device_id} - Events ({metric_name}) {start} → {end}",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_severity_duration_scatter(
    events: Sequence[EventRecord],
    device_id: int,
    metric_name: str,
    dpi: int,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    if not events:
        ax.text(0.5, 0.5, "No events detected", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.suptitle(f"Severity vs Duration - Device {device_id}")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    for pair, color in PAIR_COLORS.items():
        pair_events = [ev for ev in events if ev.culprit_pair == pair]
        if not pair_events:
            continue
        durations = [ev.duration_h for ev in pair_events]
        peaks = [ev.peak for ev in pair_events]
        sizes = [max(ev.i_sum_peak, 0.0) * 5.0 for ev in pair_events]
        ax.scatter(durations, peaks, s=sizes, alpha=0.7, label=pair, color=color, edgecolors="k", linewidths=0.3)

    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel(f"Peak {metric_name} (%)")
    ax.set_title(f"Severity vs Duration - Device {device_id}")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(title="Culprit pair")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_diurnal_heatmap(
    df: pd.DataFrame,
    device_id: int,
    metric_name: str,
    dpi: int,
    output_path: str,
) -> None:
    data = df.copy()
    data.index = pd.DatetimeIndex(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.dayofweek
    pivot = data.pivot_table(index="weekday", columns="hour", values="imbalance", aggfunc="median")

    if pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.suptitle(f"Diurnal imbalance heatmap - Device {device_id}")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.set_yticklabels([weekdays[i] for i in pivot.index])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Weekday")
    ax.set_title(f"Diurnal imbalance heatmap - Device {device_id}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Median {metric_name} (%)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_exceedance_and_taat(
    series: pd.Series,
    thresholds: Dict[str, float],
    metrics: Dict[str, float],
    device_id: int,
    metric_name: str,
    dpi: int,
    output_path: str,
) -> None:
    if series.empty:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.suptitle(f"Device {device_id} - {metric_name}")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)

    sorted_vals = np.sort(series.to_numpy())
    exceedance = 1.0 - np.arange(1, len(sorted_vals) + 1) / float(len(sorted_vals))
    ax1.plot(sorted_vals, exceedance * 100.0, color="#4daf4a", linewidth=1.5)
    for band, threshold in thresholds.items():
        ax1.axvline(threshold, color=SEVERITY_COLORS.get(band, "#cccccc"), linestyle="--", alpha=0.7)
    ax1.set_xlabel(f"{metric_name} (%)")
    ax1.set_ylabel("% time ≥ τ")
    ax1.set_title("Exceedance curve")
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    taat_values = [metrics.get(f"{band}_taat_h", 0.0) or 0.0 for band in ("mild", "moderate", "severe")]
    bottoms = 0.0
    labels = ["Mild", "Moderate", "Severe"]
    for value, label, band in zip(taat_values, labels, ("mild", "moderate", "severe")):
        ax2.bar(
            ["TAAT"],
            [value],
            bottom=bottoms,
            color=SEVERITY_COLORS.get(band, "#cccccc"),
            label=label,
        )
        bottoms += value
    ax2.set_ylabel("Hours")
    ax2.set_title("TAAT by severity")
    ax2.legend()
    fig.suptitle(f"Device {device_id} - {metric_name}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ternary(
    df: pd.DataFrame,
    device_id: int,
    dpi: int,
    output_path: str,
) -> None:

    valid = df[["Ia", "Ib", "Ic"]].dropna()
    if valid.empty:
        logging.info("No data for ternary plot.")
        return

    totals = valid.sum(axis=1).replace(0, np.nan)
    normalized = valid.divide(totals, axis=0).dropna()
    if normalized.empty:
        logging.info("No valid normalized current data for ternary plot.")
        return

    # --- Convert to numpy arrays (Pylance-friendly) ---
    x = normalized["Ia"].to_numpy(dtype=float)
    y = normalized["Ib"].to_numpy(dtype=float)
    z = normalized["Ic"].to_numpy(dtype=float)

    # Build colors aligned to normalized index
    pairs = df.loc[normalized.index, "culprit_pair"].astype(str).tolist()
    colors = [PAIR_COLORS.get(p, "#999999") for p in pairs]

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="ternary")  # type: ignore[arg-type]

    # Pylance sometimes complains about mpltern's scatter signature; this is fine at runtime.
    size = 10.0
    ax.scatter(x, y, z, c=colors, alpha=0.6, s=size)  # type: ignore[call-arg]

    # Use getattr to safely call ternary-axis label setters so static type-checkers don't flag unknown attributes.
    getattr(ax, "set_tlabel", lambda *args, **kwargs: None)("Ia fraction")
    getattr(ax, "set_llabel", lambda *args, **kwargs: None)("Ib fraction")
    getattr(ax, "set_rlabel", lambda *args, **kwargs: None)("Ic fraction")
    ax.set_title(f"Ternary current balance - Device {device_id}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Janitza device imbalance report generator")
    parser.add_argument("--device-id", required=True, help="Device identifier")
    parser.add_argument("--start", required=True, help="Start timestamp (e.g. '2025-09-01 12:00')")
    parser.add_argument("--end", required=True, help="End timestamp (e.g. '2025-10-01 12:00')")
    parser.add_argument("--timebase", required=True, help="Timebase (e.g. '15m')")
    parser.add_argument("--caps-csv", required=True, help="Path to capabilities.csv metadata")
    parser.add_argument("--outdir", default="results", help="Output directory root")
    parser.add_argument(
        "--metric",
        choices=["cur_ratio", "cur_dev_ratio"],
        default="cur_ratio",
        help="Imbalance metric to compute",
    )
    parser.add_argument("--mild", type=float, default=5.0, help="Mild severity threshold (%)")
    parser.add_argument("--moderate", type=float, default=10.0, help="Moderate severity threshold (%)")
    parser.add_argument("--severe", type=float, default=20.0, help="Severe severity threshold (%)")
    parser.add_argument(
        "--tau-off-ratio",
        type=float,
        default=0.9,
        help="Ratio applied to tau_on to obtain tau_off for hysteresis",
    )
    parser.add_argument("--gap-merge", type=int, default=1, help="Maximum gap (samples) to merge events")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum length (samples) for an event")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument("--no-ternary", action="store_true", help="Disable ternary plot even if mpltern is available")
    return parser.parse_args(argv)


def fetch_time_series(
    device_id: int,
    start: str,
    end: str,
    timebase: str,
    resolved_channels: Dict[str, Dict[str, str]],
    labels: Dict[str, str],
    sleep_s: float = 0.0,
) -> Dict[str, pd.Series]:
    series_map: Dict[str, pd.Series] = {}

    for key, info in resolved_channels.items():
        value_backend = info["value_backend"]   # we'll map this to variable_backend
        type_backend = info["type_backend"]     # we'll map this to phase_backend
        label = labels.get(key, key)

        # ✅ call with KEYWORDS and the CORRECT parameter names
        response = fetch_hist_json(
            device_id=device_id,
            variable_backend=value_backend,
            phase_backend=type_backend,
            timebase=timebase,
            start=start,
            end=end,
        )

        # Handle no data (fetcher returns None) or empty JSON (no "values")
        if not response or not _has_values(response):
            # keep an empty, correctly named series so later concat doesn’t break
            series_map[key] = pd.Series(dtype=float, name=label)
        else:
            series_map[key] = _series_from_values(response, label=label)

        if sleep_s > 0:
            time.sleep(sleep_s)

    return series_map



def assemble_dataframe(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    if not series_map:
        return pd.DataFrame()
    series_list = []
    rename_map = {"IA": "Ia", "IB": "Ib", "IC": "Ic", "VA": "Va", "VB": "Vb", "VC": "Vc"}
    for key, series in series_map.items():
        if series.empty:
            continue
        series = series.rename(rename_map.get(key, key))
        series_list.append(series)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1).sort_index()
    return df


def ensure_output_directory(base_outdir: str, device_id: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = os.path.join(base_outdir, f"report_device_{device_id}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    device_id = int(args.device_id)
    start = args.start
    end = args.end
    timebase = args.timebase

    try:
        cap_df = pd.read_csv(args.caps_csv)
    except Exception as exc:
        logging.error("Failed to read capabilities CSV: %s", exc)
        return 1

    current_channels = resolve_channels(cap_df, str(device_id), CHANNEL_SPEC_I, require_all=True)
    if not current_channels:
        logging.error("Currents could not be resolved for device %s. Aborting.", device_id)
        return 1

    voltage_channels = resolve_channels(cap_df, str(device_id), CHANNEL_SPEC_V, require_all=False)

    labels = {
        "IA": "Ia",
        "IB": "Ib",
        "IC": "Ic",
        "VA": "Va",
        "VB": "Vb",
        "VC": "Vc",
    }

    current_series = fetch_time_series(device_id, start, end, timebase, current_channels, labels, sleep_s=0.2)
    voltage_series = fetch_time_series(device_id, start, end, timebase, voltage_channels, labels, sleep_s=0.2)

    all_series = {**current_series, **voltage_series}
    df = assemble_dataframe(all_series)
    if df.empty:
        logging.error("No time series data available for device %s", device_id)
        return 1

    if not {"Ia", "Ib", "Ic"}.issubset(df.columns):
        logging.error("Required current channels Ia/Ib/Ic missing for device %s", device_id)
        return 1

    df = df.dropna(how="all")
    if df.empty:
        logging.error("No valid samples after dropna for device %s", device_id)
        return 1

    df[["Ia", "Ib", "Ic"]] = df[["Ia", "Ib", "Ic"]].astype(float)
    for col in ("Va", "Vb", "Vc"):
        if col in df.columns:
            df[col] = df[col].astype(float)


    i_max = df[["Ia", "Ib", "Ic"]].max(axis=1)
    i_min = df[["Ia", "Ib", "Ic"]].min(axis=1)

    metric_fn = cur_ratio if args.metric == "cur_ratio" else cur_dev_ratio
    df["imbalance"] = _vec_metric(metric_fn, df["Ia"], df["Ib"], df["Ic"])
    metric_name = "Current Unbalance Ratio(%)" if args.metric == "cur_ratio" else "Current Deviation Ratio Imbalance (%)"

    df["I_sum"] = df[["Ia", "Ib", "Ic"]].sum(axis=1)
    df["I_max"] = i_max
    df["I_min"] = i_min
    df["dominant_phase"] = df[["Ia", "Ib", "Ic"]].idxmax(axis=1)
    df["culprit_pair"] = df.apply(label_culprit_pair, axis=1)

    df = df.dropna(subset=["Ia", "Ib", "Ic", "imbalance"])
    if len(df) < 10:
        logging.error("Fewer than 10 valid samples after filtering; aborting.")
        return 1

    thresholds = {"mild": args.mild, "moderate": args.moderate, "severe": args.severe}
    sample_delta = _compute_sample_delta(timebase, df.index)

    events_records: Dict[str, List[EventRecord]] = {}
    for band, threshold in thresholds.items():
        tau_off = threshold * float(args.tau_off_ratio)
        events_idx = detect_events(
            df["imbalance"],
            tau_on=threshold,
            tau_off=tau_off,
            gap_merge=int(args.gap_merge),
            min_len=int(args.min_len),
        )
        events_records[band] = summarise_events(df, events_idx, sample_delta)

    mild_events = events_records.get("mild", [])

    metrics: Dict[str, float] = {}
    metrics["overall_peak"] = float(df["imbalance"].max())
    metrics["overall_p95"] = float(df["imbalance"].quantile(0.95))
    metrics["overall_p99"] = float(df["imbalance"].quantile(0.99))
    band_metrics = summarise_bands(df, df["imbalance"], events_records, thresholds, sample_delta)
    metrics.update(band_metrics)

    outdir = ensure_output_directory(args.outdir, device_id)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(outdir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)

    events_path = os.path.join(outdir, "events.csv")
    events_df = pd.DataFrame(
        [
            {
                "start_time": ev.start_time.isoformat(),
                "end_time": ev.end_time.isoformat(),
                "duration_h": ev.duration_h,
                "peak": ev.peak,
                "culprit_pair": ev.culprit_pair,
                "max_delta_I_at_peak": ev.max_delta_I_at_peak,
                "Ia_peak": ev.ia_peak,
                "Ib_peak": ev.ib_peak,
                "Ic_peak": ev.ic_peak,
                "I_sum_peak": ev.i_sum_peak,
            }
            for ev in mild_events
        ]
    )
    events_df.to_csv(events_path, index=False)

    plot_colored_cur(
        df,
        thresholds,
        device_id,
        metric_name,
        start,
        end,
        args.dpi,
        os.path.join(outdir, "01_colored_cur_t.png"),
    )

    plot_event_stripes(
        mild_events,
        device_id,
        metric_name,
        start,
        end,
        args.dpi,
        os.path.join(outdir, "02_event_stripes.png"),
        sample_delta,
    )

    plot_severity_duration_scatter(
        mild_events,
        device_id,
        metric_name,
        args.dpi,
        os.path.join(outdir, "03_severity_duration_scatter.png"),
    )

    plot_diurnal_heatmap(
        df,
        device_id,
        metric_name,
        args.dpi,
        os.path.join(outdir, "04_diurnal_heatmap.png"),
    )

    plot_exceedance_and_taat(
        df["imbalance"],
        thresholds,
        metrics,
        device_id,
        metric_name,
        args.dpi,
        os.path.join(outdir, "05_exceedance_and_taat.png"),
    )

    if not args.no_ternary:
        plot_ternary(
            df,
            device_id,
            args.dpi,
            os.path.join(outdir, "06_ternary.png"),
        )

    print_summary(device_id, metric_name, metrics, thresholds)

    logging.info("Outputs written to %s", outdir)
    return 0


def print_summary(
    device_id: int,
    metric_name: str,
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
) -> None:
    lines = [
        f"Device {device_id} - {metric_name}",
        f"Peak: {metrics.get('overall_peak', float('nan')):.2f}%",
        f"P95: {metrics.get('overall_p95', float('nan')):.2f}%",
        f"P99: {metrics.get('overall_p99', float('nan')):.2f}%",
    ]

    for band in ("mild", "moderate", "severe"):
        taat = metrics.get(f"{band}_taat_h", float("nan"))
        count = metrics.get(f"{band}_event_count", float("nan"))
        lce = metrics.get(f"{band}_lce_h", float("nan"))
        threshold = thresholds.get(band, float("nan"))
        if math.isnan(taat) or math.isnan(count) or math.isnan(lce) or math.isnan(threshold):
            lines.append(f"{band.title()}: insufficient data")
        else:
            lines.append(
                f"{band.title()} (≥{threshold:.1f}%): TAAT={taat:.2f}h, Events={int(count)}, LCE={lce:.2f}h"
            )

    summary = "\n".join(lines)
    print(summary)


if __name__ == "__main__":
    sys.exit(main())
