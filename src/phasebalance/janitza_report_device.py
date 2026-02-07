# src/phasebalance/janitza_report_device.py
"""Generate an imbalance report for a single Janitza device.

This script fetches three-phase current and voltage time series for a device
from the Janitza REST API, computes per-timestep imbalance metrics, detects
events, summarises them, and produces diagnostics (plots + CSVs).

Notes
-----
• External helpers are intentionally preserved and called exactly as imported:
  - fetch_hist_json
  - resolve_channels
  - cur_ratio, cur_dev_ratio
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse
import datetime as dt
import logging
import math
import os
import sys
import time
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import mpltern  # noqa: F401 — used by plot_ternary

try:
    from .janitza_fetch import fetch_hist_json
    from .phase_unbalance_utils import cur_ratio, cur_dev_ratio, _series_from_values,_has_values,resolve_channels
except ImportError:  # pragma: no cover - for running as a script
    from janitza_fetch import fetch_hist_json
    from phase_unbalance_utils import cur_ratio, cur_dev_ratio, _series_from_values,_has_values,resolve_channels


# ---------------------------- Config & Constants -----------------------------
CHANNEL_SPEC_I = {
    "IA": {"value_backend": "I_Effective", "type_candidates": ["Input01", "Input05", "L1"]},
    "IB": {"value_backend": "I_Effective", "type_candidates": ["Input02", "Input06", "L2"]},
    "IC": {"value_backend": "I_Effective", "type_candidates": ["Input03", "Input07", "L3"]},
}
CHANNEL_SPEC_V = {
    "VA": {"value_backend": "U_Effective", "type_candidates": ["Input01", "Input05", "L1"]},
    "VB": {"value_backend": "U_Effective", "type_candidates": ["Input02", "Input06", "L2"]},
    "VC": {"value_backend": "U_Effective", "type_candidates": ["Input03", "Input07", "L3"]},
}

PAIR_COLORS = {"Ia-Ib": "#1b9e77", "Ib-Ic": "#d95f02", "Ic-Ia": "#7570b3"}
SEVERITY_COLORS = {"mild": "#ccebc5", "moderate": "#fdb462", "severe": "#fb8072"}
RENAME_MAP = {"IA": "Ia", "IB": "Ib", "IC": "Ic", "VA": "Va", "VB": "Vb", "VC": "Vc"}
_CANON = {"IaIb": "Ia-Ib", "IbIa": "Ia-Ib", "IbIc": "Ib-Ic", "IcIb": "Ib-Ic", "IcIa": "Ic-Ia", "IaIc": "Ic-Ia"}

METRIC_FNS = {
    "cur_ratio": (cur_ratio, "Current Unbalance Ratio (%)"),
    "cur_dev_ratio": (cur_dev_ratio, "Current Deviation Ratio Imbalance (%)"),
}

# ------------------------------- Utilities ----------------------------------

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)


def _vec_metric(fn, a, b, c):
    f = np.vectorize(lambda x, y, z: float(fn(float(x), float(y), float(z))))
    return f(a, b, c)


def label_culprit_pair(row: pd.Series) -> str:
    hi = row[["Ia", "Ib", "Ic"]].idxmax()
    lo = row[["Ia", "Ib", "Ic"]].idxmin()
    return _CANON.get(f"{hi}{lo}", "Ia-Ib")



def _contiguous_indices(mask: pd.Series) -> List[pd.Index]:
    if mask.empty:
        return []
    idx = np.where(mask.to_numpy())[0]
    if idx.size == 0:
        return []
    blocks = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [mask.index[b] for b in blocks]


def _compute_sample_delta(timebase: str, index: pd.Index) -> Optional[pd.Timedelta]:
    try:
        delta = pd.to_timedelta(timebase)
        return delta if delta > pd.Timedelta(0) else None
    except Exception:
        pass
    if len(index) > 1:
        diffs = index.to_series().diff().dropna()
        return pd.to_timedelta(diffs.median(), unit="ns") if not diffs.empty else None
    return None


# ------------------------------ Event logic ---------------------------------
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


def detect_events(
    series: pd.Series, tau_on: float, tau_off: Optional[float] = None, gap_merge: int = 1, min_len: int = 1,
) -> List[Tuple[int, int]]:
    if series.empty:
        return []
    vals = series.to_numpy()
    tau_off_val = tau_off if tau_off is not None else tau_on
    events: List[Tuple[int, int]] = []
    active, start_idx, last_above = False, None, None
    for i, v in enumerate(vals):
        if not active and v >= tau_on:
            active, start_idx, last_above = True, i, i
        elif active:
            if v >= tau_off_val:
                last_above = i
            else:
                events.append((start_idx, last_above))  # type: ignore[arg-type]
                active, start_idx, last_above = False, None, None
    if active and start_idx is not None and last_above is not None:
        events.append((start_idx, last_above))
    # length filter + merge small gaps
    events = [(s, e) for s, e in events if (e - s + 1) >= max(1, min_len)]
    if not events:
        return []
    merged: List[Tuple[int, int]] = []
    cs, ce = events[0]
    for s, e in events[1:]:
        if s - ce - 1 <= max(0, gap_merge):
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def summarise_events(df: pd.DataFrame, events: List[Tuple[int, int]], sample_delta: Optional[pd.Timedelta]) -> List[EventRecord]:
    if not events:
        return []
    add_delta = sample_delta or pd.Timedelta(0)
    recs: List[EventRecord] = []
    for s, e in events:
        seg = df.iloc[s : e + 1]
        if seg.empty:
            continue
        st, et = seg.index[0], seg.index[-1]
        dur_h = ((et - st + add_delta).total_seconds() / 3600.0) if add_delta > pd.Timedelta(0) else max((et - st).total_seconds() / 3600.0, 0.0)
        peak_idx = seg["imbalance"].idxmax()
        row = df.loc[peak_idx]
        recs.append(
            EventRecord(
                start_time=st,
                end_time=et,
                duration_h=dur_h,
                peak=float(seg["imbalance"].max()),
                culprit_pair=str(row["culprit_pair"]),
                max_delta_I_at_peak=row["I_max"] - row["I_min"],
                ia_peak=row.get("Ia", np.nan),
                ib_peak=row.get("Ib", np.nan),
                ic_peak=row.get("Ic", np.nan),
                i_sum_peak=row.get("I_sum", np.nan),
            )
        )
    return recs


def summarise_bands(
    df: pd.DataFrame,
    series: pd.Series,
    events_per_band: Dict[str, List[EventRecord]],
    thresholds: Dict[str, float],
    sample_delta: Optional[pd.Timedelta],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    samp_h = sample_delta.total_seconds() / 3600.0 if sample_delta is not None else np.nan
    total = float(len(series)) if len(series) else np.nan
    for band, thr in thresholds.items():
        mask = series >= thr
        n = int(mask.sum())
        taat_h = n * samp_h if not math.isnan(samp_h) else np.nan
        duty = (n / total) if total else np.nan
        csi = float(((series - thr).clip(lower=0.0)).sum())
        evs = events_per_band.get(band, [])
        durs = [ev.duration_h for ev in evs if ev.duration_h is not None]
        out.update(
            {
                f"{band}_taat_h": taat_h,
                f"{band}_duty": duty,
                f"{band}_csi": csi,
                f"{band}_event_count": float(len(evs)),
                f"{band}_mean_duration_h": float(np.mean(durs)) if durs else np.nan,
                f"{band}_median_duration_h": float(np.median(durs)) if durs else np.nan,
                f"{band}_lce_h": float(np.max(durs)) if durs else 0.0,
                f"{band}_event_peak_p95": float(np.percentile([ev.peak for ev in evs], 95)) if evs else np.nan,
            }
        )
    return out


# ------------------------------- Plot helpers --------------------------------

def _save_text_figure(text: str, title: str, path: str, figsize=(8, 3), dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_colored_cur(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    device_id: int,
    metric_name: str,
    start: str,
    end: str,
    dpi: int,
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax.plot(df.index, df["imbalance"], color="#333333", linewidth=1.2, label=metric_name)

    for pair, color in PAIR_COLORS.items():
        mask = df["culprit_pair"] == pair
        for block in _contiguous_indices(mask):
            if len(block) < 2:
                continue
            x = mdates.date2num(pd.DatetimeIndex(block).to_pydatetime())
            y = df.loc[block, "imbalance"].to_numpy(float)
            segs = [((x[i], y[i]), (x[i + 1], y[i + 1])) for i in range(len(x) - 1)]
            ax.add_collection(LineCollection(segs, colors=color, linewidths=2.0, alpha=0.9))

    ax.set_ylabel(f"{metric_name}")
    ax.set_title(f"Device {device_id} - {metric_name} {start} → {end}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    ymin, ymax = ax.get_ylim()
    prev = 0.0
    for band in ["mild", "moderate", "severe"]:
        thr = thresholds.get(band)
        if thr is None:
            continue
        ax.axhspan(prev, thr, color=SEVERITY_COLORS[band], alpha=0.08)
        prev = thr
    ax.set_ylim(ymin, max(ymax, prev * 1.2))

    handles = [Line2D([0], [0], color=c, lw=2) for c in PAIR_COLORS.values()]
    ax.legend(handles, PAIR_COLORS.keys(), title="Largest disparity", loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()

    # --- NEW: only save+close when output_path is provided/non-empty ---
    if output_path is not None and str(output_path).strip() != "":
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return None

    # Otherwise, return the figure so the caller can display/use it
    return fig


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
        _save_text_figure("No events detected", f"Device {device_id} - Events ({metric_name}) {start} → {end}", output_path, dpi=dpi)
        return
    fig, ax = plt.subplots(figsize=(12, max(3, len(events) * 0.3)), dpi=dpi)
    y = np.arange(len(events))
    for i, ev in enumerate(events):
        s = float(mdates.date2num(ev.start_time.to_pydatetime()))
        e = float(mdates.date2num(ev.end_time.to_pydatetime()))
        if sample_delta is not None:
            e += float(sample_delta.total_seconds() / (24 * 3600))
        ax.broken_barh([(s, e - s)], (float(y[i] - 0.4), 0.8), facecolors=PAIR_COLORS.get(ev.culprit_pair, "#999999"), alpha=0.8)
        ax.text(s, float(y[i]), f"{ev.peak:.1f}%", va="center", ha="left", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Evt {i+1}" for i in range(len(events))])
    ax.set_xlabel("Time")
    ax.set_title(f"Device {device_id} - Events ({metric_name}) {start} → {end}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout(); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)


def plot_severity_duration_scatter(events: Sequence[EventRecord], device_id: int, metric_name: str, dpi: int, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    if not events:
        _save_text_figure("No events detected", f"Severity vs Duration - Device {device_id}", output_path, dpi=dpi)
        return
    for pair, color in PAIR_COLORS.items():
        evs = [ev for ev in events if ev.culprit_pair == pair]
        if not evs:
            continue
        ax.scatter([ev.duration_h for ev in evs], [ev.peak for ev in evs], s=[max(ev.i_sum_peak, 0.0) * 5.0 for ev in evs], alpha=0.7, label=pair, color=color, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel(f"Peak {metric_name}")
    ax.set_title(f"Severity vs Duration - Device {device_id}")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(title="Largest disparity")
    fig.tight_layout(); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)


def plot_diurnal_heatmap(df: pd.DataFrame, device_id: int, metric_name: str, dpi: int, output_path: str) -> None:
    data = df.copy()
    data.index = pd.DatetimeIndex(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.dayofweek
    pivot = data.pivot_table(index="weekday", columns="hour", values="imbalance", aggfunc="median")
    if pivot.empty:
        _save_text_figure("Insufficient data", f"Diurnal imbalance heatmap - Device {device_id}", output_path, dpi=dpi)
        return
    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])  # indices already 0..6
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Weekday")
    ax.set_title(f"Diurnal imbalance heatmap - Device {device_id}")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(f"Median {metric_name}")
    fig.tight_layout(); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)


def plot_exceedance_and_taat(
    series: pd.Series,
    thresholds: Dict[str, float],
    metrics: Dict[str, float],
    device_id: int,
    metric_name: str,
    dpi: int,
    output_path: str
) -> None:
    # Now: ONLY the exceedance curve (no TAAT subplot).
    if series.empty:
        _save_text_figure("No data", f"Device {device_id} - {metric_name}", output_path, dpi=dpi)
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=dpi)

    vals = np.sort(series.to_numpy())
    exc = 1.0 - np.arange(1, len(vals) + 1) / float(len(vals))

    ax1.plot(vals, exc * 100.0, color="#4daf4a", linewidth=1.5)

    for band, thr in thresholds.items():
        ax1.axvline(thr, color=SEVERITY_COLORS.get(band, "#ccc"), linestyle="--", alpha=0.7)

    ax1.set_xlabel(metric_name)
    ax1.set_ylabel("% time ≥ τ")
    ax1.set_title("Exceedance curve")
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle(f"Device {device_id} - {metric_name}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def plot_ternary(df: pd.DataFrame, device_id: int, dpi: int, output_path: str) -> None:
    valid = df[["Ia", "Ib", "Ic"]].dropna()
    if valid.empty:
        logging.info("No data for ternary plot.")
        return
    tot = valid.sum(axis=1).replace(0, np.nan)
    norm = valid.divide(tot, axis=0).dropna()
    if norm.empty:
        logging.info("No valid normalized current data for ternary plot.")
        return
    x, y, z = (norm[c].to_numpy(float) for c in ("Ia", "Ib", "Ic"))
    colors = [PAIR_COLORS.get(p, "#999999") for p in df.loc[norm.index, "culprit_pair"].astype(str).tolist()]
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="ternary")  # type: ignore[arg-type]
    ax.scatter(x, y, z, c=colors, alpha=0.6, s=10.0)  # type: ignore[call-arg]
    getattr(ax, "set_tlabel", lambda *_: None)("Ia fraction")
    getattr(ax, "set_llabel", lambda *_: None)("Ib fraction")
    getattr(ax, "set_rlabel", lambda *_: None)("Ic fraction")
    ax.set_title(f"Ternary current balance - Device {device_id}")
    fig.tight_layout(); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)

# ------------------------------ IO & orchestration ---------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Janitza device imbalance report generator")
    p.add_argument("--device-id", required=True, help="Device identifier")
    p.add_argument("--start", required=True, help="Start timestamp (e.g. '2025-09-01 12:00')")
    p.add_argument("--end", required=True, help="End timestamp (e.g. '2025-10-01 12:00')")
    p.add_argument("--timebase", required=True, help="Timebase (e.g. '15m')")
    p.add_argument("--caps-csv", required=True, help="Path to capabilities.csv metadata")
    p.add_argument("--outdir", default="results", help="Output directory root")
    p.add_argument("--metric", choices=list(METRIC_FNS.keys()), default="cur_ratio", help="Imbalance metric to compute")
    p.add_argument("--mild", type=float, default=5.0, help="Mild severity threshold (%)")
    p.add_argument("--moderate", type=float, default=10.0, help="Moderate severity threshold (%)")
    p.add_argument("--severe", type=float, default=20.0, help="Severe severity threshold (%)")
    p.add_argument("--tau-off-ratio", type=float, default=0.9, help="tau_off = tau_on * ratio (hysteresis)")
    p.add_argument("--gap-merge", type=int, default=1, help="Max gap (samples) to merge events")
    p.add_argument("--min-len", type=int, default=2, help="Minimum length (samples) for an event")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    p.add_argument("--no-ternary", action="store_true", help="Disable ternary plot even if mpltern is available")
    return p.parse_args(argv)


def fetch_time_series(
    device_id: int,
    start: str,
    end: str,
    timebase: str,
    resolved_channels: Dict[str, Dict[str, str]],
    labels: Dict[str, str],
    sleep_s: float = 0.0,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for key, info in resolved_channels.items():
        resp = fetch_hist_json(
            device_id=device_id,
            variable_backend=info["value_backend"],
            phase_backend=info["type_backend"],
            timebase=timebase,
            start=start,
            end=end,
        )
        out[key] = _series_from_values(resp, labels.get(key, key)) if resp and _has_values(resp) else pd.Series(dtype=float, name=labels.get(key, key))
        if sleep_s > 0:
            time.sleep(sleep_s)
    return out


def assemble_dataframe(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    series = [s.rename(RENAME_MAP.get(k, k)) for k, s in series_map.items() if not s.empty]
    return pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()


def ensure_output_directory(base_outdir: str, device_id: int) -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = os.path.join(base_outdir, f"report_device_{device_id}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def print_summary(device_id: int, metric_name: str, metrics: Dict[str, float], thresholds: Dict[str, float]) -> None:
    lines = [
        f"Device {device_id} - {metric_name}",
        f"Peak: {metrics.get('overall_peak', float('nan')):.2f}%",
        f"P95: {metrics.get('overall_p95', float('nan')):.2f}%",
        f"P99: {metrics.get('overall_p99', float('nan')):.2f}%",
    ]
    for band in ("mild", "moderate", "severe"):
        taat, count, lce, thr = (
            metrics.get(f"{band}_taat_h", float("nan")),
            metrics.get(f"{band}_event_count", float("nan")),
            metrics.get(f"{band}_lce_h", float("nan")),
            thresholds.get(band, float("nan")),
        )
        lines.append(
            f"{band.title()} (≥{thr:.1f}%): TAAT={taat:.2f}h, Events={int(count)}, LCE={lce:.2f}h"
            if not any(map(math.isnan, [taat, count, lce, thr]))
            else f"{band.title()}: insufficient data"
        )
    print("\n".join(lines))


# ----------------------------------- Main -----------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    device_id, start, end, timebase = int(args.device_id), args.start, args.end, args.timebase

    try:
        cap_df = pd.read_csv(args.caps_csv)
    except Exception as exc:
        logging.error("Failed to read capabilities CSV: %s", exc)
        return 1

    cur_ch = resolve_channels(cap_df, str(device_id), CHANNEL_SPEC_I, require_all=True)
    if not cur_ch:
        logging.error("Currents could not be resolved for device %s. Aborting.", device_id)
        return 1
    volt_ch = resolve_channels(cap_df, str(device_id), CHANNEL_SPEC_V, require_all=False)

    labels = {k: RENAME_MAP.get(k, k) for k in set(list(cur_ch.keys()) + list(volt_ch.keys()))}

    cur_series = fetch_time_series(device_id, start, end, timebase, cur_ch, labels, sleep_s=0.2)
    volt_series = fetch_time_series(device_id, start, end, timebase, volt_ch, labels, sleep_s=0.2)

    df = assemble_dataframe({**cur_series, **volt_series})
    if df.empty or not {"Ia", "Ib", "Ic"}.issubset(df.columns):
        logging.error("Missing or empty current channels for device %s", device_id)
        return 1

    df = df.dropna(how="all")
    if df.empty:
        logging.error("No valid samples after dropna for device %s", device_id)
        return 1

    for c in ("Ia", "Ib", "Ic", "Va", "Vb", "Vc"):
        if c in df.columns:
            df[c] = df[c].astype(float)

    df["I_sum"] = df[["Ia", "Ib", "Ic"]].sum(axis=1)
    df["I_max"] = df[["Ia", "Ib", "Ic"]].max(axis=1)
    df["I_min"] = df[["Ia", "Ib", "Ic"]].min(axis=1)

    metric_fn, metric_name = METRIC_FNS[args.metric]
    df["imbalance"] = _vec_metric(metric_fn, df["Ia"], df["Ib"], df["Ic"])
    df["dominant_phase"] = df[["Ia", "Ib", "Ic"]].idxmax(axis=1)
    df["culprit_pair"] = df.apply(label_culprit_pair, axis=1)

    df = df.dropna(subset=["Ia", "Ib", "Ic", "imbalance"])
    if len(df) < 10:
        logging.error("Fewer than 10 valid samples after filtering; aborting.")
        return 1

    thresholds = {"mild": args.mild, "moderate": args.moderate, "severe": args.severe}
    sample_delta = _compute_sample_delta(timebase, df.index)

    events_records: Dict[str, List[EventRecord]] = {}
    for band, thr in thresholds.items():
        ev_idx = detect_events(df["imbalance"], tau_on=thr, tau_off=thr * float(args.tau_off_ratio), gap_merge=int(args.gap_merge), min_len=int(args.min_len))
        events_records[band] = summarise_events(df, ev_idx, sample_delta)

    mild_events = events_records.get("mild", [])

    metrics: Dict[str, float] = {
        "overall_peak": float(df["imbalance"].max()),
        "overall_p95": float(df["imbalance"].quantile(0.95)),
        "overall_p99": float(df["imbalance"].quantile(0.99)),
    }
    metrics.update(summarise_bands(df, df["imbalance"], events_records, thresholds, sample_delta))

    outdir = ensure_output_directory(args.outdir, device_id)
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)
    pd.DataFrame([
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
    ]).to_csv(os.path.join(outdir, "events.csv"), index=False)

    plot_colored_cur(df, thresholds, device_id, metric_name, start, end, args.dpi, os.path.join(outdir, "01_colored_cur_t.png"))
    plot_event_stripes(mild_events, device_id, metric_name, start, end, args.dpi, os.path.join(outdir, "02_event_stripes.png"), sample_delta)
    plot_severity_duration_scatter(mild_events, device_id, metric_name, args.dpi, os.path.join(outdir, "03_severity_duration_scatter.png"))
    plot_diurnal_heatmap(df, device_id, metric_name, args.dpi, os.path.join(outdir, "04_diurnal_heatmap.png"))
    plot_exceedance_and_taat(df["imbalance"], thresholds, metrics, device_id, metric_name, args.dpi, os.path.join(outdir, "05_exceedance_and_taat.png"))
    if not args.no_ternary:
        plot_ternary(df, device_id, args.dpi, os.path.join(outdir, "06_ternary.png"))

    print_summary(device_id, metric_name, metrics, thresholds)
    logging.info("Outputs written to %s", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
