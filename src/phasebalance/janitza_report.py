# janitza_report.py
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting libs
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None  # plotting disabled if matplotlib isn't available

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None  # heatmap will be skipped if seaborn isn't available


# ------------ Config ------------
DEFAULT_CSV = "results/metrics_results_2025-11-05_2224.csv"

# Metrics we explicitly want bar charts for (if present)
METRICS: List[str] = [
    "cur_dev_ratio",
    "cur_ratio",
    "dib",
    "sequence_unbalance_factors.M0_mag",
    "sequence_unbalance_factors.M2_mag",
    "vuf_magnitude",
    "vuf_symmetrical",
    "neutral_from_trms_120deg",
]

# Candidate numeric columns to consider for correlation heatmap (will be filtered by presence + numeric dtype)
CANDIDATE_METRICS: List[str] = [
    # core imbalance indicators
    "cur_ratio",
    "cur_dev_ratio",
    "vuf_magnitude",
    "vuf_symmetrical",
    "sequence_unbalance_factors.M0_mag",
    "sequence_unbalance_factors.M2_mag",
    "neutral_from_trms_120deg",
    "dib",
    # raw phase means (if present)
    "Ia_avg", "Ib_avg", "Ic_avg",
    "Va_avg", "Vb_avg", "Vc_avg",
]

# ----------- Report configuration -----------
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "cur_ratio": 10.0,                    # %
    "cur_dev_ratio": 10.0,                # %
    "dib": 0.05,                          # pu
    "neutral_from_trms_120deg": 10.0,     # A
    "I4_avg": 10.0,                       # A
    "vuf_magnitude": 0.02,                # pu
    "vuf_symmetrical": 0.02,              # pu
    "sequence_unbalance_factors.M2_mag": 0.02,  # pu
    "sequence_unbalance_factors.M0_mag": 0.02,  # pu
}

CANDIDATE_METRICS_FOR_REPORT: Tuple[str, ...] = (
    "cur_ratio",
    "cur_dev_ratio",
    "dib",
    "neutral_from_trms_120deg",
    "I4_avg",
    "vuf_magnitude",
    "vuf_symmetrical",
    "sequence_unbalance_factors.M2_mag",
    "sequence_unbalance_factors.M0_mag",
)

COMPARISON_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("vuf_magnitude", "vuf_symmetrical", "pu"),
    ("neutral_from_trms_120deg", "I4_avg", "A"),
)
# ------------ I/O helpers ------------
def make_outdir(base: str = "results") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    outdir = os.path.join(base, f"report_{ts}")
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ------------ Data helpers ------------
def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def _select_existing_numeric(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    cols = [c for c in candidate_cols if c in df.columns]
    # ensure numeric dtype
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    # drop duplicates while preserving order
    seen = set()
    uniq = []
    for c in cols:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].astype(float)
    return (out - out.mean()) / out.std(ddof=0)


# ------------ Plots ------------
def barplot_by_device(
    df: pd.DataFrame,
    metric: str,
    outdir: str,
    top_n: int | None = None,
) -> None:
    """
    Create a bar chart of `metric` vs device_id.
    - Bars are ordered by descending metric value (ties keep device_id order).
    - If top_n is provided, only the top_n devices are shown.
    """
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return

    if "device_id" not in df.columns:
        print(f"[{metric}] device_id column missing; skipping.")
        return

    sub = df[["device_id", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])

    # sort by metric desc for readability
    sub = sub.sort_values(metric, ascending=False, kind="mergesort")

    if top_n is not None:
        sub = sub.head(top_n)

    n = len(sub)
    if n == 0:
        print(f"[{metric}] No valid data to plot.")
        return

    # Wider figure for many devices; clamp so it doesn't explode
    width = min(max(10, int(n * 0.25)), 36)
    height = 6

    plt.figure(figsize=(width, height))
    plt.bar(sub["device_id"].astype(str), sub[metric])
    plt.ylabel(metric)
    plt.xlabel("device_id")
    plt.title(f"{metric} by device_id (n={n})")
    plt.xticks(rotation=90)
    plt.tight_layout()

    png_path = os.path.join(outdir, f"{metric}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved plot: {png_path}")

    # Also save the sorted values for quick inspection
    csv_path = os.path.join(outdir, f"{metric}_sorted.csv")
    sub.to_csv(csv_path, index=False)
    print(f"Saved data: {csv_path}")


def plot_corr_heatmap(
    df: pd.DataFrame,
    candidate_cols: List[str],
    outdir: str,
    zscore: bool = False,
) -> None:
    """
    Make a correlation heatmap across numeric candidate cols that are present.
    If zscore=True, standardize columns before correlation.
    """
    if sns is None or plt is None:
        print("seaborn/matplotlib not available; skipping correlation heatmap.")
        return

    cols = _select_existing_numeric(df, candidate_cols)
    if not cols:
        print("No numeric candidate columns found for correlation heatmap; skipping.")
        return

    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if sub.empty:
        print("All candidate columns are empty/NaN for heatmap; skipping.")
        return

    # Optional standardization before correlation
    if zscore:
        sub = _zscore(sub, sub.columns.tolist())

    corr = sub.corr(method="pearson")

    plt.figure(figsize=(max(10, len(cols) * 0.6), max(8, len(cols) * 0.6)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Correlation heatmap of imbalance metrics"
        + (" (z-scored)" if zscore else "")
    )
    plt.tight_layout()
    fp = os.path.join(outdir, "correlation_heatmap.png")
    plt.savefig(fp, dpi=200)
    plt.close()
    print(f"Saved heatmap: {fp}")

# ------------ Report: helpers ------------
@dataclass
class MetricSummary:
    metric: str
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    threshold: Optional[float]
    exceed_count: Optional[int]

    def as_row(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "threshold": self.threshold,
            "exceed_count": self.exceed_count,
        }


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def _summarise_metric(
    series: pd.Series,
    metric: str,
    threshold: Optional[float] = None,
) -> MetricSummary:
    cleaned = series.dropna()
    if cleaned.empty:
        return MetricSummary(metric, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), threshold, None)

    exceed = None
    if threshold is not None:
        exceed = int((cleaned > threshold).sum())

    return MetricSummary(
        metric=metric,
        count=int(cleaned.count()),
        mean=float(cleaned.mean()),
        median=float(cleaned.median()),
        std=float(cleaned.std(ddof=0)),
        min=float(cleaned.min()),
        max=float(cleaned.max()),
        threshold=threshold,
        exceed_count=exceed,
    )


def _format_markdown_table(rows: Iterable[Dict[str, object]]) -> str:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return "(no metrics available)"
    return df.to_markdown(index=False, floatfmt=".4f")


def _top_devices(
    df: pd.DataFrame,
    metric: str,
    top_n: int,
) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=["device_id", "name", metric])

    cols = [c for c in ("device_id", "name", metric) if c in df.columns]
    subset = df[cols].dropna(subset=[metric])
    if subset.empty:
        return subset
    return subset.nlargest(top_n, metric)


def _dataset_overview(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    total_rows = len(df)
    lines.append(f"- Samples analysed: {total_rows:,}")

    if "device_id" in df.columns:
        device_count = int(df["device_id"].nunique())
        lines.append(f"- Unique devices: {device_count:,}")

    if {"window_start", "window_end"}.issubset(df.columns):
        start = pd.to_datetime(df["window_start"], errors="coerce").min()
        end = pd.to_datetime(df["window_end"], errors="coerce").max()
        if pd.notna(start) and pd.notna(end):
            lines.append(
                f"- Monitoring horizon: {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}"
            )

    if "sample_time" in df.columns:
        sample_series = pd.to_numeric(df["sample_time"], errors="coerce").dropna()
        if not sample_series.empty:
            mode = sample_series.mode()
            typical = mode.iloc[0] if not mode.empty else sample_series.median()
            lines.append(f"- Typical sampling interval: {typical:.0f} seconds")

    return lines


def _phase_spread_hotspots(
    df: pd.DataFrame,
    current_cols: Tuple[str, str, str] = ("Ia_avg", "Ib_avg", "Ic_avg"),
    voltage_cols: Tuple[str, str, str] = ("Va_avg", "Vb_avg", "Vc_avg"),
    top_n: int = 3,
) -> List[str]:
    lines: List[str] = []

    if all(col in df.columns for col in current_cols):
        current_spread = df[list(current_cols)].max(axis=1) - df[list(current_cols)].min(axis=1)
        current_spread = current_spread.fillna(0)
        mean_spread = float(current_spread.mean())
        lines.append(f"- Average phase current spread (max-min): {mean_spread:.2f} A")
        if top_n > 0:
            idx = current_spread.nlargest(top_n).index
            cols = [c for c in ("device_id", "name", *current_cols) if c in df.columns]
            snapshot = df.loc[idx, cols]
            lines.append("- Worst current imbalance windows:")
            lines.append(snapshot.to_markdown(index=False, floatfmt=".2f"))

    if all(col in df.columns for col in voltage_cols):
        voltage_spread = df[list(voltage_cols)].max(axis=1) - df[list(voltage_cols)].min(axis=1)
        voltage_spread = voltage_spread.fillna(0)
        mean_spread_v = float(voltage_spread.mean())
        lines.append(f"- Average phase voltage spread (max-min): {mean_spread_v:.2f} V")
        if top_n > 0:
            idx_v = voltage_spread.nlargest(top_n).index
            cols_v = [c for c in ("device_id", "name", *voltage_cols) if c in df.columns]
            snapshot_v = df.loc[idx_v, cols_v]
            lines.append("- Worst voltage imbalance windows:")
            lines.append(snapshot_v.to_markdown(index=False, floatfmt=".2f"))

    return lines


def _label_summary(df: pd.DataFrame, top_n: int = 5) -> List[str]:
    label_cols = [col for col in df.columns if col.endswith("_label")]
    lines: List[str] = []
    for label_col in label_cols:
        counts = df[label_col].fillna("(missing)").value_counts()
        head = counts.head(top_n)
        distribution = ", ".join(f"{label}: {count}" for label, count in head.items())
        lines.append(f"- `{label_col}` top states → {distribution}")
    return lines


def _neutral_channel_insights(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    top_n: int = 3,
) -> List[str]:
    lines: List[str] = []
    column = "neutral_from_trms_120deg"
    if column not in df.columns:
        return lines

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return lines

    lines.append(
        f"- Neutral current median: {series.median():.2f} A (p95 = {series.quantile(0.95):.2f} A)"
    )

    threshold = thresholds.get(column)
    if threshold is not None:
        breaches = int((series > threshold).sum())
        lines.append(f"- Samples above {threshold:.1f} A threshold: {breaches}")

    idx = series.nlargest(top_n).index
    cols = [c for c in ("device_id", "name", column) if c in df.columns]
    hottest = df.loc[idx, cols]
    lines.append("- Strongest neutral excursions:")
    lines.append(hottest.to_markdown(index=False, floatfmt=".2f"))
    return lines


def _paired_metric_checks(
    df: pd.DataFrame,
    comparisons: Iterable[Tuple[str, str, str]],
    top_n: int = 3,
) -> List[str]:
    lines: List[str] = []
    for left, right, unit in comparisons:
        if not {left, right}.issubset(df.columns):
            continue

        numeric = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if numeric.empty:
            continue

        diff = (numeric[left] - numeric[right]).abs()
        mean_abs = float(diff.mean())
        median_abs = float(diff.median())
        corr = float(numeric[left].corr(numeric[right])) if numeric.shape[0] > 1 else float("nan")

        lines.append(
            f"- `{left}` vs `{right}` → correlation {corr:.2f}, mean |Δ| {mean_abs:.4f} {unit}, median |Δ| {median_abs:.4f} {unit}"
        )

        if top_n <= 0:
            continue

        idx = diff.nlargest(top_n).index
        context_cols = [col for col in ("device_id", "name", "window_start", "window_end") if col in df.columns]
        subset_cols = context_cols + [left, right]
        snapshot = df.loc[idx, subset_cols]
        lines.append(snapshot.to_markdown(index=False, floatfmt=".4f"))

    return lines


def generate_insights(
    csv_path: str,
    output_dir: str,
    top_n: int,
    thresholds: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Build a markdown report highlighting imbalance issues."""
    df = pd.read_csv(csv_path)
    timestamp = _timestamp()
    output_dir = _ensure_output_dir(output_dir)

    summaries: List[MetricSummary] = []
    plot_paths: List[str] = []

    available_metrics = [m for m in CANDIDATE_METRICS_FOR_REPORT if m in df.columns]
    for metric in available_metrics:
        summaries.append(_summarise_metric(df[metric], metric, thresholds.get(metric)))

    # Compose markdown report
    report_lines: List[str] = []
    report_lines.append(f"# Janitza imbalance insights ({timestamp})\n")
    report_lines.append(f"Source file: `{os.path.abspath(csv_path)}`")
    report_lines.append("")

    overview = _dataset_overview(df)
    if overview:
        report_lines.append("## Dataset pulse check")
        report_lines.extend(overview)
        report_lines.append("")

    phase_lines = _phase_spread_hotspots(df, top_n=top_n)
    if phase_lines:
        report_lines.append("## Phase balance snapshots")
        report_lines.extend(phase_lines)
        report_lines.append("")

    neutral_lines = _neutral_channel_insights(df, thresholds, top_n=top_n)
    if neutral_lines:
        report_lines.append("## Neutral channel watchlist")
        report_lines.extend(neutral_lines)
        report_lines.append("")

    label_lines = _label_summary(df)
    if label_lines:
        report_lines.append("## Label sentiment overview")
        report_lines.extend(label_lines)
        report_lines.append("")

    paired_lines = _paired_metric_checks(df, COMPARISON_PAIRS, top_n=top_n)
    if paired_lines:
        report_lines.append("## Paired metric cross-checks")
        report_lines.extend(paired_lines)
        report_lines.append("")

    if summaries:
        report_lines.append("## Metric overview")
        report_lines.append(_format_markdown_table(s.as_row() for s in summaries))
    else:
        report_lines.append("No recognised metrics found in the dataset.")

    for metric in available_metrics:
        top_df = _top_devices(df, metric, top_n)
        if top_df.empty:
            continue
        report_lines.append("")
        report_lines.append(f"## Top {len(top_df)} devices by `{metric}`")
        report_lines.append(top_df.to_markdown(index=False, floatfmt=".4f"))

    if plot_paths:
        report_lines.append("")
        report_lines.append("## Generated figures")
        for path in plot_paths:
            report_lines.append(f"- {os.path.abspath(path)}")

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(output_dir, f"metrics_insights_{timestamp}.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    return report_path, plot_paths


# ------------ Orchestration ------------
def run(
    csv_path: str,
    outdir: str,
    top_n: int | None,
    zscore: bool,
    report_top_n: int,
    thresholds: Dict[str, float],
    no_plots: bool,
) -> None:
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # device_id handling
    if "device_id" in df.columns:
        df["device_id"] = df["device_id"].astype(str)

    # Light cleaning for numeric metrics we care about in bar plots
    present_metrics = [m for m in METRICS if m in df.columns]
    if present_metrics:
        df[present_metrics] = df[present_metrics].replace([np.inf, -np.inf], np.nan)

    # Plots
    if not no_plots and present_metrics:
        validate_columns(df, ["device_id"] + present_metrics)
        for metric in present_metrics:
            barplot_by_device(df, metric, outdir, top_n=top_n)
        plot_corr_heatmap(df, CANDIDATE_METRICS, outdir, zscore=zscore)
    elif no_plots:
        print("Skipping plots as requested (--no-plots).")
    else:
        print("No known METRICS found for bar charts; skipping bar section.")

    # Markdown report (always produced)
    report_path, _ = generate_insights(
        csv_path=csv_path,
        output_dir=outdir,
        top_n=report_top_n,
        thresholds=thresholds,
    )
    print(f"Saved markdown report: {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Janitza metrics report: bar charts, correlation heatmap, and a Markdown summary."
    )
    p.add_argument("--csv", default=DEFAULT_CSV, help="Path to the metrics CSV file")
    p.add_argument("--outdir", default=None, help="Output dir (default: results/report_<timestamp>/)")
    p.add_argument("--top-n", type=int, default=None, help="Show only the top-N devices per metric (bar charts).")
    p.add_argument("--zscore", action="store_true", help="Z-score normalize columns before correlation heatmap.")
    p.add_argument("--report-top-n", type=int, default=5, help="Top-N rows/devices shown in Markdown sections.")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all image plots and only generate the Markdown report.",
    )
    # Simple thresholds override via repeated --th key=value
    p.add_argument(
        "--th",
        action="append",
        default=[],
        help="Threshold override as key=value (repeatable). Example: --th cur_ratio=8 --th vuf_magnitude=0.03",
    )
    return p.parse_args()


def _parse_threshold_overrides(pairs: List[str]) -> Dict[str, float]:
    th = dict(DEFAULT_THRESHOLDS)  # copy
    for item in pairs:
        if "=" not in item:
            print(f"Ignoring malformed --th '{item}' (expected key=value).")
            continue
        k, v = item.split("=", 1)
        try:
            th[k] = float(v)
        except ValueError:
            print(f"Ignoring non-numeric threshold for '{k}': {v}")
    return th


def main():
    args = parse_args()
    outdir = args.outdir or make_outdir("results")
    print(f"Saving outputs to: {outdir}")
    thresholds = _parse_threshold_overrides(args.th)
    run(
        csv_path=args.csv,
        outdir=outdir,
        top_n=args.top_n,
        zscore=args.zscore,
        report_top_n=args.report_top_n,
        thresholds=thresholds,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()