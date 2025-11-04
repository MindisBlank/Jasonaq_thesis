"""Generate textual and visual insights from Janitza imbalance metrics.

The CSV exported by :mod:`janitza_scrapper` already contains the computed
imbalance metrics from :mod:`phase_unbalance_utils`.  This helper script loads
that CSV, summarises the key indicators, and produces simple bar charts so that
engineers can quickly spot the worst-performing devices.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

try:  # matplotlib is optional; plots are skipped if unavailable.
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    plt = None  # type: ignore
import pandas as pd

# ----------- Configuration -----------
# Default thresholds based on common utility engineering heuristics.  They can
# be overridden from the CLI if a different limit is desired.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "cur_ratio": 10.0,                    # %
    "cur_dev_ratio": 10.0,               # %
    "dib": 0.05,                         # pu
    "neutral_from_trms_120deg": 10.0,    # A
    "I4_avg": 10.0,                      # A
    "vuf_magnitude": 0.02,               # pu
    "vuf_symmetrical": 0.02,             # pu
    "sequence_unbalance_factors.M2_mag": 0.02,  # pu
    "sequence_unbalance_factors.M0_mag": 0.02,  # pu
}

# Metrics we attempt to summarise; only the ones present in the CSV will be
# processed.
CANDIDATE_METRICS: Tuple[str, ...] = (
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

# Metric pairs that should closely track each other.  Each tuple contains the
# left metric, the right metric, and a human-readable unit/label for deltas.
COMPARISON_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("vuf_magnitude", "vuf_symmetrical", "pu"),
    ("neutral_from_trms_120deg", "I4_avg", "A"),
)


@dataclass
class MetricSummary:
    """Simple container with descriptive statistics for a metric."""

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

    subset = df[["device_id", "name", metric]].dropna()
    return subset.nlargest(top_n, metric)


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    top_n: int,
    output_dir: str,
    timestamp: str,
) -> Optional[str]:
    """Create and save a bar plot for the top-N devices of a metric."""
    if plt is None:
        return None
    top_devices = _top_devices(df, metric, top_n)
    if top_devices.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_devices["name"], top_devices[metric], color="#4371B7")
    ax.set_title(f"Top {len(top_devices)} devices by {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    filename = os.path.join(output_dir, f"{metric.replace('.', '_')}_{timestamp}.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename


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
        sample_series = pd.to_numeric(df["sample_time"], errors="coerce")
        sample_series = sample_series.dropna()
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
        lines.append(
            f"- Average phase current spread (max-min): {mean_spread:.2f} A"
        )
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
        lines.append(
            f"- Average phase voltage spread (max-min): {mean_spread_v:.2f} V"
        )
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
        distribution = ", ".join(
            f"{label}: {count}" for label, count in head.items()
        )
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
        lines.append(
            f"- Samples above {threshold:.1f} A threshold: {breaches}"
        )

    idx = series.nlargest(top_n).index
    cols = [c for c in ("device_id", "name", column) if c in df.columns]
    hottest = df.loc[idx, cols]
    lines.append("- Strongest neutral excursions:")
    lines.append(hottest.to_markdown(index=False, floatfmt=".2f"))
    return lines


def _metric_correlations(
    df: pd.DataFrame,
    metrics: Iterable[str],
    top_n: int = 5,
) -> List[str]:
    columns = [m for m in metrics if m in df.columns]
    subset = df[columns].select_dtypes(include=["number"])
    subset = subset.dropna(axis=1, how="all")

    if subset.shape[1] < 2:
        return []

    corr_matrix = subset.corr(method="pearson").abs()
    pairs: List[Tuple[float, str, str]] = []
    for i, col_a in enumerate(corr_matrix.columns):
        for j, col_b in enumerate(corr_matrix.columns):
            if j <= i:
                continue
            value = corr_matrix.iloc[i, j]
            if pd.isna(value):
                continue
            pairs.append((float(value), col_a, col_b))

    pairs.sort(reverse=True, key=lambda item: item[0])
    lines = [
        f"- {a} ↔ {b}: correlation {value:.2f}"
        for value, a, b in pairs[:top_n]
    ]
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

        numeric = df[[left, right]].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna()
        if numeric.empty:
            continue

        diff = (numeric[left] - numeric[right]).abs()
        mean_abs = float(diff.mean())
        median_abs = float(diff.median())
        corr = float(numeric[left].corr(numeric[right])) if numeric.shape[0] > 1 else float("nan")

        lines.append(
            (
                f"- `{left}` vs `{right}` → correlation {corr:.2f}, "
                f"mean |Δ| {mean_abs:.4f} {unit}, median |Δ| {median_abs:.4f} {unit}"
            )
        )

        if top_n <= 0:
            continue

        idx = diff.nlargest(top_n).index
        context_cols = [
            col
            for col in ("device_id", "name", "window_start", "window_end")
            if col in df.columns
        ]
        subset_cols = context_cols + [left, right]
        snapshot = df.loc[idx, subset_cols]
        lines.append(
            snapshot.to_markdown(index=False, floatfmt=".4f")
        )

    return lines


def generate_insights(
    csv_path: str,
    output_dir: str,
    top_n: int,
    thresholds: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Build a markdown report and plots highlighting imbalance issues."""
    df = pd.read_csv(csv_path)
    timestamp = _timestamp()
    output_dir = _ensure_output_dir(output_dir)

    summaries: List[MetricSummary] = []
    plot_paths: List[str] = []

    available_metrics = [m for m in CANDIDATE_METRICS if m in df.columns]
    for metric in available_metrics:
        summaries.append(
            _summarise_metric(df[metric], metric, thresholds.get(metric))
        )
        plot_path = _plot_metric(df, metric, top_n, output_dir, timestamp)
        if plot_path:
            plot_paths.append(plot_path)

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

    corr_lines = _metric_correlations(df, CANDIDATE_METRICS)
    if corr_lines:
        report_lines.append("## Metric harmonies (strongest correlations)")
        report_lines.extend(corr_lines)
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual and textual insights from Janitza metrics.",
    )
    parser.add_argument(
        "csv_path",
        help="Path to the metrics CSV produced by janitza_scrapper.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where the report and figures will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of devices to include in the top lists and charts.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Override alert thresholds, e.g. --threshold cur_ratio=12.5.",
    )
    return parser.parse_args(argv)


def _parse_threshold_overrides(values: Iterable[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for item in values:
        if "=" not in item:
            continue
        key, raw_val = item.split("=", 1)
        key = key.strip()
        try:
            overrides[key] = float(raw_val)
        except ValueError:
            continue
    return overrides


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    thresholds = DEFAULT_THRESHOLDS.copy()
    thresholds.update(_parse_threshold_overrides(args.threshold))

    report_path, plot_paths = generate_insights(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
        thresholds=thresholds,
    )

    print(f"\n📄 Report written to {report_path}")
    if plot_paths:
        print("📊 Generated figures:")
        for path in plot_paths:
            print(f"  - {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
