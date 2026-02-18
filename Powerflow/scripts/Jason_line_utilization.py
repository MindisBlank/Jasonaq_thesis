#!/usr/bin/env python3
"""
Jason_line_utilization.py
==========================
Computes line-level peak utilization metrics to quantify how phase rebalancing
can defer grid reinforcement by reducing the bottleneck-phase loading.

Reads per-phase line currents and loading percentages from
pf_results/{sub}/unbalanced/ to compute:
  - Peak utilization   U_peak(l,t) = max(I_a, I_b, I_c) / I_lim
  - Balanced util.     U_balanced(l,t) = (I_a + I_b + I_c) / (3 * I_lim)
  - Deferrable capacity delta_U(l,t) = U_peak - U_balanced
  - Summary statistics (99th percentile, share above threshold, etc.)

Outputs saved to pf_results/{sub}/line_utilization/:
  - line_utilization_summary.parquet
  - u_peak_timeseries.parquet
  - u_balanced_timeseries.parquet
  - delta_u_timeseries.parquet
  - top_lines_utilization.png
  - top_lines_timeseries.png
  - delta_u_distribution.png

Usage:
    python Jason_line_utilization.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PF_RESULTS_DIR = PROJECT_ROOT / "Powerflow" / "output" / "pf_results"

# Load growth assumption for reinforcement deferral analysis
ANNUAL_GROWTH_RATE = 0.02  # 2% per year

# Only include these substations in the cross-substation comparison
# Set to None to include all substations with results
SELECTED_SUBSTATIONS = ['1056', '1299', '1340', '1398', '1457', '579']


# ---------------------------------------------------------------------------
#  Core computation functions
# ---------------------------------------------------------------------------

def derive_i_lim(i_a, i_b, i_c, i_lines_perc):
    """
    Derive I_lim (thermal current limit in Amps) per line from loading_percent.

    Pandapower defines:
        loading_percent = max(I_a, I_b, I_c) / I_lim * 100
    Therefore:
        I_lim = max(I_a, I_b, I_c) / (loading_percent / 100)

    Uses median over all timesteps for robustness against floating-point noise.

    Args:
        i_a, i_b, i_c: DataFrames of per-phase currents (Amps), shape (T, N_lines)
        i_lines_perc: DataFrame of loading percent, same shape

    Returns:
        pd.Series: I_lim per line (Amps), indexed by line name
    """
    # Max phase current at each timestep
    i_max = pd.DataFrame(
        np.maximum(np.maximum(i_a.values, i_b.values), i_c.values),
        index=i_a.index, columns=i_a.columns
    )

    # Mask zero-loading timesteps to avoid division by zero
    perc_safe = i_lines_perc.replace(0, np.nan)
    i_lim_ts = i_max / (perc_safe / 100)

    i_lim = i_lim_ts.median()

    nan_lines = i_lim[i_lim.isna()].index.tolist()
    if nan_lines:
        logger.warning(f"  Could not derive I_lim for {len(nan_lines)} lines (all-zero loading): {nan_lines}")

    return i_lim


def compute_utilization_timeseries(i_a, i_b, i_c, i_lim):
    """
    Compute per-line utilization timeseries.

    Args:
        i_a, i_b, i_c: DataFrames (T x N_lines) of per-phase currents in Amps
        i_lim: Series (N_lines,) of thermal limits in Amps

    Returns:
        u_peak:     DataFrame (T x N_lines) — max(I_a, I_b, I_c) / I_lim
        u_balanced: DataFrame (T x N_lines) — (I_a + I_b + I_c) / (3 * I_lim)
        delta_u:    DataFrame (T x N_lines) — u_peak - u_balanced
    """
    i_max = pd.DataFrame(
        np.maximum(np.maximum(i_a.values, i_b.values), i_c.values),
        index=i_a.index, columns=i_a.columns
    )

    i_lim_safe = i_lim.replace(0, np.nan)

    u_peak = i_max.div(i_lim_safe, axis=1)
    u_balanced = (i_a + i_b + i_c).div(3 * i_lim_safe, axis=1)
    delta_u = u_peak - u_balanced

    return u_peak, u_balanced, delta_u


def compute_line_summary(u_peak, u_balanced, delta_u, i_lim):
    """
    Compute per-line summary metrics.

    Returns DataFrame (one row per line, sorted by u_peak_99 descending) with:
        i_lim_a, u_peak_max, u_peak_99, u_peak_mean,
        u_balanced_max, u_balanced_99, u_balanced_mean,
        delta_u_max, delta_u_99, delta_u_mean,
        share_above_50, share_above_80, share_above_100
    """
    n_timesteps = len(u_peak)

    summary = pd.DataFrame(index=u_peak.columns)
    summary.index.name = 'line'

    summary['i_lim_a'] = i_lim

    # U_peak statistics
    summary['u_peak_max'] = u_peak.max()
    summary['u_peak_99'] = u_peak.quantile(0.99)
    summary['u_peak_mean'] = u_peak.mean()

    # U_balanced statistics
    summary['u_balanced_max'] = u_balanced.max()
    summary['u_balanced_99'] = u_balanced.quantile(0.99)
    summary['u_balanced_mean'] = u_balanced.mean()

    # Delta U statistics (deferrable capacity)
    summary['delta_u_max'] = delta_u.max()
    summary['delta_u_99'] = delta_u.quantile(0.99)
    summary['delta_u_mean'] = delta_u.mean()

    # Share of time above thresholds
    summary['share_above_50'] = (u_peak > 0.5).sum() / n_timesteps
    summary['share_above_80'] = (u_peak > 0.8).sum() / n_timesteps
    summary['share_above_100'] = (u_peak > 1.0).sum() / n_timesteps

    # Years to 100% threshold under assumed load growth
    # t_τ = ln(τ / U) / ln(1 + g)  — only meaningful when U < τ
    g = ANNUAL_GROWTH_RATE
    tau = 1.0  # 100% capacity threshold

    for label, u_col in [('unbal', 'u_peak_99'), ('bal', 'u_balanced_99')]:
        u = summary[u_col]
        years = np.where(
            u >= tau,
            0.0,                                       # already above threshold
            np.where(u > 0, np.log(tau / u) / np.log(1 + g), np.nan)
        )
        summary[f'years_to_100_{label}'] = years

    summary['deferral_years'] = summary['years_to_100_bal'] - summary['years_to_100_unbal']

    return summary.sort_values('u_peak_99', ascending=False)


# ---------------------------------------------------------------------------
#  Plotting functions
# ---------------------------------------------------------------------------

def plot_top_lines_bar(summary, path, sub, top_n=15):
    """
    Side-by-side bar chart: U_peak_99 vs U_balanced_99 for the top-N lines.
    The gap represents deferrable capacity from rebalancing.
    """
    top = summary.head(top_n)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    x = np.arange(len(top))
    width = 0.35

    ax.bar(x - width / 2, top['u_peak_99'] * 100, width,
           color='#D32F2F', alpha=0.8, label='$U_{peak}$ (99th pctl) — unbalanced')
    ax.bar(x + width / 2, top['u_balanced_99'] * 100, width,
           color='#1976D2', alpha=0.8, label='$U_{balanced}$ (99th pctl) — rebalanced')

    # Threshold lines
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axhline(y=100, color='darkred', linestyle='-', alpha=0.7, label='100% capacity limit')

    # Annotate delta_U on top bars
    for i, (idx, row) in enumerate(top.iterrows()):
        delta = row['delta_u_99'] * 100
        if delta > 1:
            ax.annotate(f'$\\Delta U$={delta:.1f}pp',
                        xy=(i, row['u_peak_99'] * 100),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=7, color='#D32F2F')

    ax.set_xticks(x)
    ax.set_xticklabels(top.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Line Utilization [%]')
    ax.set_title(f'Top-{top_n} Most Loaded Lines — Substation {sub}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'top_lines_utilization.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved top_lines_utilization.png")


def plot_top_lines_timeseries(u_peak, u_balanced, summary, path, sub, top_n=5):
    """
    Multi-panel timeseries: U_peak(t) and U_balanced(t) for the top-N lines.
    Orange fill shows the deferrable capacity gap.
    """
    top_lines = summary.head(top_n).index

    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3 * top_n), dpi=150, sharex=True)
    if top_n == 1:
        axes = [axes]

    for ax, line in zip(axes, top_lines):
        ax.fill_between(u_peak.index,
                         u_balanced[line] * 100,
                         u_peak[line] * 100,
                         alpha=0.3, color='#FF6B00', label='Deferrable ($\\Delta U$)')
        ax.plot(u_peak[line] * 100, color='#D32F2F', linewidth=0.8,
                alpha=0.8, label='$U_{peak}$ (unbalanced)')
        ax.plot(u_balanced[line] * 100, color='#1976D2', linewidth=0.8,
                alpha=0.8, label='$U_{balanced}$ (rebalanced)')

        ax.axhline(y=100, color='darkred', linestyle='-', alpha=0.5)
        ax.set_ylabel(f'Line {line} [%]')
        ax.grid(True, alpha=0.3)

        # Annotate I_lim
        ilim = summary.loc[line, 'i_lim_a']
        ax.text(0.98, 0.95, f'$I_{{lim}}$={ilim:.0f} A',
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=7)

    axes[0].set_title(f'Line Utilization Timeseries — Substation {sub} (Top {top_n})')
    axes[-1].set_xlabel('Time')
    fig.tight_layout()
    fig.savefig(path / 'top_lines_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved top_lines_timeseries.png")


def plot_delta_u_distribution(summary, path, sub):
    """
    Histogram + CDF of delta_U_99 across all lines.
    Shows how much capacity could be freed by rebalancing.
    """
    delta_99 = summary['delta_u_99'].dropna() * 100  # percentage points

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: histogram
    ax1.hist(delta_99, bins=30, color='#FF6B00', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(delta_99.median(), color='red', linestyle='--',
                label=f'Median: {delta_99.median():.1f} pp')
    ax1.set_xlabel('Deferrable Capacity $\\Delta U_{99}$ [percentage points]')
    ax1.set_ylabel('Number of Lines')
    ax1.set_title(f'Distribution of Deferrable Capacity — Sub {sub}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: CDF
    sorted_vals = np.sort(delta_99.values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cdf, color='#1976D2', linewidth=2)
    ax2.set_xlabel('$\\Delta U_{99}$ [percentage points]')
    ax2.set_ylabel('CDF (fraction of lines)')
    ax2.set_title(f'CDF of Deferrable Capacity — Sub {sub}')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'delta_u_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved delta_u_distribution.png")


def plot_deferral_years(summary, path, sub, top_n=15):
    """
    Horizontal bar chart showing years until 100% capacity for each line,
    comparing unbalanced (U_peak) vs balanced (U_balanced) under assumed
    annual load growth.  The gap = years of reinforcement deferral.

    Lines already above 100% show 0 years (need reinforcement now).
    """
    # Only include lines that have finite years (exclude near-zero-load lines)
    valid = summary.dropna(subset=['years_to_100_unbal', 'years_to_100_bal'])
    # Sort by unbalanced years (shortest time to threshold first = most critical)
    valid = valid.sort_values('years_to_100_unbal', ascending=True).head(top_n)

    if len(valid) == 0:
        logger.info(f"  Skipping deferral plot — no lines with finite years to threshold")
        return

    fig, ax = plt.subplots(figsize=(12, max(5, 0.5 * len(valid))), dpi=150)
    y = np.arange(len(valid))
    height = 0.35

    bars_unbal = ax.barh(y + height / 2, valid['years_to_100_unbal'], height,
                          color='#D32F2F', alpha=0.8, label='Unbalanced ($U_{peak,99}$)')
    bars_bal = ax.barh(y - height / 2, valid['years_to_100_bal'], height,
                        color='#1976D2', alpha=0.8, label='Rebalanced ($U_{balanced,99}$)')

    # Annotate deferral years
    for i, (idx, row) in enumerate(valid.iterrows()):
        defer = row['deferral_years']
        if defer > 0.5:
            x_pos = max(row['years_to_100_unbal'], row['years_to_100_bal'])
            ax.text(x_pos + 0.5, i, f'+{defer:.0f} yr',
                    va='center', fontsize=8, color='#1976D2', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(valid.index, fontsize=8)
    ax.set_xlabel(f'Years to 100% Capacity (at {ANNUAL_GROWTH_RATE*100:.0f}% annual load growth)')
    ax.set_title(f'Reinforcement Timeline — Substation {sub}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()  # most critical line at top

    fig.tight_layout()
    fig.savefig(path / 'deferral_years.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved deferral_years.png")


# ---------------------------------------------------------------------------
#  Cross-substation plotting
# ---------------------------------------------------------------------------

def plot_cross_sub_headroom(cross_df, path):
    """
    Bar chart of mean ΔU_99 (capacity headroom freed by rebalancing) per substation,
    plus a weighted-average bar.  Max ΔU_99 is annotated above each bar.
    """
    df = cross_df.copy()

    # Weighted average across all substations (weighted by number of lines)
    total_lines = df['n_lines'].sum()
    weighted_mean = (df['mean_delta_u_99'] * df['n_lines']).sum() / total_lines
    weighted_max = df['max_delta_u_99'].max()

    # Append weighted-average row
    avg_row = pd.DataFrame([{
        'substation': 'Avg',
        'n_lines': total_lines,
        'mean_delta_u_99': weighted_mean,
        'max_delta_u_99': weighted_max,
    }])
    df = pd.concat([df, avg_row], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(df))

    colors = ['#FF6B00'] * (len(df) - 1) + ['#333333']  # dark bar for average
    bars = ax.bar(x, df['mean_delta_u_99'] * 100, width=0.6,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Annotate max ΔU on top of each bar
    for i, (_, row) in enumerate(df.iterrows()):
        mean_val = row['mean_delta_u_99'] * 100
        max_val = row['max_delta_u_99'] * 100
        # Value label inside/on bar
        ax.text(i, mean_val + 0.3, f'{mean_val:.1f} pp',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Max annotation
        if max_val > mean_val + 0.5:
            ax.text(i, mean_val + 1.5, f'(max {max_val:.1f})',
                    ha='center', va='bottom', fontsize=7, color='#555555')

    ax.set_xticks(x)
    labels = [f'Sub {r["substation"]}\n({r["n_lines"]} lines)' for _, r in df.iterrows()]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean Capacity Headroom $\\overline{\\Delta U_{99}}$ [pp]')
    ax.set_title('Capacity Headroom Freed by Phase Rebalancing')
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'cross_substation_headroom.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved cross_substation_headroom.png")


def plot_cross_sub_deferral(cross_df, path):
    """
    Bar chart of mean (and max) deferral years per substation,
    plus a weighted-average bar.
    """
    df = cross_df.copy()

    # Weighted average deferral (weighted by n_lines)
    total_lines = df['n_lines'].sum()
    weighted_mean_defer = (df['mean_deferral_years'] * df['n_lines']).sum() / total_lines
    weighted_max_defer = df['max_deferral_years'].max()

    avg_row = pd.DataFrame([{
        'substation': 'Avg',
        'n_lines': total_lines,
        'mean_deferral_years': weighted_mean_defer,
        'max_deferral_years': weighted_max_defer,
    }])
    df = pd.concat([df, avg_row], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(df))
    width = 0.35

    colors_mean = ['#1976D2'] * (len(df) - 1) + ['#333333']
    colors_max = ['#90CAF9'] * (len(df) - 1) + ['#888888']

    ax.bar(x - width / 2, df['mean_deferral_years'], width,
           color=colors_mean, alpha=0.85, edgecolor='black', linewidth=0.5,
           label='Mean deferral')
    ax.bar(x + width / 2, df['max_deferral_years'], width,
           color=colors_max, alpha=0.7, edgecolor='black', linewidth=0.5,
           label='Max deferral (best-case line)')

    # Annotate values
    for i, (_, row) in enumerate(df.iterrows()):
        mean_d = row['mean_deferral_years']
        max_d = row['max_deferral_years']
        if np.isfinite(mean_d):
            ax.text(i - width / 2, mean_d + 0.3, f'{mean_d:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        if np.isfinite(max_d):
            ax.text(i + width / 2, max_d + 0.3, f'{max_d:.0f}',
                    ha='center', va='bottom', fontsize=8, color='#555555')

    ax.set_xticks(x)
    labels = [f'Sub {r["substation"]}\n({r["n_lines"]} lines)' for _, r in df.iterrows()]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f'Reinforcement Deferral [years] (at {ANNUAL_GROWTH_RATE*100:.0f}% growth)')
    ax.set_title('Reinforcement Deferral from Phase Rebalancing')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'cross_substation_deferral.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved cross_substation_deferral.png")


# ---------------------------------------------------------------------------
#  Summary printing
# ---------------------------------------------------------------------------

def _print_sub_summary(summary, sub):
    """Print a formatted summary table for one substation."""
    logger.info(f"\n  Line Utilization Summary (Substation {sub})  "
                f"[{ANNUAL_GROWTH_RATE*100:.0f}% annual growth]:")
    logger.info(f"  {'─'*90}")
    logger.info(f"  {'Line':>12} {'I_lim':>7} {'U_pk99':>8} {'U_bl99':>8} "
                f"{'ΔU_99':>7} {'>50%':>6} {'>80%':>6} {'>100%':>6} "
                f"{'Yr_unbal':>9} {'Yr_bal':>7} {'Defer':>6}")
    logger.info(f"  {'─'*90}")

    for line, row in summary.head(10).iterrows():
        yr_u = row['years_to_100_unbal']
        yr_b = row['years_to_100_bal']
        defer = row['deferral_years']
        yr_u_str = f"{yr_u:>8.0f}y" if np.isfinite(yr_u) else f"{'∞':>9}"
        yr_b_str = f"{yr_b:>6.0f}y" if np.isfinite(yr_b) else f"{'∞':>7}"
        defer_str = f"+{defer:>4.0f}y" if np.isfinite(defer) else f"{'—':>6}"
        logger.info(
            f"  {str(line):>12} {row['i_lim_a']:>6.0f}A "
            f"{row['u_peak_99']*100:>7.1f}% {row['u_balanced_99']*100:>7.1f}% "
            f"{row['delta_u_99']*100:>6.1f}pp "
            f"{row['share_above_50']*100:>5.1f}% {row['share_above_80']*100:>5.1f}% "
            f"{row['share_above_100']*100:>5.1f}% "
            f"{yr_u_str} {yr_b_str} {defer_str}"
        )

    if len(summary) > 10:
        logger.info(f"  ... ({len(summary) - 10} more lines)")


def _print_cross_sub_summary(rows):
    """Print cross-substation summary with headroom and deferral columns."""
    logger.info(f"\n{'='*100}")
    logger.info("Cross-Substation Line Utilization Summary")
    logger.info(f"{'='*100}")
    logger.info(f"  {'Sub':>5} {'Lines':>6} {'U_pk99':>8} {'U_bl99':>8} "
                f"{'MaxΔU':>7} {'MeanΔU':>8} {'N>50%':>6} {'N>80%':>6} {'N>100%':>7} "
                f"{'MnDefer':>8} {'MxDefer':>8}")
    logger.info(f"  {'─'*95}")
    for r in rows:
        mn_d = r.get('mean_deferral_years', float('nan'))
        mx_d = r.get('max_deferral_years', float('nan'))
        mn_d_str = f"{mn_d:>7.1f}y" if np.isfinite(mn_d) else f"{'—':>8}"
        mx_d_str = f"{mx_d:>7.0f}y" if np.isfinite(mx_d) else f"{'—':>8}"
        logger.info(
            f"  {r['substation']:>5} {r['n_lines']:>6} "
            f"{r['max_u_peak_99']*100:>7.1f}% {r['max_u_balanced_99']*100:>7.1f}% "
            f"{r['max_delta_u_99']*100:>6.1f}pp {r['mean_delta_u_99']*100:>7.2f}pp "
            f"{r['n_above_50_unbal']:>6} {r['n_above_80_unbal']:>6} {r['n_above_100_unbal']:>7} "
            f"{mn_d_str} {mx_d_str}"
        )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 80)
    logger.info("Jason Line Utilization Analysis")
    logger.info("=" * 80)

    if not PF_RESULTS_DIR.exists():
        logger.error(f"Results directory not found: {PF_RESULTS_DIR}")
        return

    # Auto-detect substations with unbalanced results
    substations = []
    for sub_dir in sorted(PF_RESULTS_DIR.iterdir()):
        if sub_dir.is_dir():
            unbal_dir = sub_dir / 'unbalanced'
            if (unbal_dir / 'i_lines_a.parquet').exists() and \
               (unbal_dir / 'i_lines_perc.parquet').exists():
                substations.append(sub_dir.name)

    if not substations:
        logger.error("No substations found with unbalanced power flow results.")
        return

    # Filter to selected substations if specified
    if SELECTED_SUBSTATIONS is not None:
        substations = [s for s in substations if s in SELECTED_SUBSTATIONS]
        if not substations:
            logger.error(f"None of the selected substations {SELECTED_SUBSTATIONS} have results.")
            return

    logger.info(f"Substations to process: {substations}")

    cross_sub_rows = []

    for sub in substations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Line Utilization Analysis — Substation {sub}")
        logger.info(f"{'='*60}")

        sub_path = PF_RESULTS_DIR / sub
        unbal_path = sub_path / 'unbalanced'

        # Load per-phase currents and loading percent
        i_a = pd.read_parquet(unbal_path / 'i_lines_a.parquet').apply(pd.to_numeric, errors='coerce')
        i_b = pd.read_parquet(unbal_path / 'i_lines_b.parquet').apply(pd.to_numeric, errors='coerce')
        i_c = pd.read_parquet(unbal_path / 'i_lines_c.parquet').apply(pd.to_numeric, errors='coerce')
        i_perc = pd.read_parquet(unbal_path / 'i_lines_perc.parquet').apply(pd.to_numeric, errors='coerce')

        logger.info(f"  {len(i_a.columns)} lines, {len(i_a)} timesteps")

        # Step 1: Derive I_lim per line
        i_lim = derive_i_lim(i_a, i_b, i_c, i_perc)
        logger.info(f"  I_lim range: {i_lim.min():.0f} A — {i_lim.max():.0f} A")

        # Step 2: Compute utilization timeseries
        u_peak, u_balanced, delta_u = compute_utilization_timeseries(i_a, i_b, i_c, i_lim)

        # Step 3: Compute per-line summary
        summary = compute_line_summary(u_peak, u_balanced, delta_u, i_lim)

        # Step 4: Save outputs
        out_path = sub_path / 'line_utilization'
        out_path.mkdir(parents=True, exist_ok=True)

        summary.to_parquet(out_path / 'line_utilization_summary.parquet')
        u_peak.to_parquet(out_path / 'u_peak_timeseries.parquet')
        u_balanced.to_parquet(out_path / 'u_balanced_timeseries.parquet')
        delta_u.to_parquet(out_path / 'delta_u_timeseries.parquet')
        logger.info(f"  Saved 4 parquets to {out_path}")

        # Step 5: Generate plots
        plot_top_lines_bar(summary, out_path, sub)
        plot_top_lines_timeseries(u_peak, u_balanced, summary, out_path, sub)
        plot_delta_u_distribution(summary, out_path, sub)
        plot_deferral_years(summary, out_path, sub)

        # Step 6: Print summary table
        _print_sub_summary(summary, sub)

        # Collect cross-substation stats (including deferral & headroom)
        valid_defer = summary['deferral_years'].dropna()
        cross_sub_rows.append({
            'substation': sub,
            'n_lines': len(summary),
            'max_u_peak_99': summary['u_peak_99'].max(),
            'max_u_balanced_99': summary['u_balanced_99'].max(),
            'max_delta_u_99': summary['delta_u_99'].max(),
            'mean_delta_u_99': summary['delta_u_99'].mean(),
            'n_above_50_unbal': (summary['share_above_50'] > 0).sum(),
            'n_above_80_unbal': (summary['share_above_80'] > 0).sum(),
            'n_above_100_unbal': (summary['share_above_100'] > 0).sum(),
            'mean_deferral_years': valid_defer.mean() if len(valid_defer) > 0 else np.nan,
            'max_deferral_years': valid_defer.max() if len(valid_defer) > 0 else np.nan,
            'n_above_50_bal': (summary['u_balanced_99'] > 0.5).sum(),
        })

    # Cross-substation summary
    if cross_sub_rows:
        _print_cross_sub_summary(cross_sub_rows)

        cross_df = pd.DataFrame(cross_sub_rows)

        # Generate cross-substation plots
        plot_cross_sub_headroom(cross_df, PF_RESULTS_DIR)
        plot_cross_sub_deferral(cross_df, PF_RESULTS_DIR)

        # Save cross-substation summary
        cross_df.to_parquet(PF_RESULTS_DIR / 'cross_substation_line_utilization.parquet', index=False)
        logger.info(f"\n  Saved cross_substation_line_utilization.parquet")

    logger.info("\n" + "=" * 80)
    logger.info("Line utilization analysis complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
