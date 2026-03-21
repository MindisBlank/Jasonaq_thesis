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
  - Deferral years under multiple load growth scenarios

Outputs saved to pf_results/{sub}/line_utilization/:
  - line_utilization_summary.parquet
  - u_peak_timeseries.parquet
  - u_balanced_timeseries.parquet
  - delta_u_timeseries.parquet
  - top_lines_utilization.png
  - top_lines_timeseries.png
  - delta_u_distribution.png
  - deferral_years.png

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

# Only include these substations in the cross-substation comparison
# Set to None to include all substations with results
SELECTED_SUBSTATIONS = ['1056', '1299', '1340', '1456', '1457', '579']

# Growth rates loaded from config in main(); module-level default as fallback
DEFAULT_GROWTH_RATES = [0.034, 0.039]

# Global font size bump
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9})


def _load_trafo_labels():
    """Load transformer labels from config, with fallback."""
    try:
        from Jason_config import load_transformer_labels
        return load_transformer_labels()
    except Exception:
        return {}

def _sub_label(sub, trafo_labels):
    """Return display label like '579 SP1' for a substation ID."""
    return trafo_labels.get(str(sub), f'{sub} SP1')


# ---------------------------------------------------------------------------
#  Display helpers
# ---------------------------------------------------------------------------

def rename_line_label(name):
    """
    Rename 'feederN' -> 'trunk N' for display purposes.

    In the pandapower model, 'feeder' lines are the main trunk cables from
    the transformer to junction cabinets.  The thesis reserves 'feeder' for
    the LV distribution side, so we relabel to avoid confusion.
    """
    s = str(name)
    if s.startswith('feeder'):
        return 'trunk ' + s[len('feeder'):]
    return s


def _growth_label(g):
    """Return a short human-readable label for a growth rate, e.g. '3.4%'."""
    return f'{g*100:.1f}%'


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


def compute_line_summary(u_peak, u_balanced, delta_u, i_lim, growth_rates=None):
    """
    Compute per-line summary metrics.

    Returns DataFrame (one row per line, sorted by u_peak_99 descending) with:
        i_lim_a, u_peak_max, u_peak_99, u_peak_mean,
        u_balanced_max, u_balanced_99, u_balanced_mean,
        delta_u_max, delta_u_99, delta_u_mean,
        share_above_50, share_above_80, share_above_100,
        years_to_100_unbal_g{pct}, years_to_100_bal_g{pct}, deferral_years_g{pct}
          for each growth rate
    """
    if growth_rates is None:
        growth_rates = DEFAULT_GROWTH_RATES

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

    # Years to 100% threshold under each growth scenario
    # t_τ = ln(τ / U) / ln(1 + g)  — only meaningful when U < τ
    tau = 1.0  # 100% capacity threshold

    for g in growth_rates:
        pct = int(round(g * 100))
        for label, u_col in [('unbal', 'u_peak_99'), ('bal', 'u_balanced_99')]:
            u = summary[u_col]
            years = np.where(
                u >= tau,
                0.0,                                       # already above threshold
                np.where(u > 0, np.log(tau / u) / np.log(1 + g), np.nan)
            )
            summary[f'years_to_100_{label}_g{pct}'] = years

        summary[f'deferral_years_g{pct}'] = (
            summary[f'years_to_100_bal_g{pct}'] - summary[f'years_to_100_unbal_g{pct}']
        )

    return summary.sort_values('u_peak_99', ascending=False)


# ---------------------------------------------------------------------------
#  Plotting functions
# ---------------------------------------------------------------------------

def plot_top_lines_bar(summary, path, sub_label, top_n=15):
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
    ax.set_xticklabels([rename_line_label(l) for l in top.index], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Line Utilization [%]')
    ax.set_title(f'Top-{len(top)} Most Loaded Lines — LV Transformer {sub_label}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'top_lines_utilization.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved top_lines_utilization.png")


def plot_top_lines_timeseries(u_peak, u_balanced, summary, path, sub_label,
                              min_delta_pp=15, max_n=5, fallback_n=3):
    """
    Multi-panel timeseries: U_peak(t) and U_balanced(t) for lines with
    significant deferrable capacity (ΔU₉₉ > min_delta_pp).

    Falls back to showing the top fallback_n lines if none exceed the threshold.
    Orange fill shows the deferrable capacity gap.
    """
    # Select lines with significant deferral, or fall back to top-N
    significant = summary[summary['delta_u_99'] * 100 > min_delta_pp]
    if len(significant) == 0:
        significant = summary.head(fallback_n)
        logger.info(f"  No lines exceed {min_delta_pp} pp — showing top {fallback_n}")
    else:
        significant = significant.head(max_n)

    top_lines = significant.index
    n_panels = len(top_lines)

    if n_panels == 0:
        logger.info("  Skipping timeseries plot — no lines to display")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), dpi=150, sharex=True)
    if n_panels == 1:
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
        label = rename_line_label(line)
        ax.set_ylabel(f'{label} [%]')
        ax.grid(True, alpha=0.3)

        # Annotate I_lim and ΔU
        ilim = summary.loc[line, 'i_lim_a']
        delta = summary.loc[line, 'delta_u_99'] * 100
        ax.text(0.98, 0.95,
                f'$I_{{lim}}$={ilim:.0f} A  |  $\\Delta U_{{99}}$={delta:.1f} pp',
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=7)

    threshold_note = f'$\\Delta U_{{99}}$ > {min_delta_pp} pp' if len(significant) > 0 else f'Top {fallback_n}'
    axes[0].set_title(f'Line Utilization Timeseries — LV Transformer {sub_label} ({threshold_note})')
    axes[-1].set_xlabel('Time')
    fig.tight_layout()
    fig.savefig(path / 'top_lines_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved top_lines_timeseries.png ({n_panels} panels)")


def plot_delta_u_distribution(summary, path, sub_label):
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
    ax1.set_title(f'Distribution of Deferrable Capacity — LV Transformer {sub_label}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: CDF
    sorted_vals = np.sort(delta_99.values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cdf, color='#1976D2', linewidth=2)
    ax2.set_xlabel('$\\Delta U_{99}$ [percentage points]')
    ax2.set_ylabel('CDF (fraction of lines)')
    ax2.set_title(f'CDF of Deferrable Capacity — LV Transformer {sub_label}')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'delta_u_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved delta_u_distribution.png")


def plot_deferral_years(summary, path, sub_label, growth_rates=None):
    """
    Stacked horizontal bar chart — for each line and growth scenario:
      dark segment  = years to 100 % under current (unbalanced) loading
      light segment = extra years gained by rebalancing (deferral)
    Three grouped rows per line (one per growth scenario).

    Line selection: lines where the medium growth scenario reaches 100 %
    within 50 years.  Fallback: top 5 lines by peak loading.
    """
    if growth_rates is None:
        growth_rates = DEFAULT_GROWTH_RATES

    n_scenarios = len(growth_rates)

    # Use base (first) growth scenario for line selection
    g_base = growth_rates[0]
    pct_base = int(round(g_base * 100))
    unbal_base = f'years_to_100_unbal_g{pct_base}'

    # Primary: lines that hit 100 % within 50 years at base growth
    mask = summary[unbal_base].notna() & (summary[unbal_base] <= 50)
    valid = summary.loc[mask].sort_values(unbal_base, ascending=True).head(15)

    # Fallback: top 5 most loaded lines
    if len(valid) == 0:
        valid = summary.sort_values('u_peak_99', ascending=False).head(5)
        logger.info(f"  Deferral plot: no lines reach 100 % within 50 yr "
                    f"at {pct_base} %/yr — showing top 5 by loading")

    if len(valid) == 0:
        logger.info(f"  Skipping deferral plot — no lines available")
        return

    # Colours: base (dark) + deferral (light) per scenario
    base_colours   = ['#0D47A1', '#E65100', '#B71C1C']   # dark blue / dark orange / dark red
    defer_colours  = ['#64B5F6', '#FFB74D', '#EF9A9A']   # light blue / light orange / light red
    scenario_labels_done = set()

    y = np.arange(len(valid))
    total_height = 0.82
    bar_h = total_height / n_scenarios

    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(valid))), dpi=150)

    max_end = 0
    for j, g in enumerate(growth_rates):
        pct = int(round(g * 100))
        unbal_col = f'years_to_100_unbal_g{pct}'
        defer_col = f'deferral_years_g{pct}'
        offsets = y - total_height / 2 + bar_h * (j + 0.5)

        base_vals  = valid[unbal_col].fillna(0).values
        defer_vals = valid[defer_col].fillna(0).values

        bc = base_colours[j % len(base_colours)]
        dc = defer_colours[j % len(defer_colours)]
        glabel = f'{_growth_label(g)}'

        # Base: years to capacity under unbalanced loading
        ax.barh(offsets, base_vals, bar_h * 0.88,
                color=bc, alpha=0.9, edgecolor='black', linewidth=0.3,
                label=f'{glabel} — unbalanced' if j not in scenario_labels_done else None)

        # Stacked: deferral gained by rebalancing
        ax.barh(offsets, defer_vals, bar_h * 0.88, left=base_vals,
                color=dc, alpha=0.85, edgecolor='black', linewidth=0.3,
                label=f'{glabel} — deferral' if j not in scenario_labels_done else None)
        scenario_labels_done.add(j)

        # Annotate deferral at end of stacked bar
        for i in range(len(valid)):
            end = base_vals[i] + defer_vals[i]
            if defer_vals[i] > 0.5:
                ax.text(end + 0.3, offsets[i], f'+{defer_vals[i]:.0f} yr',
                        va='center', fontsize=8, fontweight='bold', color=bc)
            max_end = max(max_end, end)

    ax.set_yticks(y)
    ax.set_yticklabels([rename_line_label(l) for l in valid.index])
    ax.set_xlabel('Years to 100 % Capacity')
    ax.set_title(f'Reinforcement Timeline — LV Transformer {sub_label}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    if np.isfinite(max_end) and max_end > 0:
        ax.set_xlim(right=max_end * 1.2)

    fig.tight_layout()
    fig.savefig(path / 'deferral_years.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved deferral_years.png ({n_scenarios} scenarios, stacked bars)")


# ---------------------------------------------------------------------------
#  Cross-substation plotting
# ---------------------------------------------------------------------------

def plot_cross_sub_headroom(cross_df, path, trafo_labels):
    """
    Bar chart of mean ΔU_99 (capacity headroom freed by rebalancing) per LV transformer,
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

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(df)), 7), dpi=150)
    x = np.arange(len(df))

    colors = ['#FF6B00'] * (len(df) - 1) + ['#333333']  # dark bar for average
    bars = ax.bar(x, df['mean_delta_u_99'] * 100, width=0.6,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Annotate max ΔU on top of each bar
    max_bar = df['max_delta_u_99'].max() * 100
    for i, (_, row) in enumerate(df.iterrows()):
        mean_val = row['mean_delta_u_99'] * 100
        max_val = row['max_delta_u_99'] * 100
        ax.text(i, mean_val + 0.3, f'{mean_val:.1f} pp',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        if max_val > mean_val + 0.5:
            ax.text(i, mean_val + max_bar * 0.08, f'(max {max_val:.1f})',
                    ha='center', va='bottom', fontsize=9, color='#555555')

    ax.set_xticks(x)
    labels = []
    for _, r in df.iterrows():
        sub = r['substation']
        sl = _sub_label(sub, trafo_labels) if sub != 'Avg' else 'Avg'
        labels.append(f'{sl}\n({r["n_lines"]:.0f} lines)')
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Capacity Headroom $\\overline{\\Delta U_{99}}$ [pp]')
    ax.set_title('Capacity Headroom Freed by Phase Rebalancing')
    ax.grid(True, axis='y', alpha=0.3)

    # Extra headroom for annotations
    ax.set_ylim(top=max_bar * 1.3)

    fig.tight_layout()
    fig.savefig(path / 'cross_substation_headroom.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved cross_substation_headroom.png")


def plot_cross_sub_deferral(cross_df, path, growth_rates=None, trafo_labels=None):
    """
    Grouped bar chart of mean deferral years per LV transformer — one bar per
    growth scenario, all in a single plot.
    """
    if growth_rates is None:
        growth_rates = DEFAULT_GROWTH_RATES
    if trafo_labels is None:
        trafo_labels = {}

    n_scenarios = len(growth_rates)
    scenario_colours = ['#1976D2', '#FF9800', '#D32F2F']

    df = cross_df.copy()

    # Weighted average row
    total_lines = df['n_lines'].sum()
    avg_dict = {'substation': 'Avg', 'n_lines': total_lines}
    for g in growth_rates:
        pct = int(round(g * 100))
        mc = f'mean_deferral_years_g{pct}'
        avg_dict[mc] = (df[mc] * df['n_lines']).sum() / total_lines
    df = pd.concat([df, pd.DataFrame([avg_dict])], ignore_index=True)

    n = len(df)
    x = np.arange(n)
    total_width = 0.75
    bar_w = total_width / n_scenarios

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * n), 7), dpi=150)

    max_bar = 0
    for j, g in enumerate(growth_rates):
        pct = int(round(g * 100))
        col = f'mean_deferral_years_g{pct}'
        vals = df[col].fillna(0).values
        offset = -total_width / 2 + bar_w * (j + 0.5)
        colour = scenario_colours[j % len(scenario_colours)]

        # Avg bar gets darker shade
        colours = [colour] * (n - 1) + ['#333333']

        ax.bar(x + offset, vals, bar_w * 0.9,
               color=colours, alpha=0.85, edgecolor='black', linewidth=0.5,
               label=f'{_growth_label(g)}')

        for i, v in enumerate(vals):
            if np.isfinite(v) and v > 0.1:
                ax.text(x[i] + offset, v + 0.2, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=colour if i < n - 1 else '#333333')
        if np.isfinite(vals.max()):
            max_bar = max(max_bar, vals.max())

    # X labels
    labels = []
    for _, r in df.iterrows():
        sub = r['substation']
        sl = _sub_label(sub, trafo_labels) if sub != 'Avg' else 'Avg'
        labels.append(f'{sl}\n({r["n_lines"]:.0f} lines)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel('Mean Reinforcement Deferral [years]')
    ax.set_title('Reinforcement Deferral from Phase Rebalancing',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    if max_bar > 0:
        ax.set_ylim(top=max_bar * 1.25)

    fig.tight_layout()
    fig.savefig(path / 'cross_substation_deferral.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved cross_substation_deferral.png")


def plot_cross_sub_delta_u_distribution(all_summaries, path):
    """
    Combined histogram + CDF of delta_U_99 across all selected substations.

    Shows aggregate totals only (no per-substation breakdown) for clarity.

    Args:
        all_summaries: dict {substation_id: summary DataFrame}
        path: output directory (PF_RESULTS_DIR)
    """
    # Pool all lines from all substations
    all_vals = []
    for sub, summary in all_summaries.items():
        vals = summary['delta_u_99'].dropna() * 100  # percentage points
        all_vals.extend(vals.tolist())
    all_vals = np.array(all_vals)
    n_subs = len(all_summaries)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: histogram
    ax1.hist(all_vals, bins=30, color='#FF6B00', alpha=0.7,
             edgecolor='black', linewidth=0.5)
    median_val = np.median(all_vals)
    ax1.axvline(median_val, color='red', linestyle='--', linewidth=1.5,
                label=f'Median: {median_val:.1f} pp')
    ax1.set_xlabel('Deferrable Capacity $\\Delta U_{99}$ [percentage points]')
    ax1.set_ylabel('Number of Lines')
    ax1.set_title(f'Distribution of Deferrable Capacity ({len(all_vals)} lines, {n_subs} LV transformers)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: CDF
    sorted_v = np.sort(all_vals)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax2.plot(sorted_v, cdf, color='#1976D2', linewidth=2)
    ax2.axvline(median_val, color='red', linestyle='--', linewidth=1,
                alpha=0.5, label=f'Median: {median_val:.1f} pp')
    ax2.set_xlabel('$\\Delta U_{99}$ [percentage points]')
    ax2.set_ylabel('CDF (fraction of lines)')
    ax2.set_title('CDF of Deferrable Capacity')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'cross_substation_delta_u_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved cross_substation_delta_u_distribution.png")


# ---------------------------------------------------------------------------
#  Summary printing
# ---------------------------------------------------------------------------

def _print_sub_summary(summary, sub, growth_rates=None):
    """Print a formatted summary table for one substation."""
    if growth_rates is None:
        growth_rates = DEFAULT_GROWTH_RATES

    g_base = growth_rates[0]  # use base scenario for display
    pct = int(round(g_base * 100))

    logger.info(f"\n  Line Utilization Summary (LV Transformer {sub})  "
                f"[showing {_growth_label(g_base)} growth scenario]:")
    logger.info(f"  {'─'*90}")
    logger.info(f"  {'Line':>12} {'I_lim':>7} {'U_pk99':>8} {'U_bl99':>8} "
                f"{'ΔU_99':>7} {'>50%':>6} {'>80%':>6} {'>100%':>6} "
                f"{'Yr_unbal':>9} {'Yr_bal':>7} {'Defer':>6}")
    logger.info(f"  {'─'*90}")

    for line, row in summary.head(10).iterrows():
        yr_u = row[f'years_to_100_unbal_g{pct}']
        yr_b = row[f'years_to_100_bal_g{pct}']
        defer = row[f'deferral_years_g{pct}']
        yr_u_str = f"{yr_u:>8.0f}y" if np.isfinite(yr_u) else f"{'∞':>9}"
        yr_b_str = f"{yr_b:>6.0f}y" if np.isfinite(yr_b) else f"{'∞':>7}"
        defer_str = f"+{defer:>4.0f}y" if np.isfinite(defer) else f"{'—':>6}"
        logger.info(
            f"  {rename_line_label(line):>12} {row['i_lim_a']:>6.0f}A "
            f"{row['u_peak_99']*100:>7.1f}% {row['u_balanced_99']*100:>7.1f}% "
            f"{row['delta_u_99']*100:>6.1f}pp "
            f"{row['share_above_50']*100:>5.1f}% {row['share_above_80']*100:>5.1f}% "
            f"{row['share_above_100']*100:>5.1f}% "
            f"{yr_u_str} {yr_b_str} {defer_str}"
        )

    if len(summary) > 10:
        logger.info(f"  ... ({len(summary) - 10} more lines)")


def _print_cross_sub_summary(rows, growth_rates=None):
    """Print cross-substation summary with headroom and deferral columns."""
    if growth_rates is None:
        growth_rates = DEFAULT_GROWTH_RATES

    g_base = growth_rates[0]
    pct = int(round(g_base * 100))

    logger.info(f"\n{'='*100}")
    logger.info(f"Cross-LV-Transformer Line Utilization Summary [showing {_growth_label(g_base)} growth]")
    logger.info(f"{'='*100}")
    logger.info(f"  {'Sub':>5} {'Lines':>6} {'U_pk99':>8} {'U_bl99':>8} "
                f"{'MaxΔU':>7} {'MeanΔU':>8} {'N>50%':>6} {'N>80%':>6} {'N>100%':>7} "
                f"{'MnDefer':>8} {'MxDefer':>8}")
    logger.info(f"  {'─'*95}")
    for r in rows:
        mn_d = r.get(f'mean_deferral_years_g{pct}', float('nan'))
        mx_d = r.get(f'max_deferral_years_g{pct}', float('nan'))
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

    # Load growth rates and transformer labels from config
    try:
        from Jason_config import load_config
        cfg = load_config()
        growth_rates = cfg.get('growth_rates', DEFAULT_GROWTH_RATES)
    except Exception:
        growth_rates = DEFAULT_GROWTH_RATES
    trafo_labels = _load_trafo_labels()
    logger.info(f"Growth rate scenarios: {[_growth_label(g) for g in growth_rates]}")

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
    all_summaries = {}  # per-substation summary for cross-sub plots

    for sub in substations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Line Utilization Analysis — LV Transformer {sub}")
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

        # Step 3: Compute per-line summary (with all growth scenarios)
        summary = compute_line_summary(u_peak, u_balanced, delta_u, i_lim, growth_rates)

        all_summaries[sub] = summary

        # Step 4: Save outputs
        out_path = sub_path / 'line_utilization'
        out_path.mkdir(parents=True, exist_ok=True)

        summary.to_parquet(out_path / 'line_utilization_summary.parquet')
        u_peak.to_parquet(out_path / 'u_peak_timeseries.parquet')
        u_balanced.to_parquet(out_path / 'u_balanced_timeseries.parquet')
        delta_u.to_parquet(out_path / 'delta_u_timeseries.parquet')
        logger.info(f"  Saved 4 parquets to {out_path}")

        # Step 5: Generate plots
        sl = _sub_label(sub, trafo_labels)
        plot_top_lines_bar(summary, out_path, sl)
        plot_top_lines_timeseries(u_peak, u_balanced, summary, out_path, sl)
        plot_delta_u_distribution(summary, out_path, sl)
        plot_deferral_years(summary, out_path, sl, growth_rates)

        # Step 6: Print summary table
        _print_sub_summary(summary, sub, growth_rates)

        # Collect cross-substation stats (including deferral for each scenario)
        row = {
            'substation': sub,
            'n_lines': len(summary),
            'max_u_peak_99': summary['u_peak_99'].max(),
            'max_u_balanced_99': summary['u_balanced_99'].max(),
            'max_delta_u_99': summary['delta_u_99'].max(),
            'mean_delta_u_99': summary['delta_u_99'].mean(),
            'n_above_50_unbal': (summary['share_above_50'] > 0).sum(),
            'n_above_80_unbal': (summary['share_above_80'] > 0).sum(),
            'n_above_100_unbal': (summary['share_above_100'] > 0).sum(),
            'n_above_50_bal': (summary['u_balanced_99'] > 0.5).sum(),
        }
        for g in growth_rates:
            pct = int(round(g * 100))
            valid_defer = summary[f'deferral_years_g{pct}'].dropna()
            row[f'mean_deferral_years_g{pct}'] = valid_defer.mean() if len(valid_defer) > 0 else np.nan
            row[f'max_deferral_years_g{pct}'] = valid_defer.max() if len(valid_defer) > 0 else np.nan

        cross_sub_rows.append(row)

    # Cross-substation summary
    if cross_sub_rows:
        _print_cross_sub_summary(cross_sub_rows, growth_rates)

        cross_df = pd.DataFrame(cross_sub_rows)

        # Generate cross-substation plots
        plot_cross_sub_headroom(cross_df, PF_RESULTS_DIR, trafo_labels)
        plot_cross_sub_deferral(cross_df, PF_RESULTS_DIR, growth_rates, trafo_labels)
        plot_cross_sub_delta_u_distribution(all_summaries, PF_RESULTS_DIR)

        # Save cross-substation summary
        cross_df.to_parquet(PF_RESULTS_DIR / 'cross_substation_line_utilization.parquet', index=False)
        logger.info(f"\n  Saved cross_substation_line_utilization.parquet")

    logger.info("\n" + "=" * 80)
    logger.info("Line utilization analysis complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
