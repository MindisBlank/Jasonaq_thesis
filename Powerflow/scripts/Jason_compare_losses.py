#!/usr/bin/env python3
"""
Jason_compare_losses.py
========================
Compares distribution losses between unbalanced and balanced power flow runs
to quantify the additional losses caused by phase imbalance.

Reads loss parquets from pf_results/{sub}/unbalanced/ and pf_results/{sub}/balanced/,
including cable I²R losses and grid throughput.

Outputs saved to pf_results/{sub}/comparison/:
  - additional_losses.parquet          (timeseries)
  - loss_summary.parquet               (single-row statistics)
  - additional_losses_timeseries.png   (instantaneous kW with context)
  - cumulative_energy_losses.png       (cumulative kWh)
  - loss_breakdown_bar.png             (stacked bar: unbalanced vs balanced)

Usage:
    python Jason_compare_losses.py
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
SELECTED_SUBSTATIONS = ['1299', '1456', '1056', '579', '1457', '1340']
# Global font size bump — applied at module level
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14,'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})


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


def load_loss_data(sub_path, run_label):
    """
    Load all loss parquets for a given run.

    Returns dict with keys:
        'phase_kw', 'neutral_kw', 'cable_total_kw' (pd.Series)
        'throughput_kw' (pd.Series from power_grid P column)
    """
    run_path = sub_path / run_label

    # Cable losses
    loss_phase = pd.read_parquet(run_path / 'loss_lines_phase.parquet')
    loss_neutral = pd.read_parquet(run_path / 'loss_lines_neutral.parquet')
    loss_phase = loss_phase.apply(pd.to_numeric, errors='coerce')
    loss_neutral = loss_neutral.apply(pd.to_numeric, errors='coerce')
    phase_kw = loss_phase.sum(axis=1)
    neutral_kw = loss_neutral.sum(axis=1)
    cable_total_kw = phase_kw + neutral_kw

    # Grid throughput
    grid_path = run_path / 'power_grid.parquet'
    if grid_path.exists():
        power_grid = pd.read_parquet(grid_path)
        power_grid = power_grid.apply(pd.to_numeric, errors='coerce')
        throughput_kw = power_grid['P'].fillna(0)
    else:
        throughput_kw = pd.Series(0.0, index=phase_kw.index)

    return {
        'phase_kw': phase_kw,
        'neutral_kw': neutral_kw,
        'cable_total_kw': cable_total_kw,
        'throughput_kw': throughput_kw,
    }


def compute_additional_losses(unbal, bal):
    """
    Compute the additional losses from phase imbalance.

    Returns pd.DataFrame with columns for both runs and the difference (cable losses only).
    """
    common_idx = unbal['phase_kw'].dropna().index.intersection(bal['phase_kw'].dropna().index)

    df = pd.DataFrame(index=common_idx)

    # Cable losses
    df['unbalanced_phase_kw'] = unbal['phase_kw'].loc[common_idx]
    df['unbalanced_neutral_kw'] = unbal['neutral_kw'].loc[common_idx]
    df['unbalanced_cable_kw'] = unbal['cable_total_kw'].loc[common_idx]
    df['balanced_phase_kw'] = bal['phase_kw'].loc[common_idx]
    df['balanced_neutral_kw'] = bal['neutral_kw'].loc[common_idx]
    df['balanced_cable_kw'] = bal['cable_total_kw'].loc[common_idx]

    # Total distribution losses (cable only)
    df['unbalanced_total_kw'] = df['unbalanced_cable_kw']
    df['balanced_total_kw'] = df['balanced_cable_kw']

    # Additional losses from imbalance
    df['additional_phase_loss_kw'] = df['unbalanced_phase_kw'] - df['balanced_phase_kw']
    df['additional_neutral_loss_kw'] = df['unbalanced_neutral_kw'] - df['balanced_neutral_kw']
    df['additional_cable_loss_kw'] = df['unbalanced_cable_kw'] - df['balanced_cable_kw']
    df['additional_total_loss_kw'] = df['unbalanced_total_kw'] - df['balanced_total_kw']

    # Throughput (from unbalanced run)
    df['throughput_kw'] = unbal['throughput_kw'].reindex(common_idx).fillna(0)

    return df


def plot_additional_losses(df, path, sub_label):
    """
    Plot the additional losses from phase imbalance over time.
    Includes a gray reference line showing total unbalanced losses for scale.
    """
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # Reference line: total unbalanced losses (for scale)
    ax.plot(df.index, df['unbalanced_total_kw'],
            color='gray', linewidth=0.8, alpha=0.4, label='Total unbalanced losses (reference)')

    # Fill: phase losses at bottom, neutral losses stacked on top
    ax.fill_between(df.index, 0, df['additional_phase_loss_kw'],
                    alpha=0.5, color='#1976D2', label='Additional phase conductor losses')
    ax.fill_between(df.index, df['additional_phase_loss_kw'],
                    df['additional_phase_loss_kw'] + df['additional_neutral_loss_kw'],
                    alpha=0.5, color='#FF6B00', label='Additional neutral conductor losses')

    # Total additional losses line
    ax.plot(df.index, df['additional_total_loss_kw'],
            color='#D32F2F', linewidth=1.0, alpha=0.8, label='Additional total losses')

    # Annotation with total additional kWh
    dt_hours = 0.25
    total_add_kwh = (df['additional_total_loss_kw'] * dt_hours).sum()
    total_bal_kwh = (df['balanced_total_kw'] * dt_hours).sum()
    pct_increase = total_add_kwh / total_bal_kwh * 100 if total_bal_kwh > 0 else 0

    ax.text(0.02, 0.95,
            f'Additional: {total_add_kwh:.1f} kWh ({pct_increase:.1f}% increase over balanced)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_ylabel('Losses [kW]')
    ax.set_xlabel('Time')
    ax.set_title(f'Additional Cable Losses due to Phase Imbalance — {sub_label}', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / 'additional_losses_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved additional_losses_timeseries.png")


def plot_cumulative_energy(df, path, sub_label, dt_hours=0.25):
    """
    Plot cumulative energy losses (kWh) over the simulation period.
    """
    cumul_phase = (df['additional_phase_loss_kw'] * dt_hours).cumsum()
    cumul_neutral = (df['additional_neutral_loss_kw'] * dt_hours).cumsum()
    cumul_total = (df['additional_total_loss_kw'] * dt_hours).cumsum()

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    ax.fill_between(df.index, 0, cumul_phase,
                    alpha=0.4, color='#1976D2', label='Phase conductor losses')
    ax.fill_between(df.index, cumul_phase, cumul_phase + cumul_neutral,
                    alpha=0.4, color='#FF6B00', label='Neutral conductor losses')
    ax.plot(df.index, cumul_total,
            color='#D32F2F', linewidth=1.5, label='Total additional losses')

    ax.set_ylabel('Cumulative Additional Energy Loss [kWh]')
    ax.set_xlabel('Time')
    ax.set_title(f'Cumulative Additional Energy Loss from Phase Imbalance — {sub_label}', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / 'cumulative_energy_losses.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved cumulative_energy_losses.png")


def plot_loss_breakdown_bar(summary, path, sub_label):
    """
    Stacked horizontal bar chart: unbalanced vs balanced loss breakdown.
    Shows cable phase and cable neutral losses.
    """
    s = summary.iloc[0]

    categories = ['Balanced', 'Unbalanced']

    # Loss components (kWh)
    cable_phase = [s['balanced_phase_kwh'], s['unbalanced_phase_kwh']]
    cable_neutral = [s['balanced_neutral_kwh'], s['unbalanced_neutral_kwh']]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    y_pos = np.arange(len(categories))
    bar_height = 0.5

    # Stacked bars
    ax.barh(y_pos, cable_phase, bar_height,
            color='#1976D2', alpha=0.8, label='Cable phase losses')
    left = np.array(cable_phase)

    ax.barh(y_pos, cable_neutral, bar_height, left=left,
            color='#FF6B00', alpha=0.8, label='Cable neutral losses')
    left = left + np.array(cable_neutral)

    # Totals and percentage annotations
    totals = left
    throughput = s.get('throughput_kwh', 0)
    for i, (cat, total) in enumerate(zip(categories, totals)):
        pct_str = f' ({total/throughput*100:.3f}% of throughput)' if throughput > 0 else ''
        ax.text(total + max(totals) * 0.01, i,
                f'{total:.1f} kWh{pct_str}',
                va='center', ha='left', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Energy Loss [kWh]')
    ax.set_title(f'Cable Loss Breakdown — {sub_label}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)

    # Expand x-axis to fit annotations
    ax.set_xlim(0, max(totals) * 1.55)

    fig.tight_layout()
    fig.savefig(path / 'loss_breakdown_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved loss_breakdown_bar.png")


def compute_summary(df, dt_hours=0.25):
    """
    Compute summary statistics for the loss comparison.

    Returns a single-row DataFrame with totals (kWh), mean (kW), peak (kW),
    and percentage increase.
    """
    n_steps = len(df)
    summary = {}

    # Cable energy losses (kWh)
    summary['unbalanced_phase_kwh'] = (df['unbalanced_phase_kw'] * dt_hours).sum()
    summary['unbalanced_neutral_kwh'] = (df['unbalanced_neutral_kw'] * dt_hours).sum()
    summary['unbalanced_cable_kwh'] = (df['unbalanced_cable_kw'] * dt_hours).sum()
    summary['balanced_phase_kwh'] = (df['balanced_phase_kw'] * dt_hours).sum()
    summary['balanced_neutral_kwh'] = (df['balanced_neutral_kw'] * dt_hours).sum()
    summary['balanced_cable_kwh'] = (df['balanced_cable_kw'] * dt_hours).sum()

    # Total distribution losses (cable only)
    summary['unbalanced_total_kwh'] = summary['unbalanced_cable_kwh']
    summary['balanced_total_kwh'] = summary['balanced_cable_kwh']

    # Additional losses
    summary['additional_total_kwh'] = (df['additional_total_loss_kw'] * dt_hours).sum()
    summary['additional_phase_kwh'] = (df['additional_phase_loss_kw'] * dt_hours).sum()
    summary['additional_neutral_kwh'] = (df['additional_neutral_loss_kw'] * dt_hours).sum()
    summary['additional_cable_kwh'] = (df['additional_cable_loss_kw'] * dt_hours).sum()

    # Percentage increase over balanced
    if summary['balanced_total_kwh'] > 0:
        summary['loss_increase_percent'] = (
            summary['additional_total_kwh'] / summary['balanced_total_kwh'] * 100
        )
    else:
        summary['loss_increase_percent'] = np.nan

    # Mean and peak instantaneous additional losses
    summary['additional_mean_kw'] = df['additional_total_loss_kw'].mean()
    summary['additional_peak_kw'] = df['additional_total_loss_kw'].max()
    summary['unbalanced_mean_kw'] = df['unbalanced_total_kw'].mean()
    summary['balanced_mean_kw'] = df['balanced_total_kw'].mean()

    # Neutral share of additional losses
    if summary['additional_total_kwh'] > 0:
        summary['neutral_share_percent'] = (
            summary['additional_neutral_kwh'] / summary['additional_total_kwh'] * 100
        )
    else:
        summary['neutral_share_percent'] = np.nan

    # Throughput (total energy delivered by grid)
    summary['throughput_kwh'] = (df['throughput_kw'] * dt_hours).sum()

    # Losses as percentage of throughput
    if summary['throughput_kwh'] > 0:
        summary['unbalanced_loss_pct'] = summary['unbalanced_total_kwh'] / summary['throughput_kwh'] * 100
        summary['balanced_loss_pct'] = summary['balanced_total_kwh'] / summary['throughput_kwh'] * 100
    else:
        summary['unbalanced_loss_pct'] = np.nan
        summary['balanced_loss_pct'] = np.nan

    summary['n_timesteps'] = n_steps
    summary['dt_hours'] = dt_hours
    summary['simulation_hours'] = n_steps * dt_hours

    return pd.DataFrame([summary])


def plot_cross_substation_bar(all_summaries, path, trafo_labels):
    """
    Grouped bar chart comparing unbalanced vs balanced losses across all LV transformers.
    Stacked by component: cable phase, cable neutral.
    Includes a 'Total' group showing the aggregate.
    """
    # Build totals row
    totals = all_summaries[['unbalanced_phase_kwh', 'unbalanced_neutral_kwh',
                            'balanced_phase_kwh', 'balanced_neutral_kwh',
                            'unbalanced_total_kwh', 'balanced_total_kwh',
                            'additional_total_kwh', 'throughput_kwh']].sum()
    totals['substation'] = 'Total'
    totals_df = pd.DataFrame([totals])
    df = pd.concat([all_summaries, totals_df], ignore_index=True)

    labels = [_sub_label(s, trafo_labels) if s != 'Total' else 'Total'
              for s in df['substation'].astype(str)]
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(14, 2 * n), 7), dpi=150)

    # Balanced (left bars) — stacked
    b_phase = df['balanced_phase_kwh'].values
    b_neutral = df['balanced_neutral_kwh'].values

    ax.bar(x - width/2, b_phase, width, color='#1976D2', alpha=0.8, label='Cable phase')
    ax.bar(x - width/2, b_neutral, width, bottom=b_phase, color='#FF6B00', alpha=0.8, label='Cable neutral')

    # Unbalanced (right bars) — stacked
    u_phase = df['unbalanced_phase_kwh'].values
    u_neutral = df['unbalanced_neutral_kwh'].values

    ax.bar(x + width/2, u_phase, width, color='#1976D2', alpha=0.4)
    ax.bar(x + width/2, u_neutral, width, bottom=u_phase, color='#FF6B00', alpha=0.4)

    # Annotate additional losses above each pair
    max_bar = max(df['unbalanced_total_kwh'].max(), df['balanced_total_kwh'].max())
    # Pre-compute rounded additional losses per substation so Total label
    # matches the visible per-substation numbers (avoids ±0.1 rounding gap).
    n_substations = n - 1  # exclude the Total row
    rounded_additional_sum = sum(
        round(df.iloc[i]['additional_total_kwh'], 1) for i in range(n_substations)
    )

    for i in range(n):
        add_kwh = df.iloc[i]['additional_total_kwh']
        bal_kwh = df.iloc[i]['balanced_total_kwh']
        unbal_kwh = df.iloc[i]['unbalanced_total_kwh']
        pct = add_kwh / bal_kwh * 100 if bal_kwh > 0 else 0
        y_top = max(unbal_kwh, bal_kwh)

        # Use the rounded-sum for the Total bar so it's consistent with
        # what you'd get by adding up the individual displayed labels.
        display_kwh = rounded_additional_sum if labels[i] == 'Total' else add_kwh

        ax.text(i, y_top + max_bar * 0.02,
                f'+{display_kwh:.1f} kWh\n({pct:.1f}%)',
                ha='center', fontsize=10, color='#D32F2F', fontweight='bold')

    # Separator line before Total
    ax.axvline(x=n - 1.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Cable Losses [kWh]')
    ax.set_title('Cable Loss Comparison: Balanced (solid) vs Unbalanced (faded)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    # Extra headroom for annotations
    ax.set_ylim(top=max_bar * 1.25)

    fig.tight_layout()
    fig.savefig(path / 'cross_substation_loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved cross_substation_loss_comparison.png")


def plot_cross_substation_loss_percent(all_summaries, path, trafo_labels):
    """
    Bar chart showing loss_increase_percent per LV transformer + weighted average.
    """
    # Weighted average: total additional / total balanced × 100
    total_add = all_summaries['additional_total_kwh'].sum()
    total_bal = all_summaries['balanced_total_kwh'].sum()
    weighted_avg = total_add / total_bal * 100 if total_bal > 0 else 0

    labels = [_sub_label(s, trafo_labels) for s in all_summaries['substation'].astype(str)] + ['Weighted\nAverage']
    values = all_summaries['loss_increase_percent'].tolist() + [weighted_avg]
    colors = ['#D32F2F'] * len(all_summaries) + ['#1976D2']

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(labels)), 5), dpi=150)
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Separator before average
    ax.axvline(x=len(all_summaries) - 0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_ylabel('Additional Cable Losses from Imbalance [%]')
    ax.set_title('Cable Loss Increase Due to Phase Imbalance by Supply Point', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / 'cross_substation_loss_percent.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved cross_substation_loss_percent.png")


def main():
    logger.info("=" * 80)
    logger.info("Jason Loss Comparison: Unbalanced vs Balanced")
    logger.info("=" * 80)

    trafo_labels = _load_trafo_labels()

    # Auto-detect substations with both unbalanced and balanced results
    if not PF_RESULTS_DIR.exists():
        logger.error(f"Results directory not found: {PF_RESULTS_DIR}")
        return

    substations = []
    for sub_dir in sorted(PF_RESULTS_DIR.iterdir()):
        if sub_dir.is_dir():
            unbal_dir = sub_dir / 'unbalanced'
            bal_dir = sub_dir / 'balanced'
            if (unbal_dir / 'loss_lines_phase.parquet').exists() and \
               (bal_dir / 'loss_lines_phase.parquet').exists():
                substations.append(sub_dir.name)

    if not substations:
        logger.error("No substations found with both unbalanced and balanced results.")
        logger.error("Run Jason_run_powerflow.py --mode both first.")
        return

    # Filter to selected substations if specified
    if SELECTED_SUBSTATIONS is not None:
        substations = [s for s in substations if s in SELECTED_SUBSTATIONS]
        if not substations:
            logger.error(f"None of the selected substations {SELECTED_SUBSTATIONS} have results.")
            return

    logger.info(f"Substations with both runs: {substations}")

    all_summaries = []

    for sub in substations:
        sl = _sub_label(sub, trafo_labels)
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing losses for Supply Point {sl}")
        logger.info(f"{'='*60}")

        sub_path = PF_RESULTS_DIR / sub

        # Load loss data
        unbal = load_loss_data(sub_path, 'unbalanced')
        bal = load_loss_data(sub_path, 'balanced')

        logger.info(f"  Unbalanced: {len(unbal['phase_kw'])} timesteps")
        logger.info(f"  Balanced:   {len(bal['phase_kw'])} timesteps")

        # Compute additional losses
        df = compute_additional_losses(unbal, bal)
        logger.info(f"  Common timesteps: {len(df)}")

        # Save
        comp_path = sub_path / 'comparison'
        comp_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(comp_path / 'additional_losses.parquet')

        # Summary statistics
        summary = compute_summary(df)
        summary.to_parquet(comp_path / 'loss_summary.parquet')

        # Plots
        plot_additional_losses(df, comp_path, sl)
        plot_cumulative_energy(df, comp_path, sl)
        plot_loss_breakdown_bar(summary, comp_path, sl)

        # Print summary
        s = summary.iloc[0]
        logger.info(f"\n  Loss Comparison Summary (Supply Point {sl}):")
        logger.info(f"  {'─'*60}")
        logger.info(f"  Simulation period:         {s['simulation_hours']:.0f} hours ({s['n_timesteps']:.0f} timesteps)")
        logger.info(f"  Grid throughput:           {s['throughput_kwh']:.0f} kWh")
        logger.info(f"  {'─'*60}")
        logger.info(f"  UNBALANCED total losses:   {s['unbalanced_total_kwh']:.2f} kWh ({s['unbalanced_loss_pct']:.3f}% of throughput)")
        logger.info(f"    Cable phase:             {s['unbalanced_phase_kwh']:.2f} kWh")
        logger.info(f"    Cable neutral:           {s['unbalanced_neutral_kwh']:.2f} kWh")
        logger.info(f"  BALANCED total losses:     {s['balanced_total_kwh']:.2f} kWh ({s['balanced_loss_pct']:.3f}% of throughput)")
        logger.info(f"    Cable phase:             {s['balanced_phase_kwh']:.2f} kWh")
        logger.info(f"    Cable neutral:           {s['balanced_neutral_kwh']:.2f} kWh")
        logger.info(f"  {'─'*60}")
        logger.info(f"  ADDITIONAL losses:         {s['additional_total_kwh']:.2f} kWh ({s['loss_increase_percent']:.1f}% increase)")
        logger.info(f"    Phase conductors:        {s['additional_phase_kwh']:.2f} kWh")
        logger.info(f"    Neutral conductor:       {s['additional_neutral_kwh']:.2f} kWh ({s['neutral_share_percent']:.1f}% of additional)")
        logger.info(f"  Peak additional loss:      {s['additional_peak_kw']:.4f} kW")
        logger.info(f"  Mean additional loss:      {s['additional_mean_kw']:.4f} kW")

        # Collect for cross-substation summary
        s_row = summary.iloc[0].to_dict()
        s_row['substation'] = sub
        all_summaries.append(s_row)

    # ------------------------------------------------------------------
    #  Cross-substation aggregate summary
    # ------------------------------------------------------------------
    if len(all_summaries) > 1:
        cross_df = pd.DataFrame(all_summaries)

        # Compute weighted totals
        total_throughput = cross_df['throughput_kwh'].sum()
        total_unbal = cross_df['unbalanced_total_kwh'].sum()
        total_bal = cross_df['balanced_total_kwh'].sum()
        total_add = cross_df['additional_total_kwh'].sum()
        weighted_pct = total_add / total_bal * 100 if total_bal > 0 else 0

        logger.info(f"\n{'='*80}")
        logger.info("Cross-Substation Loss Summary")
        logger.info(f"{'='*80}")
        logger.info(f"  {'Sub':>6} {'Throughput':>12} {'Unbal':>10} {'Balanced':>10} "
                     f"{'Additional':>11} {'Increase':>9} {'Loss%_unbal':>12} {'Loss%_bal':>10}")
        logger.info(f"  {'─'*80}")
        for _, r in cross_df.iterrows():
            logger.info(
                f"  {str(r['substation']):>6} {r['throughput_kwh']:>10.0f} kWh "
                f"{r['unbalanced_total_kwh']:>8.1f} kWh {r['balanced_total_kwh']:>8.1f} kWh "
                f"{r['additional_total_kwh']:>9.1f} kWh {r['loss_increase_percent']:>8.1f}% "
                f"{r['unbalanced_loss_pct']:>10.3f}% {r['balanced_loss_pct']:>8.3f}%"
            )
        logger.info(f"  {'─'*80}")
        logger.info(
            f"  {'TOTAL':>6} {total_throughput:>10.0f} kWh "
            f"{total_unbal:>8.1f} kWh {total_bal:>8.1f} kWh "
            f"{total_add:>9.1f} kWh {weighted_pct:>8.1f}% "
            f"{total_unbal/total_throughput*100:>10.3f}% {total_bal/total_throughput*100:>8.3f}%"
        )

        # Plots
        plot_cross_substation_bar(cross_df, PF_RESULTS_DIR, trafo_labels)
        plot_cross_substation_loss_percent(cross_df, PF_RESULTS_DIR, trafo_labels)

        # Save
        cross_df.to_parquet(PF_RESULTS_DIR / 'cross_substation_loss_summary.parquet', index=False)
        logger.info(f"Saved cross_substation_loss_summary.parquet")

    logger.info("\n" + "=" * 80)
    logger.info("Loss comparison complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
