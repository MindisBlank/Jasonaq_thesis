#!/usr/bin/env python3
"""
Jason_customer_imbalance.py
============================
Explores the relationship between customer / meter type (lysingnotkunarflokks)
and phase imbalance across selected substations.

Links smart meter customer categories with power flow imbalance metrics (VUF,
additional losses) and produces thesis-quality visualizations and summary tables.

Outputs saved to pf_results/customer_imbalance/:
  - meter_imbalance.csv             (per-meter metrics with customer types)
  - substation_summary.csv          (per-substation summary)
  - customer_type_summary.csv       (per-type aggregated statistics)
  - fig1_customer_mix_vs_imbalance.png
  - fig2_meter_imbalance_by_type.png
  - fig3_customer_type_summary_table.png
  - fig4_substation_heatmap.png
  - fig5_household_vs_losses.png

Usage:
    python Jason_customer_imbalance.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PF_RESULTS_DIR = PROJECT_ROOT / "Powerflow" / "output" / "pf_results"
SMARTMETER_DIR = PROJECT_ROOT / "data" / "smartmeter"
OUTPUT_DIR = PF_RESULTS_DIR / "customer_imbalance"

# ---------------------------------------------------------------------------
#  Target substations and their smart meter files
# ---------------------------------------------------------------------------
SELECTED_SUBSTATIONS = ['579', '1056', '1299', '1340', '1456', '1457']

SMARTMETER_FILES = {
    '579':  'smartmeter_15min__substation_0579_transformer_sp1_20250801_20250901.parquet',
    '1056': 'smartmeter_15min__substation_1056_transformer_sp1_20250801_20250901.parquet',
    '1299': 'smartmeter_15min__substation_1299_transformer_sp1_20250801_20250901.parquet',
    '1340': 'smartmeter_15min__substation_1340_transformer_sp1_20250801_20250901.parquet',
    '1456': 'smartmeter_15min__substation_1456_transformer_sp1_20250801_20250901.parquet',
    '1457': 'smartmeter_15min__substation_1457_transformer_sp1_20250801_20250901.parquet',
}

# Columns to load from smart meter parquets
SM_COLUMNS = [
    'husveita_fastanumer', 'lysingnotkunarflokks', 'substation_id',
    'numer_heimlagnar', 'P_a', 'P_b', 'P_c', 'ts',
]

# ---------------------------------------------------------------------------
#  Customer type translation (Icelandic -> English)
#  Built dynamically: keys are stripped raw strings from parquet files.
# ---------------------------------------------------------------------------
# Known translations — the keys will be matched after .strip()
_TRANSLATION_MAP = {
    'Almenn heimilisnotkun án rafhita': 'General household',
    'Sérmæld rafhitun íbúðarhúsnæðis': 'Electric space heating',
    'Önnur þjónusta og handiðnir': 'Other services & trades',
    'Byggingastarfs. og opinb. framkv': 'Construction & public works',
    'Rafknúin farartæki': 'Electric vehicles',
    'Dagheimili og leikskólar': 'Daycare & kindergarten',
    'Götu- og hafnarlýsing': 'Street & harbour lighting',
    'Almenn stjórnsýsla': 'Public administration',
    'Hitaveitur': 'District heating',
    'Heildverslun': 'Wholesale trade',
    'Menningarmál': 'Cultural affairs',
    'Póstur og sími': 'Post & telecom',
    'Veitingastaðir': 'Restaurants',
    'Félagsheimili': 'Community centres',
    'Rafmagn til skipa': 'Ship shore power',
}


def translate_customer_type(raw_label):
    """Translate an Icelandic customer type label to English."""
    stripped = raw_label.strip()
    # Try direct match first
    if stripped in _TRANSLATION_MAP:
        return _TRANSLATION_MAP[stripped]
    # Try case-insensitive / partial match
    for key, val in _TRANSLATION_MAP.items():
        if key.lower() in stripped.lower() or stripped.lower() in key.lower():
            return val
    # Fallback: return stripped original
    return stripped


# ---------------------------------------------------------------------------
#  Consistent color palette for customer types
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    'General household':       '#1976D2',
    'Other services & trades': '#FF6B00',
    'Electric vehicles':       '#4CAF50',
    'District heating':        '#9C27B0',
    'Post & telecom':          '#00BCD4',
    'Construction & public works': '#795548',
    'Daycare & kindergarten':  '#E91E63',
    'Wholesale trade':         '#607D8B',
    'Public administration':   '#FFC107',
    'Electric space heating':  '#D32F2F',
    'Street & harbour lighting': '#8BC34A',
    'Cultural affairs':        '#3F51B5',
    'Restaurants':             '#FF5722',
    'Community centres':       '#009688',
    'Ship shore power':        '#455A64',
}


def get_color(ctype_en):
    return TYPE_COLORS.get(ctype_en, '#999999')


# ===================================================================
#  Section 1: Data Loading
# ===================================================================

def load_all_data(substations):
    """
    Load smart meter data and VUF for each substation.

    Returns:
        sm_data: dict {sub_id: DataFrame} — smart meter data
        vuf_data: dict {sub_id: DataFrame} — VUF timeseries
        loss_summary: DataFrame — cross-substation loss summary
    """
    sm_data = {}
    vuf_data = {}

    for sub in substations:
        sm_file = SMARTMETER_DIR / SMARTMETER_FILES[sub]
        if not sm_file.exists():
            logger.warning(f"  Smart meter file not found for {sub}: {sm_file}")
            continue

        logger.info(f"  Loading substation {sub}...")
        sm = pd.read_parquet(sm_file, columns=SM_COLUMNS)
        sm['P_a'] = pd.to_numeric(sm['P_a'], errors='coerce').fillna(0)
        sm['P_b'] = pd.to_numeric(sm['P_b'], errors='coerce').fillna(0)
        sm['P_c'] = pd.to_numeric(sm['P_c'], errors='coerce').fillna(0)
        sm_data[sub] = sm

        vuf_path = PF_RESULTS_DIR / sub / 'unbalanced' / 'VUF.parquet'
        if vuf_path.exists():
            vuf_data[sub] = pd.read_parquet(vuf_path)
        else:
            logger.warning(f"  VUF not found for {sub}")

    # Cross-substation loss summary
    loss_path = PF_RESULTS_DIR / 'cross_substation_loss_summary.parquet'
    if loss_path.exists():
        loss_summary = pd.read_parquet(loss_path)
        loss_summary['substation'] = loss_summary['substation'].astype(str)
    else:
        logger.warning("cross_substation_loss_summary.parquet not found")
        loss_summary = pd.DataFrame()

    return sm_data, vuf_data, loss_summary


# ===================================================================
#  Section 2: Per-Meter Imbalance Computation
# ===================================================================

def compute_per_meter_imbalance(sm_data):
    """
    For each meter across all substations, compute time-averaged
    per-phase imbalance metrics from P_a, P_b, P_c.

    Returns DataFrame with one row per meter.
    """
    rows = []

    for sub, sm in sm_data.items():
        meters = sm.groupby('husveita_fastanumer')

        for meter_id, mdf in meters:
            p_a = mdf['P_a'].values
            p_b = mdf['P_b'].values
            p_c = mdf['P_c'].values

            # Customer type (static per meter)
            raw_type = mdf['lysingnotkunarflokks'].iloc[0]
            en_type = translate_customer_type(raw_type)

            # 3-phase classification: has nonzero P_b or P_c
            is_3ph = (np.abs(p_b).sum() > 0) or (np.abs(p_c).sum() > 0)

            # Per-timestep phase standard deviation
            phase_stack = np.column_stack([p_a, p_b, p_c])
            phase_std_per_t = np.std(phase_stack, axis=1)
            mean_phase_std = float(np.mean(phase_std_per_t))

            # Dominant phase fraction: fraction of total |P| on the most-loaded phase
            abs_stack = np.abs(phase_stack)
            total_per_t = abs_stack.sum(axis=1)
            max_per_t = abs_stack.max(axis=1)
            # Avoid division by zero at timesteps with no load
            mask = total_per_t > 1.0  # threshold: 1 W
            if mask.sum() > 0:
                dom_frac = float(np.mean(max_per_t[mask] / total_per_t[mask]))
            else:
                dom_frac = np.nan

            # Mean total power
            mean_total = float(np.mean(np.abs(p_a) + np.abs(p_b) + np.abs(p_c)))

            rows.append({
                'meter_id': meter_id,
                'substation': sub,
                'customer_type': raw_type.strip(),
                'customer_type_en': en_type,
                'is_3phase': is_3ph,
                'mean_total_W': mean_total,
                'phase_power_std_W': mean_phase_std,
                'dominant_phase_fraction': dom_frac,
            })

    return pd.DataFrame(rows)


# ===================================================================
#  Section 3: Substation and Customer-Type Summaries
# ===================================================================

def build_substation_summary(meter_df, vuf_data, loss_summary):
    """Build a per-substation summary table."""
    rows = []

    for sub in meter_df['substation'].unique():
        sub_meters = meter_df[meter_df['substation'] == sub]
        n_meters = len(sub_meters)
        n_types = sub_meters['customer_type_en'].nunique()

        # Dominant customer type
        type_counts = sub_meters['customer_type_en'].value_counts()
        dominant_type = type_counts.index[0]

        # Household fraction
        hh_count = type_counts.get('General household', 0)
        hh_fraction = hh_count / n_meters if n_meters > 0 else 0

        # VUF at transformer bus
        mean_vuf = max_vuf = p99_vuf = np.nan
        if sub in vuf_data and 'lv_trafo' in vuf_data[sub].columns:
            vuf_series = vuf_data[sub]['lv_trafo'].dropna()
            if len(vuf_series) > 0:
                mean_vuf = float(vuf_series.mean())
                max_vuf = float(vuf_series.max())
                p99_vuf = float(vuf_series.quantile(0.99))

        # Loss increase from cross-substation summary
        loss_pct = additional_kwh = np.nan
        if not loss_summary.empty:
            sub_loss = loss_summary[loss_summary['substation'] == sub]
            if len(sub_loss) > 0:
                loss_pct = float(sub_loss['loss_increase_percent'].iloc[0])
                additional_kwh = float(sub_loss['additional_total_kwh'].iloc[0])

        # Mean per-meter metrics
        mean_meter_std = float(sub_meters['phase_power_std_W'].mean())
        frac_3ph = float(sub_meters['is_3phase'].mean())

        rows.append({
            'substation': sub,
            'n_meters': n_meters,
            'n_customer_types': n_types,
            'dominant_type': dominant_type,
            'household_fraction': hh_fraction,
            'fraction_3phase': frac_3ph,
            'mean_VUF_pct': mean_vuf,
            'max_VUF_pct': max_vuf,
            'p99_VUF_pct': p99_vuf,
            'loss_increase_pct': loss_pct,
            'additional_loss_kwh': additional_kwh,
            'mean_meter_phase_std_W': mean_meter_std,
        })

    return pd.DataFrame(rows)


def build_customer_type_summary(meter_df):
    """Aggregate per-meter metrics by customer type."""
    grouped = meter_df.groupby('customer_type_en').agg(
        n_meters=('meter_id', 'count'),
        mean_total_W=('mean_total_W', 'mean'),
        median_total_W=('mean_total_W', 'median'),
        mean_phase_std_W=('phase_power_std_W', 'mean'),
        median_phase_std_W=('phase_power_std_W', 'median'),
        mean_dom_frac=('dominant_phase_fraction', 'mean'),
        fraction_3phase=('is_3phase', 'mean'),
        n_substations=('substation', 'nunique'),
    ).reset_index()
    grouped = grouped.sort_values('n_meters', ascending=False)
    return grouped


# ===================================================================
#  Section 4: Visualizations
# ===================================================================

def fig1_customer_mix_vs_imbalance(sub_summary, meter_df, output_path):
    """
    Stacked bar chart: customer type proportions per substation,
    with loss_increase_percent overlaid on secondary axis.
    Substations ordered by loss_increase_percent.
    """
    sub_summary = sub_summary.sort_values('loss_increase_pct', ascending=True).copy()
    subs = sub_summary['substation'].tolist()

    # Get customer type proportions per substation
    all_types = sorted(meter_df['customer_type_en'].unique())
    proportions = {}
    for ct in all_types:
        props = []
        for sub in subs:
            sub_meters = meter_df[meter_df['substation'] == sub]
            n = len(sub_meters)
            ct_count = len(sub_meters[sub_meters['customer_type_en'] == ct])
            props.append(ct_count / n if n > 0 else 0)
        proportions[ct] = props

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(subs))
    width = 0.6

    # Stacked bars
    bottom = np.zeros(len(subs))
    for ct in all_types:
        vals = np.array(proportions[ct])
        ax1.bar(x, vals, width, bottom=bottom, label=ct, color=get_color(ct),
                edgecolor='white', linewidth=0.5)
        bottom += vals

    ax1.set_ylabel('Customer Type Proportion')
    ax1.set_xlabel('Substation')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Sub {s}' for s in subs], fontsize=10)
    ax1.set_ylim(0, 1.05)

    # Secondary axis: loss increase percent
    ax2 = ax1.twinx()
    loss_vals = sub_summary['loss_increase_pct'].values
    ax2.plot(x, loss_vals, 's-', color='#D32F2F', markersize=10, linewidth=2,
             label='Loss increase (%)', zorder=5)
    for i, val in enumerate(loss_vals):
        ax2.annotate(f'{val:.1f}%', (x[i], val), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                     color='#D32F2F')
    ax2.set_ylabel('Loss Increase from Imbalance [%]', color='#D32F2F')
    ax2.tick_params(axis='y', labelcolor='#D32F2F')

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper left', fontsize=7, ncol=2)

    ax1.set_title('Customer Type Mix and Phase Imbalance by Substation')
    ax1.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / 'fig1_customer_mix_vs_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved fig1_customer_mix_vs_imbalance.png")


def fig2_meter_imbalance_by_type(meter_df, output_path):
    """
    Box plot: per-meter dominant_phase_fraction by customer type.
    Only types with >= 2 meters.
    """
    type_counts = meter_df['customer_type_en'].value_counts()
    valid_types = type_counts[type_counts >= 2].index.tolist()
    plot_df = meter_df[meter_df['customer_type_en'].isin(valid_types)].copy()

    # Order by median dominant_phase_fraction
    type_order = (plot_df.groupby('customer_type_en')['dominant_phase_fraction']
                  .median().sort_values(ascending=False).index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # Panel A: dominant phase fraction
    ax = axes[0]
    box_data = [plot_df[plot_df['customer_type_en'] == ct]['dominant_phase_fraction'].dropna().values
                for ct in type_order]
    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.6)
    for i, (patch, ct) in enumerate(zip(bp['boxes'], type_order)):
        patch.set_facecolor(get_color(ct))
        patch.set_alpha(0.7)
    ax.set_xticklabels(type_order, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Dominant Phase Fraction')
    ax.set_title('(a) Load Concentration on Dominant Phase')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Single-phase = 1.0')
    ax.axhline(y=1/3, color='green', linestyle='--', alpha=0.4, label='Perfectly balanced = 0.33')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel B: phase power std
    ax = axes[1]
    type_order_std = (plot_df.groupby('customer_type_en')['phase_power_std_W']
                      .median().sort_values(ascending=False).index.tolist())
    box_data_std = [plot_df[plot_df['customer_type_en'] == ct]['phase_power_std_W'].dropna().values
                    for ct in type_order_std]
    bp2 = ax.boxplot(box_data_std, vert=True, patch_artist=True, widths=0.6)
    for i, (patch, ct) in enumerate(zip(bp2['boxes'], type_order_std)):
        patch.set_facecolor(get_color(ct))
        patch.set_alpha(0.7)
    ax.set_xticklabels(type_order_std, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Phase Power Std. Dev. [W]')
    ax.set_title('(b) Per-Phase Power Variability')
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Per-Meter Phase Imbalance by Customer Type', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'fig2_meter_imbalance_by_type.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved fig2_meter_imbalance_by_type.png")


def fig3_customer_type_table(type_summary, output_path):
    """Render customer type summary as a thesis-quality table figure."""
    cols = ['customer_type_en', 'n_meters', 'mean_total_W', 'mean_phase_std_W',
            'mean_dom_frac', 'fraction_3phase', 'n_substations']
    headers = ['Customer Type', 'N', 'Mean Load\n[W]', 'Mean Phase\nStd [W]',
               'Dom. Phase\nFraction', '3-Phase\nFraction', 'N Subs']

    tbl = type_summary[cols].copy()
    # Format numbers
    tbl['mean_total_W'] = tbl['mean_total_W'].map('{:.0f}'.format)
    tbl['mean_phase_std_W'] = tbl['mean_phase_std_W'].map('{:.0f}'.format)
    tbl['mean_dom_frac'] = tbl['mean_dom_frac'].map('{:.2f}'.format)
    tbl['fraction_3phase'] = tbl['fraction_3phase'].map('{:.2f}'.format)
    tbl['n_substations'] = tbl['n_substations'].astype(int).astype(str)
    tbl['n_meters'] = tbl['n_meters'].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(14, 0.5 + 0.4 * len(tbl)), dpi=150)
    ax.axis('off')

    table = ax.table(
        cellText=tbl.values,
        colLabels=headers,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor('#1976D2')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(tbl) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#E3F2FD')

    ax.set_title('Customer Type Summary: Meter Count, Load, and Phase Imbalance Metrics',
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(output_path / 'fig3_customer_type_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved fig3_customer_type_summary_table.png")


def fig4_substation_heatmap(sub_summary, meter_df, output_path):
    """
    Heatmap: rows = substations (sorted by loss_increase_pct),
    columns = customer types, values = fraction of meters.
    """
    sub_summary = sub_summary.sort_values('loss_increase_pct', ascending=False).copy()
    subs = sub_summary['substation'].tolist()

    # Get all types with at least 2 meters total
    type_counts = meter_df['customer_type_en'].value_counts()
    types = type_counts[type_counts >= 2].index.tolist()

    # Build matrix
    matrix = np.zeros((len(subs), len(types)))
    for i, sub in enumerate(subs):
        sub_meters = meter_df[meter_df['substation'] == sub]
        n = len(sub_meters)
        for j, ct in enumerate(types):
            ct_count = len(sub_meters[sub_meters['customer_type_en'] == ct])
            matrix[i, j] = ct_count / n if n > 0 else 0

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(subs)):
        for j in range(len(types)):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(subs)))
    ylabels = []
    for sub in subs:
        row = sub_summary[sub_summary['substation'] == sub].iloc[0]
        ylabels.append(f"Sub {sub}\n({row['n_meters']}m, +{row['loss_increase_pct']:.1f}%)")
    ax.set_yticklabels(ylabels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Fraction of Meters')

    ax.set_title('Customer Type Distribution by Substation (sorted by imbalance loss increase)')
    fig.tight_layout()
    fig.savefig(output_path / 'fig4_substation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved fig4_substation_heatmap.png")


def fig5_household_vs_losses(sub_summary, output_path):
    """
    Scatter: household fraction vs loss increase percent.
    One point per substation, annotated.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    x = sub_summary['household_fraction'].values
    y = sub_summary['loss_increase_pct'].values
    subs = sub_summary['substation'].values
    sizes = sub_summary['n_meters'].values

    # Scale marker size by meter count
    min_size, max_size = 80, 500
    if sizes.max() > sizes.min():
        norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
        marker_sizes = min_size + norm_sizes * (max_size - min_size)
    else:
        marker_sizes = np.full_like(sizes, 200, dtype=float)

    scatter = ax.scatter(x, y, s=marker_sizes, c='#1976D2', alpha=0.7, edgecolors='black', linewidth=1)

    # Annotate each point
    for i, sub in enumerate(subs):
        n = int(sizes[i])
        ax.annotate(f'Sub {sub}\n({n} meters)',
                    (x[i], y[i]),
                    textcoords="offset points", xytext=(12, 8),
                    fontsize=8, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Trend line (if enough points)
    if len(x) >= 3:
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 3:
            z = np.polyfit(x[valid], y[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x[valid].min() - 0.05, x[valid].max() + 0.05, 100)
            ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.6, linewidth=1)

            # Spearman correlation
            from scipy import stats
            try:
                rho, pval = stats.spearmanr(x[valid], y[valid])
                ax.text(0.05, 0.95, f'Spearman r = {rho:.2f} (p = {pval:.3f})',
                        transform=ax.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
            except Exception:
                pass

    ax.set_xlabel('Household Fraction (share of meters)', fontsize=11)
    ax.set_ylabel('Loss Increase from Imbalance [%]', fontsize=11)
    ax.set_title('Relationship: Household Share vs. Imbalance-Driven Losses')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path / 'fig5_household_vs_losses.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved fig5_household_vs_losses.png")


# ===================================================================
#  Section 5: Main
# ===================================================================

def main():
    logger.info("=" * 80)
    logger.info("Customer Type vs. Phase Imbalance Analysis")
    logger.info("=" * 80)

    # Filter to substations with both SM data and PF results
    available = []
    for sub in SELECTED_SUBSTATIONS:
        sm_exists = (SMARTMETER_DIR / SMARTMETER_FILES.get(sub, '')).exists()
        pf_exists = (PF_RESULTS_DIR / sub / 'unbalanced' / 'VUF.parquet').exists()
        if sm_exists and pf_exists:
            available.append(sub)
        else:
            logger.warning(f"  Skipping substation {sub}: SM={sm_exists}, PF={pf_exists}")

    if not available:
        logger.error("No substations available with both SM data and PF results.")
        return

    logger.info(f"Substations to analyse: {available}")

    # --- Load data ---
    logger.info("\nLoading data...")
    sm_data, vuf_data, loss_summary = load_all_data(available)

    # --- Compute per-meter imbalance ---
    logger.info("\nComputing per-meter imbalance metrics...")
    meter_df = compute_per_meter_imbalance(sm_data)
    logger.info(f"  Total meters: {len(meter_df)}")
    logger.info(f"  Customer types: {meter_df['customer_type_en'].nunique()}")
    logger.info(f"  3-phase meters: {meter_df['is_3phase'].sum()}")
    logger.info(f"  Single-phase meters: {(~meter_df['is_3phase']).sum()}")

    # --- Build summaries ---
    logger.info("\nBuilding summaries...")
    sub_summary = build_substation_summary(meter_df, vuf_data, loss_summary)
    type_summary = build_customer_type_summary(meter_df)

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meter_df.to_csv(OUTPUT_DIR / 'meter_imbalance.csv', index=False)
    sub_summary.to_csv(OUTPUT_DIR / 'substation_summary.csv', index=False)
    type_summary.to_csv(OUTPUT_DIR / 'customer_type_summary.csv', index=False)
    logger.info(f"\nSaved CSVs to {OUTPUT_DIR}")

    # --- Print summary tables ---
    logger.info(f"\n{'='*80}")
    logger.info("SUBSTATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Sub':>6} {'Meters':>7} {'Types':>6} {'HH%':>6} {'3ph%':>6} "
                f"{'VUF_mean':>9} {'VUF_max':>9} {'Loss+%':>8} {'Add kWh':>10}")
    logger.info(f"  {'─'*75}")
    for _, r in sub_summary.iterrows():
        logger.info(
            f"  {str(r['substation']):>6} {int(r['n_meters']):>7} "
            f"{int(r['n_customer_types']):>6} {r['household_fraction']:>5.0%} "
            f"{r['fraction_3phase']:>5.0%} "
            f"{r['mean_VUF_pct']:>8.3f}% {r['max_VUF_pct']:>8.3f}% "
            f"{r['loss_increase_pct']:>7.1f}% {r['additional_loss_kwh']:>9.1f}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("CUSTOMER TYPE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Type':<30} {'N':>4} {'Mean W':>8} {'Std W':>8} {'DomFr':>6} {'3ph%':>5}")
    logger.info(f"  {'─'*65}")
    for _, r in type_summary.iterrows():
        logger.info(
            f"  {r['customer_type_en']:<30} {int(r['n_meters']):>4} "
            f"{r['mean_total_W']:>8.0f} {r['mean_phase_std_W']:>8.0f} "
            f"{r['mean_dom_frac']:>5.2f} {r['fraction_3phase']:>5.0%}"
        )

    # --- Generate visualizations ---
    logger.info("\nGenerating visualizations...")
    fig1_customer_mix_vs_imbalance(sub_summary, meter_df, OUTPUT_DIR)
    fig2_meter_imbalance_by_type(meter_df, OUTPUT_DIR)
    fig3_customer_type_table(type_summary, OUTPUT_DIR)
    fig4_substation_heatmap(sub_summary, meter_df, OUTPUT_DIR)
    fig5_household_vs_losses(sub_summary, OUTPUT_DIR)

    # --- Key findings ---
    logger.info(f"\n{'='*80}")
    logger.info("KEY OBSERVATIONS")
    logger.info(f"{'='*80}")

    # Most imbalanced substation
    most_imbal = sub_summary.loc[sub_summary['loss_increase_pct'].idxmax()]
    least_imbal = sub_summary.loc[sub_summary['loss_increase_pct'].idxmin()]
    logger.info(f"  Most imbalanced:  Sub {most_imbal['substation']} "
                f"(+{most_imbal['loss_increase_pct']:.1f}%, "
                f"{most_imbal['household_fraction']:.0%} household, "
                f"{int(most_imbal['n_meters'])} meters)")
    logger.info(f"  Least imbalanced: Sub {least_imbal['substation']} "
                f"(+{least_imbal['loss_increase_pct']:.1f}%, "
                f"{least_imbal['household_fraction']:.0%} household, "
                f"{int(least_imbal['n_meters'])} meters)")

    # Dominant type analysis
    hh_type = type_summary[type_summary['customer_type_en'] == 'General household']
    if not hh_type.empty:
        hh = hh_type.iloc[0]
        logger.info(f"  General household: {int(hh['n_meters'])} meters, "
                    f"dom_frac={hh['mean_dom_frac']:.2f}, "
                    f"mean_std={hh['mean_phase_std_W']:.0f} W")

    # Compare 3-phase vs single-phase
    single = meter_df[~meter_df['is_3phase']]
    three = meter_df[meter_df['is_3phase']]
    if len(three) > 0 and len(single) > 0:
        logger.info(f"  Single-phase meters: dom_frac={single['dominant_phase_fraction'].mean():.2f}, "
                    f"phase_std={single['phase_power_std_W'].mean():.0f} W (n={len(single)})")
        logger.info(f"  Three-phase meters:  dom_frac={three['dominant_phase_fraction'].mean():.2f}, "
                    f"phase_std={three['phase_power_std_W'].mean():.0f} W (n={len(three)})")

    logger.info(f"\n{'='*80}")
    logger.info("LIMITATIONS")
    logger.info(f"{'='*80}")
    logger.info("  - Only 6 substations (449 meters): too few for formal statistical testing")
    logger.info("  - 'General household' dominates (88%): rare types have N=1-2")
    logger.info("  - Phase assignment is modelled, not measured")
    logger.info("  - Single month of data (Aug-Sep 2025)")
    logger.info("  - Per-meter imbalance reflects load profile, not network-level impact")

    logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
