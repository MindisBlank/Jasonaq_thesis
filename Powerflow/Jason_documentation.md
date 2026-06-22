# Jason Data Translation: Documentation

## Overview

This document describes the data translation from Icelandic ArcGIS topology
exports and Databricks smart meter data into the power flow input format.
It covers all placeholder values, assumptions, and known limitations.

The pipeline supports multiple substations configured via `config/jason_config.yaml`.
The example below uses substation 1416 as a reference, but all values are
derived dynamically from the source data and config.

## Source Data

| File | Description |
|------|-------------|
| `cabinets.csv` | ArcGIS junction cabinet exports (Tengiskápur) |
| `lines_clean.csv` | ArcGIS LV line exports (cables) |
| `transformers.csv` | Transformer data |
| `smartmeter_*.parquet` | 15-min smart meter data from Databricks |

### Example: Substation 1416 (Jöfursbás), Reykjavik

- 1 transformer: 800 kVA, 11/0.4 kV, ABB (2019)
- 4 junction cabinets (TENGINR: 31832, 31692, 31693, 33894)
- 200 smart meters (identified by `husveita_fastanumer`)
- 15 distribution lines / home connections (identified by `numer_heimlagnar`)
- 57 cable segments (26 street lighting excluded, 31 LV lines used)

---

## Placeholder Values

### 1. Cable Resistance and Reactance (Ohm/km)

The source ArcGIS data does not include R/X values. Values are assigned
based on the cable specification from `GERD_DECODED` using IEC 60228
standard conductor resistance at 20°C and typical underground XLPE
cable reactance at 50 Hz.

| GERD_DECODED | Size (mm²) | Material | R (Ohm/km) | X (Ohm/km) | Capacity (A) |
|-------------|-----------|----------|-----------|-----------|-------------|
| 4 x 2.5 Cu | 2.5 | Copper | 7.410 | 0.110 | 30 |
| 4 x 10 Cu | 10 | Copper | 1.830 | 0.094 | 73 |
| 4 x 16 Cu | 16 | Copper | 1.150 | 0.087 | 98 |
| 4 x 25 Cu | 25 | Copper | 0.727 | 0.083 | 129 |
| 4 x 35 Cu | 35 | Copper | 0.524 | 0.081 | 152 |
| 4 x 50 Cu | 50 | Copper | 0.387 | 0.079 | 179 |
| 4 x 300 Cu | 300 | Copper | 0.060 | 0.069 | 530 |
| 4 x 25 Al | 25 | Aluminum | 1.200 | 0.088 | 96 |
| 4 x 50 Al | 50 | Aluminum | 0.641 | 0.078 | 137 |
| 4 x 70 Al | 70 | Aluminum | 0.443 | 0.075 | 169 |
| 4 x 95 Al | 95 | Aluminum | 0.320 | 0.072 | 201 |
| 4 x 120 Al | 120 | Aluminum | 0.253 | 0.072 | 234 |
| 4 x 150 Al | 150 | Aluminum | 0.207 | 0.073 | 258 |
| 4 x 185 Al | 185 | Aluminum | 0.164 | 0.072 | 294 |
| 4 x 240 Al | 240 | Aluminum | 0.125 | 0.070 | 340 |
| 4 x 300 Al | 300 | Aluminum | 0.100 | 0.069 | 385 |

**Source**: IEC 60228:2004, Table 2 (DC resistance at 20°C). AC resistance
at 50 Hz is approximately equal for these sizes. Reactance values are
typical for single-core XLPE underground cables at 0.6/1 kV.

**Important**: Values are stored as **Ohm/km** (not total Ohms), matching
pandapower's `r_ohm_per_km` parameter.

### 2. Per-Phase Reactive Power

The smart meter data provides `Q_total` (aggregate reactive power) and
per-phase columns `Q_a`, `Q_b`, `Q_c`. The power flow needs per-phase
reactive power assigned to the correct network phase.

Reactive power is handled in `Jason_prepare_powerflow.py`, which reads
directly from the source parquet (not the adapter's intermediate CSVs):

- **3-phase meters**: `Q_a`, `Q_b`, `Q_c` are used directly from the parquet.
- **Single-phase meters**: All of `Q_total` is placed on the meter's
  assigned network phase (A, B, or C). The other two phases are set to zero.

**Justification**: A single-phase meter's reactive power is physically
on one phase only, so assigning the full `Q_total` to the designated
phase is correct. For 3-phase meters, the per-phase Q columns are
already available in the source data.

### 3. Export Power and Capacitive Reactive Power

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `active_power_p23_l1/l2/l3` | 0.0 W | No generation/export data available |
| `reactive_power_q34_l1/l2/l3` | 0.0 var | Assumed purely inductive loads |

### 4. Network Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `lv_feeder_fuse_size` | Dynamic | Computed from transformer kVA (e.g., 800 kVA → 400 A) |
| `service_fuse_size` | 25 A | Default Icelandic residential service fuse (configurable) |
| `lv_feeder` | `LvFeeder.{substation_id}` | Dynamic from `jason_config.yaml` |
| `zip_code_secondary_substation` | From PNR | Read from `cabinets.csv` PNR field |

**Feeder fuse size lookup:**

| Transformer kVA | Fuse (A) |
|-----------------|----------|
| ≤ 100 | 160 |
| ≤ 200 | 200 |
| ≤ 315 | 250 |
| ≤ 500 | 315 |
| ≤ 800 | 400 |
| ≤ 1000 | 630 |
| > 1000 | 800 |

### 5. Meter-Cabinet Connection Defaults

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `delivery_point_id` | (empty) | Not available in ArcGIS data |
| `has_heat_pump` | false | Information not available |
| `has_solar_panel` | false | Information not available |
| `capacity_solar_panel` | (empty) | N/A |

### 6. Transformer Parameters (Power Flow)

Values used for the pandapower transformer model:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sn_mva` | Dynamic | Rated power from `MALRAUN` (e.g., 800 kVA → 0.8 MVA) |
| `vn_hv_kv` | 11 | HV side from `FORSPENNA` (11000 V) |
| `vn_lv_kv` | 0.4 | LV side from `EFTIRSPENNA` (400 V) |
| `vkr_percent` | 0.7 | Copper loss impedance |
| `vk_percent` | 6.0 | Short-circuit impedance |
| `pfe_kw` | 0.0 | Core losses (set to zero) |
| `i0_percent` | 0.2 | No-load current |
| `vector_group` | Dynamic | From `TENGIFLOKKUR` field (e.g., "Dyn5") |

**Note**: `SKAMMHLAUPSPENNA` in the transformer CSV contains the nameplate
short-circuit voltage, which could be used in place of the hardcoded 6%.

---

## Data Transformations

### Column Mapping: Icelandic -> Power Flow Format

#### Topology (lines_clean.csv -> lv_topology.csv)

| ArcGIS Column | Output Column | Transformation |
|---------------|---------------------|----------------|
| DNR | secondary_substation | `SecondarySubstation.{substation_id}` |
| (PNR from cabinets) | zip_code_secondary_substation | Direct value |
| DNR | transformer | `Transformer.{substation_id}` |
| MALRAUN (from transformers) | transformer_capacity | Direct (kVA) |
| TENGIFLOKKUR (from transformers) | vector_group | Direct string |
| (from config) | lv_feeder | `LvFeeder.{substation_id}` |
| (computed) | lv_feeder_fuse_size | From transformer kVA lookup |
| FROM | node1 | `Cabinet.{FROM}` or `LvFeeder.{id}` if D-prefix |
| TO | node2 | `Cabinet.{TO}` |
| OBJECTID | cable_id | `LvCable.{OBJECTID}` |
| GERD_DECODED (material) | cable_type | PEX (default fallback: MAL) |
| SHAPE_LENGD | cable_length | Direct (meters), fallback to LENGD |
| GERD_DECODED (size) | phase_size | Extract numeric (mm²) |
| GERD_DECODED (material) | phase_material | AL or CU |
| (lookup table) | cable_capacity | From IEC table (A) |
| (lookup table) | resistance | From IEC table (Ohm/km) |
| (lookup table) | reactance | From IEC table (Ohm/km) |

#### Smart Meter Data (parquet -> phase_measurements_YYYY_M.csv)

| Parquet Column | Output Column | Transformation |
|----------------|---------------------|----------------|
| husveita_fastanumer | meter_number | String (unique meter IDs) |
| ts | timestamp_dst | Parse as datetime |
| V_a | voltage_l1 | Direct (Volts) |
| V_b | voltage_l2 | Direct (Volts) |
| V_c | voltage_l3 | Direct (Volts) |
| P_a | active_power_p14_l1 | Direct (Watts) |
| P_b | active_power_p14_l2 | Direct (Watts) |
| P_c | active_power_p14_l3 | Direct (Watts) |
| (none) | active_power_p23_l1/l2/l3 | 0.0 (no export data) |
| Q_total | reactive_power_q12_l1/l2/l3 | Proportional split |
| (none) | reactive_power_q34_l1/l2/l3 | 0.0 (assumed inductive) |
| I_a | current_l1 | Direct (Amps), if available |
| I_b | current_l2 | Direct (Amps), if available |
| I_c | current_l3 | Direct (Amps), if available |

### Line Filtering

Lines are filtered by `HLUTVERK` (role):

| Icelandic Type | English | Included? |
|----------------|---------|-----------|
| Heimtaug | Home connection / service cable | Yes |
| Lágspennudreifilögn | LV distribution line | Yes |
| Lágspennustrengur | LV conductor | Yes |
| Lágspennu- og götuljósalögn | Combined LV + street lighting trunk | Yes |
| Götuljósalögn | Street lighting (pure) | **No** (separate circuit) |

**Note**: "Lágspennu- og götuljósalögn" are combined trunk cables that carry
400V power and are essential for network connectivity. They are distinct from
pure "Götuljósalögn" which are dedicated street lighting circuits.

### Node Identification

The `FROM`/`TO` columns in lines_clean.csv use these conventions:
- `D{DNR}` (e.g., `D1416`) = Distribution point at transformer (maps to `LvFeeder.{id}`)
- Cabinet TENGINR values (e.g., `31832`) = Junction cabinet nodes
- Line NR values (e.g., `194558`) = Individual meter/home connection endpoints
- `0` = Placeholder node (skipped in topology translation)

---

## Key Data Relationships

### Meter vs Line vs Cabinet

The parquet smart meter data contains three important ID columns:

| Column | Meaning | Example |
|--------|---------|---------|
| `husveita_fastanumer` | **Smart meter ID** | "205724" |
| `numer_heimlagnar` | **Line/connection ID** (Heimtaug endpoint) | "194558" |
| `tengiskapur` | **Junction cabinet ID** | "31832", "D1416" |

**Hierarchy**: Multiple meters share the same line (apartment complexes, up to 30 per line), and multiple lines connect to the same cabinet.

```
Cabinet 31832 (tengiskapur)
  ├─ Line 196866 (numer_heimlagnar) ─ 27 meters (husveita_fastanumer)
  ├─ Line 196868 (numer_heimlagnar) ─ 26 meters
  └─ Line 194981 (numer_heimlagnar) ─ 25 meters
```

**In the topology**: Each `numer_heimlagnar` maps to the TO endpoint of a Heimtaug line in `lines_clean.csv`. The meter's electrical connection point in the topology is `Cabinet.{numer_heimlagnar}`.

## File Inventory

### Pipeline Scripts

| File | Purpose |
|------|---------|
| `run_complete.py` | Master orchestration — runs all pipeline steps per substation |
| `Jason_config.py` | Shared config loader, reads `config/jason_config.yaml` |
| `Jason_data_adapter.py` | Translate ArcGIS topology + smart meter data into power flow input format |
| `Jason_prepare_powerflow.py` | Build per-phase load matrices, topology parquets, and meter-cabinet mappings |
| `Jason_run_powerflow.py` | Execute pandapower 3-phase power flow analysis (balanced + unbalanced) |
| `Jason_compare_losses.py` | Compare unbalanced vs balanced losses across substations |
| `Jason_visualize_topology.py` | Generate network topology maps (geographic, tree, spring layouts) |
| `Jason_customer_imbalance.py` | Analyze per-customer phase imbalance |
| `Jason_line_utilization.py` | Analyze cable/line utilization |

### Configuration

| File | Purpose |
|------|---------|
| `config/jason_config.yaml` | Substation IDs, phase ratios, path overrides, pipeline parameters |

### Documentation

| File | Purpose |
|------|---------|
| `Jason_documentation.md` | This file |
| `Data_info.md` | Column-by-column description of all source data files |
