# Jason Data Translation: Documentation

## Overview

This document describes the data translation from Icelandic ArcGIS topology
exports and Databricks smart meter data into the 3PhaseInsight second_batch
format. It covers all placeholder values, assumptions, and known limitations.

## Source Data

| File | Records | Description |
|------|---------|-------------|
| `cabinets.csv` | 4 | ArcGIS junction cabinet exports (Tengiskápur) |
| `lines_clean.csv` | 57 | ArcGIS LV line exports (cables) |
| `transformers.csv` | 1 | Transformer data (ABB, 800 kVA, 11/0.4 kV) |
| `smartmeter_*.parquet` | ~537K | 15-min smart meter data, Aug-Oct 2025 |

### Network: Substation 1416 (Jöfursbás), Reykjavik

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
| 4 x 10 Cu | 10 | Copper | 1.830 | 0.094 | 73 |
| 4 x 16 Cu | 16 | Copper | 1.150 | 0.087 | 98 |
| 4 x 50 Al | 50 | Aluminum | 0.641 | 0.078 | 137 |
| 4 x 150 Al | 150 | Aluminum | 0.207 | 0.073 | 258 |
| 4 x 240 Al | 240 | Aluminum | 0.125 | 0.070 | 340 |
| 4 x 300 Al | 300 | Aluminum | 0.100 | 0.069 | 385 |

**Source**: IEC 60228:2004, Table 2 (DC resistance at 20°C). AC resistance
at 50 Hz is approximately equal for these sizes. Reactance values are
typical for single-core XLPE underground cables at 0.6/1 kV.

**Note**: The existing `test/translate_to_second_batch.py` used a flat
value of 0.0018 Ohm/m (= 1.8 Ohm/km) for ALL cables, which is only
appropriate for 10 mm² Cu. This adapter uses cable-specific values.

**Important**: Values are stored as **Ohm/km** (not total Ohms), matching
the existing second_batch format and pandapower's `r_ohm_per_km` parameter.

### 2. Per-Phase Reactive Power

The smart meter data provides only `Q_total` (aggregate reactive power
across all three phases). The pipeline expects per-phase reactive power
(`reactive_power_q12_l1/l2/l3`).

**Method**: Proportional split based on active power ratio:

```
Q_phase_i = Q_total * |P_phase_i| / (|P_a| + |P_b| + |P_c|)
```

When total active power is zero, Q is set to zero for all phases.

**Justification**: Assuming load power factor is similar across phases,
the reactive power should be roughly proportional to the active power
on each phase. This is a common approximation when only aggregate Q
is available.

**Limitation**: This may not accurately represent asymmetric loads where
one phase has a significantly different power factor.

### 3. Export Power and Capacitive Reactive Power

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `active_power_p23_l1/l2/l3` | 0.0 W | No generation/export data available |
| `reactive_power_q34_l1/l2/l3` | 0.0 var | Assumed purely inductive loads |

### 4. Network Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `lv_feeder_fuse_size` | 250 A | Typical main feeder fuse for 800 kVA transformer |
| `service_fuse_size` | 25 A | Typical Icelandic residential service fuse |
| `lv_feeder` | LvFeeder.1416 | Must match jason_config.yaml `IDs: ["1416"]` |
| `zip_code_secondary_substation` | 112 | From cabinets.csv PNR field (Reykjavik postal code) |

### 5. Meter-Cabinet Connection Defaults

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `delivery_point_id` | (empty) | Not available in ArcGIS data |
| `has_heat_pump` | false | Information not available |
| `has_solar_panel` | false | Information not available |
| `capacity_solar_panel` | (empty) | N/A |

### 6. Transformer Parameters (Power Flow)

Standard values used for the pandapower transformer model:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sn_mva` | 0.8 | Rated power (800 kVA / 1000) |
| `vn_hv_kv` | 11 | HV side: 11 kV (from transformers.csv FORSPENNA=11000) |
| `vn_lv_kv` | 0.4 | LV side: 400 V (from transformers.csv EFTIRSPENNA=400) |
| `vkr_percent` | 0.65 | Copper loss impedance (standard) |
| `vk_percent` | 7.0 | Short-circuit impedance (standard) |
| `pfe_kw` | 0.45 | Core losses at rated voltage (standard) |
| `i0_percent` | 0.2 | No-load current (standard) |

**Note**: `vk_percent` = 5.96% is available from the transformer nameplate
(`SKAMMHLAUPSPENNA` field), but the standard value of 7% is used for
consistency with the existing power flow model.

---

## Data Transformations

### Column Mapping: Icelandic -> second_batch

#### Topology (lines_clean.csv -> lv_topology.csv)

| ArcGIS Column | second_batch Column | Transformation |
|---------------|---------------------|----------------|
| DNR | secondary_substation | `SecondarySubstation.{DNR}` |
| (PNR from cabinets) | zip_code_secondary_substation | Direct value |
| DNR | transformer | `Transformer.{DNR}` |
| TYPE (from transformers) | transformer_capacity | Extract numeric: 800 |
| (derived) | lv_feeder | `LvFeeder.1416` |
| (placeholder) | lv_feeder_fuse_size | 250 A |
| FROM | node1 | `Cabinet.{FROM}` or `LvFeeder.1416` |
| TO | node2 | `Cabinet.{TO}` |
| OBJECTID | cable_id | `LvCable.{OBJECTID}` |
| GERD_DECODED (material) | cable_type | Al->MAL, Cu->PEX |
| SHAPE_LENGD | cable_length | Direct (meters) |
| GERD_DECODED (size) | phase_size | Extract numeric (mm²) |
| GERD_DECODED (material) | phase_material | AL or Cu |
| (lookup table) | cable_capacity | From IEC table (A) |
| (lookup table) | resistance | From IEC table (Ohm/km) |
| (lookup table) | reactance | From IEC table (Ohm/km) |

#### Smart Meter Data (parquet -> phase_measurements_YYYY_M.csv)

| Parquet Column | second_batch Column | Transformation |
|----------------|---------------------|----------------|
| husveita_fastanumer | meter_number | String (200 unique meter IDs) |
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
| Heimtaug | Home connection | Yes |
| Lágspennudreifilögn | LV distribution line | Yes |
| Lágspennustrengur | LV conductor | Yes |
| Götuljósalögn | Street lighting | **No** (separate circuit) |

### Node Identification

The `FROM`/`TO` columns in lines_clean.csv use these conventions:
- `D1416` = Distribution point at transformer (maps to `LvFeeder.1416`)
- `31832`, `31692`, `31693`, `33894` = Junction cabinet TENGINR values
- `194558`, `196437`, etc. = Individual meter/home connection endpoints
- `1416` = Transformer/substation node

---

## Key Data Relationships

### Meter vs Line vs Cabinet

The parquet smart meter data contains three important ID columns:

| Column | Meaning | Unique Count | Example |
|--------|---------|-------------|---------|
| `husveita_fastanumer` | **Smart meter ID** | 200 | "205724" |
| `numer_heimlagnar` | **Line/connection ID** (Heimtaug endpoint) | 15 | "194558" |
| `tengiskapur` | **Junction cabinet ID** | 5 | "31832", "D1416" |

**Hierarchy**: Multiple meters share the same line (apartment complexes, up to 30 per line), and multiple lines connect to the same cabinet.

```
Cabinet 31832 (tengiskapur)
  ├─ Line 196866 (numer_heimlagnar) ─ 27 meters (husveita_fastanumer)
  ├─ Line 196868 (numer_heimlagnar) ─ 26 meters
  └─ Line 194981 (numer_heimlagnar) ─ 25 meters
```

**In the topology**: Each `numer_heimlagnar` maps to the TO endpoint of a Heimtaug line in `lines_clean.csv`. The meter's electrical connection point in the topology is `Cabinet.{numer_heimlagnar}`.

---

## Known Limitations

1. **No per-phase reactive power**: Only aggregate Q_total is available.
   The proportional split is an approximation.

2. **No export/generation data**: P23 and Qprod are set to zero.
   If there are solar panels or batteries, they are not captured.

3. **Street lighting excluded**: The Götuljósalögn network is excluded.
   These are on separate circuits with no BUN_FRA/BUN_TIL connectivity.

4. **Cable R/X from standard tables**: Actual installed cable parameters
   may differ due to aging, temperature, or non-standard installations.

5. **No THD data**: The second_batch format supports harmonic distortion
   columns (thdu_l1/l2/l3, thdi_l1/l2/l3) which are not available.

6. **Single feeder assumption**: All lines are assigned to LvFeeder.1416.
   The actual network may have multiple feeders from the transformer.

7. **Heat pump / solar panel flags**: Set to false by default as this
   information is not in the ArcGIS data.

---

## File Inventory

| File | Purpose |
|------|---------|
| `Jason_data_adapter.py` | Translate ArcGIS -> second_batch, append to source data |
| `Jason_visualize_topology.py` | Generate network topology map (PNG) |
| `Jason_run_pipeline.py` | Run DataPreprocessor pipeline via jason_config.yaml |
| `Jason_prepare_powerflow.py` | Convert pipeline output -> power flow parquets |
| `Jason_run_powerflow.py` | Execute pandapower analysis |
| `Jason_run.py` | Master orchestration script |
| `Jason_documentation.md` | This file |
