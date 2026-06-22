### Overview

The `Lines_clean`, `transformers`, and `cabinets` CSV files contain ArcGIS topology data. For power flow analysis, the transformer acts as the feeder. Note that "Götuljósalögn" (street lighting) in `Lines_clean` is currently ignored.

While this is ArcGIS topology data, it currently lacks explicit resistance and reactance values. These must be inferred from the `GERD_DECODED` column. The column `NR` can be used to map smart meters to the topology.

---

### 1. Dataset: Lines_clean (Line Topology)

- **OBJECTID:** Unique ID for the object within this CSV file.
- **HLUTVERK:** Describes the line's function:
  - "Heimtaug": Service cable connecting to end-users.
  - "Götuljósalögn": Street lighting.
  - "Lágspennudreifilögn": Low voltage distribution line (often larger cables connecting cabinets to each other or to transformers).
- **TEGUND:** Voltage type. In this dataset, it is always "lágspennustrengur" (English: Low Voltage Line).
- **DNR:** The ID number of the substation this line belongs to.
- **STRBUTUR:** Unparsed connection string indicating "From" and "To" points (e.g., "33894-194985A01" means from node 33894 to node 194985, port A01).
- **NR:** Identifier used to map the line to the `numer_heimlagnar` column in the smart meter data.
- **Spenna:** Voltage level.
- **LAGNINGARAR:** Year of installation.
- **DSPENNIR:** The transformer the cable belongs to within the substation. As there is a maximum of two transformers per substation, this value is either "1" or "2".
- **HLENGD:** Total length in meters.
- **BUN_TIL:** (Icelandic: "Component To") The destination node derived from parsing `STRBUTUR`.
- **BUN_FRA:** (Icelandic: "Component From") The origin node derived from parsing `STRBUTUR`.
- **GLOBALID:** Unique global identifier for the entire dataset (ensures no component ID overlap across different datasets).
- **AFANGASTADUR:** Address of the end destination.
- **GERD_DECODED:** Cable type code (e.g., "4x16 cu" means 4 copper wires, 16mm² cross-section).
- **FROM:** Same as `BUN_TIL` (English labeling).
- **TO:** Same as `BUN_FRA` (English labeling).

---

### 2. Dataset: Transformers

- **OBJECTID:** Unique ID for the object within this CSV file.
- **DNR:** The ID number of the substation containing the transformer.
- **NUMER:** Transformer number within the substation (either "1" or "2").
- **FRAMLEIDSLUAR:** Year of construction/manufacture.
- **GERD:** Transformer type (e.g., oil-based).
- **NETTOTHYNGD:** Net weight.
- **MALRAUN:** Rated capacity in kVA.
- **FJOLDI_FASA:** Number of phases (live wires).
- **FORSPENNA:** Primary side voltage in Volts (always 11,000 V in this dataset).
- **EFTIRSPENNA:** Secondary side voltage in Volts (always 400 V in this dataset).
- **SKAMMHLAUPSPENNA:** Short-circuit voltage.
- **GLOBALID:** Unique global identifier for the entire dataset.

---

### 3. Dataset: Cabinets

- **OBJECTID:** Unique ID for the object within this CSV file.
- **HLUTVERK:** Function of the object. In this case, it is always "Tengiskápur" (English: Junction Box).
- **GERD:** Object type (describes the specific type of junction box).
- **DNR:** The substation system ID this cabinet belongs to.
- **TENGINR:** Connection number.
- **LAGNINGARAR:** Year of installation.
- **DSPENNIR:** The transformer the cabinet belongs to (either "1" or "2").
- **DSKAPUR:** Location code for the switch within the substation. This is primarily used by on-site workers to identify which fuse box contains the breaker connected to this cabinet.
- **STADUR:** Address/Location.
- **GLOBALID:** Unique global identifier for the entire dataset.
- **SVF_DECODED:** City or municipality location.

---

### 4. Dataset: Smart Meter Data

**Description:**

This data is saved in a Parquet file and contains all smart meter measurements for a given timeframe, connected to a specific feeder (substation/transformer pair). The dataset is large, containing thousands of rows due to the sampling rate (every 15 minutes) and the number of meters.

**Important Notes:**

- There are currently no reactive power measurements; these values are missing and need to be filled.
- There are no per-phase power measurements; these will also need to be inferred/filled.
- The Icelandic power grid currently has **no** PV cells. Therefore, no users are contributing to the grid (no power injection), only consuming.

**Key Column Names:**

- **substation_id:** The ID of the substation system the meter belongs to (e.g., "1416").
- **transformer:** String indicating the connected transformer ("sp1" or "sp2").
- **husveita_fastanumer:** Unique meter ID for the smart meter dataset.
- **kennitalamaelistadar:** Another unique ID (similar to `husveita_fastanumer`). It is recommended to use `husveita_fastanumer` as the primary identifier.
- **numer_heimlagnar:** The line ID the meter connects to. Use this to link meters to the `NR` column in `Lines_clean`. WARNING: this is double datatype while "NR" is int.
- **ts:** Timestamp (Format: "YYYY-MM-DD HH:MM:SS").
- **I_a / I_b / I_c:** RMS current value for Phase A, B, and C (Unit: A).
- **I_total:** Total current, calculated by summing I_a, I_b, and I_c. Note: Since this is RMS, there are limitations to this summation method (Unit: A).
- **P_a / P_b / P_c:** Power consumed in Phase A, B, and C (Unit: W).
- **P_total:** Total active power, found by summing all phases (Unit: W).
- **PF_used:** Power factor that is simply an average over the entire feeder that was used to compute the reactive power.
- **PF_sm:** Power factor of the smart meter during the timestamp. This is only here for debugging.
- **Q_a / Q_b / Q_c:** Computed reactive power in Phase A, B, and C (Unit: var).
- **Q_total:** Total reactive power, found by summing all phases (Unit: var).
- **V_a / V_b / V_c:** Voltage on Phase A, B, and C (Unit: V).
- **V_ph_avg:** Average voltage across all three phases (Unit: V).
- **S_total:** Total apparent power (Unit: VA).
- **dspennir:** The transformer number the meter belongs to ("1" or "2").
- **latitude / longitude:** GPS coordinates.
- **tengiskapur:** The ID number of the cabinet the meter belongs to.
- **notkunarstadur:** Address of the meter.
- **lysingnotkunarflokks:** Short description of the load category (e.g., general household).
- **submeter_pairing_group:** (Definition currently unknown).
- **per_completed:** A percentage representing the progress of the smart meter rollout in the system. (might be wrong, don't trust for now)
- **n_total_mps:** Represents the total count of unique smart meters over the timeframe. It is recommended to count unique `husveita_fastanumer` entries to get an exact total instead.
