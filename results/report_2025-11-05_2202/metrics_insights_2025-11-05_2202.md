# Janitza imbalance insights (2025-11-05_2202)

Source file: `C:\Users\Notandi\Desktop\Master\Jasonaq_thesis\results\metrics_results_2025-11-04_2316.csv`

## Dataset pulse check
- Samples analysed: 194
- Unique devices: 194
- Monitoring horizon: 2025-09-01 12:00 → 2025-10-01 12:00

## Phase balance snapshots
- Average phase current spread (max-min): 213.59 A
- Worst current imbalance windows:
|   device_id | name                                                  |   Ia_avg |   Ib_avg |   Ic_avg |
|------------:|:------------------------------------------------------|---------:|---------:|---------:|
|         262 | D0006 SP2 L1 / UMG801-4700-2328 / Measurement Group 2 | 25358.67 | 36852.12 |     0.00 |
|         361 | D1398 SP1 L1                                          |   233.16 |   229.98 |    12.50 |
|         156 | D1159 SP1 L1                                          |   542.95 |   393.99 |   393.17 |
|         377 | D0579 SP1 L1                                          |   242.66 |   223.94 |    93.59 |
|         155 | D1160 SP1 L1                                          |   416.81 |   315.13 |   280.61 |
- Average phase voltage spread (max-min): 3.45 V
- Worst voltage imbalance windows:
|   device_id | name                                                  |   Va_avg |   Vb_avg |   Vc_avg |
|------------:|:------------------------------------------------------|---------:|---------:|---------:|
|         377 | D0579 SP1 L1                                          |   235.53 |   235.98 |   408.46 |
|         375 | D1416 SP1 L1                                          |   234.59 |   235.05 |   407.08 |
|         361 | D1398 SP1 L1                                          |   235.18 |   235.59 |   407.66 |
|         456 | D1029 SP1 L1 / UMG801-4700-2804 / Measurement Group 1 |   235.32 |   234.06 |   236.84 |
|         457 | D1029 SP1 L1 / UMG801-4700-2804 / Measurement Group 2 |   235.32 |   234.06 |   236.84 |

## Neutral channel watchlist
- Neutral current median: 12.75 A (p95 = 84.17 A)
- Samples above 10.0 A threshold: 109
- Strongest neutral excursions:
|   device_id | name                                                  |   neutral_from_trms_120deg |
|------------:|:------------------------------------------------------|---------------------------:|
|         262 | D0006 SP2 L1 / UMG801-4700-2328 / Measurement Group 2 |                   32659.15 |
|         361 | D1398 SP1 L1                                          |                     219.09 |
|         156 | D1159 SP1 L1                                          |                     149.37 |
|         377 | D0579 SP1 L1                                          |                     140.64 |
|         155 | D1160 SP1 L1                                          |                     122.64 |

## Label sentiment overview
- `I4_label` top states → Input04: 70, Input08: 70, (missing): 42, L4: 12
- `Ia_label` top states → Input01: 70, Input05: 70, L1: 54
- `Ib_label` top states → Input02: 70, Input06: 70, L2: 54
- `Ic_label` top states → Input03: 70, Input07: 70, L3: 54
- `Va_label` top states → L1: 194
- `Vb_label` top states → L2: 194
- `Vc_label` top states → L3: 194

## Paired metric cross-checks
- `vuf_magnitude` vs `vuf_symmetrical` → correlation 0.88, mean |Δ| 0.0005 pu, median |Δ| 0.0004 pu
|   device_id | name                                                            | window_start     | window_end       |   vuf_magnitude |   vuf_symmetrical |
|------------:|:----------------------------------------------------------------|:-----------------|:-----------------|----------------:|------------------:|
|         338 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 1           | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.0072 |
|         339 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 2           | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.0072 |
|         342 | D1030 SP1 L1 / Mod. 1 / Measurement Group 1                     | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.0072 |
|         343 | D1030 SP1 L1 / Mod. 1 / Measurement Group 2                     | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.0072 |
|          36 | D0690 SP2 L1 / D0690 Arnarsmári 28 SP2 L1 / Measurement Group 1 | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0012 |            0.0028 |
- `neutral_from_trms_120deg` vs `I4_avg` → correlation -0.04, mean |Δ| 230.2879 A, median |Δ| 8.4647 A
|   device_id | name                                                  | window_start     | window_end       |   neutral_from_trms_120deg |   I4_avg |
|------------:|:------------------------------------------------------|:-----------------|:-----------------|---------------------------:|---------:|
|         262 | D0006 SP2 L1 / UMG801-4700-2328 / Measurement Group 2 | 2025-09-01 12:00 | 2025-10-01 12:00 |                 32659.1492 |   0.0000 |
|         245 | D1411 SP1 L1 / Mod 1 / Device-245                     | 2025-09-01 12:00 | 2025-10-01 12:00 |                   112.5609 |   0.0000 |
|         208 | D0303 SP1 L1 / UMG801-4700-2320 / Measurement Group 2 | 2025-09-01 12:00 | 2025-10-01 12:00 |                   114.7506 |   6.8190 |
|         232 | D1299 SP1 L1 / UMG801-4700-1575 / Device-232          | 2025-09-01 12:00 | 2025-10-01 12:00 |                    93.2590 |   0.0000 |
|         293 | D0383 SP1 L1 / UMG801-4700-2318 / Measurement Group 1 | 2025-09-01 12:00 | 2025-10-01 12:00 |                    82.6414 |  13.3766 |

## Metric overview
| metric                            |   count |     mean |   median |       std |    min |        max |   threshold |   exceed_count |
|:----------------------------------|--------:|---------:|---------:|----------:|-------:|-----------:|------------:|---------------:|
| cur_ratio                         |     194 |  24.8229 |   7.3468 |   34.6140 | 0.0000 |   100.0000 |     10.0000 |             86 |
| cur_dev_ratio                     |     194 |  15.5034 |   4.1593 |   22.5609 | 0.0000 |    66.6667 |     10.0000 |             62 |
| dib                               |     158 |   0.1828 |   0.0601 |    0.2362 | 0.0002 |     0.6667 |      0.0500 |             89 |
| neutral_from_trms_120deg          |     194 | 190.2118 |  12.7468 | 2337.3596 | 0.0000 | 32659.1492 |     10.0000 |            109 |
| I4_avg                            |     152 |   8.9599 |   0.0000 |   19.1640 | 0.0000 |   126.8736 |     10.0000 |             33 |
| vuf_magnitude                     |     194 |   0.0078 |   0.0015 |    0.0482 | 0.0002 |     0.3930 |      0.0200 |              3 |
| vuf_symmetrical                   |     116 |   0.0022 |   0.0020 |    0.0012 | 0.0011 |     0.0072 |      0.0200 |              0 |
| sequence_unbalance_factors.M2_mag |      34 |   0.1135 |   0.0732 |    0.1415 | 0.0009 |     0.8520 |      0.0200 |             31 |
| sequence_unbalance_factors.M0_mag |      34 |   0.1148 |   0.0793 |    0.1428 | 0.0009 |     0.8535 |      0.0200 |             31 |

## Top 5 devices by `cur_ratio`
|   device_id | name                                         |   cur_ratio |
|------------:|:---------------------------------------------|------------:|
|           8 | D0029 SP1 L1 / Mod1 / Measurement Group 1    |    100.0000 |
|         153 | D0841 SP1 L1 / Mod1 / Measurement Group 1    |    100.0000 |
|         211 | D0303 SP1 L1 / Mod 1 / Measurement Group 1   |    100.0000 |
|         233 | D1299 SP1 L1 / UMG801-4700-1575 / Device-233 |    100.0000 |
|         244 | D1411 SP1 L1 / Mod 1 / Device-244            |    100.0000 |

## Top 5 devices by `cur_dev_ratio`
|   device_id | name                                                  |   cur_dev_ratio |
|------------:|:------------------------------------------------------|----------------:|
|         211 | D0303 SP1 L1 / Mod 1 / Measurement Group 1            |         66.6667 |
|         233 | D1299 SP1 L1 / UMG801-4700-1575 / Device-233          |         66.6667 |
|         249 | D0176 SP2 L1 / UMG801-4700-2673 / Measurement Group 2 |         66.6667 |
|         257 | D0173 SP1 L1 / Mod1 / Measurement Group 1             |         66.6667 |
|         302 | D0613 SP1 L1 / UMG801-4700-2675 / Measurement Group 2 |         66.6667 |

## Top 5 devices by `dib`
|   device_id | name                                                  |    dib |
|------------:|:------------------------------------------------------|-------:|
|         211 | D0303 SP1 L1 / Mod 1 / Measurement Group 1            | 0.6667 |
|         233 | D1299 SP1 L1 / UMG801-4700-1575 / Device-233          | 0.6667 |
|         249 | D0176 SP2 L1 / UMG801-4700-2673 / Measurement Group 2 | 0.6667 |
|         257 | D0173 SP1 L1 / Mod1 / Measurement Group 1             | 0.6667 |
|         302 | D0613 SP1 L1 / UMG801-4700-2675 / Measurement Group 2 | 0.6667 |

## Top 5 devices by `neutral_from_trms_120deg`
|   device_id | name                                                  |   neutral_from_trms_120deg |
|------------:|:------------------------------------------------------|---------------------------:|
|         262 | D0006 SP2 L1 / UMG801-4700-2328 / Measurement Group 2 |                 32659.1492 |
|         361 | D1398 SP1 L1                                          |                   219.0871 |
|         156 | D1159 SP1 L1                                          |                   149.3722 |
|         377 | D0579 SP1 L1                                          |                   140.6410 |
|         155 | D1160 SP1 L1                                          |                   122.6430 |

## Top 5 devices by `I4_avg`
|   device_id | name                                                  |   I4_avg |
|------------:|:------------------------------------------------------|---------:|
|         440 | D0604 SP2 L1 / UMG801-4700-7811 / Device-3            | 126.8736 |
|         383 | D1354 SP1 L1                                          |  72.7187 |
|         355 | D1554 SP1 L1 / UMG801-4700-2669 / Measurement Group 2 |  63.9037 |
|         335 | D1019 SP1 L1                                          |  59.9096 |
|         297 | D0383 SP1 L1 / Mod1 / Measurement Group 1             |  56.2697 |

## Top 5 devices by `vuf_magnitude`
|   device_id | name                                                  |   vuf_magnitude |
|------------:|:------------------------------------------------------|----------------:|
|         375 | D1416 SP1 L1                                          |          0.3930 |
|         377 | D0579 SP1 L1                                          |          0.3925 |
|         361 | D1398 SP1 L1                                          |          0.3922 |
|         344 | D1046 SP2 L1                                          |          0.0064 |
|         456 | D1029 SP1 L1 / UMG801-4700-2804 / Measurement Group 1 |          0.0061 |

## Top 5 devices by `vuf_symmetrical`
|   device_id | name                                                  |   vuf_symmetrical |
|------------:|:------------------------------------------------------|------------------:|
|         338 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 1 |            0.0072 |
|         339 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 2 |            0.0072 |
|         342 | D1030 SP1 L1 / Mod. 1 / Measurement Group 1           |            0.0072 |
|         343 | D1030 SP1 L1 / Mod. 1 / Measurement Group 2           |            0.0072 |
|         448 | D1056 SP1 L1 / UMG801-4700-2674 / Measurement Group 1 |            0.0058 |

## Top 5 devices by `sequence_unbalance_factors.M2_mag`
|   device_id | name                                                  |   sequence_unbalance_factors.M2_mag |
|------------:|:------------------------------------------------------|------------------------------------:|
|         232 | D1299 SP1 L1 / UMG801-4700-1575 / Device-232          |                              0.8520 |
|         448 | D1056 SP1 L1 / UMG801-4700-2674 / Measurement Group 1 |                              0.2499 |
|         223 | D1456 SP1 L1 / UMG801-4700-2313 / Measurement Group 1 |                              0.2288 |
|         169 | D1457 SP1 L1 / UMG801-4700-2422 / Measurement Group 1 |                              0.2202 |
|         215 | D0263 SP1 L1 / UMG801-4700-2324 / Measurement Group 1 |                              0.1620 |

## Top 5 devices by `sequence_unbalance_factors.M0_mag`
|   device_id | name                                                  |   sequence_unbalance_factors.M0_mag |
|------------:|:------------------------------------------------------|------------------------------------:|
|         232 | D1299 SP1 L1 / UMG801-4700-1575 / Device-232          |                              0.8535 |
|         223 | D1456 SP1 L1 / UMG801-4700-2313 / Measurement Group 1 |                              0.2403 |
|         448 | D1056 SP1 L1 / UMG801-4700-2674 / Measurement Group 1 |                              0.2097 |
|         169 | D1457 SP1 L1 / UMG801-4700-2422 / Measurement Group 1 |                              0.1977 |
|         317 | D1454 SP1 L1 / UMG801-4700-2819 / Measurement Group 1 |                              0.1913 |
