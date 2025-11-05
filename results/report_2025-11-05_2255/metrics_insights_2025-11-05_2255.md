# Janitza imbalance insights (2025-11-05_2255)

Source file: `C:\Users\Notandi\Desktop\Master\Jasonaq_thesis\results\metrics_results_2025-11-05_2224.csv`

## Dataset pulse check
- Samples analysed: 190
- Unique devices: 190
- Monitoring horizon: 2025-09-01 12:00 → 2025-10-01 12:00

## Phase balance snapshots
- Average phase current spread (max-min): 217.27 A
- Worst current imbalance windows:
|   device_id | name                                                  |   Ia_avg |   Ib_avg |   Ic_avg |
|------------:|:------------------------------------------------------|---------:|---------:|---------:|
|         262 | D0006 SP2 L1 / UMG801-4700-2328 / Measurement Group 2 | 25358.67 | 36852.12 |     0.00 |
|         361 | D1398 SP1 L1                                          |   233.16 |   229.98 |    12.50 |
|         156 | D1159 SP1 L1                                          |   542.95 |   393.99 |   393.17 |
|         377 | D0579 SP1 L1                                          |   242.66 |   223.94 |    93.59 |
|         155 | D1160 SP1 L1                                          |   416.81 |   315.13 |   280.61 |
- Average phase voltage spread (max-min): 3.51 V
- Worst voltage imbalance windows:
|   device_id | name                                                  |   Va_avg |   Vb_avg |   Vc_avg |
|------------:|:------------------------------------------------------|---------:|---------:|---------:|
|         377 | D0579 SP1 L1                                          |   235.53 |   235.98 |   408.46 |
|         375 | D1416 SP1 L1                                          |   234.59 |   235.05 |   407.08 |
|         361 | D1398 SP1 L1                                          |   235.18 |   235.59 |   407.66 |
|         456 | D1029 SP1 L1 / UMG801-4700-2804 / Measurement Group 1 |   235.32 |   234.06 |   236.84 |
|         457 | D1029 SP1 L1 / UMG801-4700-2804 / Measurement Group 2 |   235.32 |   234.06 |   236.84 |

## Neutral channel watchlist
- Neutral current median: 12.75 A (p95 = 83.16 A)
- Samples above 10.0 A threshold: 107
- Strongest neutral excursions:
|   device_id | name                                                  |   neutral_from_trms_120deg |
|------------:|:------------------------------------------------------|---------------------------:|
|         262 | D0006 SP2 L1 / UMG801-4700-2328 / Measurement Group 2 |                   32659.15 |
|         361 | D1398 SP1 L1                                          |                     219.09 |
|         156 | D1159 SP1 L1                                          |                     149.37 |
|         377 | D0579 SP1 L1                                          |                     140.64 |
|         155 | D1160 SP1 L1                                          |                     122.64 |

## Label sentiment overview
- `I4_label` top states → Input04: 68, Input08: 68, (missing): 42, L4: 12
- `Ia_label` top states → Input01: 68, Input05: 68, L1: 54
- `Ib_label` top states → Input02: 68, Input06: 68, L2: 54
- `Ic_label` top states → Input03: 68, Input07: 68, L3: 54
- `Va_label` top states → L1: 190
- `Vb_label` top states → L2: 190
- `Vc_label` top states → L3: 190

## Paired metric cross-checks
- `vuf_magnitude` vs `vuf_symmetrical` → correlation 0.89, mean |Δ| 0.2200 pu, median |Δ| 0.1934 pu
|   device_id | name                                                  | window_start     | window_end       |   vuf_magnitude |   vuf_symmetrical |
|------------:|:------------------------------------------------------|:-----------------|:-----------------|----------------:|------------------:|
|         338 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 1 | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.7192 |
|         339 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 2 | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.7192 |
|         342 | D1030 SP1 L1 / Mod. 1 / Measurement Group 1           | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.7192 |
|         343 | D1030 SP1 L1 / Mod. 1 / Measurement Group 2           | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0050 |            0.7192 |
|         448 | D1056 SP1 L1 / UMG801-4700-2674 / Measurement Group 1 | 2025-09-01 12:00 | 2025-10-01 12:00 |          0.0052 |            0.5765 |
- `neutral_from_trms_120deg` vs `I4_avg` → correlation -0.04, mean |Δ| 235.5633 A, median |Δ| 8.4647 A
|   device_id | name                                                  | window_start     | window_end       |   neutral_from_trms_120deg |   I4_avg |
|------------:|:------------------------------------------------------|:-----------------|:-----------------|---------------------------:|---------:|
|         262 | D0006 SP2 L1 / UMG801-4700-2328 / Measurement Group 2 | 2025-09-01 12:00 | 2025-10-01 12:00 |                 32659.1492 |   0.0000 |
|         245 | D1411 SP1 L1 / Mod 1 / Device-245                     | 2025-09-01 12:00 | 2025-10-01 12:00 |                   112.5609 |   0.0000 |
|         232 | D1299 SP1 L1 / UMG801-4700-1575 / Device-232          | 2025-09-01 12:00 | 2025-10-01 12:00 |                    93.2590 |   0.0000 |
|         293 | D0383 SP1 L1 / UMG801-4700-2318 / Measurement Group 1 | 2025-09-01 12:00 | 2025-10-01 12:00 |                    82.6414 |  13.3766 |
|         440 | D0604 SP2 L1 / UMG801-4700-7811 / Device-3            | 2025-09-01 12:00 | 2025-10-01 12:00 |                    57.9805 | 126.8736 |

## Metric overview
| metric                            |   count |     mean |   median |       std |    min |        max |   threshold |   exceed_count |
|:----------------------------------|--------:|---------:|---------:|----------:|-------:|-----------:|------------:|---------------:|
| cur_ratio                         |     190 |  24.2907 |   7.3468 |   34.1019 | 0.0000 |   100.0000 |     10.0000 |             84 |
| cur_dev_ratio                     |     190 |  15.1412 |   4.1593 |   22.2020 | 0.0000 |    66.6667 |     10.0000 |             60 |
| dib                               |     155 |   0.1779 |   0.0601 |    0.2324 | 0.0002 |     0.6667 |      0.0500 |             87 |
| neutral_from_trms_120deg          |     190 | 193.4415 |  12.7468 | 2361.7186 | 0.0000 | 32659.1492 |     10.0000 |            107 |
| I4_avg                            |     148 |   9.1559 |   0.0000 |   19.3775 | 0.0000 |   126.8736 |     10.0000 |             33 |
| vuf_magnitude                     |     190 |   0.0079 |   0.0015 |    0.0487 | 0.0002 |     0.3930 |      0.0200 |              3 |
| vuf_symmetrical                   |     112 |   0.2218 |   0.1950 |    0.1251 | 0.1146 |     0.7192 |      0.0200 |            112 |
| sequence_unbalance_factors.M2_mag |      33 |   0.1142 |   0.0730 |    0.1436 | 0.0009 |     0.8520 |      0.0200 |             30 |
| sequence_unbalance_factors.M0_mag |      33 |   0.1154 |   0.0784 |    0.1449 | 0.0009 |     0.8535 |      0.0200 |             30 |

## Top 5 devices by `cur_ratio`
|   device_id | name                                                  |   cur_ratio |
|------------:|:------------------------------------------------------|------------:|
|           8 | D0029 SP1 L1 / Mod1 / Measurement Group 1             |    100.0000 |
|         153 | D0841 SP1 L1 / Mod1 / Measurement Group 1             |    100.0000 |
|         233 | D1299 SP1 L1 / UMG801-4700-1575 / Device-233          |    100.0000 |
|         244 | D1411 SP1 L1 / Mod 1 / Device-244                     |    100.0000 |
|         249 | D0176 SP2 L1 / UMG801-4700-2673 / Measurement Group 2 |    100.0000 |

## Top 5 devices by `cur_dev_ratio`
|   device_id | name                                                  |   cur_dev_ratio |
|------------:|:------------------------------------------------------|----------------:|
|         233 | D1299 SP1 L1 / UMG801-4700-1575 / Device-233          |         66.6667 |
|         249 | D0176 SP2 L1 / UMG801-4700-2673 / Measurement Group 2 |         66.6667 |
|         257 | D0173 SP1 L1 / Mod1 / Measurement Group 1             |         66.6667 |
|         302 | D0613 SP1 L1 / UMG801-4700-2675 / Measurement Group 2 |         66.6667 |
|         349 | D0662 SP2 L1 / UMG801-4700-7805 / Measurement Group 2 |         66.6667 |

## Top 5 devices by `dib`
|   device_id | name                                                  |    dib |
|------------:|:------------------------------------------------------|-------:|
|         233 | D1299 SP1 L1 / UMG801-4700-1575 / Device-233          | 0.6667 |
|         249 | D0176 SP2 L1 / UMG801-4700-2673 / Measurement Group 2 | 0.6667 |
|         257 | D0173 SP1 L1 / Mod1 / Measurement Group 1             | 0.6667 |
|         302 | D0613 SP1 L1 / UMG801-4700-2675 / Measurement Group 2 | 0.6667 |
|         349 | D0662 SP2 L1 / UMG801-4700-7805 / Measurement Group 2 | 0.6667 |

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
|         338 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 1 |            0.7192 |
|         339 | D1030 SP1 L1 / UMG801-4700-7801 / Measurement Group 2 |            0.7192 |
|         342 | D1030 SP1 L1 / Mod. 1 / Measurement Group 1           |            0.7192 |
|         343 | D1030 SP1 L1 / Mod. 1 / Measurement Group 2           |            0.7192 |
|         448 | D1056 SP1 L1 / UMG801-4700-2674 / Measurement Group 1 |            0.5765 |

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
