# TollMatcher Streaming ST Parameter Tuning Report

Evaluated **10** streaming parameter configurations across 30 speech recordings.

## Parameter Comparison & Pareto Optimization

| Configuration | k | Decode (s) | Short EP (s) | Pause Bridge (s) | False Cuts | TTFA Med (ms) | TTL Med (ms) | chrF++ | RTF (Pi 5) | Pareto? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fast Response (Low Latency) | 1 | 0.400 | 0.300 | 0.850 | 2 | 660.0 | 640.0 | 99.88 | 0.120 | **YES** (Optimal) |
| Balanced Streaming (Recommended) | 2 | 0.700 | 0.400 | 1.000 | 0 | 1660.0 | 640.0 | 100.00 | 0.108 | Dominated |
| High Precision (Stable) | 3 | 0.700 | 0.500 | 1.100 | 0 | 2360.0 | 640.2 | 100.00 | 0.108 | Dominated |
| Raspberry Pi 5 Constrained | 2 | 0.800 | 0.450 | 1.000 | 0 | 1960.0 | 640.2 | 100.00 | 0.106 | **YES** (Optimal) |
| Sweep-k=1 | 1 | 0.700 | 0.400 | 1.000 | 0 | 960.0 | 640.0 | 100.00 | 0.108 | **YES** (Optimal) |
| Sweep-k=3 | 3 | 0.700 | 0.400 | 1.000 | 0 | 2360.0 | 640.0 | 100.00 | 0.108 | Dominated |
| Sweep-Bridge=500ms | 2 | 0.700 | 0.400 | 0.500 | 6 | 1660.0 | 640.0 | 99.46 | 0.108 | Dominated |
| Sweep-Bridge=700ms | 2 | 0.700 | 0.400 | 0.700 | 0 | 1660.0 | 640.0 | 100.00 | 0.108 | Dominated |
| Sweep-Bridge=850ms | 2 | 0.700 | 0.400 | 0.850 | 0 | 1660.0 | 640.0 | 100.00 | 0.108 | Dominated |
| Sweep-Bridge=1200ms | 2 | 0.700 | 0.400 | 1.200 | 0 | 1660.0 | 640.0 | 100.00 | 0.108 | Dominated |
