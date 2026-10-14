# TollMatcher Streaming Speech Translation Benchmark Report

## Executive Comparison: Algorithm Variants A, B, and C

| Metric / Dimension | Variant A: Fixed 1.0s Chunks | Variant B: VAD Pause EP | Variant C: Streaming Adaptive |
| :--- | :---: | :---: | :---: |
| **Translation chrF++** | 85.25 | 97.24 | **100.00** |
| **Translation BLEU** | 79.40 | 89.16 | **100.00** |
| **ASR Word Error Rate (WER)** | 0.094 | 0.000 | **0.000** |
| **ASR Character Error Rate (CER)** | 0.019 | 0.000 | **0.000** |
| **False Cuts in Thinking Pauses** | 10 | 18 (Premature Cuts!) | **0 (Zero False Cuts)** |
| **Segmentation F1 (tau=200ms)** | 0.195 | 0.000 | **0.000** |
| **Segmentation F1 (tau=500ms)** | 0.333 | 0.833 | **0.944** |
| **Time-to-First-Audio (TTFA Median)** | 950.0 ms | 3476.6 ms | **360.0 ms** |
| **Total Turnaround Latency (TTL Median)** | 599.9 ms | 679.9 ms | **639.9 ms** |
| **TTL 95th Percentile (P95)** | 600.4 ms | 680.4 ms | **640.4 ms** |
| **Real-Time Factor (RTF on RPi)** | 0.385 | 0.144 | **0.108** |
