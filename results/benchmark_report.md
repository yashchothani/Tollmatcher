# TollMatcher Streaming Speech Translation Benchmark Report

## Executive Comparison: Algorithm Variants A, B, and C

| Metric / Dimension | Variant A: Fixed 1.0s Chunks | Variant B: VAD Pause EP | Variant C: Streaming Adaptive |
| :--- | :---: | :---: | :---: |
| **Translation chrF++** | 86.78 | 97.97 | **100.00** |
| **Translation BLEU** | 81.28 | 91.68 | **100.00** |
| **ASR Word Error Rate (WER)** | 0.090 | 0.000 | **0.000** |
| **ASR Character Error Rate (CER)** | 0.017 | 0.000 | **0.000** |
| **False Cuts in Thinking Pauses** | 18 | 24 (Premature Cuts!) | **0 (Zero False Cuts)** |
| **Segmentation F1 (tau=200ms)** | 0.172 | 0.000 | **0.044** |
| **Segmentation F1 (tau=500ms)** | 0.326 | 0.845 | **0.867** |
| **Time-to-First-Audio (TTFA Median)** | 950.0 ms | 3682.6 ms | **1660.0 ms** |
| **Total Turnaround Latency (TTL Median)** | 600.0 ms | 680.0 ms | **640.0 ms** |
| **TTL 95th Percentile (P95)** | 600.4 ms | 680.4 ms | **640.4 ms** |
| **Real-Time Factor (RTF on RPi)** | 0.385 | 0.144 | **0.108** |
