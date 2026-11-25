#!/usr/bin/env python3
"""
tune_streaming_params.py
========================
Command-line interface for the TollMatcher coding team to tune, sweep, and profile
streaming speech translation algorithm parameters.

Supported Modes:
1. Preset Evaluation:
   python3 tune_streaming_params.py --preset [fast|balanced|accurate|edge_pi5]
2. Parameter Sweeps:
   python3 tune_streaming_params.py --sweep [k|decode|bridge|endpoints|curated]
3. Custom Parameter Run:
   python3 tune_streaming_params.py --custom-k 2 --custom-decode 0.700 --custom-short 0.400 --custom-bridge 1.000
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tuning.parameter_space import (
    StreamingHyperparameters,
    PRESETS,
    get_preset,
    get_curated_grid,
    get_k_sweep_grid,
    get_decode_interval_sweep_grid,
    get_pause_bridge_sweep_grid,
    get_endpoint_silence_sweep_grid
)
from src.tuning.grid_search import StreamingParameterTuner


def print_tuning_table(summaries):
    print("\n" + "=" * 120)
    print(f"{'Configuration':<30} | {'k':<2} | {'Dec(s)':<6} | {'Sh(s)':<5} | {'Brdg(s)':<7} | {'FalseCuts':<9} | {'TTFA(ms)':<8} | {'TTL(ms)':<7} | {'chrF++':<6} | {'RTF(RPi)':<8} | {'Pareto?'}")
    print("-" * 120)

    for s in summaries:
        h = s["hyperparameters"]
        pareto = "YES (Optimal)" if s.get("is_pareto_optimal") else "Dominated"
        fc_flag = f"{s['total_false_cuts_in_pause']:<9}"
        if s['total_false_cuts_in_pause'] == 0:
            fc_flag = "0 (Zero!)"
        elif s['total_false_cuts_in_pause'] > 10:
            fc_flag = f"{s['total_false_cuts_in_pause']} (Cut!)"

        print(
            f"{h['name'][:30]:<30} | "
            f"{h['k_agreement']:<2} | "
            f"{h['decode_interval_sec']:<6.3f} | "
            f"{h['short_endpoint_sec']:<5.3f} | "
            f"{h['max_pause_bridge_sec']:<7.3f} | "
            f"{fc_flag:<9} | "
            f"{s['ttfa_ms_median']:<8.1f} | "
            f"{s['ttl_ms_median']:<7.1f} | "
            f"{s['avg_chrf']:<6.2f} | "
            f"{s['avg_rtf']:<8.3f} | "
            f"{pareto}"
        )
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="TollMatcher Streaming Speech Translation Parameter Tuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all standard deployment presets
  python3 tune_streaming_params.py --presets

  # Sweep Local Agreement depth k (k=1, 2, 3)
  python3 tune_streaming_params.py --sweep k

  # Sweep periodic decode interval (400ms to 1000ms)
  python3 tune_streaming_params.py --sweep decode

  # Sweep pause bridging threshold to analyze false cuts
  python3 tune_streaming_params.py --sweep bridge

  # Run curated grid with Pareto frontier analysis
  python3 tune_streaming_params.py --sweep curated

  # Evaluate a custom parameter configuration
  python3 tune_streaming_params.py --custom-k 2 --custom-decode 0.650 --custom-short 0.350 --custom-bridge 0.950
"""
    )

    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Evaluate a specific preset configuration")
    parser.add_argument("--presets", action="store_true", help="Evaluate all pre-configured deployment presets")
    parser.add_argument("--sweep", choices=["k", "decode", "bridge", "endpoints", "curated", "all"],
                        help="Run a systematic parameter sweep")
    parser.add_argument("--custom-k", type=int, default=None, help="Custom Local Agreement depth k (e.g. 1, 2, 3)")
    parser.add_argument("--custom-decode", type=float, default=None, help="Custom ASR decode interval in seconds (e.g. 0.700)")
    parser.add_argument("--custom-short", type=float, default=None, help="Custom short silence endpoint in seconds (e.g. 0.400)")
    parser.add_argument("--custom-safe", type=float, default=None, help="Custom safe acoustic silence in seconds (e.g. 0.750)")
    parser.add_argument("--custom-bridge", type=float, default=None, help="Custom pause bridge threshold in seconds (e.g. 1.000)")
    parser.add_argument("--custom-forced", type=float, default=None, help="Custom forced endpoint duration cap in seconds (e.g. 6.000)")
    parser.add_argument("--export-dir", default=None, help="Custom output directory for JSON/CSV/MD reports")

    args = parser.parse_args()

    print("=" * 80)
    print("TOLLMATCHER STREAMING ALGORITHM PARAMETER TUNING ENGINE")
    print("=" * 80)

    tuner = StreamingParameterTuner()

    configs_to_run = []

    # Custom configuration
    if any(x is not None for x in [args.custom_k, args.custom_decode, args.custom_short, args.custom_bridge]):
        cfg = StreamingHyperparameters(
            name="Custom Configuration",
            k_agreement=args.custom_k if args.custom_k is not None else 2,
            decode_interval_sec=args.custom_decode if args.custom_decode is not None else 0.700,
            short_endpoint_sec=args.custom_short if args.custom_short is not None else 0.400,
            safe_endpoint_sec=args.custom_safe if args.custom_safe is not None else 0.750,
            max_pause_bridge_sec=args.custom_bridge if args.custom_bridge is not None else 1.000,
            forced_endpoint_sec=args.custom_forced if args.custom_forced is not None else 6.000,
            description="Custom user-specified parameter combination"
        )
        configs_to_run.append(cfg)

    # Preset mode
    elif args.preset:
        configs_to_run.append(get_preset(args.preset))

    elif args.presets:
        configs_to_run.extend(PRESETS.values())

    # Sweep mode
    elif args.sweep:
        if args.sweep == "k":
            configs_to_run.extend(get_k_sweep_grid())
        elif args.sweep == "decode":
            configs_to_run.extend(get_decode_interval_sweep_grid())
        elif args.sweep == "bridge":
            configs_to_run.extend(get_pause_bridge_sweep_grid())
        elif args.sweep == "endpoints":
            configs_to_run.extend(get_endpoint_silence_sweep_grid())
        elif args.sweep in ["curated", "all"]:
            configs_to_run.extend(get_curated_grid())

    else:
        # Default: run curated grid
        print("No specific mode specified; executing curated parameter grid search...")
        configs_to_run.extend(get_curated_grid())

    print(f"Evaluating {len(configs_to_run)} parameter configurations across 30 speech clips and FLORES pairs...")
    t0 = time.perf_counter()
    summaries = tuner.evaluate_grid(configs_to_run)
    elapsed = time.perf_counter() - t0

    print_tuning_table(summaries)
    print(f"Grid evaluation finished in {elapsed:.2f} seconds.")

    # Export reports
    export_dir = args.export_dir or os.path.join(PROJECT_ROOT, "results", "tuning")
    report_paths = tuner.export_reports(summaries, output_dir=export_dir)
    print(f"\nTuning reports exported to: {export_dir}")
    print(f" - JSON: {report_paths['json_path']}")
    print(f" - CSV : {report_paths['csv_path']}")
    print(f" - MD  : {report_paths['md_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
