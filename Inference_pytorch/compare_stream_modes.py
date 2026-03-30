import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime


METRIC_PATTERNS = {
    "accuracy_percent": r"Accuracy:\s+\d+/\d+\s+\(([\d.]+)%\)",
    "test_loss": r"Average loss:\s+([\d.]+)",
    "chip_clock_ns": r"Chip clock period(?: is)?:\s*([\deE+\-.]+)\s*ns",
    "chip_latency_ns": r"Chip pipeline-system-clock-cycle \(per image\)(?: is)?:\s*([\deE+\-.]+)\s*ns",
    "read_dynamic_energy_pj": r"Chip pipeline-system readDynamicEnergy \(per image\)(?: is)?:\s*([\deE+\-.]+)\s*pJ",
    "leakage_energy_pj": r"Chip pipeline-system leakage Energy \(per image\)(?: is)?:\s*([\deE+\-.]+)\s*pJ",
    "energy_efficiency_topsw": r"Energy Efficiency(?: TOPS/W \(Pipelined Process\))?:\s*([\deE+\-.]+)",
    "throughput_fps": r"Throughput FPS \(Pipelined Process\):\s*([\deE+\-.]+)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare StreamCNN baseline and async hardware results")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--model", default="StreamCNN")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--inference", type=int, default=1)
    parser.add_argument("--wl_weight", type=int, default=8)
    parser.add_argument("--wl_activate", type=int, default=8)
    parser.add_argument("--subArray", type=int, default=32)
    parser.add_argument("--parallelRead", type=int, default=32)
    parser.add_argument("--ADCprecision", type=int, default=5)
    parser.add_argument("--cellBit", type=int, default=1)
    parser.add_argument("--onoffratio", type=float, default=10)
    parser.add_argument("--vari", type=float, default=0.0)
    parser.add_argument("--t", type=float, default=0.0)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--detect", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--stream_frequency", type=float, default=1e4)
    parser.add_argument("--stream_window", type=float, default=32.0)
    parser.add_argument("--stream_jitter", type=float, default=0.02)
    parser.add_argument("--stream_energy_per_pulse", type=float, default=1.0)
    return parser.parse_args()


def build_command(mode, args):
    return [
        sys.executable,
        "inference.py",
        "--dataset", args.dataset,
        "--model", args.model,
        "--mode", mode,
        "--batch_size", str(args.batch_size),
        "--inference", str(args.inference),
        "--wl_weight", str(args.wl_weight),
        "--wl_activate", str(args.wl_activate),
        "--subArray", str(args.subArray),
        "--parallelRead", str(args.parallelRead),
        "--ADCprecision", str(args.ADCprecision),
        "--cellBit", str(args.cellBit),
        "--onoffratio", str(args.onoffratio),
        "--vari", str(args.vari),
        "--t", str(args.t),
        "--v", str(args.v),
        "--detect", str(args.detect),
        "--target", str(args.target),
        "--stream_frequency", str(args.stream_frequency),
        "--stream_window", str(args.stream_window),
        "--stream_jitter", str(args.stream_jitter),
        "--stream_energy_per_pulse", str(args.stream_energy_per_pulse),
    ]


def parse_metrics(output_text):
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output_text)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


def format_value(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}" if abs(value) < 1e4 else f"{value:.4e}"


def format_delta(baseline, async_value):
    if baseline is None or async_value is None:
        return "N/A"
    diff = async_value - baseline
    if abs(baseline) < 1e-12:
        return f"{diff:+.4f}"
    return f"{diff:+.4f} ({(diff / baseline) * 100:+.2f}%)"


def print_table(baseline_metrics, async_metrics):
    display_order = [
        ("accuracy_percent", "Accuracy (%)"),
        ("test_loss", "Test loss"),
        ("chip_clock_ns", "Chip clock (ns)"),
        ("chip_latency_ns", "Pipeline latency/image (ns)"),
        ("read_dynamic_energy_pj", "Read dynamic energy (pJ)"),
        ("leakage_energy_pj", "Leakage energy (pJ)"),
        ("energy_efficiency_topsw", "Energy efficiency (TOPS/W)"),
        ("throughput_fps", "Throughput (FPS)"),
    ]

    header = f"{'Metric':<30} {'Baseline':>16} {'Async':>16} {'Delta (A-B)':>22}"
    print(header)
    print("-" * len(header))
    for key, label in display_order:
        baseline_value = baseline_metrics.get(key)
        async_value = async_metrics.get(key)
        print(
            f"{label:<30} {format_value(baseline_value):>16} "
            f"{format_value(async_value):>16} {format_delta(baseline_value, async_value):>22}"
        )


def run_and_capture(mode, working_directory, output_dir, args):
    command = build_command(mode, args)
    result = subprocess.run(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    log_path = os.path.join(output_dir, f"{mode.lower()}.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
    return result.stdout


def load_trace_command(working_directory, model):
    trace_path = os.path.join(working_directory, f"layer_record_{model}", "trace_command.sh")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace command not found: {trace_path}")
    with open(trace_path, "r") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Trace command is empty: {trace_path}")
    return shlex.split(content)


def run_hardware_from_existing_trace(mode, working_directory, output_dir, model):
    command = load_trace_command(working_directory, model)
    command[6] = "1" if mode == "ASYNC" else "0"
    result = subprocess.run(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    log_path = os.path.join(output_dir, f"{mode.lower()}_from_trace.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
    return result.stdout


def main():
    args = parse_args()
    working_directory = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(working_directory, "log", f"stream_compare_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    fallback_reason = None
    try:
        print("Running STREAM_BASELINE...")
        baseline_output = run_and_capture("STREAM_BASELINE", working_directory, output_dir, args)
        print("Running ASYNC...")
        async_output = run_and_capture("ASYNC", working_directory, output_dir, args)
    except subprocess.CalledProcessError as exc:
        fallback_reason = exc.stdout or str(exc)
        if "ModuleNotFoundError: No module named 'torchvision'" not in fallback_reason:
            raise
        print("Torchvision is unavailable; reusing existing StreamCNN traces for hardware-only comparison.")
        print("Running STREAM_BASELINE from existing trace...")
        baseline_output = run_hardware_from_existing_trace("STREAM_BASELINE", working_directory, output_dir, args.model)
        print("Running ASYNC from existing trace...")
        async_output = run_hardware_from_existing_trace("ASYNC", working_directory, output_dir, args.model)

    baseline_metrics = parse_metrics(baseline_output)
    async_metrics = parse_metrics(async_output)

    print()
    print_table(baseline_metrics, async_metrics)
    print()
    if fallback_reason is not None:
        print("Software accuracy fields are N/A in fallback mode because only existing hardware traces were replayed.")
        print()
    print(f"Saved raw logs to: {output_dir}")


if __name__ == "__main__":
    main()
