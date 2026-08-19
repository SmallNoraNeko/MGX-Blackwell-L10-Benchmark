#!/usr/bin/env python3
# ===========================================================================
# run_FP4_GEMM.py
# Purpose : Run FP4 (mxfp4-out-e2m1) GEMM Benchmark
#           with SDR collection:
#             -  (BEFORE)    : one snapshot before test starts
#             -  (DURING) : every 1s while benchmark runs
#             -  (AFTER)   : continue monitoring until GPU reaches idle
#                             temp (≤ IDLE_TEMP_C) AND at least POST_MIN
#                             mins elapsed; hard-stop at POST_MAX_MIN mins.
# Platform: GB300 / ARM64
# Usage   : chmod +x run_FP4_GEMM.py && ./run_FP4_GEMM.py
#           or: python3 run_FP4_GEMM.py
# ===========================================================================

import subprocess
import threading
import time
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SDR_INTERVAL   = 1          # seconds between SDR samples
POST_MIN       = 1 * 60    # minimum post-test monitoring: 10 minutes (s)
POST_MAX_MIN   = 20 * 60    # hard ceiling for post-test monitoring (s)
IDLE_TEMP_C    = 40         # GPU considered idle below this temperature (°C)
GPU_TEMP_KEYS  = [f"GPU{i}_TEMP" for i in range(4)]

DATE       = datetime.now().strftime("%Y%m%d")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
# Allow launcher to override log directory via environment variable
LOG_DIR    = os.environ.get("LOG_OUTPUT_DIR") or os.path.join(BASE_DIR, f"logs_FP4_{DATE}")

# run_bench.py lives inside tools/gemm-memread/scripts/
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "../tools/gemm-memread/scripts/run_bench.py")
BENCH_ARGS   = [
    "--dtype",       "mxfp4-out-e2m1",
    "--gemm_dim",    "min-dram",
    "--duty_cycle",  "100",
    "--gpu",         "0",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)

def run_cmd(cmd: list, ignore_error: bool = False) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        if not ignore_error:
            raise
        return str(e)

def collect_sdr() -> str:
    return run_cmd(["ipmitool", "sdr"], ignore_error=True)

def parse_gpu_temps(sdr_output: str) -> list:
    temps = []
    for line in sdr_output.splitlines():
        for key in GPU_TEMP_KEYS:
            if line.startswith(key):
                parts = line.split("|")
                if len(parts) >= 2:
                    val_str = parts[1].strip().split()[0]
                    try:
                        temps.append(int(val_str))
                    except ValueError:
                        pass
    return temps

def write_sdr_snapshot(f, label: str, sdr_output: str):
    f.write(f"=== SDR Sampled ({label}) at: {ts()} ===\n")
    f.write(sdr_output)
    f.write("\n")
    f.flush()

# ---------------------------------------------------------------------------
# SDR background thread (during test)
# ---------------------------------------------------------------------------

class SdrMonitor(threading.Thread):
    def __init__(self, filepath: str, label: str):
        super().__init__(daemon=True)
        self.filepath    = filepath
        self.label       = label
        self._stop_event = threading.Event()  # renamed: avoid collision with Thread._stop()

    def run(self):
        with open(self.filepath, "a") as f:
            while not self._stop_event.is_set():
                sdr = collect_sdr()
                write_sdr_snapshot(f, self.label, sdr)
                temps = parse_gpu_temps(sdr)
                if temps:
                    temp_str = "  ".join(f"GPU{i}: {t}°C" for i, t in enumerate(temps))
                    print(f"[{ts()}] [GPU Temp during test] {temp_str}", flush=True)
                self._stop_event.wait(SDR_INTERVAL)

    def stop(self):
        self._stop_event.set()
        self.join()

# ---------------------------------------------------------------------------
# Post-test monitoring (until idle OR timeout)
# ---------------------------------------------------------------------------

def post_test_monitor(sdr_log_path: str, label: str):
    log(f"[{label}] Post-test monitoring started "
        f"(min {POST_MIN//60} min, idle ≤ {IDLE_TEMP_C}°C, "
        f"max {POST_MAX_MIN//60} min).")
    start   = time.monotonic()
    idle_ok = False

    with open(sdr_log_path, "a") as f:
        while True:
            elapsed = time.monotonic() - start
            sdr     = collect_sdr()
            write_sdr_snapshot(f, f"{label}_POST", sdr)

            temps = parse_gpu_temps(sdr)
            max_t = max(temps) if temps else 999
            if temps:
                temp_str = "  ".join(f"GPU{i}: {t}°C" for i, t in enumerate(temps))
                print(f"[{ts()}] [GPU Temp after test] {temp_str}", flush=True)

            if elapsed >= POST_MIN and max_t <= IDLE_TEMP_C:
                if not idle_ok:
                    log(f"[{label}] GPU temps idle ({max_t}°C ≤ {IDLE_TEMP_C}°C) "
                        f"after {elapsed/60:.1f} min.")
                idle_ok = True

            if idle_ok and elapsed >= POST_MIN:
                log(f"[{label}] Post-test done ({elapsed/60:.1f} min, max GPU {max_t}°C).")
                break

            if elapsed >= POST_MAX_MIN:
                log(f"[{label}] Post-test hit hard limit ({POST_MAX_MIN//60} min). "
                    f"GPU max temp: {max_t}°C.")
                break

            time.sleep(SDR_INTERVAL)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(BENCH_SCRIPT):
        print(f"[ERROR] run_bench.py not found in {SCRIPT_DIR}. "
              "Please check the file location.")
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)

    sdr_log = os.path.join(LOG_DIR, f"sdr_FP4_{DATE}.log")
    res_log = os.path.join(LOG_DIR, f"result_FP4_{DATE}.log")

    # Clear system logs
    log("Clearing system logs before test run...")
    run_cmd(["dmesg", "-C"],             ignore_error=True)
    run_cmd(["ipmitool", "sel", "clear"], ignore_error=True)
    try:
        with open("/var/log/syslog", "w") as f:
            f.write(" ")
    except PermissionError:
        log("[WARN] Could not clear /var/log/syslog (permission denied).")
    log("Logs cleared.")

    log("="*60)
    log("  FP4 GEMM Benchmark — GB300 ARM64")
    log("="*60)

    # ── (before) ──────────────────────────────────────────────
    log("Collecting pre-test SDR snapshot (前)...")
    with open(sdr_log, "w") as f:
        sdr = collect_sdr()
        write_sdr_snapshot(f, "FP4_PRE", sdr)
    temps_pre = parse_gpu_temps(sdr)
    log(f"Pre-test GPU temps: {temps_pre}")

    # ── (during) ──────────────────────────────────────────────
    log("Starting benchmark + SDR monitoring (中)...")
    monitor = SdrMonitor(sdr_log, "FP4_DURING")
    monitor.start()

    env = os.environ.copy()
    # GEMM_DIR must point to the directory containing build/ and configs/
    env["GEMM_DIR"] = os.path.join(SCRIPT_DIR, "../tools/gemm-memread")
    bench_cmd = [sys.executable, BENCH_SCRIPT] + BENCH_ARGS
    log(f"CMD: {' '.join(bench_cmd)}")

    with open(res_log, "w") as rf:
        proc = subprocess.Popen(bench_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            rf.write(line)
        proc.wait()

    monitor.stop()
    log(f"Benchmark finished (exit code {proc.returncode}).")

    # ── (after) ────────────────────────────────────────────────
    post_test_monitor(sdr_log, "FP4")

    # Collect final system logs
    log("Collecting post-run system logs...")

    dmesg_log = os.path.join(LOG_DIR, f"dmesg_{DATE}.log")
    with open(dmesg_log, "w") as f:
        f.write(f"=== Collected at: {ts()} ===\n")
        f.write(run_cmd(["dmesg"], ignore_error=True))

    sel_log = os.path.join(LOG_DIR, f"ipmi_sel_{DATE}.log")
    with open(sel_log, "w") as f:
        f.write(f"=== Collected at: {ts()} ===\n")
        f.write(run_cmd(["ipmitool", "sel", "elist"], ignore_error=True))

    syslog_out = os.path.join(LOG_DIR, f"syslog_{DATE}.log")
    try:
        with open("/var/log/syslog") as src, open(syslog_out, "w") as dst:
            dst.write(f"=== Collected at: {ts()} ===\n")
            dst.write(src.read())
    except Exception as e:
        log(f"[WARN] Could not copy syslog: {e}")

    log("="*60)
    log(" FP4 GEMM benchmark completed.")
    log("="*60)
    log(f"Logs saved to: {LOG_DIR}")
    log(f"  Benchmark Results : {res_log}")
    log(f"  SDR History       : {sdr_log}")
    log(f"  Hardware Events   : dmesg_{DATE}.log / ipmi_sel_{DATE}.log / syslog_{DATE}.log")


if __name__ == "__main__":
    main()
