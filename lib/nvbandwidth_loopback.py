#!/usr/bin/env python3
"""
nvbandwidth_loopback.py
=======================
GB300 L10 MGX — Single-Node NVBandwidth Loopback Test
Platform : ARM64 / Ubuntu 24.04 / PCIe-only (no NVSwitch, no NVLink Fabric)
GPU count : 4 (GB300 NVL72 per tray, PCIe Gen5 x16)

Usage:
    sudo python3 nvbandwidth_loopback.py
    sudo python3 nvbandwidth_loopback.py --nvbw-path ./tools/nvbandwidth
    sudo python3 nvbandwidth_loopback.py --iters 5 --dry-run

What this script does
---------------------
1. Pre-flight : verify nvbandwidth binary, GPU count, driver, IOMMU/ACS
2. SDR PRE    : one IPMI SDR snapshot before test
3. Run        : execute nvbandwidth with all loopback-relevant test cases
                and stream output live while collecting into log
4. SDR POST   : monitor until GPU temp ≤ IDLE_TEMP_C or timeout
5. Collect    : dmesg, IPMI SEL, syslog, nvidia-smi, nvidia-smi -q

Loopback test cases selected
-----------------------------
All CE (Copy Engine) and SM (Streaming Multiprocessor) memcpy tests that
are meaningful in a single-node PCIe-only topology:

  CE tests:
    host_to_device_memcpy_ce
    device_to_host_memcpy_ce
    host_to_device_bidirectional_memcpy_ce
    device_to_host_bidirectional_memcpy_ce
    device_to_device_memcpy_read_ce
    device_to_device_memcpy_write_ce
    device_to_device_bidirectional_memcpy_read_ce
    device_to_device_bidirectional_memcpy_write_ce
    all_to_host_memcpy_ce
    all_to_host_bidirectional_memcpy_ce
    host_to_all_memcpy_ce
    host_to_all_bidirectional_memcpy_ce
    all_to_one_write_ce
    all_to_one_read_ce
    one_to_all_write_ce
    one_to_all_read_ce

  SM tests:
    host_to_device_memcpy_sm
    device_to_host_memcpy_sm
    device_to_device_memcpy_read_sm
    device_to_device_memcpy_write_sm
    device_to_device_bidirectional_memcpy_read_sm
    device_to_device_bidirectional_memcpy_write_sm
    all_to_host_memcpy_sm
    all_to_host_bidirectional_memcpy_sm
    host_to_all_memcpy_sm
    host_to_all_bidirectional_memcpy_sm
    all_to_one_write_sm
    all_to_one_read_sm
    one_to_all_write_sm
    one_to_all_read_sm
    host_device_latency_sm
    device_to_device_latency_sm
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
# nvbandwidth binary lives in tools/ (one level up from lib/)
DEFAULT_BW   = SCRIPT_DIR.parent / "tools" / "nvbandwidth"

EXPECTED_GPUS    = 4          # GB300 L10 MGX per tray
SDR_INTERVAL     = 5          # seconds between SDR samples
POST_MIN         = 1 * 60    # min post-test monitoring (s)
POST_MAX         = 20 * 60    # hard ceiling (s)
IDLE_TEMP_C      = 40         # GPU idle threshold (°C)
GPU_TEMP_KEYS    = [f"GPU{i}_TEMP" for i in range(EXPECTED_GPUS)]

# nvbandwidth test cases for single-node loopback (PCIe-only, no NVSwitch)
LOOPBACK_TESTS = [
    # CE tests
    "host_to_device_memcpy_ce",
    "device_to_host_memcpy_ce",
    "host_to_device_bidirectional_memcpy_ce",
    "device_to_host_bidirectional_memcpy_ce",
    "device_to_device_memcpy_read_ce",
    "device_to_device_memcpy_write_ce",
    "device_to_device_bidirectional_memcpy_read_ce",
    "device_to_device_bidirectional_memcpy_write_ce",
    "all_to_host_memcpy_ce",
    "all_to_host_bidirectional_memcpy_ce",
    "host_to_all_memcpy_ce",
    "host_to_all_bidirectional_memcpy_ce",
    "all_to_one_write_ce",
    "all_to_one_read_ce",
    "one_to_all_write_ce",
    "one_to_all_read_ce",
    # SM tests
    "host_to_device_memcpy_sm",
    "device_to_host_memcpy_sm",
    "device_to_device_memcpy_read_sm",
    "device_to_device_memcpy_write_sm",
    "device_to_device_bidirectional_memcpy_read_sm",
    "device_to_device_bidirectional_memcpy_write_sm",
    "all_to_host_memcpy_sm",
    "all_to_host_bidirectional_memcpy_sm",
    "host_to_all_memcpy_sm",
    "host_to_all_bidirectional_memcpy_sm",
    "all_to_one_write_sm",
    "all_to_one_read_sm",
    "one_to_all_write_sm",
    "one_to_all_read_sm",
    "host_device_latency_sm",
    "device_to_device_latency_sm",
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
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout + r.stderr
    except Exception as e:
        if not ignore_error:
            raise
        return str(e)


def collect_sdr() -> str:
    return run_cmd(["ipmitool", "sdr"], ignore_error=True)


def parse_gpu_temps(sdr: str) -> list[int]:
    temps = []
    for line in sdr.splitlines():
        for key in GPU_TEMP_KEYS:
            if line.startswith(key):
                parts = line.split("|")
                if len(parts) >= 2:
                    try:
                        temps.append(int(parts[1].strip().split()[0]))
                    except ValueError:
                        pass
    return temps


def write_sdr(f, label: str, sdr: str):
    f.write(f"=== SDR ({label}) at: {ts()} ===\n{sdr}\n")
    f.flush()


# ---------------------------------------------------------------------------
# SDR background monitor
# ---------------------------------------------------------------------------

class SdrMonitor(threading.Thread):
    def __init__(self, path: str, label: str):
        super().__init__(daemon=True)
        self.path        = path
        self.label       = label
        self._stop_event = threading.Event()

    def run(self):
        with open(self.path, "a") as f:
            while not self._stop_event.is_set():
                sdr   = collect_sdr()
                write_sdr(f, self.label, sdr)
                temps = parse_gpu_temps(sdr)
                if temps:
                    log("[GPU Temp] " + "  ".join(
                        f"GPU{i}:{t}°C" for i, t in enumerate(temps)))
                self._stop_event.wait(SDR_INTERVAL)

    def stop(self):
        self._stop_event.set()
        self.join()


# ---------------------------------------------------------------------------
# Post-test cooldown monitor
# ---------------------------------------------------------------------------

def post_monitor(sdr_path: str):
    log(f"Post-test monitor: min {POST_MIN//60}min | idle ≤{IDLE_TEMP_C}°C | max {POST_MAX//60}min")
    start   = time.monotonic()
    idle_ok = False

    with open(sdr_path, "a") as f:
        while True:
            elapsed = time.monotonic() - start
            sdr     = collect_sdr()
            write_sdr(f, "LOOPBACK_POST", sdr)

            temps = parse_gpu_temps(sdr)
            max_t = max(temps) if temps else 999
            if temps:
                log("[GPU Temp POST] " + "  ".join(
                    f"GPU{i}:{t}°C" for i, t in enumerate(temps)))

            if elapsed >= POST_MIN and max_t <= IDLE_TEMP_C:
                if not idle_ok:
                    log(f"GPU idle ({max_t}°C) after {elapsed/60:.1f}min.")
                idle_ok = True

            if idle_ok and elapsed >= POST_MIN:
                log(f"Post-monitor done ({elapsed/60:.1f}min, max {max_t}°C).")
                break

            if elapsed >= POST_MAX:
                log(f"Post-monitor hard limit ({POST_MAX//60}min). max GPU {max_t}°C.")
                break

            time.sleep(SDR_INTERVAL)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight(nvbw: Path, expected_gpus: int) -> bool:
    ok = True

    # Binary
    if not nvbw.is_file():
        log(f"[ERROR] nvbandwidth binary not found: {nvbw}")
        log("        Place the binary in tools/ or pass --nvbw-path")
        ok = False
    else:
        log(f"[OK] nvbandwidth binary: {nvbw}")

    # GPU count via nvidia-smi
    if shutil.which("nvidia-smi"):
        out = run_cmd(["nvidia-smi", "--query-gpu=index",
                       "--format=csv,noheader"], ignore_error=True)
        found = len([l for l in out.strip().splitlines() if l.strip()])
        if found != expected_gpus:
            log(f"[WARN] Expected {expected_gpus} GPUs, found {found}")
        else:
            log(f"[OK] GPU count: {found}")

        # Driver version
        drv = run_cmd(["nvidia-smi", "--query-gpu=driver_version",
                       "--format=csv,noheader"], ignore_error=True).strip().splitlines()
        if drv:
            log(f"[OK] Driver: {drv[0]}")
    else:
        log("[WARN] nvidia-smi not found — skipping GPU count check")

    # ipmitool
    if not shutil.which("ipmitool"):
        log("[WARN] ipmitool not found — SDR collection will be skipped")

    # ACS check — if SrcValid+ found, auto-call lib/disable_acs.sh
    acs_out = run_cmd(["bash", "-c",
        "lspci -vvv 2>/dev/null | grep -i 'ACSCtl' | grep -c 'SrcValid+' || true"],
        ignore_error=True).strip()
    if acs_out and acs_out != "0":
        log(f"[WARN] ACS SrcValid+ found on {acs_out} device(s) — "
            "attempting to disable ACS via lib/disable_acs.sh ...")
        # Search lib/disable_acs.sh relative to this script's directory
        acs_script = SCRIPT_DIR / "disable_acs.sh"           # standalone: lib/
        acs_alt    = SCRIPT_DIR.parent / "lib" / "disable_acs.sh"  # from root
        acs_path   = acs_script if acs_script.is_file() else (
                     acs_alt    if acs_alt.is_file()    else None)
        if acs_path:
            os.chmod(acs_path, 0o755)
            acs_result = run_cmd(["bash", str(acs_path)], ignore_error=True)
            log(f"[ACS] disable_acs.sh output:\n{acs_result.strip()}")
            # Re-check after disabling
            acs_recheck = run_cmd(["bash", "-c",
                "lspci -vvv 2>/dev/null | grep -i 'ACSCtl' | grep -c 'SrcValid+' || true"],
                ignore_error=True).strip()
            if acs_recheck and acs_recheck != "0":
                log(f"[WARN] ACS still active on {acs_recheck} device(s) after disable.")
                log("       P2P performance may be degraded.")
            else:
                log("[OK] ACS successfully disabled — P2P enabled.")
        else:
            log("[WARN] lib/disable_acs.sh not found — skipping ACS disable.")
            log("       Copy disable_acs.sh into lib/ then re-run.")
    else:
        log("[OK] ACS check passed (no SrcValid+ found)")

    return ok


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_loopback(nvbw: Path, iters: int, log_dir: Path, dry_run: bool):
    date  = datetime.now().strftime("%Y%m%d")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sdr_log  = log_dir / f"sdr_loopback_{date}.log"
    res_log  = log_dir / f"result_loopback_{stamp}.log"
    sys_dir  = log_dir

    # Build test list argument
    test_args = []
    for t in LOOPBACK_TESTS:
        test_args += ["-t", t]

    cmd = [str(nvbw)] + test_args + ["-i", str(iters)]

    # Pre SDR
    log("Collecting pre-test SDR snapshot...")
    with open(sdr_log, "w") as f:
        sdr = collect_sdr()
        write_sdr(f, "LOOPBACK_PRE", sdr)
    temps_pre = parse_gpu_temps(sdr)
    log(f"Pre-test GPU temps: {temps_pre}")

    if dry_run:
        log(f"[dry-run] Would run: {' '.join(cmd)}")
        log(f"[dry-run] Log dir  : {log_dir}")
        return

    # Start SDR monitor
    monitor = SdrMonitor(str(sdr_log), "LOOPBACK_DURING")
    monitor.start()

    log("=" * 60)
    log("  NVBandwidth Loopback — GB300 L10 MGX ARM64")
    log("=" * 60)
    log(f"CMD: {' '.join(cmd)}")

    with open(res_log, "w") as rf:
        rf.write(f"=== NVBandwidth Loopback — {ts()} ===\n")
        rf.write(f"CMD: {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            rf.write(line)
        proc.wait()

    monitor.stop()
    log(f"nvbandwidth finished (exit code {proc.returncode}).")

    # Post-test cooldown
    post_monitor(str(sdr_log))

    # Collect system logs
    log("Collecting post-run system logs...")

    with open(sys_dir / f"dmesg_{stamp}.log", "w") as f:
        f.write(f"=== dmesg at {ts()} ===\n")
        f.write(run_cmd(["dmesg", "-T"], ignore_error=True))

    with open(sys_dir / f"ipmi_sel_{stamp}.log", "w") as f:
        f.write(f"=== IPMI SEL at {ts()} ===\n")
        f.write(run_cmd(["ipmitool", "sel", "elist"], ignore_error=True))

    with open(sys_dir / f"nvidia_smi_{stamp}.log", "w") as f:
        f.write(f"=== nvidia-smi at {ts()} ===\n")
        f.write(run_cmd(["nvidia-smi"], ignore_error=True))
        f.write(f"\n=== nvidia-smi -q at {ts()} ===\n")
        f.write(run_cmd(["nvidia-smi", "-q"], ignore_error=True))
        f.write(f"\n=== nvidia-smi topo -m at {ts()} ===\n")
        f.write(run_cmd(["nvidia-smi", "topo", "-m"], ignore_error=True))

    syslog_dst = sys_dir / f"syslog_{stamp}.log"
    for src in ["/var/log/syslog", "/var/log/messages"]:
        if os.path.exists(src):
            try:
                with open(src) as s, open(syslog_dst, "w") as d:
                    d.write(f"=== {src} at {ts()} ===\n")
                    d.write(s.read())
                break
            except PermissionError:
                log(f"[WARN] Cannot read {src} (permission denied)")

    log("=" * 60)
    log("  Loopback test completed.")
    log("=" * 60)
    log(f"  Log dir      : {log_dir}")
    log(f"  BW results   : {res_log}")
    log(f"  SDR history  : {sdr_log}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="GB300 L10 MGX — NVBandwidth Single-Node Loopback Test")
    p.add_argument(
        "--nvbw-path", type=Path, default=None,
        help=f"Path to nvbandwidth binary (default: ./tools/nvbandwidth)")
    p.add_argument(
        "--iters", type=int, default=3,
        help="Number of iterations per test case (default: 3)")
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="Output log directory (default: ./logs_nvbw_loopback_<DATE>/)")
    p.add_argument(
        "--gpus", type=int, default=EXPECTED_GPUS,
        help=f"Expected GPU count (default: {EXPECTED_GPUS})")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print command without executing")
    p.add_argument(
        "--no-preflight", action="store_true",
        help="Skip pre-flight checks")
    return p.parse_args()


def main():
    args = parse_args()

    nvbw = args.nvbw_path or DEFAULT_BW

    date    = datetime.now().strftime("%Y%m%d")
    log_dir = args.log_dir or (SCRIPT_DIR / f"logs_nvbw_loopback_{date}")
    log_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  NVBandwidth Loopback Test — GB300 L10 MGX ARM64")
    log(f"  nvbandwidth : {nvbw}")
    log(f"  iterations  : {args.iters}")
    log(f"  log dir     : {log_dir}")
    log("=" * 60)

    # Clear logs
    if not args.dry_run:
        log("Clearing pre-test system logs...")
        run_cmd(["dmesg", "-C"], ignore_error=True)
        run_cmd(["ipmitool", "sel", "clear"], ignore_error=True)
        try:
            with open("/var/log/syslog", "w") as f:
                f.write("")
        except (PermissionError, FileNotFoundError):
            pass

    # Pre-flight
    if not args.no_preflight:
        ok = preflight(nvbw, args.gpus)
        if not ok and not args.dry_run:
            log("[ERROR] Pre-flight failed. Use --no-preflight to skip.")
            sys.exit(1)

    run_loopback(nvbw, args.iters, log_dir, args.dry_run)


if __name__ == "__main__":
    main()
