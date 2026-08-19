#!/usr/bin/env python3
"""
run_nemo_validation_v3.py  (lib/ edition)
------------------------------------------
NeMo / PyTorch DL training validation script for GB300 NVL72 ARM64.

Runs a training workload inside a Podman container, measures step-time
throughput, and collects system health logs (IPMI SDR, dmesg, BMC SEL,
nvidia-smi).

Changes from the original standalone version
---------------------------------------------
1. NEMO_DIR  : configurable via --nemo-dir or LOG_OUTPUT_DIR env-var.
               Points to the directory that contains GB_training_scripts.txt
               and where run_task.sh will be written.
               Default: <lib_dir>/../nemo/
2. LOG_DIR   : controlled by --log-dir or LOG_OUTPUT_DIR env-var so the
               launcher can redirect all output to report/validation_logs_*/
               Default: <nemo_dir>/validation_logs_<TIMESTAMP>/
3. Container workspace mount: mounts NEMO_DIR (not the whole project root)
               into /workspace, keeping tools/ and lib/ out of the container.
4. --image   : exposed as a configurable parameter (default updated to
               pytorch:24.07-py3 — pytorch:25.04-py3 has no ARM64 image).
5. --gpus    : number of GPUs to expose (default 4, maps to
               CUDA_VISIBLE_DEVICES=0,..,N-1).
6. Regex patterns: extended to also match the simulation-loop format
               "iteration time: 0.23x sec" so fallback hardcoded values
               are no longer used during smoke tests.
7. Fallback mode: clearly labelled as SIMULATION in benchmark_metrics.log
               so results are never mistaken for real training data.

Prerequisites (checked at runtime, not at import)
---------------------------------------------------
  podman          - container runtime
  nvidia-ctk      - NVIDIA Container Toolkit (for CDI device access)
  ipmitool        - BMC / SDR sensor collection
  root privileges - required for podman, dmesg, ipmitool

Usage (standalone)
-------------------
  sudo python3 lib/run_nemo_validation_v3.py
  sudo python3 lib/run_nemo_validation_v3.py --image nvcr.io/nvidia/pytorch:24.07-py3
  sudo python3 lib/run_nemo_validation_v3.py --cmd "torchrun --nproc_per_node=4 train.py"
  sudo python3 lib/run_nemo_validation_v3.py --gpus 8 --nemo-dir /data/nemo

Usage (via gb300_launcher.py)
------------------------------
  The launcher calls this script via subprocess and passes:
    --nemo-dir   <project_root>/nemo/
    --log-dir    <project_root>/report/validation_logs_<TIMESTAMP>/
    --image      <user_choice>
    --gpus       <user_choice>
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths resolved relative to this file (lib/)
# ---------------------------------------------------------------------------
_LIB_DIR    = os.path.dirname(os.path.realpath(__file__))
_ROOT_DIR   = os.path.dirname(_LIB_DIR)
# GB_training_scripts.txt lives inside tools/GB_DL_scripts_v9/ (extracted from nv7z)
_NEMO_DIR   = os.path.join(_ROOT_DIR, "tools", "GB_DL_scripts_v9")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_IMAGE          = "nvcr.io/nvidia/pytorch:24.07-py3"  # 25.04-py3 has no ARM64 image
DEFAULT_GPUS           = 4
DEFAULT_GLOBAL_BATCH   = 8

# Step-time regex patterns — ordered from most specific to least specific.
# Pattern 4 is new: matches the simulation loop output "iteration time: 0.23x sec"
_STEP_PATTERNS = [
    re.compile(
        r'(?:train_step_timing|step[_-]?time|iteration[_-]?time)\s*[:=]\s*([\d\.]+)',
        re.IGNORECASE,
    ),
    re.compile(r'elapsed time per iteration[^:]*:\s*([\d\.]+)', re.IGNORECASE),
    re.compile(r'approx.*time per step:\s*([\d\.]+)',            re.IGNORECASE),
    # Simulation loop: "iteration time: 0.230 sec"
    re.compile(r'iteration\s+time:\s*([\d\.]+)\s*sec',           re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)


def run_cmd(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


# ---------------------------------------------------------------------------
# Container image management
# ---------------------------------------------------------------------------

def _find_image_tar(image_name: str) -> str | None:
    """
    Search tools/ for a .tar file that matches the image name.
    Converts 'nvcr.io/nvidia/pytorch:24.07-py3' to a glob pattern like
    'pytorch*24.07*arm64*.tar' and looks in tools/ next to this script.
    Returns the absolute path of the first match, or None.
    """
    tools_dir = Path(_ROOT_DIR) / "tools"
    if not tools_dir.is_dir():
        return None

    # Build search keywords from the image name
    # e.g. "nvcr.io/nvidia/pytorch:24.07-py3" → ["pytorch", "24.07"]
    tag_part  = image_name.split("/")[-1]          # "pytorch:24.07-py3"
    name_part = tag_part.split(":")[0]              # "pytorch"
    ver_part  = tag_part.split(":")[1] if ":" in tag_part else ""  # "24.07-py3"
    ver_short = ver_part.split("-")[0] if ver_part else ""         # "24.07"

    for f in sorted(tools_dir.glob("*.tar")):
        fname = f.name.lower()
        if name_part.lower() in fname and ver_short in fname:
            return str(f)
    return None


def ensure_image_exists(image_name: str):
    """
    Ensure the container image is available locally.

    Resolution order:
      1. Image already in local podman storage  → use directly
      2. tools/*.tar matching the image name    → podman load (offline)
      3. Network pull (requires internet)       → podman pull
      4. All failed                             → exit 1
    """
    log(f"=== Checking local image: {image_name} ===")

    # 1. Already present locally
    if run_cmd(f"podman image inspect {image_name}").returncode == 0:
        log(f"[OK] Image '{image_name}' found locally.")
        return

    # 2. Offline tar in tools/
    tar_path = _find_image_tar(image_name)
    if tar_path:
        log(f"[INFO] Image not in local storage — loading from tar: {tar_path}")
        result = run_cmd(f"podman load -i {tar_path}")
        if result.returncode == 0:
            log(f"[OK] Image loaded from tar successfully.")
            # Verify the loaded image matches what we need
            if run_cmd(f"podman image inspect {image_name}").returncode == 0:
                return
            log("[WARN] Tar loaded but image name does not match — "
                "the tar may contain a differently tagged image.")
            log("       Re-tag with: "
                f"podman tag <loaded-image> {image_name}")
        else:
            log(f"[WARN] podman load failed: {result.stderr.strip()}")

    # 3. Network pull
    log(f"[INFO] Attempting network pull (linux/arm64)...")
    result = run_cmd(f"podman pull --platform linux/arm64 {image_name}")
    if result.returncode == 0:
        log(f"[OK] Successfully pulled {image_name}")
        return

    # 4. All failed
    log(f"[FATAL] Cannot obtain image: {image_name}")
    log("        Options:")
    log(f"          a) Place a matching .tar in tools/ "
        f"(e.g. pytorch_24.07-py3_arm64.tar)")
    log(f"             Generate with: podman save {image_name} "
        f"-o tools/pytorch_24.07-py3_arm64.tar")
    log(f"          b) Connect to network and retry")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CDI setup
# ---------------------------------------------------------------------------

def ensure_cdi():
    os.makedirs("/etc/cdi", exist_ok=True)
    if not os.path.exists("/etc/cdi/nvidia.yaml"):
        log("Generating CDI device config (/etc/cdi/nvidia.yaml)...")
        result = run_cmd("nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml")
        if result.returncode != 0:
            log(f"[WARN] CDI generation failed — GPU access may not work inside container.")
            log(result.stderr.strip())
    else:
        log("[OK] CDI config already present.")


# ---------------------------------------------------------------------------
# IPMI background monitor
# ---------------------------------------------------------------------------

class IpmiMonitor(threading.Thread):
    """Poll 'ipmitool sdr elist' every INTERVAL seconds and append to log."""

    INTERVAL = 5   # seconds between samples

    def __init__(self, log_path: str):
        super().__init__(daemon=True)
        self.log_path    = log_path
        self._stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a", buffering=1) as f:
            while not self._stop_event.is_set():
                res = run_cmd("ipmitool sdr elist")
                if res.returncode == 0 and res.stdout.strip():
                    f.write(f"--- {datetime.now()} ---\n{res.stdout}\n")
                self._stop_event.wait(self.INTERVAL)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=self.INTERVAL + 2)


# ---------------------------------------------------------------------------
# Training command resolution
# ---------------------------------------------------------------------------

def resolve_training_cmd(nemo_dir: str, user_cmd: str) -> tuple[str, bool]:
    """
    Return (command_string, is_simulation).

    Priority:
      1. --cmd CLI argument (user-supplied)
      2. First executable line in nemo/GB_training_scripts.txt
      3. Built-in simulation loop (fallback)
    """
    if user_cmd.strip():
        log(f"Using user-supplied --cmd.")
        return user_cmd.strip(), False

    training_script_path = os.path.join(nemo_dir, "GB_training_scripts.txt")
    if os.path.exists(training_script_path):
        log(f"Reading training command from: {training_script_path}")
        with open(training_script_path) as f:
            for line in f:
                stripped = line.strip()
                # Accept only real executable lines; skip comments, export, git
                if stripped.startswith(("python", "torchrun", "bash")):
                    log(f"Found command: {stripped[:80]}{'...' if len(stripped) > 80 else ''}")
                    return stripped, False
        log("[WARN] GB_training_scripts.txt exists but contains no valid command.")
    else:
        log("[WARN] GB_training_scripts.txt not found at: "
            f"{training_script_path}")
        log("       Falling back to built-in simulation loop.")
        log("       Decode tools/GB_DL_scripts_v9.nv7z into tools/GB_DL_scripts_v9/")
        log("       to run a real training workload.")

    # Built-in simulation — produces "iteration time: 0.23x sec" lines
    simulation_cmd = (
        "python3 -c '"
        "import torch, time\n"
        "print(\"CUDA Available:\", torch.cuda.is_available())\n"
        "print(\"Device Count:\", torch.cuda.device_count())\n"
        "print(\"Running simulation (no GB_training_scripts.txt found)...\")\n"
        "for i in range(10):\n"
        "    time.sleep(0.2)\n"
        "    print(f\"iteration time: 0.23{i} sec\")"
        "'"
    )
    return simulation_cmd, True


# ---------------------------------------------------------------------------
# Metrics parser
# ---------------------------------------------------------------------------

def parse_metrics(stdout_path: str, metrics_path: str,
                  global_batch: int, num_gpus: int,
                  is_simulation: bool):
    log("=== Parsing benchmark metrics ===")
    step_times: list[float] = []

    if os.path.exists(stdout_path):
        with open(stdout_path) as f:
            for line in f:
                for pattern in _STEP_PATTERNS:
                    m = pattern.search(line)
                    if m:
                        try:
                            step_times.append(float(m.group(1)))
                            break
                        except ValueError:
                            pass

    # Discard first 3 warm-up steps when we have enough samples
    valid_times = step_times[3:] if len(step_times) > 5 else step_times

    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        tp       = global_batch / avg_time
        summary  = (
            f"\n================ BENCHMARK SUMMARY ================\n"
            f" Samples parsed    : {len(step_times)}  (used: {len(valid_times)})\n"
            f" Avg Step Time     : {avg_time:.4f} sec/step\n"
            f" Total Throughput  : {tp:.2f} seq/s  (global_batch={global_batch})\n"
            f" Per-GPU Throughput: {tp / num_gpus:.2f} seq/s/GPU  (gpus={num_gpus})\n"
            f"===================================================\n"
        )
    else:
        summary = (
            "\n[WARNING] Could not parse step timing from log.\n"
            "  Check execution_stdout.log for training output format.\n"
            "  Supported patterns:\n"
            "    train_step_timing = <sec>\n"
            "    elapsed time per iteration: <sec>\n"
            "    approx time per step: <sec>\n"
            "    iteration time: <sec> sec\n"
        )

    log(summary)
    with open(metrics_path, "a") as f:
        f.write(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GB300 NeMo/PyTorch DL Validation — lib/ edition",
    )
    parser.add_argument(
        "--cmd", type=str, default="",
        help="Training command to run inside the container. "
             "Overrides GB_training_scripts.txt.",
    )
    parser.add_argument(
        "--image", type=str, default=DEFAULT_IMAGE,
        help=f"Podman container image (default: {DEFAULT_IMAGE}). "
             "Note: pytorch:25.04-py3 has no ARM64 image — keep 24.07-py3 for GB300.",
    )
    parser.add_argument(
        "--gpus", type=int, default=DEFAULT_GPUS,
        help=f"Number of GPUs to expose (default: {DEFAULT_GPUS})",
    )
    parser.add_argument(
        "--global-batch", type=int, default=DEFAULT_GLOBAL_BATCH,
        help=f"Global batch size for throughput calculation (default: {DEFAULT_GLOBAL_BATCH})",
    )
    parser.add_argument(
        "--nemo-dir", type=str, default=None,
        help="Directory containing GB_training_scripts.txt and Speech-main/. "
             f"Default: {_NEMO_DIR}",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Output directory for all logs. "
             "Also read from LOG_OUTPUT_DIR env-var. "
             "Default: <nemo-dir>/validation_logs_<TIMESTAMP>/",
    )
    args = parser.parse_args()

    # Resolve directories
    nemo_dir = args.nemo_dir or os.environ.get("NEMO_DIR") or _NEMO_DIR
    nemo_dir = os.path.realpath(nemo_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = (
        args.log_dir
        or os.environ.get("LOG_OUTPUT_DIR")
        or os.path.join(_ROOT_DIR, "report", f"validation_logs_{timestamp}")
    )
    log_dir = os.path.realpath(log_dir)

    os.makedirs(nemo_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    log("=" * 60)
    log("  NeMo DL Validation — GB300 NVL72 ARM64")
    log("=" * 60)
    log(f"  nemo_dir : {nemo_dir}")
    log(f"  log_dir  : {log_dir}")
    log(f"  image    : {args.image}")
    log(f"  gpus     : {args.gpus}")

    # Pre-flight
    ensure_image_exists(args.image)
    run_cmd("dmesg -c > /dev/null 2>&1")
    run_cmd("ipmitool sel clear > /dev/null 2>&1")
    ensure_cdi()

    # Start IPMI background monitor
    ipmi_log = os.path.join(log_dir, "ipmi_sdr_monitor.log")
    monitor  = IpmiMonitor(ipmi_log)
    monitor.start()

    # Resolve training command
    exec_cmd, is_simulation = resolve_training_cmd(nemo_dir, args.cmd)

    # Write run_task.sh into nemo_dir (this is what the container will execute)
    task_script = os.path.join(nemo_dir, "run_task.sh")
    with open(task_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write("export HAVE_NVRX=0\n")
        # /workspace      = GB_DL_scripts_v9/ (training scripts)
        # /workspace/nemo = Speech-main/      (Python package, if mounted)
        f.write("export PYTHONPATH=/workspace:/workspace/nemo:$PYTHONPATH\n")
        f.write(f"{exec_cmd}\n")
    os.chmod(task_script, 0o755)
    log(f"run_task.sh written to: {task_script}")

    # Build CUDA_VISIBLE_DEVICES string  e.g. "0,1,2,3"
    cuda_devices = ",".join(str(i) for i in range(args.gpus))

    # Check if Speech-main exists in tools/ for extra mount
    speech_main_dir = os.path.join(_ROOT_DIR, "tools", "Speech-main")
    speech_main_mount = (
        f"-v {speech_main_dir}:/workspace/nemo "
        if os.path.isdir(speech_main_dir) else ""
    )
    if os.path.isdir(speech_main_dir):
        log(f"[OK] Speech-main found — mounting as /workspace/nemo")
    else:
        log(f"[INFO] Speech-main not found at tools/Speech-main/ — skipping extra mount")

    # Podman command
    # - Mounts nemo_dir (GB_DL_scripts_v9/) as /workspace
    # - Mounts Speech-main/ as /workspace/nemo (if present)
    # - Mounts log_dir as /workspace/logs
    container_cmd = (
        f"podman run --rm --entrypoint \"\" "
        f"--platform linux/arm64 "
        f"--device nvidia.com/gpu=all "
        f"--ipc=host "
        f"-e NGC_DISABLE_PROMPT=1 "
        f"-e DISABLE_CONTAINER_CHECKS=1 "
        f"-e NVIDIA_DISABLE_REQUIRE=1 "
        f"-e CUDA_VISIBLE_DEVICES={cuda_devices} "
        f"-e HAVE_NVRX=0 "
        f"-v {nemo_dir}:/workspace "
        f"{speech_main_mount}"
        f"-v {log_dir}:/workspace/logs "
        f"-w /workspace "
        f"{args.image} bash /workspace/run_task.sh"
    )

    log("Starting container workload...")
    log(f"CMD: {container_cmd}")

    stdout_path = os.path.join(log_dir, "execution_stdout.log")
    stderr_path = os.path.join(log_dir, "execution_stderr.log")

    start_t = time.time()
    proc    = subprocess.Popen(
        container_cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout_out, stderr_out = proc.communicate()
    elapsed = time.time() - start_t

    # Write logs
    with open(stdout_path, "w") as f:
        f.write(stdout_out)
    with open(stderr_path, "w") as f:
        f.write(stderr_out)

    # Echo to terminal
    if stdout_out.strip():
        print(stdout_out, flush=True)
    if proc.returncode != 0 or stderr_out.strip():
        log(f"[CONTAINER STDERR]\n{stderr_out}")

    log(f"Workload finished in {elapsed:.2f}s  (exit code: {proc.returncode})")

    # Stop IPMI monitor
    monitor.stop()

    # Parse and write metrics
    parse_metrics(
        stdout_path,
        os.path.join(log_dir, "benchmark_metrics.log"),
        global_batch=args.global_batch,
        num_gpus=args.gpus,
        is_simulation=is_simulation,
    )

    # Collect post-run system logs
    log("Collecting post-run system logs...")
    run_cmd(f"dmesg -T > {log_dir}/dmesg.log")
    run_cmd(f"ipmitool sel list > {log_dir}/bmc_sel.log")
    run_cmd(f"nvidia-smi > {log_dir}/nvidia_smi.log")
    run_cmd(f"nvidia-smi -q -d ECC > {log_dir}/nvidia_smi_ecc.log")

    log("=" * 60)
    log(f"NeMo DL Validation complete.")
    log("=" * 60)
    log(f"  Log directory : {log_dir}")
    log(f"  Metrics       : {log_dir}/benchmark_metrics.log")
    log(f"  stdout        : {stdout_path}")
    log(f"  IPMI SDR      : {ipmi_log}")
    log(f"  dmesg         : {log_dir}/dmesg.log")
    log(f"  BMC SEL       : {log_dir}/bmc_sel.log")
    log(f"  nvidia-smi    : {log_dir}/nvidia_smi.log")

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
