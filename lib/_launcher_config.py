"""
_launcher_config.py
-------------------
Central registry for all 10 benchmark tests.

Each entry defines:
  id            : int   — display number (1-10)
  key           : str   — short identifier used in log dir names
  name          : str   — display name shown in the menu
  category      : str   — section header
  script        : str   — path to the Python script, relative to ROOT_DIR
  prereqs       : list  — files/commands that must exist for this test to run
  configurable  : list  — names of parameters the user can set (shown in menu)
  defaults      : dict  — default values for all configurable parameters
  bg_capable    : bool  — True if the test can run in the background (Power Monitor)
"""

import os
import shutil

# ---------------------------------------------------------------------------
# ROOT_DIR: resolved at import time to the gb300_benchmark/ directory
# (this file lives in lib/, so ROOT_DIR is one level up)
# ---------------------------------------------------------------------------
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR   = os.path.join(ROOT_DIR, "lib")
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")


# ---------------------------------------------------------------------------
# Helper: check whether a prereq is satisfied
# ---------------------------------------------------------------------------

def _file_ok(rel_path: str) -> bool:
    """Return True if the file exists under TOOLS_DIR (or as absolute)."""
    if os.path.isabs(rel_path):
        return os.path.isfile(rel_path)
    return os.path.isfile(os.path.join(TOOLS_DIR, rel_path))


def _dir_ok(rel_path: str) -> bool:
    """Return True if the directory exists under TOOLS_DIR (or as absolute)."""
    if os.path.isabs(rel_path):
        return os.path.isdir(rel_path)
    return os.path.isdir(os.path.join(TOOLS_DIR, rel_path))


def _cmd_ok(cmd: str) -> bool:
    """Return True if a system command is on PATH."""
    return shutil.which(cmd) is not None


# ---------------------------------------------------------------------------
# Prereq specs — each entry is a (check_fn, arg, human_readable_description)
# ---------------------------------------------------------------------------

def _check_prereqs(prereqs: list) -> list[str]:
    """
    Evaluate a list of prereq specs.
    Returns a list of failure reason strings (empty = all OK).
    """
    failures = []
    for spec in prereqs:
        kind, arg, desc = spec
        if kind == "file" and not _file_ok(arg):
            failures.append(f"{desc} not found")
        elif kind == "dir" and not _dir_ok(arg):
            failures.append(f"{desc} not found")
        elif kind == "cmd" and not _cmd_ok(arg):
            failures.append(f"'{arg}' not on PATH")
    return failures


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = [
    # ------------------------------------------------------------------
    # 1  GPU Stream
    # ------------------------------------------------------------------
    {
        "id":          1,
        "key":         "GPU_stream",
        "name":        "GPU Stream",
        "category":    "GPU Compute & Memory",
        "script":      os.path.join(LIB_DIR, "run_GPU_stream.py"),
        "prereqs": [
            ("file", "stream_vectorized_float_benchmark",
             "tools/stream_vectorized_float_benchmark"),
        ],
        "configurable": [],
        "defaults":     {},
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 2  Peak TOPS
    # ------------------------------------------------------------------
    {
        "id":          2,
        "key":         "peak_tops",
        "name":        "Peak TOPS",
        "category":    "GPU Compute & Memory",
        "script":      os.path.join(LIB_DIR, "run_peak_tops.py"),
        "prereqs": [
            ("file", "peakTops",          "tools/peakTops"),
            ("file", "run_peaktops.sh",   "tools/run_peaktops.sh"),
        ],
        "configurable": [],
        "defaults":     {},
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 3  GEMM Bench (multi-precision cuBLAS)
    # ------------------------------------------------------------------
    {
        "id":          3,
        "key":         "GEMM_bench",
        "name":        "GEMM Bench",
        "category":    "GPU Compute & Memory",
        "script":      os.path.join(LIB_DIR, "run_GEMM_bench.py"),
        "prereqs": [
            ("file", "cublasMatmulBench", "tools/cublasMatmulBench"),
        ],
        "configurable": [],
        "defaults":     {},
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 4  FP4 GEMM-MemRead
    # ------------------------------------------------------------------
    {
        "id":          4,
        "key":         "FP4_GEMM",
        "name":        "FP4 GEMM-MemRead",
        "category":    "GPU Compute & Memory",
        "script":      os.path.join(LIB_DIR, "run_FP4_GEMM.py"),
        "prereqs": [
            ("dir",  "gemm-memread",                    "tools/gemm-memread/"),
            ("file", "gemm-memread/build/generic_gemm_benchmark",
             "tools/gemm-memread/build/generic_gemm_benchmark"),
            ("file", "gemm-memread/configs/commandlines.json",
             "tools/gemm-memread/configs/commandlines.json"),
        ],
        "configurable": [],
        "defaults":     {},
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 5  NCCL Loopback
    # ------------------------------------------------------------------
    {
        "id":          5,
        "key":         "NCCL",
        "name":        "NCCL Loopback",
        "category":    "GPU Interconnect",
        "script":      os.path.join(LIB_DIR, "nccl_L10_loopback_v2.py"),
        "prereqs": [
            ("file", "nccl-build/all_reduce_perf",  "tools/nccl-build/all_reduce_perf"),
            ("file", "nccl-build/all_gather_perf",  "tools/nccl-build/all_gather_perf"),
            ("file", "nccl-build/alltoall_perf",    "tools/nccl-build/alltoall_perf"),
        ],
        "configurable": ["iters", "msg_range"],
        "defaults": {
            "iters":     20,
            "msg_range": "8g",     # "8g" = 8G only (quick)  |  "full" = 2~8G full sweep
        },
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 6  RDMA Loopback IPv4
    # ------------------------------------------------------------------
    {
        "id":          6,
        "key":         "RDMA_IPv4",
        "name":        "RDMA Loopback IPv4",
        "category":    "Network / RDMA",
        "script":      os.path.join(LIB_DIR, "rdma_test_ipv4_v2.py"),
        "prereqs": [
            ("cmd", "ib_read_bw",  "ib_read_bw"),
            ("cmd", "ib_send_bw",  "ib_send_bw"),
            ("cmd", "ib_write_bw", "ib_write_bw"),
        ],
        "configurable": ["duration", "test_type"],
        "defaults": {
            "duration":  30,
            "test_type": "all",    # "all" | "read" | "send" | "write"
        },
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 7  RDMA Loopback IPv6
    # ------------------------------------------------------------------
    {
        "id":          7,
        "key":         "RDMA_IPv6",
        "name":        "RDMA Loopback IPv6",
        "category":    "Network / RDMA",
        "script":      os.path.join(LIB_DIR, "rdma_test_ipv6_v3.py"),
        "prereqs": [
            ("cmd", "ib_read_bw",  "ib_read_bw"),
            ("cmd", "ib_send_bw",  "ib_send_bw"),
            ("cmd", "ib_write_bw", "ib_write_bw"),
        ],
        "configurable": ["duration", "test_type"],
        "defaults": {
            "duration":  30,
            "test_type": "all",
        },
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 8  1G NIC iPerf
    # ------------------------------------------------------------------
    {
        "id":          8,
        "key":         "1G_iPerf",
        "name":        "1G NIC iPerf",
        "category":    "Network / RDMA",
        "script":      os.path.join(LIB_DIR, "1G_nic_iperf.py"),
        "prereqs": [
            ("cmd", "iperf", "iperf"),
        ],
        "configurable": ["client_ip", "mode", "duration"],
        "defaults": {
            "client_ip": "10.20.2.130",
            "mode":      "bidirectional",   # "bidirectional" | "unidirectional"
            "duration":  3600,
        },
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 9  NIC PCIe Health
    # ------------------------------------------------------------------
    {
        "id":          9,
        "key":         "NIC_health",
        "name":        "NIC PCIe Health",
        "category":    "Hardware Health & Monitoring",
        "script":      os.path.join(LIB_DIR, "nic_info_v3.py"),
        "prereqs": [
            ("cmd", "mst",       "mst (MFT)"),
            ("cmd", "mlxconfig", "mlxconfig (MFT)"),
            ("cmd", "lspci",     "lspci"),
        ],
        "configurable": [],
        "defaults":     {},
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 10  Power Monitor
    # ------------------------------------------------------------------
    {
        "id":          10,
        "key":         "power_monitor",
        "name":        "Power Monitor",
        "category":    "Hardware Health & Monitoring",
        "script":      os.path.join(LIB_DIR, "monitor_power1_average_v1.py"),
        "prereqs":     [],   # pure sysfs — no external binary needed
        "configurable": ["bg_mode", "csv_log"],
        "defaults": {
            "bg_mode":  True,
            "duration": 0,
            "csv_log":  False,
            "interval": 0.1,   # sampling interval in seconds (min 0.1)
        },
        "bg_capable":   True,
    },

    # ------------------------------------------------------------------
    # 11  NeMo DL Validation
    # ------------------------------------------------------------------
    {
        "id":          11,
        "key":         "nemo_validation",
        "name":        "NeMo DL Validation",
        "category":    "DL Training Validation",
        "script":      os.path.join(LIB_DIR, "run_nemo_validation_v3.py"),
        "prereqs": [
            ("cmd", "podman",      "podman"),
            ("cmd", "nvidia-ctk",  "nvidia-ctk (NVIDIA Container Toolkit)"),
            ("cmd", "ipmitool",    "ipmitool"),
        ],
        # GB_training_scripts.txt absence is a soft-warning (falls back to
        # simulation mode), NOT a hard prereq that blocks selection.
        "configurable": ["image", "gpus", "cmd"],
        "defaults": {
            "image": "nvcr.io/nvidia/pytorch:24.07-py3",
            "gpus":  4,
            "cmd":   "",   # empty = read from nemo/GB_training_scripts.txt
        },
        "bg_capable":   False,
    },

    # ------------------------------------------------------------------
    # 12  NVBandwidth Loopback
    # ------------------------------------------------------------------
    {
        "id":          12,
        "key":         "nvbandwidth_loopback",
        "name":        "NVBandwidth Loopback",
        "category":    "GPU Compute & Memory",
        "script":      os.path.join(LIB_DIR, "nvbandwidth_loopback.py"),
        "prereqs": [
            ("file", "nvbandwidth", "tools/nvbandwidth"),
        ],
        "configurable": ["iters"],
        "defaults": {
            "iters": 3,
        },
        "bg_capable":   False,
    },
]

# ---------------------------------------------------------------------------
# Public helpers used by the launcher and runner
# ---------------------------------------------------------------------------

def get_test(test_id: int) -> dict | None:
    """Return the test dict for the given id, or None."""
    for t in TESTS:
        if t["id"] == test_id:
            return t
    return None


def evaluate_availability() -> dict[int, list[str]]:
    """
    Run prereq checks for all tests.
    Returns {test_id: [failure_reason, ...]}  — empty list means available.
    """
    result = {}
    for t in TESTS:
        result[t["id"]] = _check_prereqs(t["prereqs"])
    return result


def abs_tools(rel: str) -> str:
    """Resolve a TOOLS_DIR-relative path to absolute."""
    return os.path.join(TOOLS_DIR, rel)


def abs_lib(rel: str) -> str:
    """Resolve a LIB_DIR-relative path to absolute."""
    return os.path.join(LIB_DIR, rel)
