"""
_launcher_runner.py
-------------------
Execution engine for gb300_launcher.py.

Responsibilities:
  - Create the run-scoped report directory tree
  - Set LOG_OUTPUT_DIR env-var so each sub-script knows where to write logs
  - Build the exact subprocess command for each selected test
  - Stream sub-script stdout/stderr live to the terminal
  - Record per-test timing and exit code
  - Handle background execution for Power Monitor
  - Respond to Ctrl-C gracefully (ask user whether to continue)
  - Write the final 00_summary.log
"""

import os
import sys
import signal
import subprocess
import time
from datetime import datetime

from _launcher_config import ROOT_DIR, TOOLS_DIR, LIB_DIR, get_test

# ---------------------------------------------------------------------------
# ANSI colour helpers (no emoji, pure ASCII)
# ---------------------------------------------------------------------------
G  = "\033[92m"   # bright green   — PASS / banner
R  = "\033[91m"   # bright red     — FAIL / error
Y  = "\033[93m"   # yellow         — warning / prompt
C  = "\033[96m"   # cyan           — info / log path
DIM= "\033[90m"   # dark grey      — secondary text
W  = "\033[97m"   # bright white   — primary text
RS = "\033[0m"    # reset

SEP  = "=" * 66
SEP2 = "-" * 66


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _print(msg: str = ""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Report directory setup
# ---------------------------------------------------------------------------

LOG_DIR_NAMES = {
    1:  "01_GPU_stream",
    2:  "02_peak_tops",
    3:  "03_GEMM_bench",
    4:  "04_FP4_GEMM",
    5:  "05_NCCL",
    6:  "06_RDMA_IPv4",
    7:  "07_RDMA_IPv6",
    8:  "08_1G_iPerf",
    9:  "09_NIC_health",
    10: "10_power_monitor",
    11: "11_nemo_validation",
    12: "12_nvbandwidth_loopback",
}


def create_run_dir(base_report_dir: str, selected_ids: list[int] | None = None) -> str:
    """
    Create report/run_<TIMESTAMP>/ and sub-directories only for selected tests.
    If selected_ids is None, create all sub-directories (backward compat).
    Returns the absolute path to the run directory.
    """
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_report_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    ids_to_create = selected_ids if selected_ids is not None else list(LOG_DIR_NAMES.keys())
    for tid in ids_to_create:
        if tid in LOG_DIR_NAMES:
            os.makedirs(os.path.join(run_dir, LOG_DIR_NAMES[tid]), exist_ok=True)
    return run_dir


def test_log_dir(run_dir: str, test_id: int) -> str:
    return os.path.join(run_dir, LOG_DIR_NAMES[test_id])


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(test: dict, params: dict, run_dir: str) -> list[str]:
    """
    Construct the subprocess argv list for a given test + user parameters.
    All paths are absolute.
    """
    tid    = test["id"]
    script = test["script"]
    cmd    = [sys.executable, script]

    if tid == 5:   # NCCL Loopback
        build_path = os.path.join(TOOLS_DIR, "nccl-build")
        cmd += ["--build", build_path, "--iters", str(params.get("iters", 20))]
        if params.get("msg_range") == "8g":
            # 8G only: set both begin and end to 8G for a quick single-size run
            cmd += ["--msg-begin", "8G", "--msg-end", "8G"]
        else:
            # Full sweep: 2B ~ 8G
            cmd += ["--msg-begin", "2", "--msg-end", "8G"]

    elif tid == 6:  # RDMA IPv4
        out_dir = test_log_dir(run_dir, tid)
        cmd += ["--output-dir", out_dir,
                "--duration",   str(params.get("duration", 30))]
        tt = params.get("test_type", "all")
        if tt != "all":
            cmd += ["--test", tt]

    elif tid == 7:  # RDMA IPv6
        out_dir = test_log_dir(run_dir, tid)
        cmd += ["--output-dir", out_dir,
                "--duration",   str(params.get("duration", 30))]
        tt = params.get("test_type", "all")
        if tt != "all":
            cmd += ["--test", tt]

    elif tid == 8:  # 1G NIC iPerf
        mode = params.get("mode", "bidirectional")
        cmd += [f"--{mode}",
                "--client-ip", params.get("client_ip", "10.20.2.130"),
                "--duration",  str(params.get("duration", 3600))]

    elif tid == 10:  # Power Monitor
        interval = str(params.get("interval", 1.0))
        cmd += [interval]
        if params.get("csv_log", False):
            cmd += ["--log"]

    elif tid == 11:  # NeMo DL Validation
        nemo_dir = os.path.join(TOOLS_DIR, "GB_DL_scripts_v9")
        cmd += [
            "--nemo-dir",    nemo_dir,
            "--log-dir",     test_log_dir(run_dir, tid),
            "--image",       params.get("image", "nvcr.io/nvidia/pytorch:24.07-py3"),
            "--gpus",        str(params.get("gpus", 4)),
        ]
        user_cmd = params.get("cmd", "").strip()
        if user_cmd:
            cmd += ["--cmd", user_cmd]

    elif tid == 12:  # NVBandwidth Loopback
        nvbw_bin = os.path.join(TOOLS_DIR, "nvbandwidth")
        cmd += [
            "--nvbw-path", nvbw_bin,
            "--iters",     str(params.get("iters", 3)),
            "--log-dir",   test_log_dir(run_dir, tid),
        ]

    return cmd


def build_env(test: dict, run_dir: str) -> dict:
    """
    Return a copy of os.environ with LOG_OUTPUT_DIR set to the test's log
    directory, plus any test-specific env vars.
    """
    env = os.environ.copy()
    env["LOG_OUTPUT_DIR"] = test_log_dir(run_dir, test["id"])

    if test["id"] == 4:   # FP4 GEMM-MemRead — set GEMM_DIR
        gemm_dir = os.path.join(TOOLS_DIR, "gemm-memread")
        env["GEMM_DIR"]  = gemm_dir
        env["CUDA_HOME"] = env.get("CUDA_HOME", "/usr/local/cuda")

    if test["id"] == 11:  # NeMo DL Validation — pass nemo dir via env
        env["NEMO_DIR"] = os.path.join(TOOLS_DIR, "GB_DL_scripts_v9")

    return env


# ---------------------------------------------------------------------------
# Single-test execution
# ---------------------------------------------------------------------------

def _print_test_header(seq: int, total: int, test: dict, params: dict):
    _print()
    _print(f"{W}{SEP}{RS}")
    _print(f"{W}  [{seq}/{total}]  {test['name']}{RS}")
    if params:
        summary_parts = []
        for k, v in params.items():
            summary_parts.append(f"{k}={v}")
        _print(f"{DIM}  Params: {', '.join(summary_parts)}{RS}")
    _print(f"{DIM}  Started: {_ts()}{RS}")
    _print(f"{W}{SEP}{RS}")


def _print_test_footer(test: dict, elapsed: float, rc: int, log_dir: str):
    status = f"{G}  [PASS]{RS}" if rc == 0 else f"{R}  [FAIL]{RS}"
    _print(f"{W}{SEP}{RS}")
    _print(f"{status}  {test['name']}  elapsed: {_elapsed(elapsed)}  exit: {rc}")
    _print(f"{C}  Log --> {log_dir}{RS}")
    _print(f"{W}{SEP}{RS}")


_interrupted = False   # set by SIGINT handler during test run


def _run_single(test: dict, params: dict, run_dir: str,
                seq: int, total: int, dry_run: bool = False) -> dict:
    """
    Execute one test.  Returns a result dict:
      {id, name, start, end, elapsed, rc, log_dir, params, skipped}
    """
    global _interrupted
    _interrupted = False

    log_dir = test_log_dir(run_dir, test["id"])
    _print_test_header(seq, total, test, params)

    if dry_run:
        cmd = build_command(test, params, run_dir)
        _print(f"{Y}  [dry-run] would run: {' '.join(cmd)}{RS}")
        return {
            "id": test["id"], "name": test["name"],
            "start": _ts(), "end": _ts(), "elapsed": 0,
            "rc": 0, "log_dir": log_dir, "params": params, "skipped": True,
        }

    cmd = build_command(test, params, run_dir)
    env = build_env(test, run_dir)

    # Temporary SIGINT handler — catch Ctrl-C during the subprocess
    original_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(sig, frame):
        global _interrupted
        _interrupted = True
        if proc and proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT, _sigint_handler)

    start_time = time.monotonic()
    start_ts   = _ts()
    proc       = None
    rc         = -1

    # Power Monitor (tid=10) uses ANSI cursor-positioning TUI (ESC[2J + ESC[row;colH).
    # Capturing its stdout via PIPE strips the escape codes and renders blank.
    # Run it with the terminal inherited (no PIPE) so the TUI displays correctly.
    _is_tui = (test["id"] == 10)

    try:
        if _is_tui:
            proc = subprocess.Popen(cmd, env=env)
        else:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
        rc = proc.returncode
    except Exception as exc:
        _print(f"{R}  [ERROR] Failed to launch {test['name']}: {exc}{RS}")
        rc = -1
    finally:
        elapsed = time.monotonic() - start_time
        signal.signal(signal.SIGINT, original_handler)

    _print_test_footer(test, elapsed, rc, log_dir)

    return {
        "id": test["id"], "name": test["name"],
        "start": start_ts, "end": _ts(), "elapsed": elapsed,
        "rc": rc, "log_dir": log_dir, "params": params, "skipped": False,
    }


# ---------------------------------------------------------------------------
# Background Power Monitor
# ---------------------------------------------------------------------------

def start_background_monitor(test: dict, params: dict,
                              run_dir: str) -> subprocess.Popen:
    """Launch Power Monitor as a background process. Returns the Popen handle."""
    cmd = build_command(test, params, run_dir)
    env = build_env(test, run_dir)
    _print(f"{DIM}  [bg] Power Monitor started (PID: ...){RS}")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _print(f"{DIM}  [bg] Power Monitor PID: {proc.pid}{RS}")
    return proc


def stop_background_monitor(proc: subprocess.Popen):
    """Send SIGINT to the background monitor process and wait."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run_all(selected: list[dict], user_params: dict, run_dir: str,
            dry_run: bool = False) -> list[dict]:
    """
    Execute all selected tests in order.

    selected    : list of test dicts (in execution order)
    user_params : {test_id: {param_name: value}}
    run_dir     : absolute path to the run's report directory

    Returns a list of result dicts.
    """
    global _interrupted
    _interrupted = False
    results = []

    # Separate Power Monitor (bg) from foreground tests
    fg_tests  = [t for t in selected if not t["bg_capable"]]
    bg_test   = next((t for t in selected if t["bg_capable"]), None)
    bg_proc   = None
    bg_params = user_params.get(bg_test["id"], {}) if bg_test else {}

    total = len(selected)

    # Catch outer Ctrl-C (outside of _run_single's inner handler)
    def _outer_sigint(sig, frame):
        _print(f"\n{Y}  Ctrl-C received outside of test run.  Aborting.{RS}")
        if bg_proc:
            stop_background_monitor(bg_proc)
        sys.exit(1)

    signal.signal(signal.SIGINT, _outer_sigint)

    # Start background monitor before any foreground test
    if bg_test and fg_tests and not bg_test.get("bg_mode_solo"):
        bg_params["bg_mode"] = True
        bg_proc = start_background_monitor(bg_test, bg_params, run_dir)

    # Run foreground tests
    for seq, test in enumerate(fg_tests, start=1):
        params = user_params.get(test["id"], {})
        result = _run_single(test, params, run_dir, seq, total, dry_run)
        results.append(result)

        # Handle failure
        if result["rc"] != 0 and not result["skipped"]:
            print(f"{Y}  Test failed (exit {result['rc']}).  "
                  f"Continue to next test? [y/N]: {RS}", end="", flush=True)
            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer != "y":
                _print(f"{R}  Aborting remaining tests.{RS}")
                break

        # Handle mid-test Ctrl-C (flag set by inner handler)
        if _interrupted:
            print(f"{Y}  Test was interrupted.  Continue to next test? [y/N]: {RS}",
                  end="", flush=True)
            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            _interrupted = False
            if answer != "y":
                _print(f"{R}  Aborting remaining tests.{RS}")
                break

    # Power Monitor runs solo if no foreground tests were selected
    if bg_test and not fg_tests:
        bg_params["bg_mode"] = False
        result = _run_single(bg_test, bg_params, run_dir,
                             seq=1, total=1, dry_run=dry_run)
        results.append(result)

    # Record background monitor result
    if bg_proc:
        stop_background_monitor(bg_proc)
        results.append({
            "id":      bg_test["id"],
            "name":    bg_test["name"],
            "start":   "(background)",
            "end":     _ts(),
            "elapsed": 0,
            "rc":      bg_proc.returncode if bg_proc.returncode is not None else 0,
            "log_dir": test_log_dir(run_dir, bg_test["id"]),
            "params":  bg_params,
            "skipped": False,
        })

    return results


# ---------------------------------------------------------------------------
# Summary report writer
# ---------------------------------------------------------------------------

def write_summary(results: list[dict], run_dir: str,
                  platform_info: str = ""):
    """Write 00_summary.log to run_dir."""
    summary_path = os.path.join(run_dir, "00_summary.log")

    passed  = sum(1 for r in results if r["rc"] == 0 and not r["skipped"])
    failed  = sum(1 for r in results if r["rc"] != 0 and not r["skipped"])
    skipped = sum(1 for r in results if r["skipped"])
    total   = len(results)

    lines = [
        SEP,
        "  GB300 Benchmark Launcher -- Execution Summary",
        f"  Completed : {_ts()}",
    ]
    if platform_info:
        lines.append(f"  Platform  : {platform_info}")
    lines += [
        f"  Report    : {run_dir}",
        SEP,
        "",
    ]

    for i, r in enumerate(results, start=1):
        status = "PASS" if r["rc"] == 0 else ("SKIP" if r["skipped"] else "FAIL")
        mark   = "[PASS]" if r["rc"] == 0 else ("[SKIP]" if r["skipped"] else "[FAIL]")
        param_str = ""
        if r["params"]:
            param_str = "  params: " + ", ".join(
                f"{k}={v}" for k, v in r["params"].items()
            )
        lines += [
            f"  [{i}]  {r['name']}",
            f"       {r['start']} --> {r['end']}"
            + (f"  ({_elapsed(r['elapsed'])})" if r["elapsed"] else ""),
            f"       exit: {r['rc']}  {mark}",
        ]
        if param_str:
            lines.append(f"      {param_str}")
        lines += [
            f"       Log --> {r['log_dir']}",
            "",
        ]

    lines += [
        SEP2,
        f"  Total: {total} test(s)   PASS: {passed}   FAIL: {failed}   SKIP: {skipped}",
        SEP,
        "",
    ]

    body = "\n".join(lines)

    # Write to file
    with open(summary_path, "w") as f:
        f.write(body)

    # Also print to terminal
    _print()
    _print(body)
    _print(f"{C}  Summary written --> {summary_path}{RS}")


# ---------------------------------------------------------------------------
# Platform info helper
# ---------------------------------------------------------------------------

def get_platform_info() -> str:
    """Return a short one-liner describing the current platform."""
    import platform
    uname = platform.uname()
    # Try to get NVIDIA driver version
    driver = ""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip().splitlines()
        if out:
            driver = f" / Driver {out[0]}"
    except Exception:
        pass

    return f"{uname.machine} / {uname.system} {uname.release}{driver}"
