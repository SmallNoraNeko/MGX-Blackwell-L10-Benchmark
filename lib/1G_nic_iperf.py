#!/usr/bin/env python3
"""
1G NIC iPerf Network Performance Test  (Test ID: 1002935 V8.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Architecture:
  SUT    (su1) = system10  : acts as iperf server (port 5002) AND client
  Client (su2) = system13  : acts as iperf server (port 5001) AND client

Test Modes:
  --unidirectional   One-way:  su1 → su2  (su1=client, su2=server port 5001)
  --bidirectional    Two-way:  su1 ⇄ su2  (su1 client→su2:5001 + su2 client→su1:5002)

Log files collected per run (stored under <script_dir>/1G_iperf_<timestamp>/):
  before/
    before_ifconfig_<iface>.log        ifconfig <iface>
    before_ethtool-S_<iface>.log       ethtool -S <iface>  (NIC counters)
    before_ethtool_<iface>.log         ethtool <iface> + ethtool -i <iface>
  iperf/
    su1_client_undirectional.log       Unidirectional: SUT client log
    su2_server_undirectional.log       Unidirectional: Client-side server log
    su1_server_Bidirectional.log       Bidirectional: SUT server log
    su1_client_Bidirection.log         Bidirectional: SUT client log
    su2_server_Bidirection.log         Bidirectional: Client-side server log
    su2_client_Bidirection.log         Bidirectional: Client-side client log
  after/
    after_ifconfig_<iface>.log
    after_ethtool-S_<iface>.log
    after_ethtool_<iface>.log
  logs/
    dmesg_log_<timestamp>.txt
    syslog_log_<timestamp>.txt
    ipmi_sel_log_<timestamp>.txt

Usage (run on SUT / system10 as root):
  sudo ./1G_nic_iperf.py --unidirectional
  sudo ./1G_nic_iperf.py --bidirectional
  sudo ./1G_nic_iperf.py --unidirectional --duration 3600
  sudo ./1G_nic_iperf.py --bidirectional --client-ip 10.20.2.130 --client-user root

Prerequisites:
  1. Passwordless SSH must be configured from SUT to Client
     (run: ssh-copy-id root@<client-ip>)
  2. iperf v2 must be installed on both machines
     (run: apt install iperf)
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
import re

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — modify this section if your environment differs
# ══════════════════════════════════════════════════════════════════════════════

# iperf v2 test parameters (matching observed reference logs)
IPERF_THREADS  = 10          # number of parallel streams (-P)
IPERF_DURATION = 3600        # test duration in seconds (-t)
IPERF_INTERVAL = 5           # reporting interval in seconds (-i)
IPERF_TCP_WIN  = "1000K"     # TCP window size (-w); kernel may clamp to 416K

# Port assignments
IPERF_PORT_UNI = 5001        # unidirectional: su2 server port
IPERF_PORT_BI1 = 5001        # bidirectional: su2 server port (su1 client → su2)
IPERF_PORT_BI2 = 5002        # bidirectional: su1 server port (su2 client → su1)

# SSH connection to Client (su2 / system13)
# These defaults can be overridden at runtime with --client-ip / --client-user
CLIENT_IP   = "10.20.2.130"
CLIENT_USER = "root"
# Disable host key checking and set connection timeout for non-interactive use
SSH_OPTS    = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# ══════════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════════════════════

# Resolve the directory where this script lives so logs are stored alongside it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Timestamp used for the output folder and log file names
TS         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Root output directory for this test run
OUT_DIR    = os.path.join(SCRIPT_DIR, f"1G_iperf_{TS}")

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def ts_now():
    """Return the current timestamp string used in log file names."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def banner(msg):
    """Print a section header to the terminal."""
    print(f"\n{'='*66}\n  {msg}\n{'='*66}")


def run(cmd, timeout=60):
    """
    Execute a shell command on the local machine.

    Returns:
        (stdout, stderr, returncode)
    On timeout, returncode is set to -1.
    """
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, timeout=timeout
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT after {timeout}s", -1
    except Exception as e:
        return "", str(e), -1


def ssh(client_ip, client_user, remote_cmd, timeout=60):
    """
    Execute a command on the remote machine (su2) via SSH.

    Returns:
        (stdout, stderr, returncode)
    """
    cmd = f'ssh {SSH_OPTS} {client_user}@{client_ip} "{remote_cmd}"'
    return run(cmd, timeout=timeout)


def write_file(path, content):
    """
    Write content to the specified file path.
    Parent directories are created automatically if they do not exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"    -> {path}")


def popen_local(cmd):
    """
    Launch a background process on the local machine.
    stdout and stderr are merged into a single pipe.
    Used when multiple iperf processes need to run concurrently.
    """
    return subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT LOCAL 1G NIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def detect_1g_iface() -> str | None:
    """
    Automatically locate the local 1G NIC interface name.

    Detection priority:
      1. en* interfaces reported at Speed: 1000Mb/s AND using the igb driver
         (Intel I210 — the expected NIC on GB300 systems)
      2. Fallback: any en* interface at Speed: 1000Mb/s regardless of driver

    Returns the interface name string, or None if nothing is found.
    """
    # List all network interfaces
    out, _, _ = run("ip -br link show")
    candidates = []
    for line in out.splitlines():
        cols = line.split()
        # Consider only Ethernet interfaces (en*); skip lo, bond, etc.
        if cols and cols[0].startswith("en"):
            candidates.append(cols[0])

    # First pass: prefer igb driver (Intel I210 1G NIC)
    for iface in candidates:
        eth_out, _, _ = run(f"ethtool {iface} 2>/dev/null")
        if "Speed: 1000Mb/s" in eth_out or "1000baseT" in eth_out:
            drv_out, _, _ = run(f"ethtool -i {iface} 2>/dev/null")
            if "igb" in drv_out or "i210" in drv_out.lower():
                return iface

    # Second pass: any 1G interface
    for iface in candidates:
        eth_out, _, _ = run(f"ethtool {iface} 2>/dev/null")
        if "Speed: 1000Mb/s" in eth_out:
            return iface

    return None


def get_iface_ip(iface: str) -> str | None:
    """
    Retrieve the first IPv4 address assigned to the given interface.
    Returns the IP string, or None if the interface has no IPv4 address.
    """
    out, _, _ = run(f"ip -4 addr show dev {iface} 2>/dev/null")
    m = re.search(r"inet\s+([\d.]+)/", out)
    return m.group(1) if m else None


# ══════════════════════════════════════════════════════════════════════════════
# CLEAR LOGS
# ══════════════════════════════════════════════════════════════════════════════

def clear_logs():
    """
    Clear all system logs before the test begins so that only events
    generated during this test run are captured afterwards.

    Items cleared:
      - dmesg ring buffer     (dmesg -C)
      - /var/log/syslog       (echo " " > /var/log/syslog)
      - IPMI System Event Log (ipmitool sel clear)

    Non-fatal: a warning is printed if a command fails (e.g. ipmitool
    not installed), but execution continues.
    """
    banner("Clearing logs (dmesg / syslog / IPMI SEL)")

    cmds = [
        ("dmesg -C",                   "dmesg ring buffer"),
        ('echo " " > /var/log/syslog', "syslog"),
        ("ipmitool sel clear",          "IPMI SEL"),
    ]
    for cmd, label in cmds:
        _, err, rc = run(cmd, timeout=30)
        if rc == 0:
            print(f"  [{label}] cleared OK")
        else:
            print(f"  [{label}] WARN (rc={rc}): {err.strip()}")


# ══════════════════════════════════════════════════════════════════════════════
# NIC SNAPSHOT  (before / after)
# ══════════════════════════════════════════════════════════════════════════════

def snapshot(label: str, iface: str):
    """
    Capture a NIC state snapshot for the given interface.

    Args:
        label : "before" or "after" — determines the sub-directory and
                file name prefix.
        iface : network interface name (e.g. enP5p9s0)

    Files written:
      1. <label>_ifconfig_<iface>.log   — IP address, packet counters, errors
      2. <label>_ethtool-S_<iface>.log  — hardware-level NIC statistics
                                          (rx/tx packets, bytes, error counters)
      3. <label>_ethtool_<iface>.log    — link speed, FEC, Auto-Neg status,
                                          driver version, firmware version,
                                          PCI bus info  (ethtool + ethtool -i
                                          merged into one file)
    """
    banner(f"[{label.upper()}] NIC Snapshot  —  interface: {iface}")
    base = os.path.join(OUT_DIR, label)   # e.g. 1G_iperf_xxx/before/

    # 1. ifconfig — high-level interface statistics
    out, err, rc = run(f"ifconfig {iface}")
    write_file(
        os.path.join(base, f"{label}_ifconfig_{iface}.log"),
        out if rc == 0 else f"ERROR: {err}"
    )

    # 2. ethtool -S — low-level NIC hardware counters
    out, err, rc = run(f"ethtool -S {iface}")
    write_file(
        os.path.join(base, f"{label}_ethtool-S_{iface}.log"),
        out if rc == 0 else f"ERROR: {err}"
    )

    # 3. ethtool (link info) + ethtool -i (driver/FW info) — merged into one file
    #    Merging avoids an extra file while keeping all relevant NIC metadata together
    out_link,   _, _ = run(f"ethtool {iface}")
    out_driver, _, _ = run(f"ethtool -i {iface}")
    write_file(
        os.path.join(base, f"{label}_ethtool_{iface}.log"),
        out_link + "\n" + out_driver
    )

    print(f"  [{label}] snapshot complete -> {base}/")


# ══════════════════════════════════════════════════════════════════════════════
# COLLECT SYSTEM LOGS  (after test)
# ══════════════════════════════════════════════════════════════════════════════

def collect_system_logs():
    """
    Collect system logs after the test completes.

    Because clear_logs() was called before the test, the content captured
    here reflects only events that occurred during this test run.

    Files written to <OUT_DIR>/logs/:
      dmesg_log_<timestamp>.txt       — full dmesg output
      syslog_log_<timestamp>.txt      — journalctl (preferred) or /var/log/syslog
      ipmi_sel_log_<timestamp>.txt    — IPMI System Event Log entries
    """
    banner("Collecting system logs")
    base  = os.path.join(OUT_DIR, "logs")
    stamp = ts_now()   # timestamp matching the original log file naming convention

    cmds = [
        (
            f"dmesg_log_{stamp}.txt",
            "dmesg",
            "dmesg"
        ),
        (
            f"syslog_log_{stamp}.txt",
            # Prefer journalctl; fall back to /var/log/syslog on older systems
            "journalctl --no-pager -n 50000 2>/dev/null "
            "|| cat /var/log/syslog 2>/dev/null "
            "|| echo 'syslog not found'",
            "syslog / journal"
        ),
        (
            f"ipmi_sel_log_{stamp}.txt",
            "ipmitool sel list 2>/dev/null || echo 'ipmitool not available'",
            "IPMI SEL"
        ),
    ]

    for fname, cmd, label in cmds:
        print(f"  [{label}] collecting ...")
        out, err, _ = run(cmd, timeout=120)
        write_file(
            os.path.join(base, fname),
            out or f"(no output)\nSTDERR: {err}"
        )

    print(f"  System logs collected -> {base}/")


# ══════════════════════════════════════════════════════════════════════════════
# UNIDIRECTIONAL TEST   su1 → su2
# ══════════════════════════════════════════════════════════════════════════════

def run_unidirectional(su1_ip, su2_ip, client_ip, client_user, duration, interval):
    """
    Run a one-way iperf test: SUT (su1) as client, Client (su2) as server.

    Flow:
      1. Start iperf server on su2 in the background via SSH (port 5001)
      2. Run iperf client on su1, connecting to su2:5001
      3. After client finishes, retrieve the server log from su2 via SSH

    Log files produced (names match original reference logs):
      su1_client_undirectional.log
      su2_server_undirectional.log
    """
    banner(f"iperf Unidirectional  su1({su1_ip}) → su2({su2_ip})  port {IPERF_PORT_UNI}")
    base = os.path.join(OUT_DIR, "iperf")
    os.makedirs(base, exist_ok=True)

    # File name matches original reference log naming convention
    mode = "undirectional"

    # ── Step 1: Start iperf server on su2 ────────────────────────────────────
    print(f"\n  [1/3] Starting iperf server on su2 ({su2_ip}) port {IPERF_PORT_UNI} ...")
    srv_log_remote = f"/tmp/su2_server_{mode}.log"
    # Kill any leftover iperf process, then start server with nohup so it
    # survives if the SSH control connection is closed
    ssh(
        client_ip, client_user,
        f"pkill iperf 2>/dev/null; sleep 1; "
        f"nohup iperf -s -p {IPERF_PORT_UNI} -i {interval} "
        f"> {srv_log_remote} 2>&1 &",
        timeout=15
    )
    time.sleep(2)   # wait for server to be fully ready before client connects

    # ── Step 2: Run iperf client on su1 ──────────────────────────────────────
    print(f"  [2/3] Running iperf client on su1 → su2 "
          f"(duration={duration}s, threads={IPERF_THREADS}) ...")
    cli_log = os.path.join(base, f"su1_client_{mode}.log")
    cli_cmd = (
        f"iperf -c {su2_ip} -p {IPERF_PORT_UNI} "
        f"-t {duration} -i {interval} "
        f"-P {IPERF_THREADS} -w {IPERF_TCP_WIN}"
    )
    # Use Popen so output streams in real time; communicate() blocks until done
    proc = popen_local(cli_cmd)
    cli_output = proc.communicate()[0]
    write_file(cli_log, cli_output)

    # ── Step 3: Stop su2 server and retrieve its log ──────────────────────────
    print(f"  [3/3] Retrieving server log from su2 ...")
    ssh(client_ip, client_user, "pkill iperf 2>/dev/null; sleep 1", timeout=10)
    srv_out, _, _ = ssh(
        client_ip, client_user,
        f"cat {srv_log_remote}",
        timeout=30
    )
    write_file(os.path.join(base, f"su2_server_{mode}.log"), srv_out)

    # ── Print result summary ──────────────────────────────────────────────────
    # Parse the [SUM] line for the aggregate bandwidth
    m = re.search(r"\[SUM\].*?(\d+\.?\d*)\s+Mbits/sec", cli_output)
    bw = m.group(1) if m else "N/A"
    verdict = "PASS (≥900 Mbits/sec)" if m and float(bw) >= 900 else "review result manually"
    print(f"\n  Unidirectional result: {bw} Mbits/sec  [{verdict}]")


# ══════════════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL TEST   su1 ⇄ su2
# ══════════════════════════════════════════════════════════════════════════════

def run_bidirectional(su1_ip, su2_ip, client_ip, client_user, duration, interval):
    """
    Run a simultaneous two-way iperf test between su1 and su2.

    Port assignment:
      su2 server port 5001  ←  su1 client sends to
      su1 server port 5002  ←  su2 client sends to

    Flow:
      1. Start iperf server on su1 locally (port 5002)
      2. Start iperf server on su2 remotely via SSH (port 5001)
      3. Launch su1 client (→ su2:5001) and su2 client (→ su1:5002) simultaneously
      4. Wait for su1 client to finish
      5. Stop all servers and retrieve remote logs from su2

    Log files produced (names match original reference logs):
      su1_server_Bidirectional.log
      su1_client_Bidirection.log
      su2_server_Bidirection.log
      su2_client_Bidirection.log
    """
    banner(f"iperf Bidirectional  su1({su1_ip}) ⇄ su2({su2_ip})")
    base = os.path.join(OUT_DIR, "iperf")
    os.makedirs(base, exist_ok=True)

    # File name prefix matches original reference log naming convention
    mode = "Bidirection"

    # ── Step 1: Start iperf server on su1 locally (port 5002) ────────────────
    print(f"\n  [1/4] Starting su1 iperf server (port {IPERF_PORT_BI2}) ...")
    su1_srv_log = os.path.join(base, f"su1_server_{mode}al.log")
    # Redirect output directly to log file; process runs in background via Popen
    su1_srv_proc = popen_local(
        f"iperf -s -p {IPERF_PORT_BI2} -i {interval} > {su1_srv_log} 2>&1"
    )
    print(f"    su1 server  port {IPERF_PORT_BI2}  PID={su1_srv_proc.pid}")

    # ── Step 2: Start iperf server on su2 remotely (port 5001) ───────────────
    print(f"  [2/4] Starting su2 iperf server (port {IPERF_PORT_BI1}) via SSH ...")
    su2_srv_log_remote = f"/tmp/su2_server_{mode}.log"
    ssh(
        client_ip, client_user,
        f"pkill iperf 2>/dev/null; sleep 1; "
        f"nohup iperf -s -p {IPERF_PORT_BI1} -i {interval} "
        f"> {su2_srv_log_remote} 2>&1 &",
        timeout=15
    )
    print(f"    su2 server  port {IPERF_PORT_BI1}  (remote)")
    time.sleep(2)   # allow both servers to be ready before clients connect

    # ── Step 3: Launch clients on both sides simultaneously ───────────────────
    print(f"\n  [3/4] Launching bidirectional iperf "
          f"(duration={duration}s, threads={IPERF_THREADS}) ...")

    # su1 client → su2:5001  (local Popen, non-blocking)
    su1_cli_log = os.path.join(base, f"su1_client_{mode}.log")
    su1_cli_proc = popen_local(
        f"iperf -c {su2_ip} -p {IPERF_PORT_BI1} "
        f"-t {duration} -i {interval} "
        f"-P {IPERF_THREADS} -w {IPERF_TCP_WIN}"
    )
    print(f"    su1 client → su2:{IPERF_PORT_BI1}  PID={su1_cli_proc.pid}")

    # su2 client → su1:5002  (remote SSH background process)
    su2_cli_log_remote = f"/tmp/su2_client_{mode}.log"
    ssh(
        client_ip, client_user,
        f"nohup iperf -c {su1_ip} -p {IPERF_PORT_BI2} "
        f"-t {duration} -i {interval} "
        f"-P {IPERF_THREADS} -w {IPERF_TCP_WIN} "
        f"> {su2_cli_log_remote} 2>&1 &",
        timeout=15
    )
    print(f"    su2 client → su1:{IPERF_PORT_BI2}  (remote)")

    # ── Step 4: Wait for su1 client then collect all logs ────────────────────
    print(f"\n  [4/4] Waiting {duration}s for test to complete ...")
    # communicate() blocks until su1 client finishes
    su1_cli_out = su1_cli_proc.communicate()[0]
    write_file(su1_cli_log, su1_cli_out)
    time.sleep(10)   # allow su2 client extra time to flush its log

    # Stop su1 server and close the process
    su1_srv_proc.terminate()
    try:
        su1_srv_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        su1_srv_proc.kill()
    # Server log was written directly to file during execution; no extra write needed

    # Stop su2 remote processes and retrieve logs
    ssh(client_ip, client_user, "pkill iperf 2>/dev/null; sleep 1", timeout=10)

    su2_srv_out, _, _ = ssh(
        client_ip, client_user,
        f"cat {su2_srv_log_remote}",
        timeout=30
    )
    write_file(os.path.join(base, f"su2_server_{mode}.log"), su2_srv_out)

    su2_cli_out, _, _ = ssh(
        client_ip, client_user,
        f"cat {su2_cli_log_remote}",
        timeout=30
    )
    write_file(os.path.join(base, f"su2_client_{mode}.log"), su2_cli_out)

    # ── Print result summary ──────────────────────────────────────────────────
    def extract_bw(text):
        """Parse aggregate bandwidth from the iperf [SUM] line."""
        m = re.search(r"\[SUM\].*?(\d+\.?\d*)\s+Mbits/sec", text)
        return m.group(1) if m else "N/A"

    bw_su1 = extract_bw(su1_cli_out)   # su1 → su2 throughput
    bw_su2 = extract_bw(su2_cli_out)   # su2 → su1 throughput
    print(f"\n  Bidirectional result:")
    print(f"    su1 → su2 : {bw_su1} Mbits/sec")
    print(f"    su2 → su1 : {bw_su2} Mbits/sec")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="1G NIC iPerf Test (1002935 V8.0) — SUT=system10, Client=system13",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Test mode — exactly one must be specified
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument(
        "--unidirectional", action="store_true",
        help="One-way test: su1 → su2"
    )
    mode_grp.add_argument(
        "--bidirectional", action="store_true",
        help="Two-way simultaneous test: su1 ⇄ su2"
    )

    # Optional overrides
    parser.add_argument(
        "--iface", default=None,
        help="Local 1G NIC interface name (default: auto-detect)"
    )
    parser.add_argument(
        "--client-ip", default=CLIENT_IP,
        help=f"su2 IP address used for SSH and as iperf target (default: {CLIENT_IP})"
    )
    parser.add_argument(
        "--client-user", default=CLIENT_USER,
        help=f"SSH login user on su2 (default: {CLIENT_USER})"
    )
    parser.add_argument(
        "--duration", type=int, default=IPERF_DURATION,
        help=f"iperf test duration in seconds (default: {IPERF_DURATION})"
    )
    parser.add_argument(
        "--interval", type=int, default=IPERF_INTERVAL,
        help=f"iperf reporting interval in seconds (default: {IPERF_INTERVAL})"
    )

    args = parser.parse_args()

    # ── Initialise output directory ───────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*66}")
    print(f"  1G NIC iPerf Test  (1002935 V8.0)")
    print(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output  : {OUT_DIR}")
    print(f"{'='*66}")

    # Warn if not root — some commands (dmesg -C, ipmitool) require root
    if os.geteuid() != 0:
        print("[WARNING] Not running as root — some commands may fail.")

    # ── Detect or validate the local 1G NIC interface ────────────────────────
    iface = args.iface or detect_1g_iface()
    if not iface:
        print("[ERROR] Could not auto-detect a 1G NIC interface. "
              "Specify one with --iface <name>.")
        sys.exit(1)

    # Resolve the SUT's test IP from the chosen interface
    su1_ip = get_iface_ip(iface)
    if not su1_ip:
        print(f"[ERROR] Interface {iface} has no IPv4 address. "
              "Configure the interface before running this test.")
        sys.exit(1)

    # The client-side iperf target IP is the same as the SSH destination
    su2_ip = args.client_ip

    # Print test environment summary
    print(f"\n  SUT interface  : {iface}  ({su1_ip})")
    print(f"  Client (su2)   : {args.client_user}@{args.client_ip}  "
          f"iperf target={su2_ip}")
    print(f"  Test mode      : "
          f"{'Unidirectional' if args.unidirectional else 'Bidirectional'}")
    print(f"  Duration       : {args.duration}s  "
          f"threads={IPERF_THREADS}  window={IPERF_TCP_WIN}")

    # ── Verify SSH connectivity to su2 ───────────────────────────────────────
    print(f"\n  Verifying SSH connectivity to {args.client_ip} ...")
    _, err, rc = ssh(args.client_ip, args.client_user, "echo ok", timeout=15)
    if rc != 0:
        print(f"[ERROR] SSH to {args.client_user}@{args.client_ip} failed: {err.strip()}")
        print("        Ensure passwordless SSH is configured: "
              f"ssh-copy-id {args.client_user}@{args.client_ip}")
        sys.exit(1)
    print("  SSH OK")

    # ── Verify iperf is installed on both machines ────────────────────────────
    _, _, rc = run("which iperf")
    if rc != 0:
        print("[ERROR] 'iperf' not found locally. Install with: apt install iperf")
        sys.exit(1)

    _, _, rc = ssh(args.client_ip, args.client_user, "which iperf", timeout=10)
    if rc != 0:
        print("[ERROR] 'iperf' not found on su2. Install with: apt install iperf")
        sys.exit(1)

    # ══ TEST SEQUENCE ═════════════════════════════════════════════════════════

    # Step 1: Clear dmesg / syslog / IPMI SEL before the test
    clear_logs()

    # Step 2: Capture NIC state before the test
    snapshot("before", iface)

    # Step 3: Run the selected iperf test mode
    if args.unidirectional:
        run_unidirectional(
            su1_ip, su2_ip,
            args.client_ip, args.client_user,
            args.duration, args.interval
        )
    else:
        run_bidirectional(
            su1_ip, su2_ip,
            args.client_ip, args.client_user,
            args.duration, args.interval
        )

    # Step 4: Capture NIC state after the test
    snapshot("after", iface)

    # Step 5: Collect dmesg / syslog / IPMI SEL generated during the test
    collect_system_logs()

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  All done.  Results saved to: {OUT_DIR}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
