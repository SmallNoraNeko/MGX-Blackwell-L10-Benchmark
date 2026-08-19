"""
_rdma_common.py
---------------
Shared utilities for rdma_test_ipv4_v2.py and rdma_test_ipv6_v3.py.

Provides:
  - ANSI colour helpers
  - Logging wrappers  (info / ok / warn / error / title / section)
  - Shell command runner
  - NIC / IP discovery  (IPv4 + IPv6, GID table)
  - Network setup       (MTU, kernel params, IP assign/release, ping)
  - NCCL / perftest helpers (wait_for_port, stream_output, generate_summary)
  - Log collection      (clear_all, collect_all)
  - Prerequisite check
"""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colour constants
# ---------------------------------------------------------------------------
_R  = "\033[0m"
_G  = "\033[92m"
_Y  = "\033[93m"
_RE = "\033[91m"
_C  = "\033[96m"
_W  = "\033[97m"
_D  = "\033[90m"
_B  = "\033[94m"

C = _C   # exported alias used by rdma scripts


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(prefix: str, colour: str, msg: str):
    print(f"{colour}[{ts()}] {prefix}{msg}{_R}", flush=True)


def info(msg: str):    _log("", _D,  msg)
def ok(msg: str):      _log("[OK]  ", _G,  msg)
def warn(msg: str):    _log("[WARN] ", _Y,  msg)
def error(msg: str):   _log("[ERR]  ", _RE, msg)
def title(msg: str):   print(f"\n{_W}{'='*66}\n  {msg}\n{'='*66}{_R}", flush=True)
def section(msg: str): print(f"\n{_D}{'─'*66}\n  {msg}\n{'─'*66}{_R}", flush=True)


# ---------------------------------------------------------------------------
# Shell command runner
# ---------------------------------------------------------------------------

def run(cmd: str, check: bool = False, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a shell command and return CompletedProcess. Errors are logged, not raised."""
    try:
        result = subprocess.run(
            cmd, shell=True, text=True,
            capture_output=True, timeout=timeout,
        )
        if check and result.returncode != 0:
            error(f"Command failed (rc={result.returncode}): {cmd}")
            if result.stderr:
                error(result.stderr.strip())
        return result
    except subprocess.TimeoutExpired:
        warn(f"Command timed out after {timeout}s: {cmd}")
        return subprocess.CompletedProcess(cmd, -1, "", "timeout")
    except Exception as exc:
        error(f"Command exception: {exc}")
        return subprocess.CompletedProcess(cmd, -1, "", str(exc))


# ---------------------------------------------------------------------------
# NIC / IP helpers
# ---------------------------------------------------------------------------

def rdma_to_netdev(rdma_name: str) -> str | None:
    """Return the net device name for an RDMA device (e.g. mlx5_0 -> enp...). """
    sysfs = f"/sys/class/infiniband/{rdma_name}/device/net"
    try:
        entries = os.listdir(sysfs)
        return entries[0] if entries else None
    except FileNotFoundError:
        return None


def get_nic_primary_ipv4(rdma_name: str) -> tuple[str | None, int | None]:
    """
    Return (ip_address, prefix_len) for the first non-loopback IPv4 address
    on the net device corresponding to rdma_name.
    """
    netdev = rdma_to_netdev(rdma_name)
    if not netdev:
        return None, None
    result = run(f"ip -4 addr show dev {netdev}")
    for line in result.stdout.splitlines():
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)/(\d+)", line)
        if m:
            ip  = m.group(1)
            prefix = int(m.group(2))
            if not ipaddress.ip_address(ip).is_loopback:
                return ip, prefix
    return None, None


def get_nic_primary_ipv6(rdma_name: str) -> tuple[str | None, int | None]:
    """
    Return (ipv6_address, prefix_len) for the first non-link-local IPv6
    address on the net device corresponding to rdma_name.
    """
    netdev = rdma_to_netdev(rdma_name)
    if not netdev:
        return None, None
    result = run(f"ip -6 addr show dev {netdev}")
    for line in result.stdout.splitlines():
        m = re.search(r"inet6 ([0-9a-f:]+)/(\d+)", line)
        if m:
            addr   = m.group(1)
            prefix = int(m.group(2))
            try:
                obj = ipaddress.ip_address(addr)
                if not obj.is_link_local and not obj.is_loopback:
                    return addr, prefix
            except ValueError:
                pass
    return None, None


def dump_gid_table(rdma_name: str) -> list[dict]:
    """
    Return a list of GID entries for rdma_name.
    Each entry: {index, gid, type, netdev}
    """
    entries = []
    sysfs_base = f"/sys/class/infiniband/{rdma_name}/ports/1/gids"
    try:
        indices = sorted(int(x) for x in os.listdir(sysfs_base))
    except FileNotFoundError:
        return entries
    for idx in indices:
        try:
            gid = Path(f"{sysfs_base}/{idx}").read_text().strip()
            if gid == "0000:0000:0000:0000:0000:0000:0000:0000":
                continue
            gid_type_path = (
                f"/sys/class/infiniband/{rdma_name}/ports/1/gid_attrs/types/{idx}"
            )
            try:
                gid_type = Path(gid_type_path).read_text().strip()
            except FileNotFoundError:
                gid_type = "unknown"
            entries.append({"index": idx, "gid": gid,
                             "type": gid_type, "netdev": rdma_to_netdev(rdma_name)})
        except Exception:
            pass
    return entries


def wait_for_gid_index(rdma_name: str, ip: str | None,
                        default_gid: int | None = None,
                        timeout: int = 10) -> int:
    """
    Find the GID index whose address matches the given IPv4/IPv6 string.
    Falls back to default_gid if provided, or 0 if nothing matches.
    """
    if default_gid is not None:
        return default_gid

    if not ip:
        return 0

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for entry in dump_gid_table(rdma_name):
            gid = entry["gid"]
            # IPv4-mapped: 0000:0000:0000:0000:0000:ffff:c0a8:xxxx
            if "ffff" in gid.lower():
                try:
                    packed = bytes.fromhex(gid.replace(":", ""))
                    mapped = ipaddress.ip_address(packed[-4:])
                    if str(mapped) == ip:
                        return entry["index"]
                except Exception:
                    pass
            # Direct IPv6 match
            try:
                if ipaddress.ip_address(gid) == ipaddress.ip_address(ip):
                    return entry["index"]
            except ValueError:
                pass
        time.sleep(0.5)

    warn(f"GID index for {ip} on {rdma_name} not found — using 0")
    return 0


def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    """Return True once TCP port is reachable, False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# NIC pairing / topology discovery
# ---------------------------------------------------------------------------

def discover_and_pair_nics() -> tuple[list[tuple], dict]:
    """
    Auto-discover RDMA NICs from /sys/class/infiniband and attempt to
    pair them by NUMA node.  Falls back to sequential pairing.

    Returns (pairs, numa_map):
      pairs    : [(srv_nic, cli_nic, pair_name), ...]
      numa_map : {nic_name: numa_node_id}
    """
    sysfs = Path("/sys/class/infiniband")
    nics  = sorted(d.name for d in sysfs.iterdir() if d.is_symlink()) \
            if sysfs.exists() else []

    numa_map: dict[str, int] = {}
    for nic in nics:
        try:
            numa_node = int(
                Path(f"/sys/class/infiniband/{nic}/device/numa_node").read_text()
            )
            numa_map[nic] = max(numa_node, 0)
        except Exception:
            numa_map[nic] = 0

    # Sequential pairing: (nic[0], nic[1]), (nic[2], nic[3]), ...
    pairs = []
    for i in range(0, len(nics) - 1, 2):
        srv, cli = nics[i], nics[i + 1]
        pairs.append((srv, cli, f"pair{i // 2 + 1}"))

    return pairs, numa_map


# ---------------------------------------------------------------------------
# Network / kernel setup
# ---------------------------------------------------------------------------

def setup_kernel_params():
    """Apply recommended kernel parameters for RDMA performance."""
    params = {
        "net.core.rmem_max":          "134217728",
        "net.core.wmem_max":          "134217728",
        "net.core.rmem_default":      "67108864",
        "net.core.wmem_default":      "67108864",
        "net.core.netdev_max_backlog":"250000",
        "net.ipv4.tcp_timestamps":    "0",
    }
    for key, val in params.items():
        run(f"sysctl -w {key}={val} >/dev/null 2>&1")
    info("Kernel parameters applied.")


def setup_mtu(pairs: list, numa_map: dict, mtu: int):
    """Set MTU on all NIC net-devices involved in test pairs."""
    nics = set()
    for srv, cli, _ in pairs:
        nics.add(srv)
        nics.add(cli)
    for nic in nics:
        netdev = rdma_to_netdev(nic)
        if netdev:
            run(f"ip link set dev {netdev} mtu {mtu} 2>/dev/null")
    info(f"MTU set to {mtu} on {len(nics)} NIC(s).")


def get_card_type_from_rdma(rdma_name: str) -> str:
    """
    Identify NIC card type (CX8 / BF3 / CX7 / CX6 / UNKNOWN)
    by inspecting sysfs and mst status output.

    GB300 L10 MGX known device IDs (from mst status -v):
      CX8  →  mt4131   (ConnectX8)
      BF3  →  mt41692  (BlueField3)

    Returns one of: "CX8", "BF3", "CX7", "CX6", "UNKNOWN"
    """
    # --- Fast path: infer from MST device path in sysfs ---
    # /sys/class/infiniband/<rdma>/device symlink target contains the PCI BDF
    # We can read the subsystem device ID to identify the card.
    try:
        device_path = f"/sys/class/infiniband/{rdma_name}/device"
        # Read vendor and device ID
        vendor = Path(f"{device_path}/vendor").read_text().strip()
        device = Path(f"{device_path}/device").read_text().strip()
        if vendor in ("0x15b3", "0x10de"):  # Mellanox / NVIDIA
            dev_id = int(device, 16)
            # ConnectX-8: 0x101e, 0x101f, 0x1020, 0x1021
            if 0x101e <= dev_id <= 0x1021:
                return "CX8"
            # BlueField-3: 0xa2dc, 0xa2d6, 0xa2d2, 0xa2d0
            if dev_id in (0xa2d0, 0xa2d2, 0xa2d6, 0xa2dc):
                return "BF3"
            # ConnectX-7: 0x1017, 0x1018, 0x1019, 0x101a, 0x101b
            if 0x1017 <= dev_id <= 0x101b:
                return "CX7"
            # ConnectX-6 / 6Dx: 0x101c, 0x101d
            if dev_id in (0x101c, 0x101d, 0x1013, 0x1015):
                return "CX6"
    except OSError:
        pass

    # --- Fallback: parse mst status -v output ---
    # mst reports "ConnectX8(rev:0)" / "BlueField3(rev:1)" etc.
    # Use exact word match on the RDMA column to avoid mlx5_1 matching mlx5_10/11.
    result = run("mst status -v", timeout=15)
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            # Match rdma_name as a complete whitespace-delimited token
            tokens = line.split()
            if rdma_name not in tokens:
                continue
            first = tokens[0].upper() if tokens else ""
            if "CONNECTX8" in first or "MT4131" in first:
                return "CX8"
            if "BLUEFIELD3" in first or "MT41692" in first:
                return "BF3"
            if "CONNECTX7" in first or "MT28" in first:
                return "CX7"
            if "CONNECTX6" in first or "MT27" in first:
                return "CX6"

    # --- Last resort: sysfs board_id (not always present) ---
    try:
        board_id = Path(
            f"/sys/class/infiniband/{rdma_name}/device/board_id"
        ).read_text().strip().upper()
        if "MT4131" in board_id or "MCX75" in board_id:
            return "CX8"
        if "MT41692" in board_id or "MBF3" in board_id:
            return "BF3"
        if "MT28" in board_id or "MCX74" in board_id:
            return "CX7"
        if "MT27" in board_id or "MCX62" in board_id:
            return "CX6"
    except OSError:
        pass

    return "UNKNOWN"


def use_rdma_cm_for_card(card_type: str) -> bool:
    """
    Return True if the card type requires the RDMA CM flag (-R).
    BF3 requires -R; CX6/CX7/CX8 do not.
    """
    return card_type == "BF3"


def assign_ipv4_addr(rdma_name: str, ip: str, prefix: int):
    """Assign an IPv4 address to the net device for rdma_name."""
    netdev = rdma_to_netdev(rdma_name)
    if not netdev:
        warn(f"Cannot assign IP: no netdev for {rdma_name}")
        return
    run(f"ip addr flush dev {netdev} 2>/dev/null")
    run(f"ip addr add {ip}/{prefix} dev {netdev}")
    run(f"ip link set dev {netdev} up")


def release_ipv4_addr(rdma_name: str):
    """Remove all IPv4 addresses from the net device for rdma_name."""
    netdev = rdma_to_netdev(rdma_name)
    if netdev:
        run(f"ip addr flush dev {netdev} 2>/dev/null")


def ping_ipv4_from_nic(src_rdma: str, dst_ip: str,
                        count: int = 3, timeout: int = 2) -> bool:
    """Ping dst_ip from the net device of src_rdma. Returns True on success."""
    netdev = rdma_to_netdev(src_rdma)
    if not netdev:
        return False
    result = run(
        f"ping -c {count} -W {timeout} -I {netdev} {dst_ip}",
        timeout=count * timeout + 5,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Test process helpers
# ---------------------------------------------------------------------------

def stream_output(proc: subprocess.Popen, tag: str, log_lines: list):
    """
    Read proc.stdout line-by-line, print with tag prefix, append to log_lines.
    Intended to run in a daemon thread.
    """
    prefix = f"{_D}[{tag}]{_R} "
    for line in proc.stdout:
        sys.stdout.write(prefix + line)
        sys.stdout.flush()
        log_lines.append(line)


# ---------------------------------------------------------------------------
# Log management
# ---------------------------------------------------------------------------

def clear_all(output_base: str):
    """Remove the output_base directory tree (start fresh)."""
    import shutil as _shutil
    if os.path.isdir(output_base):
        _shutil.rmtree(output_base, ignore_errors=True)
    os.makedirs(output_base, exist_ok=True)


def collect_all(result_base: str, output_dir: str):
    """
    Copy all result log files from result_base into output_dir,
    preserving relative directory structure.
    """
    import shutil as _shutil
    os.makedirs(output_dir, exist_ok=True)
    for root, _dirs, files in os.walk(result_base):
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, result_base)
            dst = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Summary generator
# ---------------------------------------------------------------------------

def generate_summary(result_base: str, mode_label: str):
    """
    Scan result_base for client log files, extract BW numbers,
    and write a human-readable summary.txt.
    """
    summary_path = os.path.join(result_base, f"summary_{mode_label}.txt")
    lines = [
        f"RDMA {mode_label.upper()} Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 66,
    ]

    bw_pattern = re.compile(r"(\d+)\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)")

    for root, _dirs, files in os.walk(result_base):
        for fname in sorted(files):
            if "client" not in fname or not fname.endswith(".log"):
                continue
            fpath = os.path.join(root, fname)
            peak_bw = None
            try:
                with open(fpath) as f:
                    for line in f:
                        m = bw_pattern.search(line)
                        if m:
                            peak_bw = float(m.group(3))
            except OSError:
                pass
            if peak_bw is not None:
                lines.append(f"  {fname:<60} {peak_bw:>10.2f} Gb/s")

    lines += ["=" * 66, ""]
    body = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(body)
    info(f"Summary written: {summary_path}")


# ---------------------------------------------------------------------------
# Prerequisite checker
# ---------------------------------------------------------------------------

def check_prerequisites(pairs: list):
    """
    Verify that all required perftest binaries are on PATH and
    that at least one mlx5 RDMA device exists.
    Exits with code 1 if critical checks fail.
    """
    tools = ["ib_read_bw", "ib_send_bw", "ib_write_bw"]
    missing = [t for t in tools if not shutil.which(t)]
    if missing:
        error(f"Missing perftest binaries: {', '.join(missing)}")
        error("Install perftest or ensure the binaries are on PATH.")
        sys.exit(1)

    sysfs = Path("/sys/class/infiniband")
    if not sysfs.exists() or not any(sysfs.iterdir()):
        error("No RDMA devices found under /sys/class/infiniband")
        sys.exit(1)

    ok(f"Prerequisites OK — {len(pairs)} loopback pair(s) configured.")
