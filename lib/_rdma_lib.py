#!/usr/bin/env python3
"""
_rdma_common.py — Universal RDMA Loopback 共用函式庫 (Dynamic Topology & RoCE v2 Support)
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import socket
import subprocess
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Terminal 顏色與輸出 Formatting
# ─────────────────────────────────────────────────────────────────────────────

class C:
    HEADER = '\033[95m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    ENDC   = '\033[0m'
    BOLD   = '\033[1m'

def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def info(msg: str):
    print(f"[{ts()}] {C.CYAN}[INFO]{C.ENDC} {msg}", flush=True)

def ok(msg: str):
    print(f"[{ts()}] {C.GREEN}[OK]{C.ENDC} {msg}", flush=True)

def warn(msg: str):
    print(f"[{ts()}] {C.YELLOW}[WARN]{C.ENDC} {msg}", flush=True)

def error(msg: str):
    print(f"[{ts()}] {C.RED}[ERROR]{C.ENDC} {msg}", flush=True)

def title(msg: str):
    print(f"\n{C.BOLD}{C.BLUE}{'='*80}\n {msg}\n{'='*80}{C.ENDC}", flush=True)

def section(msg: str):
    print(f"\n{C.BOLD}{C.HEADER}--- {msg} ---{C.ENDC}", flush=True)

def run(cmd: str, check: bool = False, capture: bool = True) -> tuple[int, str]:
    res = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    out = res.stdout.strip() if capture and res.stdout else ""
    if check and res.returncode != 0:
        error(f"Command failed (code {res.returncode}): {cmd}\nOutput: {out}")
    return res.returncode, out

def stream_output(proc: subprocess.Popen, prefix: str, log_lines: list):
    if proc.stdout is None:
        return
    relevant_keywords = ["#bytes", "BW peak", "BW average", "Conf", "Gb/sec", "Gbit/s", "MB/sec", "MsgRate"]
    for line in iter(proc.stdout.readline, ''):
        if not line:
            break
        log_lines.append(line)
        if any(kw in line for kw in relevant_keywords) or re.search(r'^\s*\d+\s+\d+\s+', line):
            print(f"  {C.YELLOW}[{prefix}]{C.ENDC} {line.strip()}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 動態 Topology 偵測
# ─────────────────────────────────────────────────────────────────────────────

def get_nic_numa(rdma_dev: str) -> int:
    path = f"/sys/class/infiniband/{rdma_dev}/device/numa_node"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                numa = int(f.read().strip())
                return numa if numa >= 0 else 0
        except Exception:
            pass
    return 0

def get_best_cuda_for_nic(rdma_dev: str) -> int | None:
    code, out = run("nvidia-smi topo -m")
    if code == 0 and out:
        lines = out.splitlines()
        header = None
        for line in lines:
            if line.startswith(" ") and ("GPU0" in line or "NIC" in line or "mlx5" in line):
                header = line.split()
                break

        if header:
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == rdma_dev:
                    best_gpu = None
                    best_score = -1
                    affinity_scores = {"PIX": 4, "PXB": 3, "NODE": 2, "SYS": 1}

                    for idx, conn_type in enumerate(parts[1:]):
                        if idx < len(header) and header[idx].startswith("GPU"):
                            try:
                                gpu_id = int(header[idx].replace("GPU", ""))
                                score = affinity_scores.get(conn_type, 0)
                                if score > best_score:
                                    best_score = score
                                    best_gpu = gpu_id
                            except ValueError:
                                continue
                    if best_gpu is not None:
                        return best_gpu

    nic_numa = get_nic_numa(rdma_dev)
    try:
        m = re.search(r'\d+', rdma_dev)
        if m:
            nic_idx = int(m.group(0))
            return (nic_idx // 2) % 8
    except Exception:
        pass
    return 0

def discover_and_pair_nics() -> tuple[list, dict, dict, list]:
    # WARNING: pairs adjacent device numbers (0-1, 2-3, ...). On GB300 system10 the
    # physical loopback cabling is 0↔6, 1↔7, 2↔4, 3↔5, 8↔10, 9↔11 — do NOT use this
    # for that platform; use a static pair table instead.
    code, out = run("ibv_devinfo -l")
    if code != 0:
        return [], {}, {}, []

    def _sort_key(dev_str: str) -> int:
        m = re.search(r'\d+', dev_str)
        return int(m.group(0)) if m else 0

    devices = sorted([line.strip() for line in out.splitlines() if "mlx5_" in line], key=_sort_key)
    numa_map = {}
    cuda_map = {}

    for dev in devices:
        numa_map[dev] = get_nic_numa(dev)
        cuda_map[dev] = get_best_cuda_for_nic(dev)

    pairs = []
    visited = set()
    
    for i in range(0, len(devices) - 1, 2):
        dev1 = devices[i]
        dev2 = devices[i+1]
        pairs.append((dev1, dev2, f"pair{len(pairs)+1}"))
        visited.add(dev1)
        visited.add(dev2)

    unassigned = [dev for dev in devices if dev not in visited]
    return pairs, numa_map, cuda_map, unassigned

# ─────────────────────────────────────────────────────────────────────────────
# GID / IP / Netdev 相關處理 (精準單獨 Port RoCE v2 定位)
# ─────────────────────────────────────────────────────────────────────────────

def rdma_to_netdev(rdma_dev: str) -> str | None:
    path = f"/sys/class/infiniband/{rdma_dev}/device/net"
    if os.path.exists(path):
        entries = os.listdir(path)
        if entries:
            return entries[0]
    return None

def _parse_gid_to_ipv4(gid_str: str) -> str | None:
    """
    Convert a GID string to dotted-quad IPv4. Supports all sysfs formats:

    Format A — 8-group 4-hex-digit (standard sysfs output, GB300/system10):
      0000:0000:0000:0000:0000:ffff:0101:0101  → 1.1.1.1
      0000:0000:0000:0000:0000:ffff:0d88:bbf6  → 13.136.187.246

    Format B — 16-byte colon-separated DECIMAL (seen in perftest log output):
      00:00:00:00:00:00:00:00:00:00:255:255:01:01:01:01  → 1.1.1.1

    Format C — 16-byte colon-separated HEX (older kernel sysfs):
      00:00:00:00:00:00:00:00:00:00:ff:ff:01:01:01:01  → 1.1.1.1

    Format D — trailing dotted-quad regex fallback:
      ::ffff:1.1.1.1  → 1.1.1.1
    """
    if not gid_str or gid_str == "0000:0000:0000:0000:0000:0000:0000:0000":
        return None
    parts = gid_str.split(":")

    # Format A: 8 groups × 4 hex digits (primary sysfs format on GB300)
    if len(parts) == 8:
        try:
            p6, p7 = int(parts[6], 16), int(parts[7], 16)
            if parts[5].lower() == "ffff" and all(int(p, 16) == 0 for p in parts[:5]):
                return f"{p6 >> 8}.{p6 & 0xff}.{p7 >> 8}.{p7 & 0xff}"
        except ValueError:
            pass

    # Format B: 16 colon-separated DECIMAL bytes (perftest log format)
    # e.g. 00:00:00:00:00:00:00:00:00:00:255:255:01:01:01:01
    if len(parts) == 16:
        try:
            b = [int(x, 10) for x in parts]
            if all(x == 0 for x in b[:10]) and b[10] == 255 and b[11] == 255:
                return f"{b[12]}.{b[13]}.{b[14]}.{b[15]}"
        except ValueError:
            pass
        # Format C: 16 colon-separated HEX bytes (older kernel)
        # e.g. 00:00:00:00:00:00:00:00:00:00:ff:ff:01:01:01:01
        try:
            b = [int(x, 16) for x in parts]
            if all(x == 0 for x in b[:10]) and b[10] == 0xff and b[11] == 0xff:
                return f"{b[12]}.{b[13]}.{b[14]}.{b[15]}"
        except ValueError:
            pass

    # Format D: trailing dotted-quad regex fallback (::ffff:a.b.c.d style)
    m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$', gid_str)
    return m.group(1) if m else None

def dump_gid_table(rdma_dev: str) -> list[dict]:
    gids = []
    gid_dir = f"/sys/class/infiniband/{rdma_dev}/ports/1/gids"
    type_dir = f"/sys/class/infiniband/{rdma_dev}/ports/1/gid_attrs/types"

    if not os.path.exists(gid_dir):
        return gids

    for idx_file in sorted(os.listdir(gid_dir), key=lambda x: int(x) if x.isdigit() else 999):
        try:
            with open(os.path.join(gid_dir, idx_file), "r") as f:
                gid_val = f.read().strip()
            if gid_val == "0000:0000:0000:0000:0000:0000:0000:0000":
                continue

            gid_type = "Unknown"
            type_path = os.path.join(type_dir, idx_file)
            if os.path.exists(type_path):
                with open(type_path, "r") as tf:
                    gid_type = tf.read().strip()

            gids.append({
                "index": int(idx_file),
                "gid": gid_val,
                "type": gid_type,
                "ipv4": _parse_gid_to_ipv4(gid_val)
            })
        except Exception:
            continue
    return gids

def get_nic_primary_ipv4(rdma_dev: str, target_ip: str | None = None) -> tuple[str | None, int]:
    """
    動態抓取指定 rdma_dev 獨立對應的 RoCE v2 GID Index。
    """
    gids = dump_gid_table(rdma_dev)
    
    # 策略 1: 精準匹配目標 IP，且必須是 RoCE v2
    if target_ip:
        for item in gids:
            if "v2" in item["type"].lower() and item.get("ipv4") == target_ip:
                return item["ipv4"], item["index"]

    # 策略 2: 找到第一個帶有任意有效 IPv4 的 RoCE v2 GID Index
    for item in gids:
        if "v2" in item["type"].lower() and item.get("ipv4") is not None:
            return item["ipv4"], item["index"]

    # 策略 3: 找第一個型態標註為 RoCE v2 的 Index
    for item in gids:
        if "v2" in item["type"].lower():
            return item.get("ipv4"), item["index"]

    # 保底返回 Index 3
    return None, 3

def get_nic_primary_ipv6(rdma_dev: str) -> tuple[str | None, int]:
    gids = dump_gid_table(rdma_dev)
    for item in gids:
        if "v2" in item["type"].lower() and not item.get("ipv4"):
            return item["gid"], item["index"]
    return None, 1

def find_rocev2_gid_for_ip(rdma_dev: str, target_ip: str) -> int | None:
    """Exact match only: RoCE v2 GID whose IPv4 == target_ip, else None."""
    for item in dump_gid_table(rdma_dev):
        if "v2" in item["type"].lower() and item.get("ipv4") == target_ip:
            return item["index"]
    return None

def wait_for_gid_index(rdma_dev: str, target_ip: str | None = None, default_gid: int | None = None,
                       timeout: int = 5, strict: bool = False) -> int | None:
    """
    即時且精準獲取當前 Port 獨立與 IPv4 綁定的 RoCE v2 GID Index

    strict=True: 只接受與 target_ip 完全相符的 RoCE v2 GID；逾時回傳 None（不默默 fallback 到別的 IP 的 GID）
    strict=False: 舊行為（相容其他腳本）
    """
    if default_gid is not None:
        return default_gid

    if strict and target_ip:
        deadline = time.time() + timeout
        while time.time() < deadline:
            idx = find_rocev2_gid_for_ip(rdma_dev, target_ip)
            if idx is not None:
                return idx
            time.sleep(0.3)
        error(f"{rdma_dev}: no RoCE v2 GID for {target_ip} after {timeout}s")
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        _, idx = get_nic_primary_ipv4(rdma_dev, target_ip)
        if idx is not None:
            return idx
        time.sleep(0.3)
    _, fallback_idx = get_nic_primary_ipv4(rdma_dev, target_ip)
    return fallback_idx if fallback_idx is not None else 3

def wait_for_port(port: int, timeout: int = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return True
        time.sleep(0.1)
    return False

def assign_ipv4_addr(rdma_dev: str, ip: str, subnet_mask: int = 24) -> bool:
    netdev = rdma_to_netdev(rdma_dev)
    if not netdev:
        return False
    run(f"ip link set dev {netdev} up")
    run(f"sysctl -w net.ipv4.conf.{netdev}.rp_filter=0 >/dev/null 2>&1")
    run(f"sysctl -w net.ipv4.conf.{netdev}.accept_local=1 >/dev/null 2>&1")
    run(f"sysctl -w net.ipv4.conf.{netdev}.arp_ignore=1 >/dev/null 2>&1")
    run(f"sysctl -w net.ipv4.conf.{netdev}.arp_announce=2 >/dev/null 2>&1")
    code, out = run(f"ip addr add {ip}/{subnet_mask} dev {netdev}")
    return code == 0 or "File exists" in out

def release_ipv4_addr(rdma_dev: str, ip: str | None = None, subnet_mask: int = 24):
    """
    只移除測試自己加的那個 IP。
    舊版用 `ip addr flush dev` 會把介面上「所有」IP 清掉（包含其他用途的 IP），已移除。
    """
    netdev = rdma_to_netdev(rdma_dev)
    if not netdev:
        return
    if not ip:
        warn(f"release_ipv4_addr({rdma_dev}) called without ip — skipped (no flush)")
        return
    run(f"ip addr del {ip}/{subnet_mask} dev {netdev}")

def ping_ipv4_from_nic(src_rdma_dev: str, target_ip: str, count: int = 1, timeout: int = 2) -> bool:
    """
    注意：同一台機器上的 loopback 測試，ping 本機 IP 會走 lo，
    通過也不代表實體接線正確。實體接線請用 RDMA 短測驗證。
    （舊版在 ping 失敗時以 PORT_ACTIVE 判定成功，已移除）
    """
    netdev = rdma_to_netdev(src_rdma_dev)
    if not netdev:
        return False
    code, _ = run(f"ping -I {netdev} -c {count} -W {timeout} {target_ip}")
    return code == 0

def setup_kernel_params():
    params = [
        "sysctl -w net.core.rmem_max=134217728",
        "sysctl -w net.core.wmem_max=134217728",
        "sysctl -w net.ipv4.conf.all.rp_filter=0",
        "sysctl -w net.ipv4.conf.default.rp_filter=0",
        "sysctl -w net.ipv4.conf.all.accept_local=1",
        "sysctl -w net.ipv4.conf.all.arp_ignore=1",
        "sysctl -w net.ipv4.conf.all.arp_announce=2",
    ]
    for p in params: run(p)

def setup_mtu(pairs: list, numa_map: dict | None = None, mtu: int = 9000) -> dict[str, str]:
    """
    設定 netdev MTU，回傳 {netdev: 原本的 MTU} 供 restore_mtu() 還原。
    netdev MTU 9000 → RoCE active_mtu 4096（1500 時只有 1024，BF3 352→394、CX8 740→784 Gb/s）
    """
    original: dict[str, str] = {}
    for p in pairs:
        for dev in (p[0], p[1]):
            netdev = rdma_to_netdev(dev)
            if not netdev or netdev in original:
                continue
            _, cur = run(f"cat /sys/class/net/{netdev}/mtu")
            original[netdev] = cur
            run(f"ip link set dev {netdev} mtu {mtu}")
            _, active = run(f"ibv_devinfo -d {dev} | awk '/active_mtu/{{print $2}}'")
            msg = f"{netdev} ({dev}): MTU {cur} -> {mtu}, RoCE active_mtu {active}"
            (info if active == "4096" or mtu < 4200 else warn)(msg)
    return original

def restore_mtu(original: dict[str, str]):
    for netdev, mtu in (original or {}).items():
        if mtu:
            run(f"ip link set dev {netdev} mtu {mtu}")
            info(f"{netdev}: MTU restored to {mtu}")

def clear_all(output_base: str):
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

def generate_summary(output_base: str, proto: str):
    """
    Parse all client_*.log files and print a BW summary table.

    Supports two client log formats:

    Format 1 — Full table (no --output=bandwidth flag):
      #bytes  #iterations  BW peak[Gb/sec]  BW average[Gb/sec]  MsgRate[Mpps]
      65536   4481059      0.00             783.12              1.493684
      → Parsed from the line following "#bytes" header.

    Format 2 — Single float (--output=bandwidth flag on client):
      WARNING: BW peak won't be measured in this run.
      783.12
      → Only BW average is printed; BW peak shown as "-".
      → Detected by finding a lone numeric line (no "#bytes" header present).
    """
    section(f"SUMMARY RESULT ({proto.upper()})")
    log_files = glob.glob(f"{output_base}/*/*.log")
    if not log_files:
        warn("No log files found.")
        return

    print(f"\n{C.BOLD}{'LOG FILE':<55} | {'BW Peak (Gb/s)':<18} | {'BW Average (Gb/s)':<18}{C.ENDC}")
    print("-" * 98)

    for lf in sorted(log_files):
        filename = os.path.basename(lf)
        if not filename.startswith("client_"):
            continue
        bw_peak, bw_avg = "N/A", "N/A"
        try:
            with open(lf, "r") as f:
                lines = f.readlines()

            # Format 1: full table — look for "#bytes" header then parse next line
            for i, line in enumerate(lines):
                if "#bytes" in line and i + 1 < len(lines):
                    parts = lines[i + 1].strip().split()
                    if len(parts) >= 4 and parts[0].replace('.', '', 1).isdigit():
                        peak_val = float(parts[2])
                        avg_val  = float(parts[3])
                        # perftest -b mode: BW peak = 0 → substitute with avg
                        bw_peak = f"{avg_val:.2f}" if peak_val == 0.0 else f"{peak_val:.2f}"
                        bw_avg  = f"{avg_val:.2f}"
                        break

            # Format 2: single float (--output=bandwidth) — no "#bytes" header
            if bw_avg == "N/A":
                for line in lines:
                    stripped = line.strip()
                    # Skip warning / empty lines, match bare float
                    if stripped and not stripped.startswith("W") and not stripped.startswith("["):
                        try:
                            val = float(stripped)
                            if val > 0:
                                bw_avg  = f"{val:.2f}"
                                bw_peak = "-"          # peak not available in this format
                                break
                        except ValueError:
                            continue

        except Exception:
            pass

        avg_str = f"{C.GREEN}{bw_avg:<18}{C.ENDC}" if bw_avg != "N/A" else f"{C.RED}{'N/A':<18}{C.ENDC}"
        print(f"{filename:<55} | {bw_peak:<18} | {avg_str}")
    print("-" * 98)

def collect_all(output_base: str, target_dir: str):
    if os.path.exists(output_base):
        os.makedirs(target_dir, exist_ok=True)
        archive_name = os.path.join(target_dir, f"rdma_logs_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.make_archive(archive_name, 'gztar', output_base)
        ok(f"Logs collected to: {archive_name}.tar.gz")

def check_prerequisites(pairs: list | None = None):
    for cmd in ["ibv_devinfo", "numactl", "ip", "ping"]:
        if not shutil.which(cmd):
            error(f"Missing required tool: {cmd}")
            sys.exit(1)
            
    if pairs:
        for srv, cli, _ in pairs:
            for dev in (srv, cli):
                netdev = rdma_to_netdev(dev)
                if not netdev:
                    warn(f"RDMA device {dev} has no mapped netdev under /sys/class/infiniband/{dev}/device/net")

# ─────────────────────────────────────────────────────────────────────────────
# Platform profiles (rdma_platforms.json)
# ─────────────────────────────────────────────────────────────────────────────
#
# 每個平台可設定：pairs（必填）、cards、link_gbps、gpu_map、waves、line_rate_pct
# 沒填的項目會自動偵測：
#   cards     → sysfs PCI device id / mst status / board_id
#   link_gbps → /sys/class/infiniband/<dev>/ports/1/rate
#   gpu_map   → nvidia-smi + sysfs NUMA（server/client 盡量分配同 NUMA 的不同 GPU）
#   waves     → 同一張實體卡的 port 不放在同一批（貪婪排程）
# 跨卡干擾（例如 L10 上 BF3 與 NUMA1 CX8 互相影響）無法自動偵測，需實測後寫進 waves。

import json

# 依卡型的 perftest 行為（2026-10-16 system10 驗證）
CARD_BEHAVIOR = {
    "BF3":     {"rdma_cm": True,  "data_direct": False},  # -R；不支援 --use_data_direct
    "CX8":     {"rdma_cm": False, "data_direct": True},   # -m 4096；data_direct 兩端需不同 GPU
    "CX7":     {"rdma_cm": False, "data_direct": False},
    "CX6":     {"rdma_cm": False, "data_direct": False},
    "UNKNOWN": {"rdma_cm": False, "data_direct": False},
}

# 內建 GB300 L10 MGX (system10) 驗證設定 — 找不到 rdma_platforms.json 時使用
BUILTIN_PLATFORMS = {
    "GB300_L10": {
        "description": "GB300 L10 MGX (system10), validated 2026-10-16",
        "pairs": [["mlx5_0", "mlx5_6"], ["mlx5_1", "mlx5_7"], ["mlx5_2", "mlx5_4"],
                  ["mlx5_3", "mlx5_5"], ["mlx5_8", "mlx5_10"], ["mlx5_9", "mlx5_11"]],
        "cards": {"mlx5_0": "BF3", "mlx5_1": "BF3", "mlx5_6": "BF3", "mlx5_7": "BF3",
                  "mlx5_2": "CX8", "mlx5_3": "CX8", "mlx5_4": "CX8", "mlx5_5": "CX8",
                  "mlx5_8": "CX8", "mlx5_9": "CX8", "mlx5_10": "CX8", "mlx5_11": "CX8"},
        "link_gbps": {"BF3": 400, "CX8": 800},
        "gpu_map": {"mlx5_0": "0008:06:00.0", "mlx5_1": "0008:06:00.0",
                    "mlx5_6": "0018:06:00.0", "mlx5_7": "0018:06:00.0",
                    "mlx5_2": "0008:06:00.0", "mlx5_3": "0008:06:00.0",
                    "mlx5_4": "0009:06:00.0", "mlx5_5": "0009:06:00.0",
                    "mlx5_8": "0018:06:00.0", "mlx5_9": "0018:06:00.0",
                    "mlx5_10": "0019:06:00.0", "mlx5_11": "0019:06:00.0"},
        "waves": [["pair1", "pair3"], ["pair2", "pair4"], ["pair5"], ["pair6"]],
    },
}
DEFAULT_PLATFORM = "GB300_L10"
PLATFORM_FILE = "rdma_platforms.json"


def _sysfs(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def detect_card_type(dev: str) -> str:
    """BF3 / CX8 / CX7 / CX6 / UNKNOWN"""
    base = f"/sys/class/infiniband/{dev}/device"
    dev_id = _sysfs(f"{base}/device")
    if dev_id:
        try:
            i = int(dev_id, 16)
            if i in (0xa2d0, 0xa2d2, 0xa2d6, 0xa2dc):
                return "BF3"
            if i == 0x1023:
                return "CX8"
            if i == 0x1021:
                return "CX7"
            if i in (0x101b, 0x101d, 0x101f):
                return "CX6"
        except ValueError:
            pass
    board = (_sysfs(f"/sys/class/infiniband/{dev}/board_id") or "").upper()
    for key, card in (("MBF3", "BF3"), ("MT41692", "BF3"), ("MT4131", "CX8"), ("MCX8", "CX8")):
        if key in board:
            return card
    _, out = run("mst status -v 2>/dev/null")
    for line in out.splitlines():
        tok = line.split()
        if dev in tok and tok:
            first = tok[0].upper()
            if "BLUEFIELD3" in first:
                return "BF3"
            if "CONNECTX8" in first:
                return "CX8"
            if "CONNECTX7" in first:
                return "CX7"
            if "CONNECTX6" in first:
                return "CX6"
    return "UNKNOWN"


def detect_link_gbps(dev: str) -> int | None:
    """/sys/class/infiniband/<dev>/ports/1/rate → '400 Gb/sec (4X NDR)' → 400"""
    rate = _sysfs(f"/sys/class/infiniband/{dev}/ports/1/rate")
    if rate:
        try:
            return int(float(rate.split()[0]))
        except (ValueError, IndexError):
            pass
    return None


def nic_card_id(dev: str) -> str:
    """實體卡識別：PCI BDF 去掉 function（0000:01:00.1 → 0000:01:00）"""
    target = os.path.realpath(f"/sys/class/infiniband/{dev}/device")
    bdf = os.path.basename(target)
    return bdf.rsplit(".", 1)[0] if "." in bdf else dev


def list_gpus() -> list[tuple[str, int]]:
    """[(bus_id 'DDDD:BB:DD.F', numa), ...]，依 nvidia-smi index 排序"""
    code, out = run("nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null")
    gpus = []
    if code != 0:
        return gpus
    for line in out.splitlines():
        bus = line.strip()
        if not bus:
            continue
        dom, rest = bus.split(":", 1)
        bus = f"{int(dom, 16):04x}:{rest}".lower()
        numa = _sysfs(f"/sys/bus/pci/devices/{bus}/numa_node")
        gpus.append((bus, int(numa) if numa and numa.lstrip("-").isdigit() else 0))
    return gpus


def _auto_gpu_map(pairs: list, numa_map: dict) -> dict[str, str]:
    """每個 pair：server 用該 NUMA 的 GPU[k]，client 盡量用同 NUMA 的另一顆"""
    gpus = list_gpus()
    if not gpus:
        return {}
    by_numa: dict[int, list[str]] = {}
    for bus, numa in gpus:
        by_numa.setdefault(numa, []).append(bus)
    all_bus = [b for b, _ in gpus]
    gmap = {}
    for k, (s, c, *_rest) in enumerate(pairs):
        s_list = by_numa.get(numa_map.get(s, 0)) or all_bus
        c_list = by_numa.get(numa_map.get(c, 0)) or all_bus
        s_gpu = s_list[k % len(s_list)]
        c_cands = [g for g in c_list if g != s_gpu] or c_list
        c_gpu = c_cands[k % len(c_cands)]
        gmap.setdefault(s, s_gpu)
        gmap.setdefault(c, c_gpu)
    return gmap


def _auto_waves(pairs: list) -> list[list[str]]:
    """同一張實體卡的 port 不同批"""
    waves: list[tuple[set, list]] = []
    for s, c, pname, *_ in pairs:
        cards = {nic_card_id(s), nic_card_id(c)}
        for used, names in waves:
            if not (used & cards):
                used |= cards
                names.append(pname)
                break
        else:
            waves.append((set(cards), [pname]))
    return [names for _, names in waves]


def find_platform_file(search_dirs: list[str]) -> str | None:
    for d in search_dirs:
        p = os.path.join(d, PLATFORM_FILE)
        if os.path.isfile(p):
            return p
    return None


def load_platforms(search_dirs: list[str]) -> tuple[dict, str, str | None]:
    """回傳 (platforms, default_name, file_path)"""
    path = find_platform_file(search_dirs)
    if not path:
        return dict(BUILTIN_PLATFORMS), DEFAULT_PLATFORM, None
    with open(path) as f:
        data = json.load(f)
    plats = {k: v for k, v in data.get("platforms", {}).items() if not k.startswith("_")}
    for k, v in BUILTIN_PLATFORMS.items():
        plats.setdefault(k, v)
    return plats, data.get("default", DEFAULT_PLATFORM), path


def resolve_platform(search_dirs: list[str], name: str | None,
                     line_rate_pct: float, dry_run: bool = False) -> dict:
    """
    回傳:
      name, source, pairs [(server, client, pairN, card)], card_cfg {card: {...}},
      gpu_map {dev: bus}, waves [[pairN, ...]], numa_map {dev: numa}, auto [欄位名稱]
    """
    plats, default, path = load_platforms(search_dirs)
    name = name or default
    if name not in plats:
        raise SystemExit(f"[ERROR] platform '{name}' not found (available: {', '.join(sorted(plats))})")
    prof = plats[name]
    pct = float(prof.get("line_rate_pct", line_rate_pct))
    auto = []

    raw_pairs = prof.get("pairs") or []
    if not raw_pairs:
        raise SystemExit(f"[ERROR] platform '{name}' has no 'pairs'")
    devs = [d for p in raw_pairs for d in p[:2]]
    numa_map = {d: get_nic_numa(d) for d in devs}

    cards = dict(prof.get("cards", {}))
    if any(d not in cards for d in devs):
        auto.append("cards")
        for d in devs:
            cards.setdefault(d, detect_card_type(d))

    pairs = []
    for i, p in enumerate(raw_pairs, start=1):
        s, c = p[0], p[1]
        if cards[s] != cards[c]:
            warn(f"pair{i} {s}<->{c}: card types differ ({cards[s]} / {cards[c]}), using {cards[s]}")
        pairs.append((s, c, f"pair{i}", cards[s]))

    link = dict(prof.get("link_gbps", {}))
    card_cfg = {}
    for card in sorted({p[3] for p in pairs}):
        beh = dict(CARD_BEHAVIOR.get(card, CARD_BEHAVIOR["UNKNOWN"]))
        beh.update(prof.get("card_behavior", {}).get(card, {}))
        gbps = link.get(card)
        if gbps is None:
            rates = [detect_link_gbps(p[0]) for p in pairs if p[3] == card]
            rates = [r for r in rates if r]
            gbps = min(rates) if rates else None
            if "link_gbps" not in auto:
                auto.append("link_gbps")
        beh["link_gbps"] = gbps
        beh["threshold"] = gbps * pct if gbps else None
        card_cfg[card] = beh

    gpu_map = dict(prof.get("gpu_map", {}))
    if any(d not in gpu_map for d in devs):
        auto.append("gpu_map")
        for d, b in _auto_gpu_map(pairs, numa_map).items():
            gpu_map.setdefault(d, b)

    waves = prof.get("waves")
    if not waves:
        auto.append("waves")
        waves = _auto_waves(pairs)

    return {"name": name, "source": path or "built-in", "description": prof.get("description", ""),
            "pairs": pairs, "card_cfg": card_cfg, "gpu_map": gpu_map, "waves": waves,
            "numa_map": numa_map, "line_rate_pct": pct, "auto": auto}


def print_platform(plat: dict):
    section(f"Platform: {plat['name']}  ({plat['source']})")
    if plat["description"]:
        info(f"  {plat['description']}")
    if plat["auto"]:
        warn(f"  auto-detected (not in profile): {', '.join(plat['auto'])} — verify, then pin in {PLATFORM_FILE}")
    for s, c, pname, card in plat["pairs"]:
        cfg = plat["card_cfg"][card]
        thr = f"{cfg['threshold']:.0f}" if cfg["threshold"] else "N/A"
        info(f"  {pname}: {s:<8} <-> {c:<8} {card:<4} link={cfg['link_gbps'] or '?'}G thr={thr} "
             f"GPU {plat['gpu_map'].get(s, '-')} / {plat['gpu_map'].get(c, '-')}")
    info(f"  waves: " + " | ".join("+".join(w) for w in plat["waves"]))
