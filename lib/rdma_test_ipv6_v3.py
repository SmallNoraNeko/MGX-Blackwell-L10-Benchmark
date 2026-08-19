#!/usr/bin/env python3
"""
rdma_test_ipv6_v3.py — GB300 MGX RDMA Loopback 效能測試（IPv6 v3）
===================================================================
v3 修正重點：
  1. 補齊 make_server_cmd / make_client_cmd / run_test（v2 完全缺失）
  2. 捨棄 fe80:: link-local + %netdev 方案（ib_*_bw 不支援 scope ID）
  3. 改用 IPv4 assign mode IP → 轉成 IPv4-mapped IPv6 格式傳給 client
     格式：0000:0000:0000:0000:0000:ffff:<hi16>:<lo16>
     與 .sh baseline 完全對齊
  4. Server / Client 皆加 --ipv6 flag，GID 繼續用 -x 3
  5. NUMA map 修正：mlx5_6/7 改回 Node 1（與 IPv4 v2 對齊）
  6. Topology 對齊 IPv4 v2：6 對 CX8 loopback（mlx5_0~11 全部使用）
"""

from __future__ import annotations

import atexit
import argparse
import ipaddress
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rdma_common import (
    C, info, ok, warn, error, title, section,
    run, ts,
    get_nic_primary_ipv6,
    dump_gid_table,
    rdma_to_netdev,
    discover_and_pair_nics,
    setup_kernel_params,
    setup_mtu,
    clear_all,
    collect_all,
    stream_output,
    generate_summary,
    check_prerequisites,
    assign_ipv4_addr,
    release_ipv4_addr,
    ping_ipv4_from_nic,
    wait_for_gid_index,
    get_card_type_from_rdma,
    use_rdma_cm_for_card,
)

# ──────────────────────────────────────────────
# 靜態拓撲（對齊 IPv4 v2 的 6 對 CX8 loopback 接線）
# ──────────────────────────────────────────────
_STATIC_PAIRS = [
    ("mlx5_0",  "mlx5_1",   "pair1"),
    ("mlx5_2",  "mlx5_4",   "pair2"),
    ("mlx5_3",  "mlx5_5",   "pair3"),
    ("mlx5_6",  "mlx5_7",   "pair4"),
    ("mlx5_8",  "mlx5_10",  "pair5"),
    ("mlx5_9",  "mlx5_11",  "pair6"),
]

# 修正：mlx5_6/7 改為 Node 1，與 IPv4 v2 對齊
_STATIC_NUMA_MAP = {
    "mlx5_0": 0,  "mlx5_1": 0,  "mlx5_2": 0,  "mlx5_3": 0,
    "mlx5_4": 0,  "mlx5_5": 0,  "mlx5_6": 1,  "mlx5_7": 1,
    "mlx5_8": 1,  "mlx5_9": 1,  "mlx5_10": 1, "mlx5_11": 1,
}

# fe80:: 表僅保留供 show_gids 參考，不再用於打流
_STATIC_GID1: dict[str, str] = {
    "mlx5_0":  "fe80::92e3:17ff:fe25:96ee",
    "mlx5_1":  "fe80::92e3:17ff:fe25:96ef",
    "mlx5_2":  "fe80::da94:24ff:fe9d:4b40",
    "mlx5_3":  "fe80::da94:24ff:fe9d:4b41",
    "mlx5_4":  "fe80::da94:24ff:fe9d:4b10",
    "mlx5_5":  "fe80::da94:24ff:fe9d:4b11",
    "mlx5_6":  "fe80::92e3:17ff:fe25:3730",
    "mlx5_7":  "fe80::92e3:17ff:fe25:3731",
    "mlx5_8":  "fe80::da94:24ff:feaf:4a8e",
    "mlx5_9":  "fe80::da94:24ff:feaf:4a8f",
    "mlx5_10": "fe80::da94:24ff:feaf:4a5e",
    "mlx5_11": "fe80::da94:24ff:feaf:4a5f",
}

CUDA_MAP: dict[str, int] = {
    "mlx5_0": 0, "mlx5_2": 1, "mlx5_4": 2, "mlx5_5": 2,
    "mlx5_6": 3, "mlx5_8": 1, "mlx5_10": 3, "mlx5_11": 3,
}

MSG_SIZES_FULL = [
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
]

PORT_BASE    = {"read": 14000, "send": 15000, "write": 16000}
DURATION_SEC = 30
QPS          = 16
SERVER_WAIT  = 3
MTU_TARGET   = 0        # 0 = do not force MTU (recommended); set via --mtu if needed
GID_INDEX    = 3          # IPv6 模式下與 .sh baseline 一致用 -x 3
OUTPUT_BASE  = "./rdma_results_ipv6"

# ──────────────────────────────────────────────
# IPv4-mapped IPv6 轉換工具
# ──────────────────────────────────────────────
def ipv4_to_mapped_ipv6(ipv4: str) -> str:
    """
    將 IPv4 字串轉為 IPv4-mapped IPv6 全展開格式。
    例：192.168.100.1 → 0000:0000:0000:0000:0000:ffff:c0a8:6401
    與 .sh baseline 的格式完全一致。
    """
    addr = ipaddress.IPv4Address(ipv4)
    packed = addr.packed                          # 4 bytes
    hi = (packed[0] << 8) | packed[1]            # 前兩 bytes
    lo = (packed[2] << 8) | packed[3]            # 後兩 bytes
    return f"0000:0000:0000:0000:0000:ffff:{hi:04x}:{lo:04x}"

# ──────────────────────────────────────────────
# 動態 IPv4 指派（借用 IPv4 v2 相同邏輯，base_cidr=200 避免衝突）
# ──────────────────────────────────────────────
def assign_dynamic_ipv4(pairs: list, base_cidr: int = 200) -> dict[str, str]:
    ip_map: dict[str, str] = {}
    for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
        srv_ip = f"192.168.{base_cidr + idx}.1"
        cli_ip = f"192.168.{base_cidr + idx}.2"
        assign_ipv4_addr(srv_nic, srv_ip, 24)
        assign_ipv4_addr(cli_nic, cli_ip, 24)
        ip_map[srv_nic] = srv_ip
        ip_map[cli_nic] = cli_ip
    time.sleep(2.0)
    return ip_map

def teardown_dynamic_ipv4(pairs: list):
    for srv_nic, cli_nic, _ in pairs:
        release_ipv4_addr(srv_nic)
        release_ipv4_addr(cli_nic)

# ──────────────────────────────────────────────
# 指令組合
# ──────────────────────────────────────────────
def _tool(test_type: str) -> str:
    return {"read": "ib_read_bw", "send": "ib_send_bw", "write": "ib_write_bw"}[test_type]

def _cuda_flags(nic: str) -> str:
    dev = CUDA_MAP.get(nic)
    if dev is None:
        return ""
    return f"--use_cuda={dev} --use_cuda_dmabuf"

def kill_existing_rdma_tools():
    for tool in ["ib_read_bw", "ib_send_bw", "ib_write_bw"]:
        run(f"pkill -9 -f {tool} >/dev/null 2>&1")
    time.sleep(0.5)

def make_server_cmd(
    test_type: str, srv_nic: str, port: int, msg_size: int,
    numa_map: dict, duration: int, qps: int,
    dyn_ip_map: dict[str, str],
    enable_cuda: bool = True,
) -> str:
    numa    = numa_map.get(srv_nic, 0)
    tool    = _tool(test_type)
    srv_ip  = dyn_ip_map.get(srv_nic, "")
    gid_idx = wait_for_gid_index(srv_nic, srv_ip, default_gid=GID_INDEX)

    # BF3 requires RDMA CM flag (-R); CX8/CX7/CX6 do not
    card_type = get_card_type_from_rdma(srv_nic)
    rdma_cm   = " -R" if use_rdma_cm_for_card(card_type) else ""

    flags = f"-F -q {qps} -s {msg_size} -D {duration} -x {gid_idx} --report_gbits -b{rdma_cm} --ipv6"
    cuda  = _cuda_flags(srv_nic) if (enable_cuda and test_type == "write") else ""

    cmd = f"numactl --cpunodebind={numa} --membind={numa} {tool} -d {srv_nic} -p {port} {flags}"
    if cuda:
        cmd += f" {cuda}"
    return cmd

def make_client_cmd(
    test_type: str, srv_nic: str, cli_nic: str, port: int, msg_size: int,
    numa_map: dict, duration: int, qps: int,
    dyn_ip_map: dict[str, str],
    enable_cuda: bool = True,
) -> tuple[str, str | None]:
    srv_ipv4 = dyn_ip_map.get(srv_nic)
    cli_ipv4 = dyn_ip_map.get(cli_nic)
    if not srv_ipv4:
        return "", None

    srv_ipv6_mapped = ipv4_to_mapped_ipv6(srv_ipv4)

    numa    = numa_map.get(cli_nic, 0)
    tool    = _tool(test_type)
    gid_idx = wait_for_gid_index(cli_nic, cli_ipv4, default_gid=GID_INDEX)

    # BF3 requires RDMA CM flag (-R); CX8/CX7/CX6 do not
    card_type = get_card_type_from_rdma(cli_nic)
    rdma_cm   = " -R" if use_rdma_cm_for_card(card_type) else ""

    flags = f"-F -q {qps} -s {msg_size} -D {duration} -x {gid_idx} --report_gbits -b{rdma_cm} --ipv6"
    cuda  = _cuda_flags(cli_nic) if (enable_cuda and test_type == "write") else ""

    cmd = f"numactl --cpunodebind={numa} --membind={numa} {tool} -d {cli_nic} -p {port} {flags}"
    if cuda:
        cmd += f" {cuda}"
    cmd += f" {srv_ipv6_mapped}"
    return cmd, srv_ipv6_mapped

# ──────────────────────────────────────────────
# 連通性驗證（ping IPv4，確保 IP 已生效）
# ──────────────────────────────────────────────
def verify_connectivity(pairs: list, dyn_ip_map: dict[str, str]) -> bool:
    info("Running IPv4 connectivity check before IPv6 test...")
    all_ok = True
    for srv_nic, cli_nic, pname in pairs:
        srv_ip = dyn_ip_map.get(srv_nic)
        if not srv_ip:
            error(f"  {pname}: server {srv_nic} IP unavailable")
            all_ok = False
            continue
        if not ping_ipv4_from_nic(cli_nic, srv_ip, count=2, timeout=2):
            error(f"  {pname}: ping failed {cli_nic} -> {srv_ip}")
            all_ok = False
    if not all_ok:
        error("Connectivity check failed — aborting")
        return False
    ok("Connectivity check passed")
    return True

# ──────────────────────────────────────────────
# 核心測試執行
# ──────────────────────────────────────────────
def run_test(
    test_type: str, pairs: list, numa_map: dict, msg_sizes: list,
    duration: int, qps: int, dyn_ip_map: dict[str, str],
    dry_run: bool = False, enable_cuda: bool = True,
    serial: bool = False,
):
    title(f"Running: {test_type.upper()}_IPv6  ({'serial' if serial else 'parallel'})")
    base_port = PORT_BASE[test_type]
    out_dir   = os.path.join(OUTPUT_BASE, f"{test_type}_ipv6")
    os.makedirs(out_dir, exist_ok=True)

    for msg_size in msg_sizes:
        print(f"\n  --> [{test_type}_ipv6]  msg_size={msg_size} B", flush=True)
        kill_existing_rdma_tools()

        if serial:
            # ── Serial mode: one pair at a time ──────────────────────────
            for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
                port  = base_port + idx * 50
                scmd  = make_server_cmd(test_type, srv_nic, port, msg_size, numa_map, duration, qps, dyn_ip_map, enable_cuda)
                ccmd, target = make_client_cmd(test_type, srv_nic, cli_nic, port, msg_size, numa_map, duration, qps, dyn_ip_map, enable_cuda)
                srv_log = os.path.join(out_dir, f"server_{pname}_{srv_nic}_{test_type}_ipv6_msg{msg_size}.log")
                cli_log = os.path.join(out_dir, f"client_{pname}_{cli_nic}_{test_type}_ipv6_msg{msg_size}.log")

                card_type = get_card_type_from_rdma(srv_nic)
                info(f"  [{card_type}] SRV {srv_nic} <-> CLI {cli_nic}  port={port}  target={target}")
                if dry_run:
                    info(f"    [dry-run] {scmd}")
                    info(f"    [dry-run] {ccmd}")
                    continue

                srv_lines = [f"COMMAND: {scmd}\n"]
                proc_s = subprocess.Popen(scmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                t_s = threading.Thread(target=stream_output, args=(proc_s, f"SRV-{srv_nic}", srv_lines), daemon=True)
                t_s.start()
                time.sleep(SERVER_WAIT)

                cli_lines = [f"COMMAND: {ccmd}\n"]
                proc_c = subprocess.Popen(ccmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                t_c = threading.Thread(target=stream_output, args=(proc_c, f"CLI-{cli_nic}", cli_lines), daemon=True)
                t_c.start()

                max_timeout = duration + 25
                try:
                    proc_c.wait(timeout=max_timeout)
                except subprocess.TimeoutExpired:
                    warn("Client timeout — killing")
                    proc_c.kill()
                t_c.join(timeout=3)
                with open(cli_log, "w") as f: f.writelines(cli_lines)

                try:
                    proc_s.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc_s.kill()
                t_s.join(timeout=3)
                with open(srv_log, "w") as f: f.writelines(srv_lines)

                kill_existing_rdma_tools()
        else:
            # ── Parallel mode: all pairs at once ─────────────────────────
            srv_jobs, cli_jobs = [], []

            for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
                port  = base_port + idx * 50
                scmd  = make_server_cmd(test_type, srv_nic, port, msg_size, numa_map, duration, qps, dyn_ip_map, enable_cuda)
                srv_log   = os.path.join(out_dir, f"server_{pname}_{srv_nic}_{test_type}_ipv6_msg{msg_size}.log")
                log_lines = [f"COMMAND: {scmd}\n"]

                card_type = get_card_type_from_rdma(srv_nic)
                info(f"  [{card_type}] SRV {srv_nic}: port={port}  {scmd}")
                if dry_run:
                    continue

                proc = subprocess.Popen(scmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                t    = threading.Thread(target=stream_output, args=(proc, f"SRV-{srv_nic}", log_lines), daemon=True)
                t.start()
                srv_jobs.append((proc, t, log_lines, srv_log))
                time.sleep(0.1)

            if dry_run:
                continue

            time.sleep(SERVER_WAIT)

            for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
                port  = base_port + idx * 50
                ccmd, target = make_client_cmd(test_type, srv_nic, cli_nic, port, msg_size, numa_map, duration, qps, dyn_ip_map, enable_cuda)
                cli_log   = os.path.join(out_dir, f"client_{pname}_{cli_nic}_{test_type}_ipv6_msg{msg_size}.log")
                log_lines = [f"COMMAND: {ccmd}\n"]

                info(f"  CLI {cli_nic}: port={port}  target={target}  {ccmd}")
                proc_c = subprocess.Popen(ccmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                t_c    = threading.Thread(target=stream_output, args=(proc_c, f"CLI-{cli_nic}", log_lines), daemon=True)
                t_c.start()
                cli_jobs.append((proc_c, t_c, log_lines, cli_log))
                time.sleep(0.1)

            max_timeout = duration + 25

            for proc, t, log_lines, log_path in cli_jobs:
                if proc:
                    try:
                        proc.wait(timeout=max_timeout)
                    except subprocess.TimeoutExpired:
                        warn("Client timeout — killing")
                        proc.kill()
                if t: t.join(timeout=3)
                with open(log_path, "w") as f: f.writelines(log_lines)

            for proc, t, log_lines, log_path in srv_jobs:
                if proc:
                    try:
                        proc.wait(timeout=max_timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                if t: t.join(timeout=3)
                with open(log_path, "w") as f: f.writelines(log_lines)

# ──────────────────────────────────────────────
# show_gids（保留參考用）
# ──────────────────────────────────────────────
def show_gids(pairs: list):
    title("IPv6 GID Reference Table (fe80:: link-local, info only)")
    for nic in sorted({n for p in pairs for n in (p[0], p[1])}):
        gid     = _STATIC_GID1.get(nic, "(unknown)")
        netdev  = rdma_to_netdev(nic) or "(unknown)"
        ok(f"  {nic}: {gid}  netdev={netdev}")

# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    global OUTPUT_BASE

    parser = argparse.ArgumentParser(description="GB300 MGX — RDMA Loopback IPv6 Performance Suite v3")
    parser.add_argument("--test",        nargs="+", choices=["read", "send", "write"], default=["read", "send", "write"])
    parser.add_argument("--msg-size",    type=int,  default=None)
    parser.add_argument("--duration",    type=int,  default=DURATION_SEC)
    parser.add_argument("--qps",         type=int,  default=QPS)
    parser.add_argument("--server-wait", type=int,  default=SERVER_WAIT)
    parser.add_argument("--mtu",         type=int,  default=MTU_TARGET)
    parser.add_argument("--output-dir",  default="./collected_logs_ipv6")
    parser.add_argument("--no-cuda",     action="store_true")
    parser.add_argument("--show-gids",   action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--no-clear",    action="store_true")
    parser.add_argument("--no-collect",  action="store_true")
    parser.add_argument("--serial",      action="store_true",
                        help="Run one pair at a time (serial mode). "
                             "Eliminates resource contention, gives peak single-pair BW.")
    args = parser.parse_args()

    if os.geteuid() != 0 and not args.dry_run:
        error("請以 root 權限執行 (sudo)")
        sys.exit(1)

    pairs    = list(_STATIC_PAIRS)
    numa_map = dict(_STATIC_NUMA_MAP)

    if args.show_gids:
        show_gids(pairs)
        return

    check_prerequisites(pairs)

    # ── atexit 清理 IPv4 ──
    dyn_ip_map: dict[str, str] = {}
    def cleanup():
        teardown_dynamic_ipv4(pairs)
    atexit.register(cleanup)

    if not args.dry_run:
        # 0. 清空 pre-test logs
        info("Clearing pre-test system logs...")
        run("dmesg -C >/dev/null 2>&1")
        run("journalctl --rotate >/dev/null 2>&1 && journalctl --vacuum-time=1s >/dev/null 2>&1")
        run("ipmitool sel clear >/dev/null 2>&1")

        # 1. 清理舊結果目錄
        if not args.no_clear:
            clear_all(OUTPUT_BASE)

        # 2. Kernel 參數
        setup_kernel_params()

        # 3. MTU
        if args.mtu > 0:
            setup_mtu(pairs, numa_map, args.mtu)

        # 4. 指派 IPv4（base_cidr=200 避免與 IPv4 v2 的 100-105 衝突）
        dyn_ip_map = assign_dynamic_ipv4(pairs, base_cidr=200)

        # 5. 連通性確認
        if not verify_connectivity(pairs, dyn_ip_map):
            sys.exit(2)

        # 6. 顯示 IPv4-mapped IPv6 對照表
        section("IPv4 → IPv4-mapped IPv6 target table")
        for srv_nic, cli_nic, pname in pairs:
            srv_ipv4 = dyn_ip_map.get(srv_nic, "N/A")
            mapped   = ipv4_to_mapped_ipv6(srv_ipv4) if srv_ipv4 != "N/A" else "N/A"
            info(f"  {pname}: {srv_nic}({srv_ipv4}) → client target = {mapped}")

    msg_sizes   = [args.msg_size] if args.msg_size else MSG_SIZES_FULL
    enable_cuda = not args.no_cuda

    for test_type in args.test:
        run_test(test_type, pairs, numa_map, msg_sizes, args.duration, args.qps,
                 dyn_ip_map, dry_run=args.dry_run, enable_cuda=enable_cuda,
                 serial=args.serial)

    if not args.dry_run:
        generate_summary(OUTPUT_BASE, "ipv6")

        if not args.no_collect:
            info("Collecting post-test logs...")
            os.makedirs(args.output_dir, exist_ok=True)
            run(f"dmesg > {os.path.join(args.output_dir, 'dmesg.log')} 2>&1")
            run(f"journalctl --no-pager > {os.path.join(args.output_dir, 'system.log')} 2>&1")
            run(f"ipmitool sel list > {os.path.join(args.output_dir, 'sel.log')} 2>&1")
            run(f"ipmitool sdr elist > {os.path.join(args.output_dir, 'sdr.log')} 2>&1")
            collect_all(OUTPUT_BASE, args.output_dir)

if __name__ == "__main__":
    main()
