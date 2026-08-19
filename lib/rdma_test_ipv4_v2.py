#!/usr/bin/env python3
"""
rdma_test_ipv4_v2.py — GB300 MGX RDMA Loopback 效能測試（IPv4 動態拓撲與 GID 修正版）
"""

from __future__ import annotations

import atexit
import argparse
import datetime
import ipaddress
import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rdma_common import (
    C, info, ok, warn, error, title, section,
    run, ts,
    get_nic_primary_ipv4,
    get_nic_primary_ipv6,
    dump_gid_table,
    wait_for_gid_index,
    wait_for_port,
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
    get_card_type_from_rdma,
    use_rdma_cm_for_card,
)

# 根據 CX8 設定檔 (ibstress topology) 修正為真正在機櫃內部的實體 Loopback 連線拓撲
_STATIC_PAIRS = [
    ("mlx5_0",  "mlx5_1",   "pair1"),
    ("mlx5_2",  "mlx5_4",   "pair2"),
    ("mlx5_3",  "mlx5_5",   "pair3"),
    ("mlx5_6",  "mlx5_7",   "pair4"),
    ("mlx5_8",  "mlx5_10",  "pair5"),
    ("mlx5_9",  "mlx5_11",  "pair6"),
]

_STATIC_NUMA_MAP = {
    "mlx5_0": 0,  "mlx5_1": 0,  "mlx5_2": 0,  "mlx5_3": 0,
    "mlx5_4": 0,  "mlx5_5": 0,  "mlx5_6": 1,  "mlx5_7": 1,
    "mlx5_8": 1,  "mlx5_9": 1,  "mlx5_10": 1, "mlx5_11": 1,
}

CUDA_MAP: dict[str, int] = {}

MSG_SIZES_FULL = [
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
]

PORT_BASE = {
    "read":  11000,
    "send":  12000,
    "write": 13000,
}

DURATION_SEC = 30
QPS          = 16
SERVER_WAIT  = 3
MTU_TARGET   = 0        # 0 = do not force MTU (recommended); set via --mtu if needed
OUTPUT_BASE  = "./rdma_results_ipv4"

def _is_usable_ipv4_literal(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and not (addr.is_unspecified or addr.is_loopback or addr.is_multicast)

def resolve_ipv4(nic: str) -> str | None:
    ip, idx = get_nic_primary_ipv4(nic)
    if ip and _is_usable_ipv4_literal(ip):
        return ip
    return None

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

def make_server_cmd(test_type: str, srv_nic: str, port: int, msg_size: int, numa_map: dict, duration: int, qps: int, ip_mode: str = "assign", dyn_ip_map: dict[str, str] | None = None, enable_cuda: bool = True, user_gid: int | None = None) -> str:
    numa = numa_map.get(srv_nic, 0)
    tool = _tool(test_type)

    srv_ip  = dyn_ip_map.get(srv_nic) if (ip_mode == "assign" and dyn_ip_map) else resolve_ipv4(srv_nic)
    gid_idx = wait_for_gid_index(srv_nic, srv_ip, default_gid=user_gid)

    # BF3 requires RDMA CM flag (-R); CX8/CX7/CX6 do not
    card_type = get_card_type_from_rdma(srv_nic)
    rdma_cm   = " -R" if use_rdma_cm_for_card(card_type) else ""

    flags = f"-F -q {qps} -s {msg_size} -D {duration} -x {gid_idx} --report_gbits -b{rdma_cm}"

    cuda = _cuda_flags(srv_nic) if (enable_cuda and test_type == "write") else ""
    cmd  = f"numactl --cpunodebind={numa} --membind={numa} {tool} -d {srv_nic} -p {port} {flags}"
    if cuda:
        cmd += f" {cuda}"
    return cmd

def make_client_cmd(test_type: str, srv_nic: str, cli_nic: str, port: int, msg_size: int, numa_map: dict, duration: int, qps: int, ip_mode: str = "assign", dyn_ip_map: dict[str, str] | None = None, enable_cuda: bool = True, user_gid: int | None = None) -> tuple[str, str | None]:
    if ip_mode == "assign":
        if not dyn_ip_map: return "", None
        srv_ip = dyn_ip_map.get(srv_nic)
        cli_ip = dyn_ip_map.get(cli_nic)
    else:
        srv_ip = resolve_ipv4(srv_nic)
        cli_ip = resolve_ipv4(cli_nic)

    if not srv_ip: return "", None

    numa        = numa_map.get(cli_nic, 0)
    tool        = _tool(test_type)
    cli_gid_idx = wait_for_gid_index(cli_nic, cli_ip, default_gid=user_gid)

    # BF3 requires RDMA CM flag (-R); CX8/CX7/CX6 do not
    card_type = get_card_type_from_rdma(cli_nic)
    rdma_cm   = " -R" if use_rdma_cm_for_card(card_type) else ""

    flags = f"-F -q {qps} -s {msg_size} -D {duration} -x {cli_gid_idx} --report_gbits -b{rdma_cm}"

    cuda = _cuda_flags(cli_nic) if (enable_cuda and test_type == "write") else ""
    cmd  = f"numactl --cpunodebind={numa} --membind={numa} {tool} -d {cli_nic} -p {port} {flags}"
    if cuda:
        cmd += f" {cuda}"
    cmd += f" {srv_ip}"
    return cmd, srv_ip

def assign_dynamic_ipv4(pairs: list, base_cidr: int = 100) -> dict[str, str]:
    ip_map: dict[str, str] = {}
    for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
        srv_ip = f"192.168.{base_cidr + idx}.1"
        cli_ip = f"192.168.{base_cidr + idx}.2"
        subnet = 24

        assign_ipv4_addr(srv_nic, srv_ip, subnet)
        assign_ipv4_addr(cli_nic, cli_ip, subnet)

        ip_map[srv_nic] = srv_ip
        ip_map[cli_nic] = cli_ip
    
    time.sleep(2.0)
    return ip_map

def teardown_dynamic_ipv4(pairs: list):
    for srv_nic, cli_nic, _ in pairs:
        release_ipv4_addr(srv_nic)
        release_ipv4_addr(cli_nic)

def verify_assign_mode_connectivity(pairs: list, ip_mode: str, dyn_ip_map: dict[str, str] | None) -> bool:
    info(f"Running connectivity check ({ip_mode} mode)...")
    all_ok = True
    for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
        srv_ip = dyn_ip_map.get(srv_nic) if ip_mode == "assign" else resolve_ipv4(srv_nic)
        if not srv_ip:
            error(f"  {pname}: server {srv_nic} IP unavailable")
            all_ok = False
            continue

        if not ping_ipv4_from_nic(cli_nic, srv_ip, count=2, timeout=2):
            error(f"  {pname}: connectivity test failed: {cli_nic} -> {srv_ip}")
            all_ok = False

    if not all_ok:
        error("Connectivity test failed — aborting execution")
        return False

    ok("Connectivity test passed")
    return True

def run_test(test_type: str, pairs: list, numa_map: dict, msg_sizes: list, duration: int, qps: int, dry_run: bool = False, ip_mode: str = "assign", dyn_ip_map: dict[str, str] | None = None, enable_cuda: bool = True, user_gid: int | None = None, serial: bool = False):
    title(f"Running: {test_type.upper()}_IPv4  ({'serial' if serial else 'parallel'})")
    base_port = PORT_BASE[test_type]
    out_dir   = os.path.join(OUTPUT_BASE, f"{test_type}_ipv4")
    os.makedirs(out_dir, exist_ok=True)

    for msg_size in msg_sizes:
        print(f"\n  --> [{test_type}_ipv4]  msg_size={msg_size} B", flush=True)
        kill_existing_rdma_tools()

        if serial:
            # ── Serial mode: one pair at a time ──────────────────────────
            for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
                port  = base_port + idx * 50
                scmd  = make_server_cmd(test_type, srv_nic, port, msg_size, numa_map, duration, qps, ip_mode, dyn_ip_map, enable_cuda, user_gid)
                ccmd, srv_ip = make_client_cmd(test_type, srv_nic, cli_nic, port, msg_size, numa_map, duration, qps, ip_mode, dyn_ip_map, enable_cuda, user_gid)
                srv_log = os.path.join(out_dir, f"server_{pname}_{srv_nic}_{test_type}_ipv4_msg{msg_size}.log")
                cli_log = os.path.join(out_dir, f"client_{pname}_{cli_nic}_{test_type}_ipv4_msg{msg_size}.log")

                card_type = get_card_type_from_rdma(srv_nic)
                info(f"  [{card_type}] SRV {srv_nic} <-> CLI {cli_nic}  port={port}  target={srv_ip}")
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
                    warn(f"Client timeout, killing...")
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
                scmd  = make_server_cmd(test_type, srv_nic, port, msg_size, numa_map, duration, qps, ip_mode, dyn_ip_map, enable_cuda, user_gid)
                srv_log = os.path.join(out_dir, f"server_{pname}_{srv_nic}_{test_type}_ipv4_msg{msg_size}.log")
                log_lines = [f"COMMAND: {scmd}\n"]

                card_type = get_card_type_from_rdma(srv_nic)
                info(f"  [{card_type}] SRV {srv_nic}: port={port}  {scmd}")
                if dry_run: continue

                proc = subprocess.Popen(scmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                t    = threading.Thread(target=stream_output, args=(proc, f"SRV-{srv_nic}", log_lines), daemon=True)
                t.start()
                srv_jobs.append((proc, t, log_lines, srv_log))
                time.sleep(0.1)

            if dry_run: continue
            time.sleep(SERVER_WAIT)

            for idx, (srv_nic, cli_nic, pname) in enumerate(pairs):
                port  = base_port + idx * 50
                ccmd, srv_ip = make_client_cmd(test_type, srv_nic, cli_nic, port, msg_size, numa_map, duration, qps, ip_mode, dyn_ip_map, enable_cuda, user_gid)
                cli_log   = os.path.join(out_dir, f"client_{pname}_{cli_nic}_{test_type}_ipv4_msg{msg_size}.log")
                log_lines = [f"COMMAND: {ccmd}\n"]

                info(f"  CLI {cli_nic}: port={port}  target={srv_ip}  {ccmd}")
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
                        warn(f"Client timeout reached, killing process...")
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

def main():
    global DURATION_SEC, QPS, SERVER_WAIT, MTU_TARGET, OUTPUT_BASE, CUDA_MAP

    parser = argparse.ArgumentParser(description="GB300 MGX — RDMA Loopback IPv4 Performance Suite")
    parser.add_argument("--test", nargs="+", choices=["read", "send", "write"], default=["read", "send", "write"])
    parser.add_argument("--msg-size", type=int, default=None)
    parser.add_argument("--duration", type=int, default=DURATION_SEC)
    parser.add_argument("--qps", type=int, default=QPS)
    parser.add_argument("--server-wait", type=int, default=SERVER_WAIT)
    parser.add_argument("--mtu", type=int, default=MTU_TARGET)
    parser.add_argument("--gid", type=int, default=None, help="Explicitly specify GID Index for all ports if needed")
    parser.add_argument("--output-dir", default="./collected_logs_ipv4")
    parser.add_argument("--ip-mode", choices=["gid", "assign"], default="assign")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA GPUDirect flags")
    parser.add_argument("--static", action="store_true", help="Force using static CX8 topology mapping")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--serial", action="store_true",
                        help="Run one pair at a time (serial mode). "
                             "Eliminates resource contention between pairs, "
                             "gives peak single-pair BW. Default: parallel (all pairs at once).")

    args = parser.parse_args()

    if os.geteuid() != 0 and not args.dry_run:
        error("Please run with root (sudo)")
        sys.exit(1)

    pairs, numa_map = list(_STATIC_PAIRS), dict(_STATIC_NUMA_MAP)

    check_prerequisites(pairs)

    dyn_ip_map = None
    def cleanup():
        if args.ip_mode == "assign":
            teardown_dynamic_ipv4(pairs)

    atexit.register(cleanup)

    if not args.dry_run:
        # 0. 測試前清空 dmesg、system log、sel log
        info("Clearing pre-test system logs (dmesg, journalctl, ipmitool sel)...")
        run("dmesg -C >/dev/null 2>&1")
        run("journalctl --rotate >/dev/null 2>&1 && journalctl --vacuum-time=1s >/dev/null 2>&1")
        run("ipmitool sel clear >/dev/null 2>&1")

        # 1. 清理舊日誌
        if not args.no_clear:
            clear_all(OUTPUT_BASE)

        # 2. 優先設定 Kernel 參數
        setup_kernel_params()

        # 3. 設定 MTU
        if args.mtu > 0:
            setup_mtu(pairs, numa_map, args.mtu)

        # 4. Kernel 參數生效後綁定 IP
        if args.ip_mode == "assign":
            dyn_ip_map = assign_dynamic_ipv4(pairs)

        # 5. IP 與 Kernel 參數皆到位後，執行連通性測試
        if not verify_assign_mode_connectivity(pairs, args.ip_mode, dyn_ip_map):
            sys.exit(2)

    msg_sizes = [args.msg_size] if args.msg_size else MSG_SIZES_FULL
    enable_cuda = not args.no_cuda

    for test_type in args.test:
        run_test(test_type, pairs, numa_map, msg_sizes, args.duration, args.qps,
                 args.dry_run, args.ip_mode, dyn_ip_map, enable_cuda,
                 user_gid=args.gid, serial=args.serial)

    if not args.dry_run:
        generate_summary(OUTPUT_BASE, "ipv4")
        if not args.no_collect:
            # 測試結束後額外收集 dmesg、system log、sel log、sdr log 存至指定的 output-dir
            info("Collecting post-test logs (dmesg, system log, sel log, sdr log)...")
            os.makedirs(args.output_dir, exist_ok=True)
            run(f"dmesg > {os.path.join(args.output_dir, 'dmesg.log')} 2>&1")
            run(f"journalctl -u syslog -u systemd --no-pager > {os.path.join(args.output_dir, 'system.log')} 2>&1 || journalctl --no-pager > {os.path.join(args.output_dir, 'system.log')} 2>&1")
            run(f"ipmitool sel list > {os.path.join(args.output_dir, 'sel.log')} 2>&1")
            run(f"ipmitool sdr elist > {os.path.join(args.output_dir, 'sdr.log')} 2>&1")
            
            collect_all(OUTPUT_BASE, args.output_dir)

if __name__ == "__main__":
    main()