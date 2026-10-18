#!/usr/bin/env python3
"""
rdma_test_ipv4_v4.py — GB300 MGX RDMA Loopback 效能測試（IPv4 v4）

v4 修正重點（依 2026-10-15 system10 實測 / gdr_test_v6 結論）:
  1. BF3 配對改為跨卡 mlx5_0↔mlx5_6、mlx5_1↔mlx5_7（v3 的 0↔1 / 6↔7 是同卡兩 port）
  2. BF3 使用 rdma_cm (-R)，與 v6 驗證過的 393.8 Gb/s 一致；CX8 加 -m 4096
  3. MTU 9000（RoCE active_mtu 4096）測完自動還原原值
  4. 只刪除測試自己加的 IP（v3 會 flush 掉介面上所有 IP）
  5. GID 嚴格比對 IP，找不到就報錯，不默默用錯的 GID
  6. 連通性檢查改為實際 RDMA 短測（同機 loopback 的 ping 會走 lo，無法驗證接線）
  7. NIC→GPU Bus ID 依 NUMA 對齊（v3 把 NUMA1 的 NIC 對到 NUMA0 的 GPU），啟動時檢查
  8. NUMA 從 sysfs 讀取，不再寫死
  9. PASS/FAIL 門檻判定 + exit code（供 Benchmark launcher 使用）
 10. SIGTERM / SIGINT 也會執行清理（kill ib_* / 移除 IP / 還原 MTU）
 11. --quick 模式：只跑 64KB / 256KB
 12. 門檻 = link rate 90%（BF3 400G → 360、CX8 800G → 720）
 14. BF3 的 GPU Direct 不加 --use_data_direct（BF3 不支援，會讓 MR 建立失敗而卡住）
 15. CX8 data_direct：pair 兩端用不同 GPU（server GPU0/2、client GPU1/3），同 GPU 會 run_iter_bw 失敗
 13. 預設分 4 批跑（--schedule waves），避免同卡雙 port 互搶、BF3 與 NUMA1 CX8 互相影響；
     --schedule all = 全部同時跑（整機壓力測試，數值不適用單 port 門檻）

結果為雙向 BW（-b flag），與 PD 單向值比較時需除以 2。

平台設定：--platform <name>（rdma_platforms.json），--list-platforms 列出可用平台。

Exit code: 0 = PASS, 1 = FAIL, 2 = pre-flight / link check 失敗, 130/143 = 被中斷
"""

from __future__ import annotations

import atexit
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rdma_lib import (   # 驗證版共用庫（launcher 的 _rdma_common.py 保留給拓撲探索）
    C, info, ok, warn, error, title, section,
    run,
    resolve_platform,
    load_platforms,
    print_platform,
    get_nic_numa,
    wait_for_gid_index,
    rdma_to_netdev,
    setup_kernel_params,
    setup_mtu,
    restore_mtu,
    clear_all,
    stream_output,
    generate_summary,
    check_prerequisites,
    assign_ipv4_addr,
    release_ipv4_addr,
)

# ─────────────────────────────────────────────────────────────────────────────
# 平台設定：pairs / 卡型 / 門檻 / GPU 對應 / 分批 由 rdma_platforms.json 載入
# （找不到檔案時使用 _rdma_lib.BUILTIN_PLATFORMS["GB300_L10"] — 2026-10-16 system10 驗證版）
#   pairs: (server, client, "pairN", card_type)
#   CARD_CFG[card]: rdma_cm / data_direct / link_gbps / threshold
# ─────────────────────────────────────────────────────────────────────────────
LINE_RATE_PCT = 0.90   # 門檻 = link rate 的 90%（平台檔可用 line_rate_pct 覆蓋）
PAIRS: list = []
CARD_CFG: dict = {}
WAVES: list = []
NIC_TO_GPU_BUS_ID: dict = {}
PAIR_INDEX: dict = {}

MSG_SIZES_FULL = [
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
]
MSG_SIZES_QUICK = [65536, 262144]   # BF3 最佳 64KB / CX8 最佳 256KB

# PASS/FAIL 只看 >= 此大小的結果（小封包本來就跑不滿）
JUDGE_MIN_MSG = 65536

PORT_BASE = {"read": 11000, "send": 12000, "write": 13000, "link": 14000}

DURATION_SEC = 30
QPS          = 16
SERVER_WAIT  = 3
MTU_TARGET   = 9000
IB_MTU       = 4096          # -m 值（非 rdma_cm 路徑）
OUTPUT_BASE  = "./collected_logs_ipv4/rdma_results_ipv4"   # main() 依 --output-dir 重設
SUBNET       = 24

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup state
# ─────────────────────────────────────────────────────────────────────────────
_STATE = {"assigned": [], "orig_mtu": {}, "cleaned": False}


def kill_existing_rdma_tools():
    for tool in ["ib_read_bw", "ib_send_bw", "ib_write_bw"]:
        run(f"pkill -9 -x {tool} >/dev/null 2>&1")
    time.sleep(0.5)


def cleanup():
    if _STATE["cleaned"]:
        return
    _STATE["cleaned"] = True
    kill_existing_rdma_tools()
    for dev, ip in _STATE["assigned"]:
        release_ipv4_addr(dev, ip, SUBNET)
    if _STATE["assigned"]:
        info(f"Removed {len(_STATE['assigned'])} test IPs")
    restore_mtu(_STATE["orig_mtu"])


def _on_signal(signum, _frame):
    warn(f"Received signal {signum}, cleaning up...")
    sys.exit(128 + signum)   # 觸發 atexit → cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tool(test_type: str) -> str:
    return {"read": "ib_read_bw", "send": "ib_send_bw", "write": "ib_write_bw", "link": "ib_write_bw"}[test_type]


def _cuda_flags(nic: str, card: str) -> str:
    bus_id = NIC_TO_GPU_BUS_ID.get(nic)
    if not bus_id:
        return ""
    flags = f"--use_cuda_bus_id {bus_id} --use_cuda_dmabuf"
    if CARD_CFG[card]["data_direct"]:
        flags += " --use_data_direct"
    return flags


def plan_ipv4(pairs: list, base: int = 100) -> dict[str, str]:
    """每個 pair 一個獨立 /24：192.168.(100+idx).1 / .2"""
    ip_map = {}
    for idx, p in enumerate(pairs):
        ip_map[p[0]] = f"192.168.{base + idx}.1"
        ip_map[p[1]] = f"192.168.{base + idx}.2"
    return ip_map


def assign_ips(pairs: list, ip_map: dict[str, str]):
    for p in pairs:
        for dev in (p[0], p[1]):
            ip = ip_map[dev]
            netdev = rdma_to_netdev(dev)
            _, existing = run(f"ip -4 addr show dev {netdev} | grep -w 'inet {ip}/'")
            if assign_ipv4_addr(dev, ip, SUBNET) and not existing:
                _STATE["assigned"].append((dev, ip))
    time.sleep(2.0)


def check_gpu_numa(pairs: list, numa_map: dict) -> bool:
    good = True
    for p in pairs:
        for dev in (p[0], p[1]):
            bus = NIC_TO_GPU_BUS_ID.get(dev)
            if not bus:
                continue
            path = f"/sys/bus/pci/devices/{_full_bdf(bus)}/numa_node"
            if not os.path.exists(path):
                error(f"GPU {bus} (for {dev}) not found in sysfs")
                good = False
                continue
            gpu_numa = int(open(path).read().strip())
            if gpu_numa != numa_map[dev]:
                warn(f"{dev} is NUMA{numa_map[dev]} but GPU {bus} is NUMA{gpu_numa}")
        # data_direct 時同一 pair 兩端不可共用 GPU
        s, c, pname, card = p
        if CARD_CFG[card]["data_direct"] and NIC_TO_GPU_BUS_ID.get(s) == NIC_TO_GPU_BUS_ID.get(c):
            error(f"{pname} {s}<->{c}: both ends on GPU {NIC_TO_GPU_BUS_ID.get(s)} with --use_data_direct (will fail)")
            good = False
    return good


def _full_bdf(bus: str) -> str:
    """'0008:06:00.0' -> '0008:06:00.0'；'06:00.0' -> '0000:06:00.0'"""
    return bus if bus.count(":") == 2 else f"0000:{bus}"


def build_cmd(test_type: str, dev: str, port: int, gid: int | str, msg_size: int,
              card: str, numa: int, duration: int, qps: int,
              cuda: bool, server_ip: str | None = None, link_check: bool = False) -> str:
    cfg   = CARD_CFG[card]
    tool  = _tool(test_type)
    flags = f"-d {dev} -p {port} -x {gid} -F"
    if cfg["rdma_cm"]:
        flags += " -R"                       # BF3: rdma_cm，path MTU 跟隨 netdev MTU
    else:
        flags += f" -m {IB_MTU}"             # CX8: 明確指定 4096
    if link_check:
        flags += f" -q 1 -s 65536 -n 2000 --report_gbits"
    else:
        flags += f" -q {qps} -s {msg_size} -D {duration} --report_gbits -b"
    if cuda and test_type == "write" and not link_check and _cuda_flags(dev, card):
        flags += " " + _cuda_flags(dev, card)
    cmd = f"numactl --cpunodebind={numa} --membind={numa} {tool} {flags}"
    if server_ip:
        cmd += f" {server_ip}"
        if not link_check:
            cmd += " --output=bandwidth"
    return cmd


def parse_bw(log_lines: list[str]) -> float | None:
    """支援完整表格與 --output=bandwidth 單一數值兩種格式"""
    for i, line in enumerate(log_lines):
        if "#bytes" in line and i + 1 < len(log_lines):
            parts = log_lines[i + 1].split()
            if len(parts) >= 4:
                try:
                    return float(parts[3])
                except ValueError:
                    pass
    for line in log_lines:
        s = line.strip()
        if s and not s.startswith(("W", "[", "COMMAND")):
            try:
                v = float(s)
                if v > 0:
                    return v
            except ValueError:
                continue
    return None


def _wait_and_save(jobs: list, timeout: int):
    for proc, t, log_lines, log_path in jobs:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            warn(f"Timeout, killing: {os.path.basename(log_path)}")
            proc.kill()
        t.join(timeout=3)
        if log_path:
            with open(log_path, "w") as f:
                f.writelines(log_lines)


def _launch(cmd: str, prefix: str, log_path: str | None):
    log_lines = [f"COMMAND: {cmd}\n"]
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    t = threading.Thread(target=stream_output, args=(proc, prefix, log_lines), daemon=True)
    t.start()
    return proc, t, log_lines, log_path


def resolve_gids(pairs: list, ip_map: dict[str, str], user_gid: int | None, dry_run: bool) -> dict[str, int | str] | None:
    gids = {}
    for p in pairs:
        for dev in (p[0], p[1]):
            if dry_run:
                gids[dev] = user_gid if user_gid is not None else "<gid>"
                continue
            g = wait_for_gid_index(dev, ip_map[dev], default_gid=user_gid, strict=True)
            if g is None:
                return None
            gids[dev] = g
    return gids


# ─────────────────────────────────────────────────────────────────────────────
# RDMA link check（取代 ping）
# ─────────────────────────────────────────────────────────────────────────────
def rdma_link_check(pairs, ip_map, gids, numa_map) -> bool:
    info("RDMA link check (ib_write_bw, 1 QP, 2000 iters per pair)...")
    kill_existing_rdma_tools()
    srv_jobs, cli_jobs = [], []
    for idx, (s, c, pname, card) in enumerate(pairs):
        port = PORT_BASE["link"] + idx * 10
        scmd = build_cmd("link", s, port, gids[s], 65536, card, numa_map[s], 0, 1, False, link_check=True)
        srv_jobs.append(_launch(scmd, f"LINK-SRV-{s}", None))
    time.sleep(SERVER_WAIT)
    for idx, (s, c, pname, card) in enumerate(pairs):
        port = PORT_BASE["link"] + idx * 10
        ccmd = build_cmd("link", c, port, gids[c], 65536, card, numa_map[c], 0, 1, False,
                         server_ip=ip_map[s], link_check=True)
        cli_jobs.append(_launch(ccmd, f"LINK-CLI-{c}", None))
    _wait_and_save(cli_jobs, 30)
    _wait_and_save(srv_jobs, 10)

    all_ok = True
    for (s, c, pname, card), (_, _, lines, _) in zip(pairs, cli_jobs):
        bw = parse_bw(lines)
        if bw and bw > 1.0:
            ok(f"  {pname} {s}<->{c} [{card}]: link OK ({bw:.1f} Gb/s)")
        else:
            error(f"  {pname} {s}<->{c} [{card}]: RDMA link FAILED — check cabling / IP / GID")
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main test loop
# ─────────────────────────────────────────────────────────────────────────────
def build_waves(pairs: list, schedule: str) -> list[list]:
    """schedule='waves' → 依 WAVES 分批；'all' → 全部同時跑（整機壓力測試）"""
    if schedule == "all":
        return [pairs]
    by_name = {p[2]: p for p in pairs}
    waves = [[by_name[n] for n in w if n in by_name] for w in WAVES]
    listed = {n for w in WAVES for n in w}
    extra = [p for p in pairs if p[2] not in listed]
    waves += [[p] for p in extra]
    return [w for w in waves if w]


def run_test(test_type, waves, numa_map, msg_sizes, duration, qps, dry_run,
             ip_map, gids, enable_cuda, results):
    title(f"Running: {test_type.upper()}_IPv4")
    out_dir = os.path.join(OUTPUT_BASE, f"{test_type}_ipv4")
    os.makedirs(out_dir, exist_ok=True)

    for w_idx, wave in enumerate(waves, start=1):
        names = " + ".join(p[2] for p in wave)
        section(f"[{test_type}] Wave {w_idx}/{len(waves)}: {names}")
        for msg_size in msg_sizes:
            _run_one(test_type, wave, numa_map, msg_size, duration, qps, dry_run,
                     ip_map, gids, enable_cuda, results, out_dir)


def _run_one(test_type, pairs, numa_map, msg_size, duration, qps, dry_run,
             ip_map, gids, enable_cuda, results, out_dir):
    base_port = PORT_BASE[test_type]
    print(f"\n  --> [{test_type}_ipv4]  msg_size={msg_size} B", flush=True)
    if not dry_run:
        kill_existing_rdma_tools()
    srv_jobs, cli_jobs = [], []

    for s, c, pname, card in pairs:
        port = base_port + PAIR_INDEX[pname] * 50
        scmd = build_cmd(test_type, s, port, gids[s], msg_size, card, numa_map[s],
                         duration, qps, enable_cuda)
        info(f"  SRV {s}: {scmd}")
        if dry_run:
            continue
        log = os.path.join(out_dir, f"server_{pname}_{s}_{test_type}_ipv4_msg{msg_size}.log")
        srv_jobs.append(_launch(scmd, f"SRV-{s}", log))
        time.sleep(0.1)

    if not dry_run:
        time.sleep(SERVER_WAIT)

    for s, c, pname, card in pairs:
        port = base_port + PAIR_INDEX[pname] * 50
        ccmd = build_cmd(test_type, c, port, gids[c], msg_size, card, numa_map[c],
                         duration, qps, enable_cuda, server_ip=ip_map[s])
        info(f"  CLI {c}: {ccmd}")
        if dry_run:
            continue
        log = os.path.join(out_dir, f"client_{pname}_{c}_{test_type}_ipv4_msg{msg_size}.log")
        cli_jobs.append(_launch(ccmd, f"CLI-{c}", log))
        time.sleep(0.1)

    if dry_run:
        return

    timeout = duration + 25
    _wait_and_save(cli_jobs, timeout)
    _wait_and_save(srv_jobs, timeout)

    for (s, c, pname, card), (_, _, lines, _) in zip(pairs, cli_jobs):
        results[(test_type, pname, msg_size)] = parse_bw(lines)


def judge(results, pairs, tests) -> bool:
    section("PASS / FAIL (best BW at msg_size >= %d B, bidirectional, threshold = %d%% of link rate)" % (JUDGE_MIN_MSG, LINE_RATE_PCT * 100))
    print(f"{C.BOLD}{'TEST':<7} {'PAIR':<7} {'NICs':<20} {'CARD':<5} {'BEST Gb/s':>10} {'@MSG':>9} {'THRESH':>8}  RESULT{C.ENDC}")
    print("-" * 82)
    overall = True
    for t in tests:
        for s, c, pname, card in pairs:
            cands = [(bw, m) for (tt, pn, m), bw in results.items()
                     if tt == t and pn == pname and m >= JUDGE_MIN_MSG and bw is not None]
            thr = CARD_CFG[card]["threshold"]
            if thr is None:
                passed, bw_s, m_s = False, (f"{max(cands)[0]:.2f}" if cands else "N/A"), "-"
                warn(f"{pname}: link rate unknown — no threshold (set link_gbps in rdma_platforms.json)")
            elif cands:
                bw, m = max(cands)
                passed = bw >= thr
                bw_s, m_s = f"{bw:.2f}", str(m)
            else:
                passed, bw_s, m_s = False, "N/A", "-"
            overall &= passed
            res = f"{C.GREEN}PASS{C.ENDC}" if passed else f"{C.RED}FAIL{C.ENDC}"
            print(f"{t:<7} {pname:<7} {s + '<->' + c:<20} {card:<5} {bw_s:>10} {m_s:>9} {(f"{thr:.0f}" if thr else "N/A"):>8}  {res}")
    print("-" * 82)
    print(f"  Overall: {C.GREEN + 'PASS' if overall else C.RED + 'FAIL'}{C.ENDC}")
    return overall


def main():
    global OUTPUT_BASE

    parser = argparse.ArgumentParser(description="GB300 MGX — RDMA Loopback IPv4 Performance Suite v4")
    parser.add_argument("--test", nargs="+", choices=["read", "send", "write"], default=["read", "send", "write"])
    parser.add_argument("--msg-size", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Only 64KB and 256KB")
    parser.add_argument("--platform", default=None,
                        help="Platform profile in rdma_platforms.json (default: file's 'default', else GB300_L10)")
    parser.add_argument("--list-platforms", action="store_true", help="List platform profiles and exit")
    parser.add_argument("--pairs", nargs="+", default=None, help="Subset, e.g. --pairs pair1 pair3")
    parser.add_argument("--schedule", choices=["waves", "all"], default="waves",
                        help="waves: 分 4 批避免互搶（單 port 效能，預設）；all: 全部同跑（整機壓力）")
    parser.add_argument("--duration", type=int, default=DURATION_SEC)
    parser.add_argument("--qps", type=int, default=QPS)
    parser.add_argument("--mtu", type=int, default=MTU_TARGET, help="netdev MTU (0 = don't touch)")
    parser.add_argument("--gid", type=int, default=None, help="Force GID index for all ports")
    parser.add_argument("--output-dir", default="./collected_logs_ipv4")
    parser.add_argument("--no-cuda", action="store_true", help="Disable GPU Direct flags on write")
    parser.add_argument("--skip-link-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    args = parser.parse_args()
    # 結果放在 --output-dir 底下（launcher 會傳 report/run_<TS>/0x_RDMA_*）
    OUTPUT_BASE = os.path.join(os.path.abspath(args.output_dir), "rdma_results_ipv4")

    global PAIRS, CARD_CFG, WAVES, NIC_TO_GPU_BUS_ID, PAIR_INDEX
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [here, os.path.dirname(here)]          # lib/ 或專案根目錄
    if args.list_platforms:
        plats, default, path = load_platforms(search_dirs)
        info(f"Platform file: {path or '(none — built-in only)'}")
        for n, p in plats.items():
            info(f"  {'*' if n == default else ' '} {n:<14} {len(p.get('pairs', []))} pairs  {p.get('description', '')}")
        return
    plat = resolve_platform(search_dirs, args.platform, LINE_RATE_PCT, args.dry_run)
    PAIRS, CARD_CFG, WAVES = plat["pairs"], plat["card_cfg"], plat["waves"]
    NIC_TO_GPU_BUS_ID = plat["gpu_map"]
    PAIR_INDEX = {p[2]: i for i, p in enumerate(PAIRS)}
    print_platform(plat)

    if os.geteuid() != 0 and not args.dry_run:
        error("Please run with root (sudo)")
        sys.exit(1)

    pairs = [p for p in PAIRS if not args.pairs or p[2] in args.pairs]
    if not pairs:
        error(f"No pairs selected (available: {', '.join(p[2] for p in PAIRS)})")
        sys.exit(2)

    numa_map = {dev: get_nic_numa(dev) for p in pairs for dev in (p[0], p[1])}
    ip_map   = plan_ipv4(pairs)
    enable_cuda = not args.no_cuda

    if not args.dry_run:
        check_prerequisites([p[:3] for p in pairs])
        atexit.register(cleanup)
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT,  _on_signal)

        info("Clearing pre-test system logs (dmesg, syslog, journalctl, ipmitool sel)...")
        run("dmesg -C >/dev/null 2>&1")
        run("echo ' ' > /var/log/syslog 2>/dev/null")
        run("journalctl --rotate >/dev/null 2>&1 && journalctl --vacuum-time=1s >/dev/null 2>&1")
        run("ipmitool sel clear >/dev/null 2>&1")

        if not args.no_clear:
            clear_all(OUTPUT_BASE)
        setup_kernel_params()
        if args.mtu > 0:
            _STATE["orig_mtu"] = setup_mtu(pairs, numa_map, args.mtu)
        assign_ips(pairs, ip_map)

        if enable_cuda and "write" in args.test and not check_gpu_numa(pairs, numa_map):
            sys.exit(2)

    gids = resolve_gids(pairs, ip_map, args.gid, args.dry_run)
    if gids is None:
        error("GID resolution failed — aborting")
        sys.exit(2)

    if not args.dry_run and not args.skip_link_check:
        if not rdma_link_check(pairs, ip_map, gids, numa_map):
            error("RDMA link check failed — aborting")
            sys.exit(2)

    msg_sizes = [args.msg_size] if args.msg_size else (MSG_SIZES_QUICK if args.quick else MSG_SIZES_FULL)
    waves = build_waves(pairs, args.schedule)
    info(f"Schedule: {args.schedule} — " + " | ".join("+".join(p[2] for p in w) for w in waves))
    results: dict = {}
    for t in args.test:
        run_test(t, waves, numa_map, msg_sizes, args.duration, args.qps, args.dry_run,
                 ip_map, gids, enable_cuda, results)

    if args.dry_run:
        return

    generate_summary(OUTPUT_BASE, "ipv4")
    passed = judge(results, pairs, args.test)

    if not args.no_collect:
        info(f"Collecting post-test logs (dmesg, system log, sel log, sdr log) → {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        run(f"dmesg > {os.path.join(args.output_dir, 'dmesg.log')} 2>&1")
        run(f"journalctl --no-pager > {os.path.join(args.output_dir, 'system.log')} 2>&1")
        run(f"ipmitool sel list > {os.path.join(args.output_dir, 'sel.log')} 2>&1")
        run(f"ipmitool sdr elist > {os.path.join(args.output_dir, 'sdr.log')} 2>&1")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
