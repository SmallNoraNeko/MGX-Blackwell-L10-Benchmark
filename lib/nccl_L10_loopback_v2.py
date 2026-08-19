#!/usr/bin/env python3
"""
nccl_l10_loopback.py — GB300 L10 單機台 NCCL Loopback 效能測試
================================================================
平台   : NVIDIA Blackwell GB300 / ARM64 Ubuntu 24.04
層級   : L10（單一 Compute Node，4 GPU，純 PCIe，無 NVLink Fabric / 無 IB）
測試項 : all_reduce / all_gather / alltoall（2B ~ 8GB 全掃）
流程   :
  Step 0  disable_acs（內建，對齊 NVIDIA disable_acs.sh 邏輯）
  Step 1  載入 nvidia_peermem（GDR 必要）
  Step 2  清場（dmesg / journalctl / IPMI SEL）
  Step 3  環境確認（GPU / PCIe / CUDA / NCCL / binary）
  Step 4  執行 NCCL 測試
  Step 5  收集 Log（dmesg / system / sel / sdr / nvidia-smi / lspci / acs_status）
build  : 預設為腳本同層的 build/ 資料夾，可用 --build 覆蓋
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

# ──────────────────────────────────────────────
# 預設組態
# ──────────────────────────────────────────────
DEFAULT_GPU_COUNT  = 4
DEFAULT_ITERS      = 20
DEFAULT_BUILD_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
DEFAULT_TESTS      = ["all_reduce", "all_gather", "alltoall"]

MSG_BEGIN  = "2"
MSG_END    = "8G"
MSG_FACTOR = 2

NCCL_ENV = {
    "NCCL_DEBUG":        "INFO",
    "NCCL_IB_DISABLE":   "1",
    "NCCL_P2P_DISABLE":  "1",       # 顯式關閉 P2P，告訴 NCCL 這台機器設計上就沒有 P2P
    #"NCCL_P2P_LEVEL":    "5",       # 放開至 SYS 層級，允許跨 Socket PCIe P2P
    "NCCL_SHM_DISABLE":  "0",       # 強制使用 Shared Memory (SHM) 進行跨卡通訊
    #"NCCL_ALGO":         "Ring",   # 建議註解掉，讓 NCCL 自動選擇最佳演算法 (Tree/Ring)
}

# ──────────────────────────────────────────────
# 顏色輸出
# ──────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"

def info(msg):  print(f"{C.CYAN}[INFO]{C.RESET} {msg}", flush=True)
def ok(msg):    print(f"{C.GREEN}[OK]{C.RESET}   {msg}", flush=True)
def warn(msg):  print(f"{C.YELLOW}[WARN]{C.RESET} {msg}", flush=True)
def error(msg): print(f"{C.RED}[ERROR]{C.RESET} {msg}", flush=True)

def title(msg):
    bar = "=" * 70
    print(f"\n{C.BOLD}{bar}\n  {msg}\n{bar}{C.RESET}", flush=True)

def section(msg):
    print(f"\n{C.BOLD}── {msg} {'─' * max(0, 65 - len(msg))}{C.RESET}", flush=True)

def ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def run_stream(cmd: str, log_lines: list, env: dict = None) -> int:
    _env = os.environ.copy()
    if env:
        _env.update(env)
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=_env,
    )
    for line in iter(proc.stdout.readline, ""):
        print(f"  {line}", end="", flush=True)
        log_lines.append(line)
    proc.stdout.close()
    proc.wait()
    return proc.returncode

# ──────────────────────────────────────────────
# Step 0: disable_acs（移植自 NVIDIA disable_acs.sh）
# ──────────────────────────────────────────────
def disable_acs() -> bool:
    """
    對所有支援 ACS 的 PCIe 裝置關閉 ACS，啟用 GPU P2P Direct。
    邏輯完全對齊 NVIDIA 官方 disable_acs.sh：
      1. lspci 列出所有 BDF
      2. setpci ECAP_ACS+0x6.w 確認支援 ACS
      3. 寫入 0x0 關閉
      4. 回讀確認值為 0
    """
    section("Step 0 — Disable ACS（啟用 GPU P2P Direct）")

    res = run("lspci -d '*:*:*' | awk '{print $1}'")
    bdfs = [b.strip() for b in res.stdout.splitlines() if b.strip()]
    if not bdfs:
        warn("找不到 PCIe 裝置，跳過 disable_acs")
        return True

    disabled = 0
    skipped  = 0
    failed   = 0

    for bdf in bdfs:
        # 確認是否支援 ACS
        chk = run(f"setpci -v -s {bdf} ECAP_ACS+0x6.w 2>/dev/null")
        if chk.returncode != 0:
            skipped += 1
            continue

        # 寫入 0x0 關閉 ACS
        write = run(f"setpci -v -s {bdf} ECAP_ACS+0x6.w=0x0 2>/dev/null")
        if write.returncode != 0:
            warn(f"  ACS disable 失敗（write）：{bdf}")
            failed += 1
            continue

        # 回讀確認
        after_res = run(f"setpci -v -s {bdf} ECAP_ACS+0x6.w 2>/dev/null")
        after_val = after_res.stdout.strip().split()[-1] if after_res.stdout.strip() else "ffff"
        try:
            after_dec = int(after_val, 16)
        except ValueError:
            after_dec = -1

        if after_dec != 0:
            warn(f"  ACS disable 確認失敗：{bdf} → 值={after_val}（預期 0x0）")
            failed += 1
        else:
            disabled += 1

    ok(f"disable_acs 完成：已關閉={disabled}，不支援ACS跳過={skipped}，失敗={failed}")
    return failed == 0

# ──────────────────────────────────────────────
# Step 1: 載入 nvidia_peermem
# ──────────────────────────────────────────────
def load_peermem() -> bool:
    section("Step 1 — 載入 nvidia_peermem（GDR 必要模組）")
    if run("lsmod | grep nvidia_peermem").stdout.strip():
        ok("nvidia_peermem 已載入 ✓")
        return True
    info("nvidia_peermem 未載入，嘗試 modprobe...")
    res = run("modprobe nvidia_peermem")
    if res.returncode == 0:
        ok("nvidia_peermem 載入成功 ✓")
        return True
    warn(f"nvidia_peermem 載入失敗：{res.stderr.strip()}（測試仍繼續）")
    return True  # 非阻斷性

# ──────────────────────────────────────────────
# Step 2: 清場
# ──────────────────────────────────────────────
def clear_all():
    section("Step 2 — 測試前清場（dmesg / journalctl / IPMI SEL）")
    run("dmesg -C")
    run("journalctl --rotate")
    run("journalctl --vacuum-time=1s")
    run("ipmitool sel clear")
    run("pkill -9 -f 'all_reduce_perf|all_gather_perf|alltoall_perf' 2>/dev/null")
    ok("清場完成")

# ──────────────────────────────────────────────
# Step 3: 環境確認
# ──────────────────────────────────────────────
def check_env(build_dir: str, gpu_count: int) -> bool:
    section("Step 3 — 環境確認")
    all_ok = True

    # 架構
    arch = run("uname -m").stdout.strip()
    ok(f"架構：{arch} ✓") if arch == "aarch64" else warn(f"架構：{arch}（預期 aarch64）")

    # GPU 數量
    detected = len([l for l in run(
        "nvidia-smi --query-gpu=name --format=csv,noheader"
    ).stdout.strip().splitlines() if l.strip()])
    if detected >= gpu_count:
        ok(f"GPU 偵測：{detected} 顆（測試使用 {gpu_count} 顆）✓")
    else:
        error(f"GPU 偵測：{detected} 顆，需要 {gpu_count} 顆")
        all_ok = False

    # PCIe 速度 + ACS 狀態
    bdfs = [
        l.strip().replace("00000000:", "")
        for l in run(
            "nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader"
        ).stdout.strip().splitlines() if l.strip()
    ]
    for bdf in bdfs[:gpu_count]:
        lnk = run(f"lspci -s {bdf} -vvv 2>/dev/null | grep 'LnkSta:'").stdout.strip()
        if "32GT/s" in lnk and "x16" in lnk:
            ok(f"PCIe [{bdf}] Speed 32GT/s x16 ✓")
        elif lnk:
            warn(f"PCIe [{bdf}] {lnk.strip()} ← 請確認")
        else:
            warn(f"PCIe [{bdf}] 無法取得 LnkSta")

        acs_raw = run(f"setpci -v -s {bdf} ECAP_ACS+0x6.w 2>/dev/null").stdout.strip()
        if acs_raw:
            val_str = acs_raw.split()[-1]
            try:
                val_dec = int(val_str, 16)
                if val_dec == 0:
                    ok(f"ACS  [{bdf}] = 0x0 ✓")
                else:
                    warn(f"ACS  [{bdf}] = {val_str}（非 0，P2P 可能受限）")
            except ValueError:
                warn(f"ACS  [{bdf}] 值無法解析：{val_str}")

    # CUDA
    cuda_ver = next(
        (l for l in run("nvcc --version").stdout.splitlines() if "release" in l.lower()),
        "unknown"
    )
    ok(f"CUDA：{cuda_ver}")

    # NCCL
    nccl = run("dpkg -l libnccl* 2>/dev/null | grep '^ii'").stdout.strip()
    if nccl:
        ok("NCCL 套件已安裝 ✓")
        for line in nccl.splitlines():
            info(f"  {line.split()[1]}  {line.split()[2]}")
    else:
        warn("找不到 libnccl* 套件")

    # nvidia_peermem
    if run("lsmod | grep nvidia_peermem").stdout.strip():
        ok("nvidia_peermem 已載入 ✓")
    else:
        warn("nvidia_peermem 未載入")

    # ECC
    ecc_vals = [
        v.strip() for v in run(
            "nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total "
            "--format=csv,noheader"
        ).stdout.strip().splitlines() if v.strip()
    ]
    if all(v in ("0", "N/A", "[N/A]") for v in ecc_vals):
        ok(f"ECC Uncorrected：{ecc_vals} ✓")
    else:
        warn(f"ECC Uncorrected：{ecc_vals} ← 有非零值")

    # binary 確認
    for t in ["all_reduce_perf", "all_gather_perf", "alltoall_perf"]:
        path = os.path.join(build_dir, t)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            ok(f"binary ✓ {path}")
        else:
            error(f"binary 不存在或不可執行：{path}")
            all_ok = False

    return all_ok

# ──────────────────────────────────────────────
# Step 4: 執行 NCCL 測試
# ──────────────────────────────────────────────
def run_nccl_test(
    test_type: str, build_dir: str, log_dir: str,
    gpu_count: int, iters: int, dry_run: bool,
    msg_begin: str = MSG_BEGIN, msg_end: str = MSG_END,
) -> bool:
    tool_map = {
        "all_reduce": "all_reduce_perf",
        "all_gather": "all_gather_perf",
        "alltoall":   "alltoall_perf",
    }
    tool     = os.path.join(build_dir, tool_map[test_type])
    log_path = os.path.join(log_dir, f"nccl_{test_type}_{ts()}.log")
    cmd = (
        f"{tool} "
        f"-b {msg_begin} -e {msg_end} -f {MSG_FACTOR} "
        f"-g {gpu_count} -n {iters} --op sum"
    )

    title(f"執行測試：{test_type.upper()}")
    info(f"指令：{cmd}")
    info(f"Log ：{log_path}")

    if dry_run:
        warn("dry-run 模式，跳過實際執行")
        return True

    log_lines = [
        f"COMMAND: {cmd}\n",
        f"NCCL_ENV: {NCCL_ENV}\n",
        f"START: {datetime.datetime.now()}\n\n",
    ]
    rc = run_stream(cmd, log_lines, env=NCCL_ENV)

    out_of_bounds  = any("Out of bounds values" in l and ": 0" not in l for l in log_lines)
    abort_detected = any("Abort" in l or "NCCL WARN" in l for l in log_lines)

    log_lines.append(f"\nEND: {datetime.datetime.now()}\nRETURNCODE: {rc}\n")
    with open(log_path, "w") as f:
        f.writelines(log_lines)

    if rc != 0:
        error(f"{test_type}: 執行失敗 (returncode={rc})")
        return False
    if out_of_bounds:
        error(f"{test_type}: Out of bounds values 非零 → 資料正確性錯誤")
        return False
    if abort_detected:
        warn(f"{test_type}: 偵測到 NCCL WARN / Abort，請確認 log")
    ok(f"{test_type}: 完成 → {log_path}")
    return True

# ──────────────────────────────────────────────
# Step 5: 收集 Log
# ──────────────────────────────────────────────
def collect_logs(log_dir: str, gpu_count: int):
    section("Step 5 — 收集系統 Log")

    bdfs = [
        l.strip().replace("00000000:", "")
        for l in run(
            "nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader"
        ).stdout.strip().splitlines() if l.strip()
    ]

    cmds: dict[str, str] = {
        "dmesg.log":            "dmesg",
        "system.log":           "journalctl --no-pager",
        "sel.log":              "ipmitool sel elist",
        "sdr.log":              "ipmitool sdr elist",
        "nvidia_smi.log":       "nvidia-smi",
        "nvidia_smi_query.log": "nvidia-smi -q",
        "gpu_status.log": (
            "nvidia-smi --query-gpu="
            "index,name,driver_version,"
            "pcie.link.gen.current,pcie.link.width.current,"
            "temperature.gpu,power.draw,"
            "ecc.errors.uncorrected.volatile.total "
            "--format=csv"
        ),
        "lspci_vt.log":      "lspci -tv",
        "cuda_version.log":  "nvcc --version",
        "nccl_version.log":  "dpkg -l libnccl*",
        "driver_version.log": (
            "nvidia-smi --query-gpu=driver_version "
            "--format=csv,noheader | head -1"
        ),
        "kernel_arch.log":   "uname -a",
        "peermem.log":       "lsmod | grep nvidia_peermem",
        "acs_status.log": (
            "for bdf in $(lspci -d '*:*:*' | awk '{print $1}'); do "
            "  val=$(setpci -v -s $bdf ECAP_ACS+0x6.w 2>/dev/null); "
            "  [ -n \"$val\" ] && echo \"$bdf $val\"; "
            "done"
        ),
    }
    for i, bdf in enumerate(bdfs[:gpu_count]):
        cmds[f"lspci_gpu{i}_{bdf}.log"] = f"lspci -s {bdf} -vvv"

    for filename, cmd in cmds.items():
        res = run(cmd)
        content = res.stdout if res.stdout else res.stderr
        out_path = os.path.join(log_dir, filename)
        with open(out_path, "w") as f:
            f.write(f"# CMD: {cmd}\n# TIME: {datetime.datetime.now()}\n\n")
            f.write(content)
        ok(f"已收集：{filename}")

    info(f"\n全部 Log 存放於：{log_dir}")

def print_log_summary(log_dir: str):
    section("Log 清單")
    files = sorted(os.listdir(log_dir))
    for i, f in enumerate(files, 1):
        size = os.path.getsize(os.path.join(log_dir, f))
        print(f"  {i:>3}. {f:<55} {size:>8,} bytes")

# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GB300 L10 NCCL Loopback 自動化測試腳本"
    )
    parser.add_argument("--build", default=DEFAULT_BUILD_DIR,
                        help="nccl-tests build 目錄（預設：腳本同層的 build/）")
    parser.add_argument("--gpu",   type=int, default=DEFAULT_GPU_COUNT,
                        help=f"GPU 數量（預設 {DEFAULT_GPU_COUNT}）")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                        help=f"每個 msg size 迭代次數（預設 {DEFAULT_ITERS}）")
    parser.add_argument("--test",  nargs="+",
                        choices=["all_reduce", "all_gather", "alltoall"],
                        default=DEFAULT_TESTS)
    parser.add_argument("--msg-begin", default=MSG_BEGIN,
                        help=f"起始訊息大小（預設 {MSG_BEGIN}）")
    parser.add_argument("--msg-end",   default=MSG_END,
                        help=f"結束訊息大小（預設 {MSG_END}）")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--no-clear",      action="store_true")
    parser.add_argument("--no-collect",    action="store_true")
    parser.add_argument("--skip-check",    action="store_true")
    parser.add_argument("--skip-acs",      action="store_true",
                        help="跳過 disable_acs 步驟")
    parser.add_argument("--skip-peermem",  action="store_true",
                        help="跳過 nvidia_peermem 載入步驟")
    args = parser.parse_args()

    if os.geteuid() != 0 and not args.dry_run:
        error("請以 root 權限執行（sudo ./nccl_l10_loopback.py）")
        sys.exit(1)

    # Allow launcher to redirect log output via environment variable
    log_dir = (os.environ.get("LOG_OUTPUT_DIR")
               or f"./nccl_L10_logs_{ts()}")
    os.makedirs(log_dir, exist_ok=True)

    title("GB300 L10 NCCL Loopback 效能測試")
    info(f"GPU 數量  : {args.gpu}")
    info(f"Build 路徑: {args.build}")
    info(f"測試項目  : {', '.join(args.test)}")
    info(f"MSG 範圍  : {args.msg_begin} ~ {args.msg_end}（步進 ×{MSG_FACTOR}）")
    info(f"迭代次數  : {args.iters}")
    info(f"Log 目錄  : {log_dir}")

    if not args.skip_acs     and not args.dry_run: disable_acs()
    else: info("[dry-run/skip] 跳過 disable_acs")

    if not args.skip_peermem and not args.dry_run: load_peermem()
    else: info("[dry-run/skip] 跳過 nvidia_peermem 載入")

    if not args.no_clear     and not args.dry_run: clear_all()
    else: info("[dry-run/skip] 跳過清場")

    if not args.skip_check:
        if not check_env(args.build, args.gpu):
            error("環境確認失敗，請修正後重試（或加 --skip-check 跳過）")
            sys.exit(1)
    else:
        warn("跳過環境確認（--skip-check）")

    section("開始執行 NCCL 測試")
    results: dict[str, bool] = {}
    for test_type in args.test:
        passed = run_nccl_test(
            test_type, args.build, log_dir, args.gpu, args.iters, args.dry_run,
            msg_begin=args.msg_begin, msg_end=args.msg_end,
        )
        results[test_type] = passed
        if not passed:
            error(f"{test_type} 測試失敗，停止後續測試")
            break
        time.sleep(2)

    if not args.no_collect and not args.dry_run:
        collect_logs(log_dir, args.gpu)
    else:
        info("[dry-run/skip] 跳過 Log 收集")

    title("測試結果摘要")
    all_passed = True
    for test_type, passed in results.items():
        (ok if passed else error)(f"  {test_type:<15} {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    if not args.dry_run:
        print_log_summary(log_dir)

    print()
    if all_passed and results:
        ok(f"全部測試通過 ✓  Log 目錄：{log_dir}")
        sys.exit(0)
    else:
        error(f"有測試失敗，請確認 Log：{log_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
