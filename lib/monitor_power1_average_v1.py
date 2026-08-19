#!/usr/bin/env python3
"""
Power Monitor — dynamically reads power1_average (or power1_input) from OEM sysfs nodes.

Usage:
    ./monitor_power1_average.py            # default 0.1s interval per sensor, NO CSV log
    ./monitor_power1_average.py 0.2        # 0.2s interval per sensor, NO CSV log
    ./monitor_power1_average.py --log      # enable CSV logging (default interval)
    ./monitor_power1_average.py 0.1 --log  # 0.1s interval with CSV logging
"""

import sys
import os
import time
import signal
import csv
import re
from datetime import datetime
from pathlib import Path

# ── ANSI colour palette & Screen Control ──────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BG_HEADER = "\033[48;5;233m"   # near-black background for header
FG_CYAN   = "\033[38;5;87m"    # bright cyan  — label accent
FG_GREEN  = "\033[38;5;82m"    # bright green — values in normal range
FG_YELLOW = "\033[38;5;220m"   # amber        — moderate load
FG_RED    = "\033[38;5;196m"   # red          — high load
FG_WHITE  = "\033[38;5;255m"   # bright white — primary text
FG_GRAY   = "\033[38;5;244m"   # mid-grey     — secondary / dim text
FG_TOTAL  = "\033[38;5;214m"   # orange       — total power

# ANSI Cursor Control
CURSOR_HIDE     = "\033[?25l"    # 隱藏光標
CURSOR_SHOW     = "\033[?25h"    # 顯示光標
SCREEN_CLEAR    = "\033[2J"      # 清空全屏 (僅初始化時執行一次)

def move_cursor(row: int, col: int) -> str:
    """跳躍光標至指定行列 (基底為 1)"""
    return f"\033[{row};{col}H"

# ── Dynamic Discovery & Helpers ───────────────────────────────────────────────

def get_oem_info_path(power_path: Path) -> Path | None:
    """尋找與 power 檔案同目錄下的 power1_oem_info"""
    candidate = power_path.parent / "power1_oem_info"
    if candidate.exists():
        return candidate
    return None


def extract_hwmon_name(path: Path) -> str:
    """從 power1_oem_info 讀取自訂設備名稱 (例如: Grace Power Socket 0)"""
    oem_file = get_oem_info_path(path)
    if oem_file and oem_file.exists():
        try:
            name = oem_file.read_text().strip()
            if name:
                return name
        except Exception:
            pass
            
    # 備用機制：若找不到 oem_info 則退回原本抓取 hwmonXX
    for part in path.parts:
        if re.match(r"^hwmon\d+$", part):
            return part
    return path.parent.name


def get_hwmon_number(path: Path) -> int:
    """提取裝置數字編號用於正向排序 (如: hwmon2 -> 2)"""
    for part in path.parts:
        if re.match(r"^hwmon\d+$", part):
            match = re.search(r"\d+", part)
            return int(match.group()) if match else 0
    return 0


def resolve_power_file(dir_path: Path) -> Path | None:
    """優先回傳 power1_average 以取得動態平滑更新數值，若不存在則退回 power1_input"""
    avg_file = dir_path / "power1_average"
    if avg_file.is_file():
        return avg_file
    input_file = dir_path / "power1_input"
    if input_file.is_file():
        return input_file
    return None


def discover_power_paths() -> list[Path]:
    """僅抓取包含 power1_oem_info 且可讀取的指定 power 節點 (優先取 power1_average)"""
    found_paths = set()
    hwmon_base = Path("/sys/class/hwmon")
    
    if hwmon_base.exists():
        # 搜尋 hwmonX/device/ 以及 hwmonX/ 兩層結構
        for hwmon_dir in hwmon_base.glob("*"):
            # 優先搜尋 device/ 目錄
            device_dir = hwmon_dir / "device"
            target_dir = device_dir if (device_dir / "power1_oem_info").exists() else hwmon_dir
            
            if (target_dir / "power1_oem_info").exists():
                p_file = resolve_power_file(target_dir)
                if p_file:
                    found_paths.add(p_file)

    valid_paths = [p for p in found_paths if p.is_file()]
    return sorted(valid_paths, key=get_hwmon_number)


# Thresholds per device (watts) for colour coding
WARN_W  = 150.0   # amber above this
CRIT_W  = 250.0   # red   above this

# ── Display Helpers ───────────────────────────────────────────────────────────

def bar(value_w: float, max_w: float = 300.0, width: int = 20) -> str:
    """Return a compact ASCII power bar."""
    filled = int(round(value_w / max_w * width))
    filled = max(0, min(filled, width))
    empty  = width - filled

    if value_w >= CRIT_W:
        colour = FG_RED
    elif value_w >= WARN_W:
        colour = FG_YELLOW
    else:
        colour = FG_GREEN

    return f"{colour}{'█' * filled}{FG_GRAY}{'░' * empty}{RESET}"


def value_colour(value_w: float) -> str:
    if value_w >= CRIT_W:
        return FG_RED
    if value_w >= WARN_W:
        return FG_YELLOW
    return FG_GREEN


def read_single_hwmon(p: Path) -> dict:
    """讀取單一 Sensor 的數值"""
    hwmon_name = extract_hwmon_name(p)
    if p.exists():
        try:
            raw = int(p.read_text().strip())
            watt = raw / 1_000_000
            return {"name": hwmon_name, "raw_uw": raw, "watt": watt, "ok": True}
        except (ValueError, OSError) as exc:
            return {"name": hwmon_name, "raw_uw": 0, "watt": 0.0, "ok": False, "error": str(exc)}
    else:
        return {"name": hwmon_name, "raw_uw": 0, "watt": 0.0, "ok": False, "error": "not found"}


def render_layout(device_names: list[str], log_path: Path | None) -> None:
    """僅在啟動時執行一次：繪製靜態 UI 骨架與 Device 欄位"""
    width = 78
    buf = []
    L = buf.append

    # Line 1: Header Background Space Holder
    L(move_cursor(1, 1) + f"{BG_HEADER}{' ' * width}{RESET}")
    L(move_cursor(1, 1) + f"{BG_HEADER}{FG_CYAN}{BOLD}  ⚡ Power Monitor{RESET}")
    
    # Line 2: Header Divider
    L(move_cursor(2, 1) + f"{FG_GRAY}{'─' * width}{RESET}")

    # Line 3: Column Titles
    L(move_cursor(3, 1) + f"  {FG_GRAY}{'Device':<22}  {'µW':>14}   {'Watts':>8}   {'Load':>22}{RESET}")
    
    # Line 4: Column Divider
    L(move_cursor(4, 1) + f"{FG_GRAY}{'╌' * width}{RESET}")

    # Line 5 ~ 5+N: Device Rows (只印左側 Device 名稱，右邊保留佔位)
    for i, name in enumerate(device_names):
        row = 5 + i
        L(move_cursor(row, 1) + f"  {FG_WHITE}{name:<22}{RESET}")

    # Totals Divider
    total_row = 5 + len(device_names)
    L(move_cursor(total_row, 1) + f"{FG_GRAY}{'─' * width}{RESET}")

    # Totals Label Row
    total_val_row = total_row + 1
    L(move_cursor(total_val_row, 1) + f"  {FG_TOTAL}{BOLD}Total{RESET}")

    # Footer
    footer_row = total_val_row + 2
    if log_path:
        L(move_cursor(footer_row, 1) + f"  {FG_GRAY}Log → {log_path}{RESET}")
    else:
        L(move_cursor(footer_row, 1) + f"  {FG_GRAY}Log → Disabled (Use --log to enable){RESET}")
    L(move_cursor(footer_row + 1, 1) + f"  {FG_GRAY}Press Ctrl+C to stop.{RESET}\n")

    sys.stdout.write("".join(buf))
    sys.stdout.flush()


def update_sensor_value(index: int, r: dict | None) -> None:
    """只跳到指定 Sensor 所在行的第 26 欄，進行紅框數據的 Partial Update"""
    row = 5 + index
    col = 26  # 紅框區域起始欄位

    if r is None:
        content = f"{FG_GRAY}{'—':>14}{RESET}   {FG_GRAY}{'—':>8}{RESET}"
    elif r["ok"]:
        vc = value_colour(r["watt"])
        b  = bar(r["watt"])
        content = f"{FG_GRAY}{r['raw_uw']:>14,} µW{RESET}   {vc}{r['watt']:>7.3f} W{RESET}   {b}"
    else:
        err = r.get("error", "unknown error")
        content = f"{FG_RED}✖  {err}{RESET}"

    sys.stdout.write(move_cursor(row, col) + content)
    sys.stdout.flush()


def update_header_totals(results: list[dict], timestamp: str, iteration: int, device_count: int) -> None:
    """更新 Header 時間軸與 Total 加總列"""
    # 1. 更新 Header 時間與 Sweep 次數 (Line 1 Col 25)
    header_info = f"{FG_GRAY}{timestamp}   {DIM}sweep #{iteration}{RESET}"
    sys.stdout.write(move_cursor(1, 25) + header_info)

    # 2. 更新 Total 總數 (Total 行 Col 11)
    total_uw  = sum(r["raw_uw"] for r in results if r and r["ok"])
    total_w   = total_uw / 1_000_000
    gpu_count = sum(1 for r in results if r and r["ok"])
    avg_w     = total_w / gpu_count if gpu_count else 0.0

    total_row = 5 + device_count + 1
    total_str = (
        f"{FG_GRAY}{gpu_count} device{'s' if gpu_count != 1 else ''}   {RESET}"
        f"{FG_TOTAL}{BOLD}{total_w:>10.3f} W{RESET}  "
        f"{FG_GRAY}avg {avg_w:.3f} W / device{RESET}   "
    )
    sys.stdout.write(move_cursor(total_row, 11) + total_str)
    sys.stdout.flush()


def write_csv_header(log_path: Path, names: list[str]) -> None:
    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp"] + [f"{n}_uW" for n in names] + [f"{n}_W" for n in names] + ["total_W"])


def append_csv_row(log_path: Path, results: list[dict], timestamp: str, total_w: float) -> None:
    with log_path.open("a", newline="") as f:
        w = csv.writer(f)
        row = [timestamp]
        row += [r["raw_uw"] if (r and r["ok"]) else "" for r in results]
        row += [f"{r['watt']:.3f}" if (r and r["ok"]) else "" for r in results]
        row += [f"{total_w:.3f}"]
        w.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    enable_log = any(arg in sys.argv for arg in ["--log", "--enable-log"])
    
    interval_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    interval = float(interval_args[0]) if interval_args else 0.1

    script_dir = Path(__file__).resolve().parent
    current_paths = discover_power_paths()
    device_names = [extract_hwmon_name(p) for p in current_paths]

    if enable_log:
        # Use LOG_OUTPUT_DIR if set by the launcher, otherwise fall back to script directory
        _log_base = Path(os.environ.get("LOG_OUTPUT_DIR") or str(script_dir))
        _log_base.mkdir(parents=True, exist_ok=True)
        log_path = _log_base / f"power_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        write_csv_header(log_path, device_names)
    else:
        log_path = None

    stop_requested = False

    def handle_exit(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # 首次執行：清屏、隱藏光標、畫出靜態 UI 框架
    sys.stdout.write(SCREEN_CLEAR + CURSOR_HIDE)
    sys.stdout.flush()
    render_layout(device_names, log_path)

    iteration = 0
    device_count = len(current_paths)

    try:
        while not stop_requested:
            iteration += 1
            results = [None] * device_count
            
            for i, p in enumerate(current_paths):
                if stop_requested:
                    break

                now = datetime.now()
                timestamp = f"{now.strftime('%Y-%m-%d %H:%M:%S')}.{now.microsecond // 1000:03d}"
                
                # 1. 讀取單一 Sensor
                results[i] = read_single_hwmon(p)

                # 2. 僅更新該行的紅框區域數據 (In-place Partial Update)
                update_sensor_value(i, results[i])

                # 3. 更新頂部 Sweep/Time 和底部 Total
                update_header_totals(results, timestamp, iteration, device_count)

                if not stop_requested:
                    time.sleep(interval)

            if log_path and not stop_requested:
                now = datetime.now()
                timestamp = f"{now.strftime('%Y-%m-%d %H:%M:%S')}.{now.microsecond // 1000:03d}"
                total_w = sum(r["raw_uw"] for r in results if r and r["ok"]) / 1_000_000
                append_csv_row(log_path, results, timestamp, total_w)

    finally:
        # 結束時恢復游標，並將游標移到畫面最下方
        exit_row = 5 + device_count + 5
        sys.stdout.write(move_cursor(exit_row, 1) + CURSOR_SHOW)
        sys.stdout.flush()
        if log_path:
            print(f"\n{FG_CYAN}Stopped. Log saved → {log_path}{RESET}\n")
        else:
            print(f"\n{FG_CYAN}Stopped.{RESET}\n")


if __name__ == "__main__":
    main()