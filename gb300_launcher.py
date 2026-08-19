#!/usr/bin/env python3
"""
gb300_launcher.py
-----------------
GB300 Benchmark Launcher — single entry point.

Usage:
    sudo python3 gb300_launcher.py             # interactive mode
    sudo python3 gb300_launcher.py --dry-run   # simulate without running
    sudo python3 gb300_launcher.py --select 1,3,6 --yes   # non-interactive
    sudo python3 gb300_launcher.py --no-prereq # skip binary checks
    sudo python3 gb300_launcher.py --log-dir /tmp/myreport

Controls (interactive menu):
    UP / DOWN    Move cursor
    Space        Toggle selection
    a            Select all available tests
    n            Clear all selections
    Enter        Confirm and proceed
    q            Quit
"""

import sys
import os
import curses
import argparse
import shutil
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure lib/ is importable regardless of working directory
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR  = os.path.join(ROOT_DIR, "lib")
sys.path.insert(0, LIB_DIR)

from _launcher_config import TESTS, evaluate_availability, ROOT_DIR as CFG_ROOT
from _launcher_runner  import (
    create_run_dir, run_all, write_summary, get_platform_info,
)

# ---------------------------------------------------------------------------
# ANSI colour codes (used outside curses — parameter prompts, confirm screen)
# ---------------------------------------------------------------------------
G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
C   = "\033[96m"
DIM = "\033[90m"
W   = "\033[97m"
B   = "\033[94m"
RS  = "\033[0m"
SEP = "=" * 66

# ---------------------------------------------------------------------------
# ASCII Art Banner (figlet "Big" font — Benchmark)
# ---------------------------------------------------------------------------
BANNER = r"""
  ____                  _                          _
 | __ )  ___ _ __   ___| |__  _ __ ___   __ _ _ __| | __
 |  _ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ /
 | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <
 |____/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\
"""
SUBTITLE = "               GB300 NVL72  .  ARM64  .  Ubuntu 24.04"

# ---------------------------------------------------------------------------
# Category order for menu grouping
# ---------------------------------------------------------------------------
CATEGORIES = [
    "GPU Compute & Memory",
    "GPU Interconnect",
    "Network / RDMA",
    "Hardware Health & Monitoring",
    "DL Training Validation",
]

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight(skip_prereq: bool) -> dict[int, list[str]]:
    """
    Run environment pre-flight checks.
    Returns availability dict: {test_id: [failure_reasons]}
    """
    warnings = []

    # Root check
    if os.geteuid() != 0:
        warnings.append("Not running as root — some tests may fail (ipmitool, dmesg, etc.)")

    # Python version
    if sys.version_info < (3, 10):
        warnings.append(f"Python {sys.version_info.major}.{sys.version_info.minor} "
                        "detected — Python 3.10+ recommended")

    # ipmitool (SDR collection used by GPU benchmarks)
    if not shutil.which("ipmitool"):
        warnings.append("ipmitool not found — SDR sensor collection will be skipped")

    for w in warnings:
        print(f"{Y}  [WARN] {w}{RS}", flush=True)
    if warnings:
        print()

    if skip_prereq:
        return {t["id"]: [] for t in TESTS}

    return evaluate_availability()


# ===========================================================================
# SECTION 1 — curses-based interactive checkbox menu
# ===========================================================================

# Colour pair IDs
CP_BANNER   = 1   # bright green
CP_SUBTITLE = 2   # dark grey
CP_SECTION  = 3   # dark grey (section headers)
CP_NORMAL   = 4   # white
CP_DIM      = 5   # grey (unavailable / desc)
CP_CURSOR   = 6   # reverse highlight
CP_CHECKED  = 7   # green [x]
CP_CFG      = 8   # cyan configurable hint
CP_UNAVAIL  = 9   # red [UNAVAILABLE]
CP_STATUS   = 10  # cyan status line


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    # Colour 8 = dark grey — only available when terminal supports 256 colours.
    # Fall back to COLOR_WHITE on 8-colour terminals (e.g. plain SSH without
    # TERM=xterm-256color) to prevent ValueError.
    _grey = 8 if curses.COLORS > 8 else curses.COLOR_WHITE
    curses.init_pair(CP_BANNER,   curses.COLOR_GREEN,  -1)
    curses.init_pair(CP_SUBTITLE, _grey,               -1)
    curses.init_pair(CP_SECTION,  _grey,               -1)
    curses.init_pair(CP_NORMAL,   curses.COLOR_WHITE,  -1)
    curses.init_pair(CP_DIM,      _grey,               -1)
    curses.init_pair(CP_CURSOR,   curses.COLOR_BLACK,  curses.COLOR_WHITE)
    curses.init_pair(CP_CHECKED,  curses.COLOR_GREEN,  -1)
    curses.init_pair(CP_CFG,      curses.COLOR_CYAN,   -1)
    curses.init_pair(CP_UNAVAIL,  curses.COLOR_RED,    -1)
    curses.init_pair(CP_STATUS,   curses.COLOR_CYAN,   -1)


def _build_menu_rows(availability: dict[int, list[str]]) -> list[dict]:
    """
    Build a flat list of menu rows in display order.
    Each row: {type: "section"|"test"|"unavail", ...}
    """
    rows = []
    prev_cat = None

    for test in TESTS:
        cat = test["category"]
        if cat != prev_cat:
            rows.append({"type": "section", "label": f"-- {cat} " + "-" * max(0, 48 - len(cat))})
            prev_cat = cat

        failures = availability.get(test["id"], [])
        available = len(failures) == 0

        rows.append({
            "type":       "test",
            "test":       test,
            "available":  available,
            "failures":   failures,
            "selected":   False,   # default: all unselected, user picks manually
            "configurable": bool(test["configurable"]),
        })

    return rows


def _selectable(row: dict) -> bool:
    return row["type"] == "test" and row["available"]


def _draw_menu(stdscr, rows: list[dict], cursor: int, scroll: int):
    """Render the full menu to the curses window."""
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # Banner (first 5 lines)
    banner_lines = BANNER.strip("\n").splitlines()
    for i, line in enumerate(banner_lines):
        if i >= h:
            break
        try:
            stdscr.addstr(i, 0, line[:w], curses.color_pair(CP_BANNER) | curses.A_BOLD)
        except curses.error:
            pass

    # Subtitle
    sub_y = len(banner_lines)
    if sub_y < h:
        try:
            stdscr.addstr(sub_y, 0, SUBTITLE[:w], curses.color_pair(CP_SUBTITLE))
        except curses.error:
            pass

    # Hint line
    hint_y = sub_y + 1
    hint = "  UP/DOWN: move   Space: toggle   a: select all   n: clear   Enter: confirm   q: quit"
    if hint_y < h:
        try:
            stdscr.addstr(hint_y, 0, hint[:w], curses.color_pair(CP_DIM))
        except curses.error:
            pass

    # Blank line
    menu_start_y = hint_y + 2

    # Draw visible rows
    display_y = menu_start_y
    for idx in range(scroll, len(rows)):
        row = rows[idx]
        if display_y >= h - 3:   # reserve 3 lines for status bar
            break

        is_cursor = (idx == cursor)

        if row["type"] == "section":
            label = f"  {row['label']}"
            if display_y < h:
                try:
                    stdscr.addstr(display_y, 0, label[:w],
                                  curses.color_pair(CP_SECTION))
                except curses.error:
                    pass
            display_y += 1

        elif row["type"] == "test":
            test   = row["test"]
            avail  = row["available"]
            sel    = row["selected"]
            cfg    = row["configurable"]
            box    = "[x]" if sel else "[ ]"
            num    = f"({test['id']:2d})"
            name   = test["name"]
            desc   = test.get("desc") or ""

            # Compose the line
            left  = f"  {box} {num}  {name:<22}"
            cfg_s = "  (*)" if cfg and avail else ""
            # Pad + description
            line  = f"{left}{cfg_s:<6}  {'':<2}"

            attr_base = curses.color_pair(CP_CURSOR) if is_cursor else 0

            if not avail:
                reason = row["failures"][0] if row["failures"] else "unknown"
                unavail_line = f"  [ ] ({test['id']:2d})  {name:<22}  [UNAVAILABLE]  {reason}"
                if display_y < h:
                    try:
                        stdscr.addstr(display_y, 0,
                                      unavail_line[:w],
                                      curses.color_pair(CP_DIM))
                    except curses.error:
                        pass
                display_y += 1
                continue

            if display_y < h:
                try:
                    if is_cursor:
                        stdscr.addstr(display_y, 0, " " * min(w, 66),
                                      curses.color_pair(CP_CURSOR))
                        stdscr.addstr(display_y, 0, f"  ", curses.color_pair(CP_CURSOR))

                    # Checkbox
                    chk_attr = (curses.color_pair(CP_CHECKED) | curses.A_BOLD
                                if sel else curses.color_pair(CP_DIM))
                    if is_cursor:
                        chk_attr = curses.color_pair(CP_CURSOR)

                    stdscr.addstr(display_y, 2, box,
                                  chk_attr if not is_cursor else curses.color_pair(CP_CURSOR))

                    num_x = 6
                    name_x = num_x + 5

                    if is_cursor:
                        stdscr.addstr(display_y, num_x,
                                      f" {num}  {name:<22}",
                                      curses.color_pair(CP_CURSOR))
                    else:
                        stdscr.addstr(display_y, num_x,
                                      f" {num}  {name:<22}",
                                      curses.color_pair(CP_NORMAL))

                    cfg_x = name_x + 22
                    if cfg and avail:
                        cfg_str = "  (*)"
                        if is_cursor:
                            stdscr.addstr(display_y, cfg_x, cfg_str,
                                          curses.color_pair(CP_CURSOR))
                        else:
                            stdscr.addstr(display_y, cfg_x, cfg_str,
                                          curses.color_pair(CP_CFG))

                except curses.error:
                    pass

            display_y += 1

    # Status bar (bottom 2 lines)
    sel_ids = [r["test"]["id"] for r in rows
               if r["type"] == "test" and r.get("selected")]
    sel_str = "Selected: " + (" ".join(str(i) for i in sel_ids) if sel_ids else "(none)")
    count_str = f"{len(sel_ids)} / {sum(1 for r in rows if r['type']=='test' and r['available'])} items"

    bar_y = h - 2
    if bar_y >= 0:
        try:
            stdscr.addstr(bar_y - 1, 0, "-" * min(w, 66),
                          curses.color_pair(CP_DIM))
            stdscr.addstr(bar_y, 0, f"  {sel_str}"[:w],
                          curses.color_pair(CP_STATUS))
            stdscr.addstr(bar_y, min(w - 20, 50),
                          count_str[:20],
                          curses.color_pair(CP_DIM))
        except curses.error:
            pass

    stdscr.refresh()


def _menu_curses(stdscr, rows: list[dict]) -> list[dict] | None:
    """
    Interactive curses menu loop.
    Returns the list of selected test rows, or None if user quit.
    q, Q, Ctrl-C, Ctrl-D all exit cleanly.
    Enter with nothing selected shows a hint; press q to quit.
    """
    curses.curs_set(0)
    _init_colors()

    # Find the first selectable row
    cursor = next((i for i, r in enumerate(rows) if _selectable(r)), 0)
    scroll = 0
    h, _w  = stdscr.getmaxyx()

    while True:
        _draw_menu(stdscr, rows, cursor, scroll)
        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            return None

        if key in (ord("q"), ord("Q")):
            return None

        elif key in (curses.KEY_UP, ord("k")):
            # Move cursor up, skip section rows
            for _ in range(len(rows)):
                cursor = max(0, cursor - 1)
                if rows[cursor]["type"] == "test":
                    break
            # Scroll up if needed
            if cursor < scroll:
                scroll = cursor

        elif key in (curses.KEY_DOWN, ord("j")):
            # Move cursor down, skip section rows
            for _ in range(len(rows)):
                cursor = min(len(rows) - 1, cursor + 1)
                if rows[cursor]["type"] == "test":
                    break
            # Scroll down if needed (rough estimate: banner ~8 lines, status ~3)
            visible_rows = h - 12
            if cursor - scroll >= visible_rows:
                scroll = cursor - visible_rows + 1

        elif key == ord(" "):
            if _selectable(rows[cursor]):
                rows[cursor]["selected"] = not rows[cursor]["selected"]

        elif key in (ord("a"), ord("A")):
            for r in rows:
                if _selectable(r):
                    r["selected"] = True

        elif key in (ord("n"), ord("N")):
            for r in rows:
                if r["type"] == "test":
                    r["selected"] = False

        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            selected = [r for r in rows
                        if r["type"] == "test" and r.get("selected")]
            if not selected:
                # Nothing selected — ask whether to quit
                h2, w2 = stdscr.getmaxyx()
                msg = "  No tests selected.  Press q to quit, or Space to select a test."
                try:
                    stdscr.addstr(h2 - 1, 0, msg[:w2],
                                  curses.color_pair(CP_UNAVAIL))
                    stdscr.refresh()
                    curses.napms(1200)
                except curses.error:
                    pass
                continue
            return selected


def run_menu(availability: dict[int, list[str]]) -> list[dict] | None:
    """
    Entry point for the curses menu.
    Returns sorted list of selected test dicts, or None if user quit.
    """
    rows = _build_menu_rows(availability)
    selected_rows = curses.wrapper(_menu_curses, rows)
    if selected_rows is None:
        return None
    # Return test dicts in id order
    return sorted([r["test"] for r in selected_rows], key=lambda t: t["id"])


# ===========================================================================
# SECTION 2 — Parameter prompts (plain input(), runs after curses exits)
# ===========================================================================

def _prompt(prompt_str: str, default, cast=str, validate=None) -> object:
    """
    Display a prompt with a default value.
    Re-prompts on invalid input.
    """
    while True:
        raw = input(f"{W}  {prompt_str} [{default}]: {RS}").strip()
        if raw == "":
            return default
        try:
            value = cast(raw)
        except (ValueError, TypeError):
            print(f"{R}  Invalid input. Expected {cast.__name__}.{RS}")
            continue
        if validate and not validate(value):
            print(f"{R}  Value out of range or invalid.{RS}")
            continue
        return value


def _prompt_choice(prompt_str: str, choices: list[tuple], default_key: str) -> str:
    """
    Display a numbered choice list.  Returns the chosen key.
    choices: [(key, label), ...]
    """
    print(f"{W}  {prompt_str}{RS}")
    for i, (key, label) in enumerate(choices, start=1):
        marker = " <-- default" if key == default_key else ""
        print(f"{DIM}    [{i}] {label}{marker}{RS}")
    while True:
        raw = input(f"{W}  Choice [1]: {RS}").strip()
        if raw == "":
            return default_key
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        except ValueError:
            pass
        print(f"{R}  Invalid choice.  Enter a number between 1 and {len(choices)}.{RS}")


def _prompt_yesno(prompt_str: str, default: bool) -> bool:
    default_str = "y" if default else "n"
    raw = input(f"{W}  {prompt_str} [y/n] [{default_str}]: {RS}").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "1")


def configure_params(selected_tests: list[dict]) -> dict[int, dict]:
    """
    For each selected test that has configurable parameters,
    prompt the user and collect values.
    Returns {test_id: {param: value}}.
    """
    user_params: dict[int, dict] = {}

    for test in selected_tests:
        if not test["configurable"]:
            user_params[test["id"]] = {}
            continue

        defaults = test["defaults"].copy()
        tid = test["id"]
        p   = {}

        print()
        print(f"{W}{'-'*66}{RS}")
        print(f"{W}  Configure: {test['name']}{RS}")
        print(f"{DIM}  Press Enter to accept the default value shown in [brackets].{RS}")
        print(f"{W}{'-'*66}{RS}")

        # -----------------------------------------------------------------
        if tid == 5:   # NCCL Loopback
            p["iters"] = _prompt(
                "Iterations per message size (recommended: 5-100)",
                defaults["iters"], int,
                validate=lambda v: 1 <= v <= 1000,
            )
            p["msg_range"] = _prompt_choice(
                "Message size range:",
                [("8g",   "8G only              (quick check)"),
                 ("full", "Full sweep  2B ~ 8G  (all sizes)")],
                defaults["msg_range"],
            )

        # -----------------------------------------------------------------
        elif tid in (6, 7):   # RDMA IPv4 / IPv6
            label = "IPv4" if tid == 6 else "IPv6"
            p["duration"] = _prompt(
                f"Duration per message size, seconds (RDMA {label}, recommended: 10-120)",
                defaults["duration"], int,
                validate=lambda v: 1 <= v <= 600,
            )
            p["test_type"] = _prompt_choice(
                "Test type:",
                [("all",   "All  (ib_read + ib_send + ib_write)"),
                 ("read",  "ib_read_bw  only"),
                 ("send",  "ib_send_bw  only"),
                 ("write", "ib_write_bw only")],
                defaults["test_type"],
            )

        # -----------------------------------------------------------------
        elif tid == 8:   # 1G NIC iPerf
            p["client_ip"] = _prompt(
                "Client (su2) IP address  [Note: passwordless SSH must be configured]",
                defaults["client_ip"], str,
                validate=lambda v: len(v.split(".")) == 4,
            )
            p["mode"] = _prompt_choice(
                "Test mode:",
                [("bidirectional",   "Bidirectional  (SUT <--> Client)"),
                 ("unidirectional",  "Unidirectional (SUT --> Client)")],
                defaults["mode"],
            )
            p["duration"] = _prompt(
                "Test duration, seconds (recommended: 60-7200, default 3600 = 1 hour)",
                defaults["duration"], int,
                validate=lambda v: 10 <= v <= 86400,
            )

        # -----------------------------------------------------------------
        elif tid == 10:   # Power Monitor
            p["bg_mode"] = _prompt_choice(
                "Monitor mode:",
                [("bg",    "Background — follow other tests (auto start/stop)"),
                 ("fixed", "Fixed duration (specify seconds below)")],
                "bg",
            )
            if p["bg_mode"] == "fixed":
                p["bg_mode"]  = False
                p["duration"] = _prompt(
                    "Monitor duration, seconds",
                    defaults.get("duration", 300), int,
                    validate=lambda v: v > 0,
                )
            else:
                p["bg_mode"]  = True
                p["duration"] = 0
            p["interval"] = _prompt(
                "Sampling interval, seconds  (0.1 = default / 1.0 = light / 5.0 = minimal)",
                defaults.get("interval", 0.1), float,
                validate=lambda v: 0.1 <= v <= 60.0,
            )
            p["csv_log"] = _prompt_yesno(
                "Save CSV power log?",
                defaults.get("csv_log", False),
            )

        # -----------------------------------------------------------------
        elif tid == 11:   # NeMo DL Validation
            print()
            print(f"{DIM}  Note: GB_training_scripts.txt is read from tools/GB_DL_scripts_v9/{RS}")
            print(f"{DIM}  Decode tools/GB_DL_scripts_v9.nv7z if not yet extracted.{RS}")
            print(f"{DIM}  If not found, the script runs a built-in smoke test instead.{RS}")
            print()
            p["image"] = _prompt(
                "Container image  [Note: pytorch:25.04-py3 has no ARM64 — keep 24.07]",
                defaults["image"], str,
            )
            p["gpus"] = _prompt(
                "Number of GPUs to expose",
                defaults["gpus"], int,
                validate=lambda v: 1 <= v <= 8,
            )
            p["cmd"] = _prompt(
                "Training command  (leave blank to read from GB_training_scripts.txt)",
                "", str,
            )

        # -----------------------------------------------------------------
        elif tid == 12:   # NVBandwidth Loopback
            p["iters"] = _prompt(
                "Iterations per test case (recommended: 1-10)",
                defaults["iters"], int,
                validate=lambda v: 1 <= v <= 50,
            )

        user_params[tid] = p

    return user_params

def _fmt_params(test: dict, params: dict) -> str:
    if not params:
        return ""
    parts = []
    tid = test["id"]
    if tid == 5:
        parts.append(f"iters={params.get('iters')}")
        parts.append("msg=full" if params.get("msg_range") == "full" else "msg=8G")
    elif tid in (6, 7):
        parts.append(f"duration={params.get('duration')}s")
        parts.append(f"type={params.get('test_type')}")
    elif tid == 8:
        parts.append(f"client={params.get('client_ip')}")
        parts.append(params.get("mode", "bidirectional"))
        parts.append(f"duration={params.get('duration')}s")
    elif tid == 10:
        parts.append("bg" if params.get("bg_mode") else f"fixed={params.get('duration')}s")
        parts.append(f"interval={params.get('interval', 1.0)}s")
        parts.append("csv=yes" if params.get("csv_log") else "csv=no")
    elif tid == 11:
        parts.append(f"image={params.get('image','').split('/')[-1]}")
        parts.append(f"gpus={params.get('gpus', 4)}")
        if params.get("cmd"):
            parts.append(f"cmd=custom")
        else:
            parts.append("cmd=GB_training_scripts.txt")
    elif tid == 12:
        parts.append(f"iters={params.get('iters', 3)}")
    return "  " + ", ".join(parts) if parts else ""


def confirm_plan(selected: list[dict], user_params: dict,
                 report_base: str, dry_run: bool) -> bool:
    """
    Display the execution plan and ask for confirmation.
    Returns True if user confirms, False if they cancel.
    """
    print()
    print(f"{W}{SEP}{RS}")
    print(f"{W}  Execution Plan{RS}")
    if dry_run:
        print(f"{Y}  [DRY-RUN MODE -- no tests will actually execute]{RS}")
    print(f"{W}{SEP}{RS}")

    for i, test in enumerate(selected, start=1):
        params    = user_params.get(test["id"], {})
        param_str = _fmt_params(test, params)
        print(f"{W}  [{i}/{len(selected)}]  {test['name']}{RS}"
              + (f"\n{DIM}{param_str}{RS}" if param_str else ""))

    print()
    print(f"{DIM}  Log output: {report_base}/run_<timestamp>/{RS}")
    print(f"{W}{SEP}{RS}")
    print()

    raw = input(f"{Y}  Run all selected tests? [y/N]: {RS}").strip().lower()
    return raw in ("y", "yes")


# ===========================================================================
# SECTION 4 — CLI argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gb300_launcher.py",
        description="GB300 Benchmark Launcher — ARM64 / Ubuntu 24.04",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo python3 gb300_launcher.py
  sudo python3 gb300_launcher.py --dry-run
  sudo python3 gb300_launcher.py --select 1,3,6 --yes
  sudo python3 gb300_launcher.py --no-prereq --log-dir /tmp/reports
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate execution -- show commands but do not run them",
    )
    parser.add_argument(
        "--no-prereq", action="store_true",
        help="Skip binary / tool pre-flight checks",
    )
    parser.add_argument(
        "--select", default=None,
        help="Comma-separated test IDs to run (e.g. 1,3,6) -- skips interactive menu",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Accept all default parameter values without prompting (use with --select)",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Override the report/ output directory",
    )
    return parser.parse_args()


# ===========================================================================
# SECTION 5 — Main
# ===========================================================================

def main():
    args = parse_args()

    # Print banner (outside curses, so it stays visible)
    print(f"{G}{BANNER}{RS}")
    print(f"{DIM}{SUBTITLE}{RS}")
    print()

    # Pre-flight
    availability = preflight(args.no_prereq)

    # Determine report base directory
    report_base = args.log_dir or os.path.join(ROOT_DIR, "report")
    os.makedirs(report_base, exist_ok=True)

    # -----------------------------------------------------------------------
    # Test selection
    # -----------------------------------------------------------------------
    if args.select:
        # Non-interactive mode: parse --select IDs
        try:
            sel_ids = [int(x.strip()) for x in args.select.split(",")]
        except ValueError:
            print(f"{R}  [ERROR] --select requires comma-separated integers, e.g. 1,3,6{RS}")
            sys.exit(1)

        selected = []
        for tid in sel_ids:
            test = next((t for t in TESTS if t["id"] == tid), None)
            if test is None:
                print(f"{R}  [ERROR] Unknown test ID: {tid}{RS}")
                sys.exit(1)
            failures = availability.get(tid, [])
            if failures:
                print(f"{Y}  [WARN] Test {tid} ({test['name']}) is unavailable: "
                      f"{failures[0]}.  Skipping.{RS}")
                continue
            selected.append(test)

        if not selected:
            print(f"{R}  No valid tests selected.  Exiting.{RS}")
            sys.exit(1)
    else:
        # Interactive curses menu
        selected = run_menu(availability)
        if selected is None:
            print(f"\n{DIM}  Launcher exited.{RS}\n")
            sys.exit(0)

    # -----------------------------------------------------------------------
    # Parameter configuration
    # -----------------------------------------------------------------------
    if args.yes:
        # Use all defaults without prompting
        user_params = {t["id"]: t["defaults"].copy() for t in selected}
    else:
        user_params = configure_params(selected)

    # -----------------------------------------------------------------------
    # Execution plan confirmation
    # -----------------------------------------------------------------------
    if not args.yes:
        confirmed = confirm_plan(selected, user_params, report_base, args.dry_run)
        if not confirmed:
            print(f"\n{DIM}  Aborted by user.{RS}\n")
            sys.exit(0)

    # -----------------------------------------------------------------------
    # Create run directory and execute
    # -----------------------------------------------------------------------
    run_dir      = create_run_dir(report_base, selected_ids=[t["id"] for t in selected])
    platform_str = get_platform_info()

    print()
    print(f"{G}{SEP}{RS}")
    print(f"{G}  Starting {len(selected)} test(s) -- {_now()}{RS}")
    print(f"{DIM}  Report: {run_dir}{RS}")
    print(f"{G}{SEP}{RS}")

    results = run_all(selected, user_params, run_dir, dry_run=args.dry_run)

    write_summary(results, run_dir, platform_info=platform_str)

    # Final exit code: non-zero if any test failed
    any_failed = any(r["rc"] != 0 and not r["skipped"] for r in results)
    sys.exit(1 if any_failed else 0)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
