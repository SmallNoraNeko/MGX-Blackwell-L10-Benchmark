#!/usr/bin/env python3
"""
NIC PCIe Validation Script for GB300 ARM (Ubuntu)
─────────────────────────────────────────────────
★ Fully auto-discovers NICs from `mst status -v`
  No hardcoded PCI addresses needed — works on any server.

Per-port log layout  /var/log/nic_check/<TIMESTAMP>/
  mst_status.log
  summary.log
  <PREFIX>_<pci>_lspci.log
  <PREFIX>_mlx5_<rdma_n>_<pci>_mlxconfig.log
  <PREFIX>_<pci>_result.log
"""

import subprocess, sys, re, shutil, os
from datetime import datetime
from dataclasses import dataclass, field

# ── PCIe speed spec per known device type ─────────────────────────────────────
DEVICE_SPEED_SPEC = {
    "connectx8":  {"speed": "64GT/s", "width": "x16", "vfs": 16, "product": "ConnectX-8", "gen": "Gen6"},
    "bluefield3": {"speed": "32GT/s", "width": "x16", "vfs": 16, "product": "BlueField-3", "gen": "Gen5"},
    "connectx7":  {"speed": "32GT/s", "width": "x16", "vfs": 16, "product": "ConnectX-7",  "gen": "Gen5"},
    "connectx6":  {"speed": "16GT/s", "width": "x16", "vfs": 8,  "product": "ConnectX-6",  "gen": "Gen4"},
    # Add more device types here as needed
}

def get_spec_for(device_type: str) -> dict | None:
    """Match device type string (case-insensitive) to spec dict."""
    key = device_type.lower().replace("-", "").replace("(rev:0)", "").replace("(rev:1)", "").strip()
    for k, v in DEVICE_SPEED_SPEC.items():
        if k in key or key in k:
            return v
    return None

# ── Logging setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TS         = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR    = os.environ.get("LOG_OUTPUT_DIR") or os.path.join(SCRIPT_DIR, f"nic_check_{TS}")
os.makedirs(LOG_DIR, exist_ok=True)

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def _strip_ansi(t): return re.sub(r"\033\[[0-9;]*m", "", t)
def ok(m):   return f"{GREEN}[PASS]{RESET} {m}"
def fail(m): return f"{RED}[FAIL]{RESET} {m}"
def warn(m): return f"{YELLOW}[WARN]{RESET} {m}"
def tprint(m=""): print(m)

def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(_strip_ansi(content))

def run(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] {cmd}\n"

# ── Auto-discovered NIC entry ──────────────────────────────────────────────────
@dataclass
class NICPort:
    device_type: str        # raw string from mst, e.g. "ConnectX8(rev:0)"
    mst_dev:     str        # /dev/mst/mt4131_pciconf0
    pci:         str        # 0002:03:00.0
    rdma:        str        # mlx5_2  (may be empty)
    net:         str        # enP2p3s0f0np0  (may be empty)
    numa:        str        # 0 or 1
    spec:        dict = field(default_factory=dict)   # matched speed/width/vfs

    @property
    def prefix(self):
        dt = self.device_type.lower()
        if "bluefield" in dt: return "BF3"
        if "connectx8" in dt: return "CX8"
        if "connectx7" in dt: return "CX7"
        if "connectx6" in dt: return "CX6"
        return "NIC"

    @property
    def pci_tag(self):
        return self.pci.replace(":", "_").replace(".", "_")

    @property
    def rdma_tag(self):
        return self.rdma.replace("mlx5_", "").replace("bond_", "bond") if self.rdma else "x"

    @property
    def label(self):
        return f"{self.prefix} {self.pci}"

    def lspci_logname(self):    return f"{self.prefix}_{self.pci_tag}_lspci.log"
    def mlx_logname(self):      return f"{self.prefix}_mlx5_{self.rdma_tag}_{self.pci_tag}_mlxconfig.log"
    def result_logname(self):   return f"{self.prefix}_{self.pci_tag}_result.log"


# ── Parse mst status -v ───────────────────────────────────────────────────────
def parse_mst_status(output: str) -> list[NICPort]:
    """
    Parse the PCI devices table from `mst status -v`.

    Expected columns (space-separated, variable width):
      DEVICE_TYPE  MST  PCI  RDMA  NET  NUMA  VFIO  FWCTL  STATE
    """
    ports = []
    in_table = False
    header_found = False

    for line in output.splitlines():
        line_s = line.strip()

        # Detect header row
        if "DEVICE_TYPE" in line and "MST" in line and "PCI" in line:
            header_found = True
            in_table = True
            continue

        if not in_table:
            continue

        # Skip separator / empty lines after header
        if not line_s or line_s.startswith("-"):
            continue

        # Stop if we hit another section
        if line_s.startswith("MST modules") or line_s.startswith("Non MST"):
            break

        # Split on 2+ spaces to handle variable-width columns
        cols = re.split(r"\s{2,}", line_s)

        # Need at least DEVICE_TYPE, MST, PCI
        if len(cols) < 3:
            continue

        device_type = cols[0]
        mst_dev     = cols[1] if len(cols) > 1 else ""
        pci         = cols[2] if len(cols) > 2 else ""
        rdma        = cols[3] if len(cols) > 3 else ""
        net         = cols[4] if len(cols) > 4 else ""
        numa        = cols[5] if len(cols) > 5 else ""

        # Validate PCI looks like a BDF (xx:xx.x or xxxx:xx:xx.x)
        if not re.match(r"[\da-fA-F]{4}:[\da-fA-F]{2}:[\da-fA-F]{2}\.\d", pci):
            # try short form xx:xx.x — not expected from mst but just in case
            if not re.match(r"[\da-fA-F]{2}:[\da-fA-F]{2}\.\d", pci):
                continue

        spec = get_spec_for(device_type)
        if spec is None:
            tprint(warn(f"Unknown device type '{device_type}' at {pci} — skipping (add to DEVICE_SPEED_SPEC)"))
            continue

        ports.append(NICPort(
            device_type=device_type,
            mst_dev=mst_dev,
            pci=pci,
            rdma=rdma,
            net=net,
            numa=numa,
            spec=spec,
        ))

    return ports


# ── 7 Checks ─────────────────────────────────────────────────────────────────
def chk_vf(lspci_out, port):
    m = re.search(r"Total VFs:\s*(\d+)", lspci_out)
    if not m: return False, "SR-IOV / Total VFs not found"
    v = int(m.group(1))
    exp = port.spec["vfs"]
    return v == exp, f"Total VFs = {v} (expected {exp})"

def chk_aspm(lspci_out, port):
    for line in lspci_out.splitlines():
        if "LnkCtl:" in line and "ASPM" in line:
            return "ASPM Disabled" in line, line.strip()
    return False, "LnkCtl / ASPM line not found"

def chk_error(lspci_out, port):
    has_u = any("UEMsk" in l for l in lspci_out.splitlines())
    has_c = any("CEMsk" in l for l in lspci_out.splitlines())
    if has_u and has_c: return True, "UEMsk + CEMsk present (AER confirmed)"
    missing = [x for x, f_ in [("UEMsk", has_u), ("CEMsk", has_c)] if not f_]
    return False, f"Missing: {', '.join(missing)}"

def chk_io(lspci_out, port):
    for line in lspci_out.splitlines():
        if line.strip().startswith("Control:"):
            if "I/O-" in line: return True,  "I/O- (not occupied)"
            if "I/O+" in line: return False, f"I/O+ (occupied): {line.strip()}"
    if "I/O port" in lspci_out: return False, "I/O port entry found"
    return True, "No I/O port usage"

def chk_link(lspci_out, port):
    exp_speed = port.spec["speed"]
    exp_width = port.spec["width"]
    for line in lspci_out.splitlines():
        if re.match(r"\s+LnkSta:", line) and "LnkSta2" not in line:
            s_ok = exp_speed in line
            w_ok = exp_width in line
            issues = ([] if s_ok else [f"speed expected {exp_speed}"]) + \
                     ([] if w_ok else [f"width expected {exp_width}"])
            return (s_ok and w_ok), line.strip() + (f"  ← {', '.join(issues)}" if issues else "")
    return False, "LnkSta line not found"

def chk_product(lspci_out, port):
    substr = port.spec["product"]
    for line in lspci_out.splitlines():
        if "Product Name:" in line:
            name = line.split("Product Name:")[-1].strip()
            return substr.lower() in name.lower(), name[:100]
    return False, "Product Name not found"

def chk_net(port):
    if not port.net:
        return None, "NET column empty in mst status — skipping"
    bdf_full  = port.pci
    bdf_short = ":".join(port.pci.split(":")[-2:])
    for bdf in [bdf_full, bdf_short]:
        out = run(f'ls $(find /sys/devices/ -name "net" -type d | grep "{bdf}") 2>/dev/null').strip()
        if out:
            names = out.split()
            matched = port.net in names or any(port.net in n or n in port.net for n in names)
            return matched, f"Net device: {' '.join(names)}" + \
                   ("" if matched else f"  ← expected '{port.net}'")
    return False, f"No net device found for {bdf_full}"

CHECKS = [
    ("[1] VF count",      chk_vf),
    ("[2] ASPM disabled", chk_aspm),
    ("[3] Error report",  chk_error),
    ("[4] I/O space",     chk_io),
    ("[5] Link status",   chk_link),
    ("[6] Product name",  chk_product),
]


# ── Per-port runner ───────────────────────────────────────────────────────────
_summary_lines = []

def run_port(port: NICPort) -> tuple[int, int]:
    header = f"{'='*60}\n{port.label}  [mst: {port.mst_dev}]\n{'='*60}"
    tprint(f"\n{BOLD}{header}{RESET}")

    lspci_out = run(f"lspci -vvv -s {port.pci}")
    mlx_out   = run(f"mlxconfig -d {port.mst_dev} query")

    lspci_path  = os.path.join(LOG_DIR, port.lspci_logname())
    mlx_path    = os.path.join(LOG_DIR, port.mlx_logname())
    result_path = os.path.join(LOG_DIR, port.result_logname())

    write_file(lspci_path, lspci_out)
    write_file(mlx_path,   mlx_out)

    result_lines = [header, f"PCI     : {port.pci}",
                    f"MST dev : {port.mst_dev}",
                    f"RDMA    : {port.rdma}",
                    f"NET     : {port.net}",
                    f"NUMA    : {port.numa}",
                    f"Spec    : {port.spec['gen']}  {port.spec['speed']} {port.spec['width']}  VFs={port.spec['vfs']}",
                    ""]

    passed = failed = 0

    if not lspci_out.strip():
        msg = f"lspci returned nothing for {port.pci}"
        tprint(fail(msg))
        result_lines.append(f"[FAIL] {msg}")
        write_file(result_path, "\n".join(result_lines))
        _summary_lines.append(f"  {port.label:<32} [FAIL] no lspci output  (0/7)")
        return 0, 7

    for name, fn in CHECKS:
        p, detail = fn(lspci_out, port)
        tprint(f"  {name}: {ok(detail) if p else fail(detail)}")
        result_lines.append(f"  {name}: {'[PASS]' if p else '[FAIL]'} {detail}")
        passed += p; failed += (not p)

    # Check 7
    p, detail = chk_net(port)
    if p is None:   # skipped
        tprint(f"  [7] Net device: {warn(detail)}")
        result_lines.append(f"  [7] Net device: [SKIP] {detail}")
    else:
        tprint(f"  [7] Net device: {ok(detail) if p else fail(detail)}")
        result_lines.append(f"  [7] Net device: {'[PASS]' if p else '[FAIL]'} {detail}")
        passed += p; failed += (not p)

    result_lines += ["",
                     f"Result  : {passed}/7 passed",
                     f"lspci   : {lspci_path}",
                     f"mlxcfg  : {mlx_path}"]

    write_file(result_path, "\n".join(result_lines))
    tprint(f"  {CYAN}→ {result_path}{RESET}")

    status = "[PASS] ALL PASS" if failed == 0 else f"[FAIL] {failed} item(s) failed"
    _summary_lines.append(f"  {port.label:<32} {status}  ({passed}/7)")
    return passed, failed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tprint(f"\n{BOLD}{'='*60}")
    tprint(f"  GB300 NIC PCIe Validation Script  (auto-discovery)")
    tprint(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tprint(f"  Log dir : {LOG_DIR}")
    tprint(f"{'='*60}{RESET}")

    # ── Pre-check: mst ────────────────────────────────────────────────────────
    tprint(f"\n{BOLD}PRE-CHECK: mst status -v{RESET}")
    if not shutil.which("mst"):
        tprint(fail("'mst' not found. Install MFT first."))
        sys.exit(1)

    mst_out = run("mst status -v")
    write_file(os.path.join(LOG_DIR, "mst_status.log"), mst_out)
    tprint(mst_out)

    ports = parse_mst_status(mst_out)

    if not ports:
        tprint(fail("No supported NICs discovered from mst status -v. Aborting."))
        sys.exit(1)

    # Print discovered port table
    tprint(f"\n{BOLD}Discovered {len(ports)} port(s):{RESET}")
    tprint(f"  {'DEVICE_TYPE':<18} {'PCI':<18} {'MST_DEV':<30} {'RDMA':<14} NET")
    tprint(f"  {'-'*100}")
    for p in ports:
        tprint(f"  {p.device_type:<18} {p.pci:<18} {p.mst_dev:<30} {p.rdma:<14} {p.net}")

    # Confirm before proceeding
    tprint(f"\n{YELLOW}Press Enter to start checks, or Ctrl-C to abort...{RESET}", )
    try:
        input()
    except KeyboardInterrupt:
        tprint("\nAborted.")
        sys.exit(0)

    # ── Collect whole-system lspci -vvv ──────────────────────────────────────
    tprint(f"\n{BOLD}Collecting full lspci -vvv ...{RESET}")
    lspci_full = run("lspci -vvv")
    lspci_full_path = os.path.join(LOG_DIR, "lspci_vvv.txt")
    write_file(lspci_full_path, lspci_full)
    tprint(f"  {CYAN}-> {lspci_full_path}{RESET}")

    # ── Collect NIC device names (all ports in one file) ──────────────────────
    tprint(f"\n{BOLD}Collecting NIC device names ...{RESET}")
    dev_name_lines = []
    for port in ports:
        bdf_full  = port.pci
        bdf_short = ":".join(port.pci.split(":")[-2:])
        cmd = f'ls $(find /sys/devices/ -name "net" -type d | grep "{bdf_full}") 2>/dev/null'
        out = run(cmd).strip()
        if not out:
            cmd = f'ls $(find /sys/devices/ -name "net" -type d | grep "{bdf_short}") 2>/dev/null'
            out = run(cmd).strip()
        dev_name_lines.append(f'# ls $(find /sys/devices/ -name "net" -type d | grep "{bdf_full}")')
        dev_name_lines.append(out if out else "(no output)")
        dev_name_lines.append("")
        tprint(f"  {bdf_full:<18} -> {out if out else '(no output)'}")

    dev_name_path = os.path.join(LOG_DIR, "NIC_device_name.log")
    write_file(dev_name_path, "\n".join(dev_name_lines))
    tprint(f"  {CYAN}-> {dev_name_path}{RESET}")

    # ── Per-port checks ───────────────────────────────────────────────────────
    total_pass = total_fail = 0
    for port in ports:
        p, f_ = run_port(port)
        total_pass += p; total_fail += f_

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_text = "\n".join([
        "=" * 60, "  SUMMARY", "=" * 60,
        f"  Timestamp : {TS}",
        f"  Log dir   : {LOG_DIR}",
        f"  NICs found: {len(ports)}",
        "",
    ] + _summary_lines + [
        "",
        f"  Total: {total_pass}/{total_pass+total_fail} checks passed",
        "=" * 60,
    ])

    tprint(f"\n{BOLD}{'='*60}\n  SUMMARY\n{'='*60}{RESET}")
    for line in _summary_lines:
        colour = GREEN if "[PASS]" in line else RED
        tprint(f"{colour}{line}{RESET}")
    tprint(f"\n{BOLD}Total: {total_pass}/{total_pass+total_fail} passed{RESET}")

    summary_path = os.path.join(LOG_DIR, "summary.log")
    write_file(summary_path, summary_text)
    tprint(f"\n{CYAN}Summary : {summary_path}{RESET}")
    tprint(f"{CYAN}All logs: {LOG_DIR}/{RESET}\n")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
