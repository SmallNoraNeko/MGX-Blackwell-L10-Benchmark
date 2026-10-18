# RDMA Loopback Test — README (v1.0.5)

> Scripts: `lib/rdma_test_ipv4_v4.py` (test 6) / `lib/rdma_test_ipv6_v5.py` (test 7)
> Shared library: `lib/_rdma_lib.py`
> Platform: GB300 L10 MGX (system10) / ARM64 / Ubuntu 24.04
> Validated: 2026-10-16 on system10 (all pairs PASS for read / send / write incl. GPU Direct)

---

## What changed vs v1.0.4 (ipv4_v2 / ipv6_v3)

| Item | v1.0.4 | v1.0.5 |
|---|---|---|
| BF3 pairs | mlx5_0↔1, mlx5_6↔7 (same card, wrong) | **mlx5_0↔6, mlx5_1↔7** (cross-card, verified by traffic) |
| netdev MTU | 1500 → RoCE active_mtu 1024 | **9000 → active_mtu 4096**, restored on exit (BF3 352→394, CX8 740→784 Gb/s) |
| BF3 | raw path | **rdma_cm (`-R`)**, GPU Direct **without** `--use_data_direct` (BF3 unsupported → MR fails, hangs) |
| CX8 | — | `-m 4096`; GPU Direct with `--use_data_direct`, **server/client on different GPUs** (same GPU → run_iter_bw fails) |
| Schedule | all pairs at once | **4 waves**: pair1+3 / pair2+4 / pair5 / pair6 (same-card ports and BF3 + NUMA1 CX8 contend) |
| GID | fallback to wrong GID silently (ipv6: always GID 3) | strict match to test IP, abort if missing |
| Connectivity | ping (always passes on same host) | **short RDMA write per pair** |
| Platform | hard-coded / topology.json | **`rdma_platforms.json` profiles** (`--platform`), auto-detect for missing fields |
| Cleanup | `ip addr flush` (wiped every IP on the port) | removes only the test IPs; SIGTERM/SIGINT also clean up |
| Verdict | none | **PASS/FAIL per pair** (best BW at ≥64KB vs 90% of link rate) + exit code |

## Platform profiles (`rdma_platforms.json`)

Pairs, card types, thresholds, GPU mapping and wave schedule come from `rdma_platforms.json`
(project root or `lib/`). Without the file, the built-in validated `GB300_L10` profile is used.
`topology.json` is NOT read by these scripts (it is only written by the menu's `d` discovery).

```bash
sudo python3 lib/rdma_test_ipv4_v4.py --list-platforms
sudo python3 lib/rdma_test_ipv4_v4.py --platform TTM --dry-run     # shows detected cards/links/GPUs/waves
```

| Key | Required | When omitted (auto) |
|---|---|---|
| `pairs` | yes | — (named pair1..N in order) |
| `cards` | no | sysfs PCI device id → board_id → `mst status -v` |
| `link_gbps` | no | `/sys/class/infiniband/<nic>/ports/1/rate` |
| `gpu_map` | no | GPU on the NIC's NUMA node; server/client get different GPUs when possible |
| `waves` | no | ports of the same physical card never run in the same wave |
| `line_rate_pct` | no | 0.90 |
| `card_behavior` | no | BF3: `-R`, no data_direct · CX8: `-m 4096`, data_direct · others: neither |

Cross-card interference (on L10, BF3 slows down next to the NUMA1 CX8) cannot be detected
automatically: run the `--schedule all` / subset tests, then pin `waves` in the profile.
The startup banner warns which fields were auto-detected.

### GB300_L10 (validated 2026-10-16)

| Pair | Server | Client | Card | Threshold (bidir) | Measured |
|---|---|---|---|---|---|
| pair1 | mlx5_0 | mlx5_6 | BF3 400G | 360 Gb/s | ~379–394 |
| pair2 | mlx5_1 | mlx5_7 | BF3 400G | 360 Gb/s | ~379–394 |
| pair3 | mlx5_2 | mlx5_4 | CX8 800G | 720 Gb/s | ~784 |
| pair4 | mlx5_3 | mlx5_5 | CX8 800G | 720 Gb/s | ~784 |
| pair5 | mlx5_8 | mlx5_10 | CX8 800G | 720 Gb/s | ~784 |
| pair6 | mlx5_9 | mlx5_11 | CX8 800G | 720 Gb/s | ~784 |

GPU (write only): mlx5_0/1/2/3 → GPU0 `0008:06:00.0`, mlx5_4/5 → GPU1 `0009`, mlx5_6/7/8/9 → GPU2 `0018`, mlx5_10/11 → GPU3 `0019`.
Waves: pair1+pair3 | pair2+pair4 | pair5 | pair6.

### TTM (not yet validated)

Only `pairs` are set (from the v1.0.4 README); everything else is auto-detected.
Run `--platform TTM --dry-run`, check the banner, run `--quick`, then pin the values.

Results are bidirectional (`-b`); divide by 2 to compare with PartnerDiag uni-directional values.

## Launcher parameters (tests 6 / 7)

| Param | Default | Meaning |
|---|---|---|
| `duration` | 30 | seconds per message size |
| `test_type` | all | all / read / send / write |
| `quick` | false | true = only 64KB & 256KB (~10 min per test type); false = 2B~8MB (~50 min per test type) |
| `platform` | "" | profile name in `rdma_platforms.json` ("" = file default, GB300_L10) |

```bash
sudo python3 gb300_launcher.py --select 6,7 --yes
sudo python3 gb300_launcher.py --select 6 --yes --params "6.quick=true,6.test_type=write"
sudo python3 gb300_launcher.py --select 6,7 --yes --params "6.platform=TTM,7.platform=TTM"
```

Run 6 and 7 one after another (the launcher does); never in parallel.

## Standalone

```bash
sudo python3 lib/rdma_test_ipv4_v4.py                 # full read/send/write
sudo python3 lib/rdma_test_ipv4_v4.py --test write --quick
sudo python3 lib/rdma_test_ipv4_v4.py --pairs pair1 pair3 --no-cuda
sudo python3 lib/rdma_test_ipv4_v4.py --dry-run
sudo python3 lib/rdma_test_ipv4_v4.py --schedule all  # all pairs at once (stress; per-pair thresholds don't apply)
```
Same options for `rdma_test_ipv6_v5.py` (IPv4-mapped IPv6 + `--ipv6`, IPs 192.168.200–205, ports 15000+).

## Output

```
<output-dir>/                      (launcher: report/run_<TS>/06_RDMA_IPv4/)
├── rdma_results_ipv4/
│   ├── read_ipv4/   server_* / client_* per pair per msg size
│   ├── send_ipv4/
│   └── write_ipv4/
├── dmesg.log  system.log  sel.log  sdr.log
```

Exit code: 0 = PASS, 1 = FAIL, 2 = pre-flight / link-check failure, 130/143 = interrupted.
