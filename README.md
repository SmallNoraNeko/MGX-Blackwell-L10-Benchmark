# GB300 Benchmark Launcher

Single-entry automation script for GB300 NVL72 / ARM64 / Ubuntu 24.04 benchmark validation.

## Quick Start

```bash
# 1. Place benchmark binaries in tools/  (see tools/README.txt)
# 2. Run as root
sudo python3 gb300_launcher.py
```

## Usage

```
sudo python3 gb300_launcher.py [OPTIONS]

Options:
  --dry-run            Simulate execution — show commands, do not run
  --no-prereq          Skip binary / tool pre-flight checks
  --select 1,3,6       Run specific tests by ID (skips interactive menu)
  --yes                Accept all default parameters (use with --select)
  --log-dir PATH       Override report/ output path
```

## Interactive Menu Controls

| Key       | Action                     |
|-----------|----------------------------|
| UP / DOWN | Move cursor                |
| Space     | Toggle selection           |
| a         | Select all available tests |
| n         | Clear all selections       |
| Enter     | Confirm and proceed        |
| q         | Quit                       |

## Available Tests

| ID | Category               | Name                 | Configurable Parameters       |
|----|------------------------|----------------------|-------------------------------|
|  1 | GPU Compute & Memory   | GPU Stream           | —                             |
|  2 | GPU Compute & Memory   | Peak TOPS            | —                             |
|  3 | GPU Compute & Memory   | GEMM Bench           | —                             |
|  4 | GPU Compute & Memory   | FP4 GEMM-MemRead     | —                             |
|  5 | GPU Interconnect       | NCCL Loopback        | iters, msg size range         |
|  6 | Network / RDMA         | RDMA Loopback IPv4   | duration, test type           |
|  7 | Network / RDMA         | RDMA Loopback IPv6   | duration, test type           |
|  8 | Network / RDMA         | 1G NIC iPerf         | client IP, mode, duration     |
|  9 | Hardware Health        | NIC PCIe Health      | —                             |
| 10 | Hardware Health        | Power Monitor        | duration mode, CSV log        |
| 11 | DL Training Validation | NeMo DL Validation   | image, gpus, training command |
| 12 | GPU Compute & Memory   | NVBandwidth Loopback | iters                         |

## Directory Structure

```
gb300_benchmark/
├── gb300_launcher.py           <- Entry point (run this)
│
├── lib/                        <- All test scripts and launcher modules
│   ├── _launcher_config.py
│   ├── _launcher_runner.py
│   ├── _rdma_common.py
│   ├── disable_acs.sh          <- Auto-called by NVBandwidth test if ACS detected
│   ├── run_nemo_validation_v3.py
│   └── ...
│
├── tools/                      <- Benchmark binaries + offline resources
│   ├── README.txt
│   ├── gemm-memread/           <- Extract from GEMM_v6.zip (GB300 path)
│   ├── nccl-build/             <- Build from nccl-tests-master.zip
│   ├── nvbandwidth             <- Build from nvbandwidth-main.zip
│   ├── GB_DL_scripts_v9/       <- Decode from GB_DL_scripts_v9.nv7z
│   │   ├── GB_training_scripts.txt   <- NeMo test reads this
│   │   ├── GB_inference_scripts.txt
│   │   └── build-llm-containers.sh
│   ├── Speech-main/            <- Extract from Speech-main.zip (optional)
│   ├── pytorch_24.07-py3_arm64.tar   <- Offline container image (optional)
│   └── ...
│
└── report/                     <- All logs (auto-created per test run)
    └── run_<TIMESTAMP>/
        ├── 00_summary.log
        ├── 01_GPU_stream/      <- Only created for tests that were selected
        ├── 02_peak_tops/
        ├── ...
        └── 12_nvbandwidth_loopback/
```

## NeMo DL Validation Setup (Test 11)

Test 11 runs a DL training workload inside a Podman container and measures
step-time throughput.

### Step 1 — Decode GB_training_scripts.txt

```bash
cd gb300_benchmark/tools/
mv GB_DL_scripts_v9.nv7z GB_DL_scripts_v9.7z
7z x GB_DL_scripts_v9.7z          # NDA password required

# Result: tools/GB_DL_scripts_v9/ containing:
#   GB_training_scripts.txt   <- launcher reads this
#   GB_inference_scripts.txt
#   build-llm-containers.sh
```

### Step 2 — Container Image (Online or Offline)

**Online (pull from NGC):**

```bash
# pytorch:24.07-py3 is the only ARM64-compatible image for GB300
# pytorch:25.04-py3 does NOT have an ARM64 image
sudo podman pull --platform linux/arm64 nvcr.io/nvidia/pytorch:24.07-py3
```

**Offline (recommended for air-gapped deployment):**

On a node with network access, export the image once:

```bash
podman save nvcr.io/nvidia/pytorch:24.07-py3 \
    -o gb300_benchmark/tools/pytorch_24.07-py3_arm64.tar
```

Place `pytorch_24.07-py3_arm64.tar` in `tools/`. The launcher automatically
detects and loads it — no manual `podman load` required.

Image resolution order (fully automatic):

```
1. Image already in local podman storage  →  use directly
2. Matching .tar found in tools/          →  podman load automatically
3. Network available                      →  podman pull
4. All failed                             →  error with instructions
```

### Step 3 — Extract Speech-main (optional)

Required only if the training command references the NeMo Speech framework.

```bash
cd gb300_benchmark/tools/
unzip Speech-main.zip
# Result: tools/Speech-main/
```

If `tools/GB_DL_scripts_v9/GB_training_scripts.txt` is not found, the
launcher automatically falls back to a built-in smoke test.

## NVBandwidth Loopback (Test 12)

Measures GPU memory bandwidth across 32 CE and SM test cases.
ACS is automatically disabled via `lib/disable_acs.sh` if detected — no
manual `setpci` needed.

**Build nvbandwidth:**

```bash
cd gb300_benchmark/tools/
unzip nvbandwidth-main.zip
cd nvbandwidth-main/
cmake -DCMAKE_BUILD_TYPE=Release .
make -j$(nproc)
cp nvbandwidth ../
cd .. && rm -rf nvbandwidth-main/
```

## Offline / Air-gapped Deployment

```bash
# On a networked node — save image and package everything
podman save nvcr.io/nvidia/pytorch:24.07-py3 \
    -o gb300_benchmark/tools/pytorch_24.07-py3_arm64.tar

# Pack without compression (tar is already non-compressible, saves time)
tar -cf gb300_benchmark_v1.0.0_offline.tar gb300_benchmark/

# Transfer to target node (USB drive, scp, etc.)
scp gb300_benchmark_v1.0.0_offline.tar root@<target>:/root/

# On the target node — extract and run directly
tar -xf gb300_benchmark_v1.0.0_offline.tar
cd gb300_benchmark/
sudo python3 gb300_launcher.py
```

The launcher auto-loads the container image from `tools/*.tar` on first run.

## Requirements

- Python 3.10+
- Ubuntu 24.04 / ARM64
- Root privileges (sudo)
- CUDA 13 / Driver 580.82+

**System tools** (install as needed):

| Tool | Used by | Install |
|------|---------|---------|
| `ipmitool` | All GPU benchmark scripts (SDR) | `apt install ipmitool` |
| `iperf` | 1G NIC iPerf (test 8) | `apt install iperf` |
| `mst`, `mlxconfig` | NIC PCIe Health (test 9) | Install Mellanox MFT |
| `lspci` | NIC PCIe Health (test 9) | `apt install pciutils` |
| `ib_read_bw`, `ib_send_bw`, `ib_write_bw` | RDMA tests (6, 7) | Build from `tools/perftest-master.zip` |
| `podman` | NeMo DL Validation (test 11) | `apt install podman` |
| `nvidia-ctk` | NeMo DL Validation (test 11) | Install NVIDIA Container Toolkit |

See `tools/README.txt` and `GB300_Benchmark_Setup_Guide.md` for full
build and installation instructions.
