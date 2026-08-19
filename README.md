# MGX Architecture Blackwell L10 — Automated Benchmark Suite

> **PERSONAL PORTFOLIO PROJECT**
> Customized benchmark suite for a specific MGX Architecture Blackwell L10 Standalone
> deployment. Not a general-purpose tool. Configuration: Standalone — Mini-Rack
> Cartridges only. No Rack. No NVSwitch.

---

## Demo

https://github.com/SmallNoraNeko/MGX-Blackwell-L10-Benchmark/releases/download/v1.0.1/GB300_Automated_Benchmark_in_Action_Demo.mp4

---

## Overview

A 12-in-1 automated benchmark suite with an interactive TUI launcher, designed to
validate the complete hardware stack of an MGX Architecture Blackwell L10 Standalone
system — covering GPU compute, memory bandwidth, RDMA networking, NVLink, and
end-to-end deep learning validation.

---

## Benchmark Modules

| No. | Module | Scope |
|-----|--------|-------|
| 01 | GPU Stream | Memory bandwidth validation |
| 02 | Peak TOPS | AI compute peak throughput |
| 03 | GEMM Bench | cuBLAS matrix multiply performance |
| 04 | FP4 GEMM | FP4 precision GEMM validation |
| 05 | NCCL Loopback | GPU interconnect collective operations |
| 06 | RDMA IPv4 | InfiniBand RDMA throughput (IPv4) |
| 07 | RDMA IPv6 | InfiniBand RDMA throughput (IPv6) |
| 08 | 1G NIC iPerf | Management NIC throughput |
| 09 | NIC PCIe Health | PCIe link state and topology check |
| 10 | Power Monitor | GPU TDP and average power draw |
| 11 | NeMo DL Validation | End-to-end deep learning training validation |
| 12 | NVBandwidth Loopback | NVLink bandwidth loopback test |

---

## Hardware & Firmware Specification

| Component | Specification |
|-----------|--------------|
| Architecture | MGX Architecture |
| GPU Module | Blackwell L10 |
| Form Factor | Standalone — Mini-Rack (no Rack, no NVSwitch) |
| Cooling | Liquid Cooling via Mini-Cartridge |
| Interconnect | InfiniBand RDMA (IPv4 + IPv6) — CX8 / BF3 |
| Host Platform | Intel NUC |
| GPU Driver | 580.126.20 (ARM64 SBSA / Ubuntu 24.04) |
| CUDA | 13.0.2 |
| DOCA Host | 3.2.1-044413 |
| HMC | GB200Nvl-25.08-B |
| GPU VBIOS | 97.10.4A.00.1F |
| SBIOS (NV BIOS) | 2.05.05 |
| FPGA (SMR) | 1.60 |
| CX8 Firmware | 40.47.2526 |
| BF3 Firmware | 32.47.2526 |

Full firmware stack: [docs/setup/Standalone_L10_Setup_Spec.md](./docs/setup/Standalone_L10_Setup_Spec.md)

---

## Docker

Base image: `nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04`

Proprietary binary tools are **not included** in the image.
Mount `tools/` as a volume at runtime.

Prerequisites: Docker Engine · NVIDIA Container Toolkit · `tools/` populated

Build:

\`\`\`bash
docker build -t night-kuronos/mgx-blackwell-l10-benchmark:1.0.1 .
\`\`\`

Run:

\`\`\`bash
docker compose run benchmark
\`\`\`

---

## Repository Structure

\`\`\`
MGX-Blackwell-L10-Benchmark/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── gb300_launcher.py                TUI main entry point
├── lib/                             12 benchmark modules
├── docs/setup/                      Standalone_L10_Setup_Spec.md
├── tools/                           NOT tracked — place binaries here manually
└── report/                          NOT tracked — benchmark output logs
\`\`\`

---

## Disclaimer

Published as a personal technical portfolio piece. Engineered for a single specific
hardware deployment. Not maintained as an open-source project.

---

## License

MIT License — see [LICENSE](./LICENSE) for details.
