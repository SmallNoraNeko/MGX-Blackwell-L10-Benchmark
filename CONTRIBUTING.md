# Contributing

This repository is a personal portfolio project, not an open-source community tool.
It documents a customized benchmark suite built for a specific MGX Architecture
Blackwell L10 Standalone deployment and is published for reference and portfolio
purposes only. Source package: gb300_benchmark_v1.0.1.zip

---

## Scope

The benchmark suite operates against a fixed hardware topology and depends on
proprietary binary tools not distributed in this repository. All 12 benchmark
modules, the TUI launcher, and supporting scripts were written and tuned for
this specific deployment. There is no abstraction layer designed for reuse.

---

## What Is Not Accepted

| Category                      | Status            |
|-------------------------------|-------------------|
| Pull Requests                 | Not accepted      |
| Feature Requests              | Not applicable    |
| Bug Reports (general use)     | Not applicable    |
| Hardware porting requests     | Out of scope      |
| Binary tool distribution      | Not distributable |
| General deployment support    | Not provided      |

---

## Author

Night-Kuronos / SmallNoraNeko — GPU Infrastructure Validation Engineering

Specialization: MGX Architecture hardware validation, NVIDIA GPU stack
(NCCL, RDMA, NeMo, cuBLAS, NVBandwidth), Docker, ROS 2, ARM64,
InfiniBand, PCIe topology validation.

---

## License

MIT License. Permitted for portfolio viewing and technical reference only.
Deployment on other hardware requires independent re-engineering of all
hardware assumptions, binary dependencies, and network configurations.
