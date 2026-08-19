# MGX Architecture Blackwell L10 — Standalone Validation Spec

> PERSONAL PORTFOLIO PROJECT
> Single-unit standalone configuration. No Rack, no NVSwitch.
> Mini-Rack Cartridges only. Source: gb300_benchmark_v1.0.1.zip

---

## Platform Boundary

| Item             | Configuration                        |
|------------------|--------------------------------------|
| Form Factor      | Standalone — not rack-mounted        |
| NVSwitch         | Not connected                        |
| Interconnect     | Mini-Rack Cartridges only            |
| Multi-node       | No                                   |

---

## Compute Tray Firmware — V1.0.6 (PID: 1152046, 2026/03/17)

| Component       | Version                  |
|-----------------|--------------------------|
| ERoT            | 01.04.0031_0000_n04      |
| HMC             | GB200Nvl-25.08-B         |
| SBIOS (NV BIOS) | 2.05.05                  |
| FPGA (SMR)      | 1.60                     |
| GPU VBIOS       | 97.10.4A.00.1F           |
| HMC CPLD        | 0.22                     |
| E1.S BP CPLD    | 02 (2025/09/19)          |

---

## CX8 / BF3 — V1.0.5 (PID: 1149372, 2026/02/06)

| Component      | Version        | Notes                                  |
|----------------|----------------|----------------------------------------|
| CX8            | 40.47.2526     |                                        |
| BF3            | 32.47.2526     |                                        |
| DOCA Host      | 3.2.1-044413   |                                        |
| MFT            | 4.34.1-12      |                                        |
| CUDA           | 13.0.2         | 搭配新的 DOCA 可以不用綁 CUDA 12.9     |
| BF3 Ubuntu BFB | 3.2.1-42       |                                        |

Package note: Validated against Compute Tray V1.0.6.
CX8/BF3 portion remains at V1.0.5. Package name identical to 1.0.5_dev00.

---

## GPU Drivers

| Component   | Version    | Platform                            |
|-------------|------------|-------------------------------------|
| GPU Driver  | 580.126.20 | Linux / ARM64 SBSA / Ubuntu 24.04  |
| IMEX Driver | 580.126.20 |                                     |

Download: https://developer.nvidia.com/datacenter-driver-580-126-20-download-archive?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=24.04&target_type=deb_local

---

## Storage

| Device                                          | Firmware  | Flash Method |
|-------------------------------------------------|-----------|--------------|
| Samsung 3.84TB Gen5x4 MZTL63T8HFLT-00AW7       | LDDJ3U2Q  | nvme-cli     |
| Samsung 1.92TB Gen4x4 22110 MZ1L21T9HCLS-00A07 | GDC7502Q  |              |

---

## Version Lock Notice

All 12 benchmark modules were written and validated against the exact firmware
versions documented above. Behavior on other firmware versions is undefined
and out of scope for this portfolio project.
