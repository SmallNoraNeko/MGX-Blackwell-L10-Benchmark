# GB300 Benchmark Launcher

> ⚠️ **此為客制化專案，並非通用工具。**
>
> 專為 **GB300 L10 MGX / ARM64 / Ubuntu 24.04** 平台建置。
> RDMA 拓撲支援透過 `--discover` 或 `topology.json` 自動探索，
> 可在不修改程式碼的情況下適應不同平台。
> 其他平台相關設定（GPU 數量、iPerf client IP 等）仍需手動調整。

---

## 展示影片

https://github.com/user-attachments/assets/REPLACE_WITH_YOUR_VIDEO_ASSET_ID

---

## 平台規格

| 項目 | 規格 |
|------|------|
| 硬體 | NVIDIA GB300 L10 MGX |
| 架構 | ARM64 |
| 作業系統 | Ubuntu 24.04.2 |
| Driver | 580.82+ |
| CUDA | 13.0 |
| Python | 3.10+ |

---

## 快速開始

```bash
# 1. 將 benchmark binary 放入 tools/（詳見 tools/README.txt）
# 2. 以 root 身份執行
sudo python3 gb300_launcher.py
```

---

## 使用方式

```
sudo python3 gb300_launcher.py [OPTIONS]
```

### 參數說明

| 參數 | 說明 |
|------|------|
| `--list` | 列出所有測試 ID 與可設定參數後離開 |
| `--select <IDs>` | 以逗號分隔的測試 ID 直接執行，不進選單（例如 `1,5,6,12`） |
| `--yes` | 全部使用預設參數，跳過所有互動提示（搭配 `--select` 使用） |
| `--params <OVERRIDES>` | 覆蓋特定測試參數（搭配 `--select --yes` 使用） |
| `--dry-run` | 僅顯示指令，不實際執行 |
| `--no-prereq` | 跳過 binary / 工具前置條件檢查 |
| `--log-dir <PATH>` | 覆蓋 `report/` 輸出目錄 |

---

## 執行模式

### 模式 1 — 互動式選單（預設）

```bash
sudo python3 gb300_launcher.py
```

進入 curses 互動選單，使用方向鍵和空白鍵選擇測試項目。

**選單操作說明：**

| 按鍵 | 功能 |
|------|------|
| `↑` / `↓` | 移動游標 |
| `Space` | 切換勾選 |
| `a` | 全選所有可用測試 |
| `n` | 取消所有選取 |
| `d` | 探索 RDMA 拓撲（寫入 `topology.json`） |
| `Enter` | 確認選取並繼續 |
| `q` / `Ctrl-C` | 隨時離開 |

---

### 模式 2 — 非互動式，使用預設參數

```bash
sudo python3 gb300_launcher.py --select 1,2,3 --yes
```

以所有預設參數執行測試 1、2、3，無選單、無提示。
適合排程自動化或 CI/CD 流程使用。

---

### 模式 3 — 非互動式，覆蓋指定參數

```bash
sudo python3 gb300_launcher.py --select 5,6,13 --yes \
    --params "5.iters=5,6.duration=60,6.serial=true,13.iters=5"
```

以自訂參數執行選定測試。`--params` 覆蓋指定的預設值，
其他未指定的參數維持預設值不變。

---

### 模式 4 — Dry-run 模擬執行

```bash
sudo python3 gb300_launcher.py --select 6,7,13 --yes --dry-run
```

顯示所有將執行的指令，但不實際執行。
適合在正式跑測試前確認 `--params` 設定是否正確。

---

### 模式 5 — 列出所有測試

```bash
sudo python3 gb300_launcher.py --list
```

印出完整測試目錄，含 ID、可用狀態、所有可設定參數及其預設值。
使用此模式查詢 `--select` 的 ID 和 `--params` 的 key 名稱。

---

## `--params` 參數說明

格式：`<測試ID>.<參數名>=<值>`，多個設定以逗號分隔。

**自動型別轉換：**

| 輸入字串 | Python 型別 |
|---|---|
| `true` / `yes` / `1` | `bool` → `True` |
| `false` / `no` / `0` | `bool` → `False` |
| `5`、`30`、`120` | `int` |
| `0.1`、`3.5` | `float` |
| 其他字串 | `str` |

**所有可設定參數一覽：**

| 測試名稱 | ID | 參數名 | 型別 | 預設值 | 說明 |
|----------|----|--------|------|--------|------|
| NCCL Loopback | 5 | `iters` | int | `20` | 每個訊息大小的迭代次數 |
| NCCL Loopback | 5 | `msg_range` | str | `"8g"` | `"8g"` = 僅 8G，`"full"` = 2B~8G 全掃 |
| RDMA IPv4 / IPv6 | 6 / 7 | `duration` | int | `30` | 每個訊息大小的測試秒數 |
| RDMA IPv4 / IPv6 | 6 / 7 | `test_type` | str | `"all"` | `all` / `read` / `send` / `write` |
| RDMA IPv4 / IPv6 | 6 / 7 | `quick` | bool | `False` | `True` = 只跑 64KB / 256KB（每種測試約 10 分鐘） |
| RDMA IPv4 / IPv6 | 6 / 7 | `platform` | str | `""` | `rdma_platforms.json` 的平台名稱（空白 = 預設 GB300_L10） |
| 1G NIC iPerf | 8 | `client_ip` | str | `"10.20.2.130"` | iPerf client IP 位址 |
| 1G NIC iPerf | 8 | `mode` | str | `"bidirectional"` | `"bidirectional"` 或 `"unidirectional"` |
| 1G NIC iPerf | 8 | `duration` | int | `3600` | 測試持續秒數 |
| 功耗監控 | 10 | `interval` | float | `0.1` | 採樣間隔（秒） |
| 功耗監控 | 10 | `csv_log` | bool | `False` | 是否儲存 CSV 功耗 log |
| NeMo DL Validation | 11 | `image` | str | `"nvcr.io/nvidia/pytorch:24.07-py3"` | 容器 image |
| NeMo DL Validation | 11 | `gpus` | int | `4` | 開放給容器的 GPU 數量 |
| NeMo DL Validation | 11 | `cmd` | str | `""` | 訓練指令（空白 = 讀取 GB_training_scripts.txt） |
| NVBandwidth | 12 | `iters` | int | `3` | 每個測試 case 的迭代次數 |
| MUFU.EX2 | 13 | `iters` | int | `3` | Benchmark 執行次數（穩定性驗證） |

---

## `--params` 使用範例

```bash
# NCCL：全掃訊息大小，10 次迭代
sudo python3 gb300_launcher.py --select 5 --yes \
    --params "5.iters=10,5.msg_range=full"

# RDMA IPv4：快測 write
sudo python3 gb300_launcher.py --select 6 --yes \
    --params "6.quick=true,6.test_type=write"

# RDMA IPv4 / IPv6：TTM 平台
sudo python3 gb300_launcher.py --select 6,7 --yes \
    --params "6.platform=TTM,7.platform=TTM"

# MUFU.EX2：5 次執行
sudo python3 gb300_launcher.py --select 13 --yes \
    --params "13.iters=5"

# GPU 全套測試 + MUFU
sudo python3 gb300_launcher.py --select 1,2,3,4,12,13 --yes \
    --params "12.iters=5,13.iters=5"

# 功耗監控：5 秒採樣間隔並儲存 CSV
sudo python3 gb300_launcher.py --select 10 --yes \
    --params "10.interval=5.0,10.csv_log=true"
```

---

## 測試項目一覽

| ID | 分類 | 測試名稱 | 可設定參數 |
|----|------|----------|------------|
| 1 | GPU 運算與記憶體 | GPU Stream | — |
| 2 | GPU 運算與記憶體 | Peak TOPS | — |
| 3 | GPU 運算與記憶體 | GEMM Bench | — |
| 4 | GPU 運算與記憶體 | FP4 GEMM-MemRead | — |
| 5 | GPU 互聯 | NCCL Loopback | `iters`、`msg_range` |
| 6 | 網路 / RDMA | RDMA Loopback IPv4 | `duration`、`test_type`、`quick`、`platform` |
| 7 | 網路 / RDMA | RDMA Loopback IPv6 | `duration`、`test_type`、`quick`、`platform` |
| 8 | 網路 / RDMA | 1G NIC iPerf | `client_ip`、`mode`、`duration` |
| 9 | 硬體健康 | NIC PCIe 健康檢查 | — |
| 10 | 硬體健康 | 功耗監控 | `interval`、`csv_log` |
| 11 | DL 訓練驗證 | NeMo DL Validation | `image`、`gpus`、`cmd` |
| 12 | GPU 運算與記憶體 | NVBandwidth Loopback | `iters` |
| 13 | GPU 運算與記憶體 | MUFU.EX2 Benchmark | `iters` |

---

## MUFU.EX2 Benchmark（測試 13）

### 功能說明

量測 GB300 GPU 的 **指數運算吞吐量（Exponent Op Throughput）**，
支援 FP32、FP16、BF16 三種精度，與官方 Peak 值（10068 Gop/s）比對，
通過門檻為實測值 / Peak ≥ 95%。

### 前置部署

```
tools/
├── MUFU.EX2.zip                    ← 直接放這裡（腳本自動解壓）
└── MUFU.EX2/                       ← 或預先解壓放這裡
    └── GB300 & Vera Rubin/
        └── exp2-bench

lib/
└── run_mufu_benchmark.py           ← 腳本放這裡
```

### Log 輸出結構

```
report/run_<TIMESTAMP>/
└── 13_mufu_ex2/
    ├── mufu_<TS>.log          ← exp2-bench 原始輸出（每次執行）
    ├── report_<TS>.log        ← 驗證報告（平均值、StdDev、Pass/Fail）
    ├── dmesg_<TS>.log
    ├── system_<TS>.log
    ├── sdr_<TS>.log
    ├── sel_<TS>.log
    └── nvidia_smi_<TS>.log
```

### 報告格式說明

```
================================================================
MUFU.EX2 Benchmark Result Summary
Runs completed : 3 / 3
Timestamp      : 2026-08-25 12:12:43
================================================================
Datatype    Peak(Gop/s)   Avg(Gop/s)   StdDev    Ratio   Result
----------------------------------------------------------------
FP32         10068.000     10060.133    0.246   99.92%     PASS
FP16         10068.000     10049.416    0.400   99.82%     PASS
BF16         10068.000     10057.852    0.236   99.90%     PASS
----------------------------------------------------------------
Pass threshold : Measured/Peak >= 95%
Overall result : ALL PASS
================================================================
```

### 單獨執行

```bash
# 使用預設 3 次執行
sudo python3 lib/run_mufu_benchmark.py

# 指定 5 次執行
sudo python3 lib/run_mufu_benchmark.py --iters 5

# 指定輸出目錄
sudo python3 lib/run_mufu_benchmark.py --iters 5 --log-dir /tmp/mufu_out
```

---

## RDMA 平台設定（測試 6 / 7）

RDMA 測試（`lib/rdma_test_ipv4_v4.py`、`lib/rdma_test_ipv6_v5.py`）從專案根目錄的
`rdma_platforms.json` 讀取平台設定；檔案不存在時使用內建、已在 system10 驗證的 `GB300_L10`。

| 欄位 | 必填 | 沒填時自動偵測 |
|------|------|----------------|
| `pairs` | 是 | —（依序命名 pair1..N） |
| `cards` | 否 | sysfs PCI device id → board_id → `mst status -v` |
| `link_gbps` | 否 | `/sys/class/infiniband/<nic>/ports/1/rate` |
| `gpu_map` | 否 | NIC 所在 NUMA 的 GPU，server / client 盡量用不同顆 |
| `waves` | 否 | 同一張實體卡的 port 不放在同一批 |
| `line_rate_pct` | 否 | 0.90（門檻 = link rate × 90%） |

跨卡干擾（例如 L10 上 BF3 與 NUMA1 的 CX8 同跑會掉速）無法自動偵測，
需實測後把 `waves` 寫進設定檔。啟動畫面會標示哪些欄位是自動偵測的。

```bash
sudo python3 lib/rdma_test_ipv4_v4.py --list-platforms
sudo python3 lib/rdma_test_ipv4_v4.py --platform TTM --dry-run
```

| 平台 | pairs | 狀態 |
|------|-------|------|
| GB300_L10 | `[0,6],[1,7],[2,4],[3,5],[8,10],[9,11]` | 2026-10-16 驗證 |
| TTM | `[0,2],[1,3],[6,8],[7,9]` | 只有 pairs，其餘自動偵測，尚未驗證 |

> `topology.json` 與選單的 `d`（自動探索）只給舊版腳本使用，新腳本不讀取。
> 探索是用 ping 判斷接線，同一台機器上的 NIC 互 ping 一定會通，在 L10 上會判出錯誤配對。

---

## 目錄結構

```
gb300_benchmark/
├── gb300_launcher.py           ← 唯一執行入口
├── rdma_platforms.json         ← RDMA 平台設定（測試 6 / 7）
├── topology.json               ← 舊版 RDMA 配對（選單 d 探索用）
├── README.md                   ← 英文版說明
├── README_zh-TW.md             ← 繁體中文版說明
├── GB300_Benchmark_Setup_Guide.md
│
├── lib/                        ← 所有測試腳本
│   ├── _launcher_config.py     ← 測試定義與預設參數
│   ├── _launcher_runner.py     ← 執行引擎與 log 路徑分配
│   ├── _rdma_common.py         ← 拓撲探索（選單 d）
│   ├── _rdma_lib.py            ← RDMA 測試共用函式庫
│   ├── rdma_test_ipv4_v4.py    ← RDMA IPv4（測試 6）
│   ├── rdma_test_ipv6_v5.py    ← RDMA IPv6（測試 7）
│   ├── disable_acs.sh          ← NVBandwidth 偵測到 ACS 時自動呼叫
│   ├── run_mufu_benchmark.py   ← MUFU.EX2 測試腳本
│   └── ...                     ← 其他各測試腳本
│
├── tools/                      ← 測試工具 binary（不含在 repo 中）
│   ├── MUFU.EX2.zip            ← MUFU.EX2 benchmark binary 封裝包
│   ├── MUFU.EX2/               ← 解壓後的 binary（腳本自動解壓）
│   │   └── GB300 & Vera Rubin/
│   │       └── exp2-bench
│   ├── GB_DL_scripts_v9/       ← 從 GB_DL_scripts_v9.nv7z 解密後
│   ├── Speech-main/            ← 從 Speech-main.zip 解壓後
│   └── ...
│
└── report/                     ← 測試 log（每次執行自動建立）
    └── run_<TIMESTAMP>/
        ├── 00_summary.log
        ├── 01_GPU_stream/
        ├── ...
        ├── 12_nvbandwidth_loopback/
        └── 13_mufu_ex2/        ← 只有選到的測試才建立子目錄
            ├── mufu_<TS>.log
            ├── report_<TS>.log
            ├── dmesg_<TS>.log
            ├── system_<TS>.log
            ├── sdr_<TS>.log
            ├── sel_<TS>.log
            └── nvidia_smi_<TS>.log
```

---

## NeMo DL Validation（測試 11）

**模式 A — 真實 LLM 訓練**（需 Speech-main v2.3.1）：

```bash
cd tools/Speech-main/
git fetch --tags && git checkout v2.3.1
cd ../GB_DL_scripts_v9/
ln -snf ../Speech-main/scripts ./scripts
```

**模式 B — Smoke Test**（完全離線，驗證容器 + GPU + log 收集）：

```bash
mv tools/GB_DL_scripts_v9/GB_training_scripts.txt \
   tools/GB_DL_scripts_v9/GB_training_scripts.txt.bak
```

Launcher 自動判斷執行哪個模式。若容器內找不到訓練模組
（`scripts.performance.llm`），自動 fallback 至 smoke test。

---

## 離線 / 隔離網路部署

```bash
# 在有網路的機台上，匯出 container image
podman save nvcr.io/nvidia/pytorch:24.07-py3 \
    -o tools/pytorch_24.07-py3_arm64.tar

# 打包（不壓縮，大型 binary 壓縮效益低，節省時間）
tar -cf gb300_benchmark_offline.tar gb300_benchmark/

# 傳送到目標機台後解壓並執行
tar -xf gb300_benchmark_offline.tar
cd gb300_benchmark/
sudo python3 gb300_launcher.py
# Container image 會從 tools/*.tar 自動載入，無需手動 podman load
```

---

## 系統需求

- Python 3.10+ / Ubuntu 24.04 / ARM64 / root 權限
- CUDA 13 / Driver 580.82+

| 工具 | 使用測試 | 安裝方式 |
|------|----------|----------|
| `ipmitool` | 所有 GPU benchmark（SDR / SEL 採集）| `apt install ipmitool` |
| `iperf`（v2） | 1G NIC iPerf（測試 8） | `apt install iperf` |
| `mst`、`mlxconfig` | NIC PCIe 健康 + RDMA 拓撲探索 | 安裝 Mellanox MFT |
| `lspci` | NIC PCIe 健康（測試 9） | `apt install pciutils` |
| `ib_read_bw` 等 | RDMA 測試（6、7） | 從 `tools/perftest-master.zip` 編譯 |
| `podman` | NeMo DL Validation（測試 11） | `apt install podman` |
| `nvidia-ctk` | NeMo DL Validation（測試 11） | 安裝 NVIDIA Container Toolkit |

詳細的編譯與安裝步驟請參考 `GB300_Benchmark_Setup_Guide.md`。

---

## 授權

僅限內部使用，未經授權不得對外發布。
