GB300 Benchmark Launcher — tools/ directory
===========================================

Place the benchmark binaries in this directory as shown below.
Placeholder files (zero-byte) are included to preserve the directory
structure.  Replace them with the actual binaries before running.

Required layout
---------------

tools/
├── stream_vectorized_float_benchmark   <- from stream_test_v4.nv7z
├── peakTops                            <- from peakTOPS_v9.nv7z
├── run_peaktops.sh                     <- from peakTOPS_v9.nv7z
├── cublasMatmulBench                   <- from cublasMatmulBench_v7.nv7z
│
├── gemm-memread/                       <- extract GEMM_v6.nv7z (GB300 path)
│   │   Source: GEMM_v6/GB300/gemm-memread/gemm-memread/
│   ├── build/
│   │   ├── generic_gemm_benchmark
│   │   └── lib/
│   │       ├── libcutlass.so
│   │       ├── libcutlass_gemm_sm103_void_gemm_e2m1.so
│   │       └── ...other .so files...
│   ├── configs/
│   │   ├── commandlines.json
│   │   └── ...yaml files...
│   └── scripts/
│       ├── run_bench.py
│       └── duty_cycle_controller_v2.5.py
│
└── nccl-build/                         <- build output from nccl-tests-master
    │   Build: cd nccl-tests-master && make MPI=0 CUDA_HOME=... NCCL_HOME=...
    │   Then copy build/ contents here.
    ├── all_reduce_perf
    ├── all_gather_perf
    └── alltoall_perf

Source archives (from DA-12276-001_v28 PDF attachments)
---------------------------------------------------------
  stream_test_v4.nv7z        -> rename to .7z, extract
  peakTOPS_v9.nv7z           -> rename to .7z, extract
  cublasMatmulBench_v7.nv7z  -> rename to .7z, extract
  GEMM_v6.nv7z               -> rename to .7z, extract (use GB300 path)
  NCCL_v3.nv7z               -> source only; build with make first

System tools (install via apt, not placed here)
-------------------------------------------------
  iperf          : apt install iperf
  ib_read_bw etc : apt install perftest   (or build from perftest-master.zip)
  mst / mlxconfig: install Mellanox MFT package
  ipmitool       : apt install ipmitool
