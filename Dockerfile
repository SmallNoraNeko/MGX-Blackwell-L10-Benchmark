# ============================================================
# MGX Architecture Blackwell L10 — Automated Benchmark Suite
# Personal Portfolio Project — Night-Kuronos / SmallNoraNeko
#
# Base:   NVIDIA CUDA 12.8 + Ubuntu 24.04
# Target: MGX Blackwell L10 Standalone (customized deployment)
# Source: gb300_benchmark_v1.0.1.zip
#
# NOTE: Proprietary binary tools are NOT included in this image.
#       Mount tools/ as a volume at runtime.
#       See tools/README.txt for the full binary inventory.
# ============================================================

FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04

LABEL maintainer="Night-Kuronos <https://github.com/SmallNoraNeko>"
LABEL description="MGX Blackwell L10 Benchmark Suite — Portfolio Project"
LABEL version="1.0.1"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV BENCHMARK_VERSION=1.0.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    iproute2 iputils-ping net-tools iperf3 \
    infiniband-diags rdma-core libibverbs-dev ibverbs-utils \
    pciutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

WORKDIR /workspace
COPY gb300_launcher.py .
COPY lib/ ./lib/

RUN mkdir -p /workspace/tools /workspace/report

ENTRYPOINT ["python3", "gb300_launcher.py"]
