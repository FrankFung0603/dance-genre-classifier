# 後端：FastAPI + in-proc inference worker（ffmpeg + ONNX Runtime + CAM 歸因）
# 基底含 CUDA 12.4 + cuDNN 9 runtime：有 GPU（--gpus all）走 CUDA EP，
# 無 GPU 環境 ORT 自動落 CPUExecutionProvider（fallback 模式，程式碼同一份）。
# 模型檔不進 image：runtime 以 volume 掛載 ./models → /app/models（換模型免重建）。
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FFMPEG_BIN=/usr/bin/ffmpeg

# ffmpeg（幀流）+ curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl gnupg software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12（對齊本機 dev venv 3.12.3；numpy 2.5.x 鎖 >=3.12，3.11 裝不動）
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 依賴版本對齊 requirements.txt（torch/onnx* 僅訓練用，runtime 不需要；
# onnxruntime-directml 是 Windows 主機用，容器換 onnxruntime-gpu 對 CUDA 12）。
# yt-dlp 不 pin 版本：建置時裝最新（CLAUDE.md 硬規則——YouTube 改版會弄壞舊版）。
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi==0.142.2" \
        "uvicorn[standard]==0.54.0" \
        "onnxruntime-gpu~=1.24.0" \
        "numpy==2.5.3" \
        "sqlalchemy==2.0.44" \
        "psycopg2-binary==2.9.11" \
        redis \
    && pip install --no-cache-dir --upgrade yt-dlp

WORKDIR /app
COPY server/ ./server/

EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
