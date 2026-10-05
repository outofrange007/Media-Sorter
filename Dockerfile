# Devel base required so dlib can compile against CUDA/cuDNN (FACE_MODEL=cnn)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3 python3-pip ffmpeg libgl1 libglib2.0-0 \
    cmake build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heavy layers first: cached, only reruns when dependencies change.
# torch pinned to the last cu118 build with Pascal (sm_61) kernels for the Quadro P1000.
# numpy pinned to 1.x: dlib 19.24.2 breaks with numpy 2 ("Unsupported image type").
RUN pip3 install --no-cache-dir \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir \
    numpy==1.26.4 dlib==19.24.2 \
    streamlit==1.64.0 opencv-python-headless==4.9.0.80 psutil==5.9.8 \
    pillow==10.2.0 ddgs easyocr==1.7.1 face_recognition==1.3.0 \
    paddleocr==2.8.1 paddlepaddle-gpu==2.6.1

# Application code AFTER pip install: code edits reuse the cached pip layers
COPY videoocr.py web_app.py ./

EXPOSE 8501
CMD ["streamlit", "run", "web_app.py", "--server.address=0.0.0.0"]
