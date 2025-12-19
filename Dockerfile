# Sarah Voice Assistant - Docker Image
# Auto-detects GPU and CUDA version at build time

FROM ubuntu:22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libsdl2-mixer-2.0-0 \
    libsdl2-2.0-0 \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Detect GPU and CUDA version, install appropriate PyTorch
RUN if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then \
    echo "GPU detected!" && \
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1) && \
    echo "CUDA Driver: $CUDA_VERSION" && \
    echo "GPU" > /app/.device_type && \
    if [ "$CUDA_VERSION" -ge 525 ]; then \
    echo "Installing PyTorch for CUDA 12.1..." && \
    pip install --no-cache-dir torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$CUDA_VERSION" -ge 450 ]; then \
    echo "Installing PyTorch for CUDA 11.8..." && \
    pip install --no-cache-dir torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118; \
    else \
    echo "Old CUDA, installing CPU PyTorch..." && \
    echo "CPU" > /app/.device_type && \
    pip install --no-cache-dir torch torchaudio; \
    fi; \
    else \
    echo "No GPU detected, installing CPU PyTorch..." && \
    echo "CPU" > /app/.device_type && \
    pip install --no-cache-dir torch torchaudio; \
    fi

# Install remaining requirements (without torch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_ENV=production

# Run the web app
CMD ["python", "web_app.py"]
