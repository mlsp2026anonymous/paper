# Mirrors the mamba311 conda env: Python 3.11, PyTorch 2.4.1+cu121.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl bzip2 ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniconda with Python 3.11
ENV CONDA_DIR=/opt/conda
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
RUN mkdir -p /workspace/tmp
ENV TMPDIR=/workspace/tmp

# PyTorch with CUDA 12.1 (must come before mamba-ssm)
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# mamba-ssm requires torch to already be installed at build time
#RUN pip install --no-cache-dir --no-build-isolation mamba-ssm==2.2.5

# momentfm pins huggingface-hub==0.24.0 but the actual env runs 0.36.2;
# install it without deps to match the original conda env state.
RUN pip install --no-cache-dir --no-deps momentfm==0.1.4

# Remaining packages
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project sources (adjust or remove if you mount a volume at runtime)
COPY . /workspace/

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc

CMD ["/bin/bash"]
