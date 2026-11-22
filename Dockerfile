# Runtime image with the PyTorch / CUDA / cuDNN versions used for the published results.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Pin torch/torchvision so that installing the package never replaces the base image's build.
RUN pip install . "torch==2.5.1" "torchvision==0.20.1"

COPY scripts ./scripts
COPY .env.example ./

# Mount the data bundle and an output directory, e.g.
#   docker run --gpus all -v /path/to/embryo-tempoformer_release_v1:/release:ro -e RELEASE_DIR=/release \
#     -v "$PWD/runs:/app/runs" embryo-tempoformer etf reproduce
CMD ["bash"]
