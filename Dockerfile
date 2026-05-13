ARG ML_CPU_BASE_IMAGE=local/ml-python311-cpu:bookworm
FROM ${ML_CPU_BASE_IMAGE}

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip \
    && pip install ".[cpu,dev]"

CMD ["bash"]
