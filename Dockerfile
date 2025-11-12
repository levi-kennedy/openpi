FROM ghcr.io/astral-sh/uv:0.5.7 AS uv

FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

COPY --from=uv /uv /uvx /bin/

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY packages ./packages

RUN uv venv --python 3.12 $UV_PROJECT_ENVIRONMENT && \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

COPY . .

RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install --python /.venv/bin/python -e .

RUN /.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
import transformers

target_dir = Path(transformers.__file__).parent
source_dir = Path("src/openpi/models_pytorch/transformers_replace")
for file_path in source_dir.rglob("*"):
    if not file_path.is_file():
        continue
    destination = target_dir / file_path.relative_to(source_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, destination)
PY

ENV PATH=/.venv/bin:$PATH
ENV PYTHONPATH=/workspace/src

CMD ["bash"]
