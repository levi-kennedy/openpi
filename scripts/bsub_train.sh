#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bsub_train.sh <config_name> [--exp-name NAME] [-- <train_args...>]

Submits an LSF job that runs openpi training via bsub.

Options:
  --exp-name NAME         Training exp name (passed through to train.py)
  --help                  Show this help message

Examples:
  scripts/bsub_train.sh pi05_libero --exp-name=exp1
  scripts/bsub_train.sh pi05_libero -- --overwrite --log-interval 50
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG_NAME="$1"
shift

QUEUE="gpu"
GPUS=1
CPUS=8
MEM_GB=16
WALLTIME="24:00"
LOG_DIR="logs/lsf"
JOB_NAME=""
EXP_NAME=""
RUNNER="uv"
XLA_MEM_FRAC="0.9"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-name)
      EXP_NAME="$2"
      EXTRA_ARGS+=("--exp-name" "$2")
      shift 2
      ;;
    --exp-name=*)
      EXP_NAME="${1#--exp-name=}"
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="openpi-${CONFIG_NAME}"
  if [[ -n "${EXP_NAME}" ]]; then
    JOB_NAME="${JOB_NAME}-${EXP_NAME}"
  fi
fi

mkdir -p "${LOG_DIR}"

BSUB_ARGS=(-J "${JOB_NAME}" -n "${CPUS}" -W "${WALLTIME}" -R "span[hosts=1]" -R "rusage[mem=${MEM_GB}]")
if [[ -n "${QUEUE}" ]]; then
  BSUB_ARGS+=(-q "${QUEUE}")
fi
if [[ "${GPUS}" -gt 0 ]]; then
  BSUB_ARGS+=(-gpu "num=${GPUS}:mode=exclusive:mps=yes")
  BSUB_ARGS+=(-R "select[h100 || h200]")
fi

if [[ "${RUNNER}" == "uv" ]]; then
  RUN_CMD=(uv run scripts/train.py)
elif [[ "${RUNNER}" == "python" ]]; then
  RUN_CMD=(python scripts/train.py)
else
  echo "Unsupported --runner '${RUNNER}'. Use 'uv' or 'python'."
  exit 2
fi

BSUB_ARGS+=(-oo "${LOG_DIR}/%J.out" -eo "${LOG_DIR}/%J.err")

WORKDIR="$(pwd)"

escape_cmd() {
  local out=()
  local arg
  for arg in "$@"; do
    out+=("$(printf "%q" "${arg}")")
  done
  printf "%s" "${out[*]}"
}

TRAIN_CMD="$(escape_cmd "${RUN_CMD[@]}" "${CONFIG_NAME}" "${EXTRA_ARGS[@]}")"

bsub "${BSUB_ARGS[@]}" \
  bash -lc "cd '${WORKDIR}' && export XLA_PYTHON_CLIENT_MEM_FRACTION='${XLA_MEM_FRAC}' && ${TRAIN_CMD}"
