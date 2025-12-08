#!/bin/bash -l
#SBATCH --job-name=poetry-attrs
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

set -euo pipefail

# usage:
#   sbatch poetry_attrs.sbatch                       # full run (default)
#   VAR=pilot sbatch poetry_attrs.sbatch            # 1000-sample pilot
#   PILOT_1000=1 sbatch poetry_attrs.sbatch         # alternative toggle
#   LIMIT=500 OUT_DIR=poem_attrs_500 sbatch ...     # explicit override

# ---- environment setup (tempest) ----
module purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0 CUDA/12.3.0
source .venv/bin/activate

# ---- model/server knobs ----
MODEL="${MODEL:-openai/gpt-oss-120b}"
MAXLEN="${MAXLEN:-16384}"         # poems are short; no need for giant context
MAX_SEQS="${MAX_SEQS:-16}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# ---- app settings ----
READ_TIMEOUT="${READ_TIMEOUT:-600}"

# ---- paths and container ----
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG_FILE="${IMG_FILE:-$HOME/containers/vllm-gptoss.sif}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8020}"

# ---- httpx tuning for guarded backend ----
export HTTP_POOL="${HTTP_POOL:-256}"
export HTTP_KEEPALIVE="${HTTP_KEEPALIVE:-120}"
export HTTP_POOL_TIMEOUT="${HTTP_POOL_TIMEOUT:-600}"

# ---- pilot/full mode selection (default full) ----
MODE="full"
if [[ "${VAR:-}" == "pilot" || "${PILOT_1000:-0}" == "1" ]]; then
  MODE="pilot"
fi

# ---- poetry attrs settings (defaults depend on mode) ----
# script entrypoint can be swapped via env
SCRIPT="${SCRIPT:-add_poetry_attrs.py}"

if [[ "$MODE" == "pilot" ]]; then
  OUT_DIR="${OUT_DIR:-poem_attrs_1000}"
  LIMIT="${LIMIT:-1000}"
else
  OUT_DIR="${OUT_DIR:-poem_attrs}"
  LIMIT="${LIMIT:-0}"              # 0 = no cap (full dataset)
fi

SPLITS="${SPLITS:-all}"            # can be "all" or "train,validation,test"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

CUDA_BIND="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
export PYTHONUNBUFFERED=1
STDBUF="$(command -v stdbuf >/dev/null 2>&1 && echo 'stdbuf -oL -eL' || true)"

mkdir -p logs "$HF_HOME" "$OUT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ulimit -n 65535 || true

# hf token via env or file
export HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || echo)}"

echo "[info] poetry-attrs job starting"
echo "[mode] ${MODE}"
echo "[script] ${SCRIPT}"
echo "[output] OUT_DIR=${OUT_DIR}"
echo "[model] MODEL=${MODEL}"
echo "[splits] SPLITS=${SPLITS}"
echo "[parallel] MAX_CONCURRENT=${MAX_CONCURRENT}"
echo "[limit] LIMIT=${LIMIT}"

API="http://${HOST}:${PORT}"
TP=1

echo "[server] starting vllm on ${API} (tensor-parallel=${TP})"

command -v apptainer >/dev/null || { echo "[fatal] apptainer missing"; exit 90; }
[[ -f "$IMG_FILE" ]] || { echo "[fatal] image missing: $IMG_FILE"; exit 91; }

apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  $( [[ -d "$CUDA_BIND" ]] && echo --bind "$CUDA_BIND":"$CUDA_BIND" ) \
  --env PYTHONNOUSERSITE=1,PIP_USER=no,TRANSFORMERS_NO_ADVISORY_WARNINGS=1,HF_TOKEN="${HF_TOKEN}" \
  "$IMG_FILE" bash --noprofile --norc -c "
set -euo pipefail
python3 -m pip install --no-cache-dir --force-reinstall 'tokenizers>=0.21,<0.22' >/dev/null 2>&1 || true
python3 -m vllm.entrypoints.openai.api_server \
  --model '${MODEL}' --served-model-name '${MODEL}' \
  --download-dir /root/.cache/huggingface \
  --tensor-parallel-size ${TP} --gpu-memory-utilization ${GPU_UTIL} \
  --max-model-len ${MAXLEN} --max-num-seqs ${MAX_SEQS} \
  --enable-prefix-caching --enforce-eager \
  --host 0.0.0.0 --port ${PORT}
" & SERVER_PID=$!

trap 'echo "[info] stopping..."; kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "[server] waiting for readiness..."
for ((t=1;t<=300;t++)); do
  if curl -fsS "${API}/v1/models" >/dev/null 2>&1; then
    echo "[server] ready after ${t}s"
    break
  fi
  sleep 1
done

echo "[server] warming up..."
curl -fsS "${API}/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' >/dev/null || true

echo "[poetry] starting annotation..."
START_TS="$(date +%s)"

CMD="$STDBUF python ${SCRIPT} \
  --base_url \"${API}\" \
  --model \"${MODEL}\" \
  --read_timeout \"${READ_TIMEOUT}\" \
  --out_dir \"${OUT_DIR}\" \
  --splits \"${SPLITS}\" \
  --max_concurrent \"${MAX_CONCURRENT}\" \
  --limit \"${LIMIT}\""

echo "[cmd] $CMD"
eval "$CMD"

END_TS=$(date +%s)
DUR=$((END_TS - START_TS))

echo ""
echo "=== poetry attrs complete ==="
echo "[duration] ${DUR}s"
echo "[model] ${MODEL}"
echo "[output] ${OUT_DIR}/"
echo "[server] ${API}"
echo "[parallel] ${MAX_CONCURRENT}"
