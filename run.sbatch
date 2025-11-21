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

# ---- environment setup (tempest) ----
module purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0 CUDA/12.3.0
source .venv/bin/activate

# ---- model/server knobs ----
MODEL="${MODEL:-openai/gpt-oss-120b}"
# poems are short; no need for a giant context
MAXLEN="${MAXLEN:-16384}"
MAX_SEQS="${MAX_SEQS:-64}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# ---- app settings ----
READ_TIMEOUT="${READ_TIMEOUT:-1800}"

# ---- paths and container ----
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG_FILE="${IMG_FILE:-$HOME/containers/vllm-gptoss.sif}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8020}"

# ---- poetry attrs settings ----
OUT_DIR="${OUT_DIR:-poem_attrs}"
SPLIT="${SPLIT:-train}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
LIMIT="${LIMIT:-0}"  # 0 means no limit

CUDA_BIND="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
export PYTHONUNBUFFERED=1
STDBUF="$(command -v stdbuf >/dev/null 2>&1 && echo 'stdbuf -oL -eL' || true)"

mkdir -p logs "$HF_HOME" "$OUT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ulimit -n 65535 || true

# HF token via env or file
export HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || echo)}"

echo "[info] poetry-attrs job starting"
echo "[output] OUT_DIR=${OUT_DIR}"
echo "[model] MODEL=${MODEL}"
echo "[split] SPLIT=${SPLIT}"
echo "[parallel] MAX_CONCURRENT=${MAX_CONCURRENT}"
echo "[limit] LIMIT=${LIMIT}"

API="http://${HOST}:${PORT}"
TP=1

echo "[server] starting vLLM on ${API} (tensor-parallel=${TP})"

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

CMD="$STDBUF python add_poetry_attrs.py \
  --base_url \"${API}\" \
  --model \"${MODEL}\" \
  --read_timeout \"${READ_TIMEOUT}\" \
  --out_dir \"${OUT_DIR}\" \
  --split \"${SPLIT}\" \
  --max_concurrent \"${MAX_CONCURRENT}\" \
  --limit \"${LIMIT}\""

echo "[cmd] $CMD"
eval "$CMD"

END_TS=$(date +%s)
DUR=$((END_TS - START_TS))

echo ""
echo "=== POETRY ATTRS COMPLETE ==="
echo "[duration] ${DUR}s"
echo "[model] ${MODEL}"
echo "[output] ${OUT_DIR}/"
echo "[server] ${API}"
echo "[parallel] ${MAX_CONCURRENT}"
