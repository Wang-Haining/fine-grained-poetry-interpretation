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
MAXLEN="${MAXLEN:-124000}"
MAX_SEQS="${MAX_SEQS:-512}"
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

# cuda bind and logging
CUDA_BIND="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
export PYTHONUNBUFFERED=1
STDBUF="$(command -v stdbuf >/dev/null 2>&1 && echo 'stdbuf -oL -eL' || true)"

mkdir -p logs "$HF_HOME" "$OUT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ulimit -n 65535 || true

# make env token also honor file-based token without branching
export HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || echo)}"

echo "[info] poetry-attrs job starting"
echo "[output] OUT_DIR=${OUT_DIR}"
echo "[model] MODEL=${MODEL}"
echo "[split] SPLIT=${SPLIT}"
echo "[parallel] MAX_CONCURRENT=${MAX_CONCURRENT}"
echo "[limit] LIMIT=${LIMIT}"

# ============================================================================
# SERVER SETUP
# ============================================================================

# ---- collision-proof port selection ----
port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:]$1\$" -q
    else
        netstat -ltn 2>/dev/null | grep -E "[: ]$1 " -q
    fi
}

choose_free_port() {
    p="${1:-8020}"
    for _ in $(seq 0 100); do
        port_in_use "$p" || { echo "$p"; return; }
        p=$((p+1))
    done
    echo "${1:-8020}"
}

# ---- GPU detection ----
detect_gpu_count() {
    [[ -n "${GPU_COUNT:-}" && "${GPU_COUNT}" -gt 0 ]] && { echo "${GPU_COUNT}"; return; }
    if command -v nvidia-smi >/dev/null 2>&1; then
        n=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
        [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]] && { echo "$n"; return; }
    fi
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && {
        IFS=, read -ra g <<< "${CUDA_VISIBLE_DEVICES}"
        [[ "${#g[@]}" -gt 0 ]] && { echo "${#g[@]}"; return; }
    }
    echo 1  # default for this job (1x H100)
}

GPUC="$(detect_gpu_count)"
TP="${TP:-1}"
PORT="$(choose_free_port "$PORT")"
API="http://${HOST}:${PORT}"

echo "[server] starting vLLM on ${API} (tensor-parallel=${TP})"

# ---- start vLLM server ----
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

# ---- wait for server ready  ----
echo "[server] waiting for readiness..."
for ((t=1;t<=600;t++)); do
  curl -fsS "${API}/v1/models" >/dev/null 2>&1 && { echo "[server] ready after ${t}s"; break; }
  (( t > 60 )) && sleep 2 || sleep 1
done

# ---- warm up ----
echo "[server] warming up..."
curl -fsS "${API}/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' >/dev/null || true

# ============================================================================
# RUN POETRY ATTR EXTRACTION
# ============================================================================
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
