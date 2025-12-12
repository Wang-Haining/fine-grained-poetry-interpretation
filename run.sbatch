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
#   sbatch poetry_attrs.sbatch                          # full run
#   VAR=rest SPLITS=train sbatch poetry_attrs.sbatch    # only missing rows for train
#   LIMIT=500 OUT_DIR=poem_attrs_500 sbatch ...         # explicit override

# ---- environment setup (tempest) ----
module --force purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0 CUDA/12.3.0
source .venv/bin/activate

# ---- model/server knobs ----
MODEL="${MODEL:-openai/gpt-oss-120b}"
MAXLEN="${MAXLEN:-16384}"
MAX_SEQS="${MAX_SEQS:-16}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# ---- app settings ----
READ_TIMEOUT="${READ_TIMEOUT:-1800}"

# ---- paths and container ----
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG_FILE="${IMG_FILE:-$HOME/containers/vllm-gptoss.sif}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8020}"

# ---- httpx tuning for guarded backend ----
export HTTP_POOL="${HTTP_POOL:-256}"
export HTTP_KEEPALIVE="${HTTP_KEEPALIVE:-120}"
export HTTP_POOL_TIMEOUT="${HTTP_POOL_TIMEOUT:-600}"

# ---- mode selection ----
# "full" processes everything (LIMIT=0); "rest" finishes only missing rows
MODE="full"
if [[ "${VAR:-}" == "rest" || "${MISSING_ONLY:-0}" == "1" ]]; then
  MODE="rest"
fi

# ---- script + io ----
SCRIPT="${SCRIPT:-add_poetry_attrs.py}"
DATASET_ID="${DATASET_ID:-haining/poem_interpretation_corpus}"

if [[ "$MODE" == "full" ]]; then
  OUT_DIR="${OUT_DIR:-poem_attrs}"
  LIMIT="${LIMIT:-0}"
else
  OUT_DIR="${OUT_DIR:-poem_attrs}"  # continue writing to the same dir
  LIMIT="${LIMIT:-0}"               # we will pass --include_file instead of LIMIT
fi

SPLITS="${SPLITS:-all}"             # "all" or "train,validation,test"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

CUDA_BIND="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
export PYTHONUNBUFFERED=1
STDBUF="$(command -v stdbuf >/dev/null 2>&1 && echo 'stdbuf -oL -eL' || true)"

mkdir -p logs "$HF_HOME" "$OUT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ulimit -n 65535 || true

# hf token via env or file
export HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || echo)}"

echo "[info] poetry-attrs starting"
echo "[mode] ${MODE}"
echo "[dataset] ${DATASET_ID}"
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

cleanup() {
  echo "[info] stopping..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[server] waiting for readiness..."
ready=0
for ((t=1;t<=300;t++)); do
  if curl -fsS "${API}/v1/models" >/dev/null 2>&1; then
    echo "[server] ready after ${t}s"
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  echo "[fatal] vllm server did not become ready"; exit 92
fi

echo "[server] warming up..."
curl -fsS "${API}/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' >/dev/null || true

START_TS="$(date +%s)"

# ---- run annotations ----
if [[ "$MODE" == "full" ]]; then
  echo "[run] full dataset"
  CMD="$STDBUF python ${SCRIPT} \
    --dataset_id \"${DATASET_ID}\" \
    --base_url \"${API}\" \
    --model \"${MODEL}\" \
    --read_timeout \"${READ_TIMEOUT}\" \
    --out_dir \"${OUT_DIR}\" \
    --splits \"${SPLITS}\" \
    --max_concurrent \"${MAX_CONCURRENT}\" \
    --limit \"${LIMIT}\""
  echo "[cmd] $CMD"
  eval "$CMD"
else
  echo "[run] missing rows only"
  # expand splits
  if [[ "${SPLITS}" == "all" ]]; then
    SPLIT_LIST="train validation test"
  else
    IFS=',' read -r -a arr <<< "${SPLITS}"
    SPLIT_LIST="${arr[*]}"
  fi

  for split in ${SPLIT_LIST}; do
    echo "[rest] preparing todo list for ${split}"

    # compute missing list for this split into OUT_DIR/<split>/todo_missing.txt
    python - "$DATASET_ID" "$OUT_DIR" "$split" <<'PY'
import sys, json, math, glob
from pathlib import Path
from datasets import load_dataset

ds_id, out_dir, split = sys.argv[1], sys.argv[2], sys.argv[3]
rows = Path(out_dir) / split / "rows"
rows.mkdir(parents=True, exist_ok=True)

def empty(v):
    return v is None or (isinstance(v,float) and math.isnan(v)) or (isinstance(v,str) and not v.strip()) or (isinstance(v,(list,dict)) and not v)

def filled(d):
    return any(not empty(d.get(k)) for k in ("emotions","primary_emotion","sentiment","themes","themes_50"))

n = len(load_dataset(ds_id, split=split))
seen = set()
for p in rows.glob("*.json"):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if filled(d):
            i = int(d.get("row_index", p.stem))
            if 0 <= i < n: seen.add(i)
    except Exception:
        pass

missing = [str(i) for i in range(n) if i not in seen]
todo = rows.parent / "todo_missing.txt"
todo.write_text("\n".join(missing))
print(f"[todo_missing] split={split} total={n} filled={len(seen)} missing={len(missing)} -> {todo}")
PY

    TODO_FILE="${OUT_DIR}/${split}/todo_missing.txt"
    N_MISSING=$( (test -f "$TODO_FILE" && wc -l < "$TODO_FILE") || echo 0 )

    if [[ "$N_MISSING" -gt 0 ]]; then
      echo "[rest] ${split}: ${N_MISSING} rows to process"
      CMD="$STDBUF python ${SCRIPT} \
        --dataset_id \"${DATASET_ID}\" \
        --base_url \"${API}\" \
        --model \"${MODEL}\" \
        --read_timeout \"${READ_TIMEOUT}\" \
        --out_dir \"${OUT_DIR}\" \
        --splits \"${split}\" \
        --include_file \"${TODO_FILE}\" \
        --skip_filled \
        --limit 0 \
        --max_concurrent \"${MAX_CONCURRENT}\""
      echo "[cmd] $CMD"
      eval "$CMD"
    else
      echo "[rest] ${split}: nothing to do"
    fi
  done
fi

# ---- merge into parquet ----
echo "[merge] writing merged parquet(s)"
MERGE_SPLITS="${SPLITS}"
if [[ "${MERGE_SPLITS}" == "all" ]]; then
  MERGE_SPLITS="train,validation,test"
fi
$STDBUF python merge_poem_attrs.py \
  --dataset_id "${DATASET_ID}" \
  --attrs_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/merged" \
  --splits "${MERGE_SPLITS}"

# ---- quick summary ----
echo "[summary] annotated row counts:"
python - <<'PY'
import glob, pandas as pd, os
out_root=os.environ.get("OUT_DIR","poem_attrs")
root=os.path.join(out_root,"merged")
for p in sorted(glob.glob(f"{root}/*.parquet")):
    df=pd.read_parquet(p)
    got=(df["emotions"].notna() | df["primary_emotion"].notna() | df["sentiment"].notna())
    print(os.path.basename(p), "rows:", len(df), "annotated:", int(got.sum()))
PY

END_TS=$(date +%s)
DUR=$((END_TS - START_TS))

echo ""
echo "=== poetry attrs complete ==="
echo "[duration] ${DUR}s"
echo "[model] ${MODEL}"
echo "[output dir] ${OUT_DIR}/"
echo "[server] ${API}"
echo "[parallel] ${MAX_CONCURRENT}"
