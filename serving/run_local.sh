#!/usr/bin/env bash
# T-003 로컬 서빙 기동 (dev 서버, A100 x8)
# - vLLM: conda env handbook5 (vllm 0.23.0), 모델 heri2026/heri_tts 로컬 사본
# - Flask 앱: venv_app, app_vllm_sep_t003.py (운영 코드 + 경로/호스트 env 패치)
#
# 인스턴스 2개 구성 (전체 유물 고속 수집용):
#   ./run_local.sh vllm 0   → GPU 0-3, vLLM 포트 9000
#   ./run_local.sh vllm 1   → GPU 4-7, vLLM 포트 9001
#   ./run_local.sh app  0   → Flask 5000 → vLLM 9000
#   ./run_local.sh app  1   → Flask 5001 → vLLM 9001
#
# 기본은 운영 동일(순차, MAX_NUM_SEQS=1). 고속 수집 시:
#   MAX_NUM_SEQS=24 ./run_local.sh vllm <i>
#   APP_THREADED=1  ./run_local.sh app  <i>
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 타 서버 이식: 아래 네 개는 환경변수로 오버라이드 가능
MODEL="${MODEL_PATH:-/home/heri/etri_server/llm/hf_model_tts}"
VLLM_BIN="${VLLM_BIN:-/home/heri/.conda/envs/handbook5/bin/vllm}"
APP_PY="${APP_PY:-$DIR/venv_app/bin/python}"
DATA_DIR="${DATA_DIR:-$DIR/data}"
IDX="${2:-0}"
mkdir -p "$DIR/logs"

case "${1:-}" in
  vllm)
    if [[ "$IDX" == "0" ]]; then GPUS=0,1,2,3; PORT=9000; else GPUS=4,5,6,7; PORT=9001; fi
    # flash_attn ABI 불일치 우회 (pyshim/sitecustomize.py 참조; 문제 없는 서버에서도 무해)
    export PYTHONPATH="$DIR/pyshim${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"
    exec "$VLLM_BIN" serve "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --tensor-parallel-size 4 \
      --max-model-len 12288 \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.6}" \
      --max-num-seqs "${MAX_NUM_SEQS:-1}" \
      --kv-offloading-size 8 \
      --disable-hybrid-kv-cache-manager
    ;;
  app)
    if [[ "$IDX" == "0" ]]; then APP_PORT=5000; VURL=http://127.0.0.1:9000; else APP_PORT=5001; VURL=http://127.0.0.1:9001; fi
    export MODEL_PATH="$MODEL"
    export EVIDENCE_FILE="${EVIDENCE_FILE:-$DATA_DIR/aimuse_items_enriched_final_v6.json}"
    export PERSONA_PROMPTS_DIR="${PERSONA_PROMPTS_DIR:-$DATA_DIR/prompts/rephrase_prompts}"
    export DOCENT_RAG_FILE="${DOCENT_RAG_FILE:-$DATA_DIR/docent_rag_flat_ordered.json}"
    export APP_HOST="${APP_HOST:-127.0.0.1}"
    export APP_PORT VLLM_SERVER_URL="$VURL"
    exec "$APP_PY" "$DIR/app_vllm_sep_t003.py"
    ;;
  judge)
    # NLI 판정용 공개 LLM (생성 모델과 별개). 기본 1 GPU(80GB)면 충분.
    export PYTHONPATH="$DIR/pyshim${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec "$VLLM_BIN" serve "${JUDGE_MODEL:-google/gemma-3-27b-it}" \
      --host 127.0.0.1 --port "${JUDGE_PORT:-9100}" \
      --tensor-parallel-size "${JUDGE_TP:-1}" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.92 \
      --max-num-seqs 48
    ;;
  *)
    echo "usage: $0 [vllm|app|judge] [0|1]"
    exit 2
    ;;
esac
