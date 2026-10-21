#!/usr/bin/env bash
# T-003 시연 패키지 초기 설치 (신규 서버용) — 순서: venv → 모델 → 데이터
# 요구: Python 3.10+, NVIDIA GPU(80GB급 4장 권장, TP 조정 가능), 외부망,
#       비공개 모델 접근용 HF 토큰:  export HF_TOKEN=hf_...
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="${MODEL_REPO:-heri2026/heri_tts_v2}"
DATA_REPO="${DATA_REPO:-heri2026/heri_t003_data}"   # evidence v6가 올라간 HF dataset repo
PY="${PYTHON:-python3}"

echo "[1/3] 가상환경(.venv) 구성 및 패키지 설치"
if [ ! -x "$BASE/.venv/bin/python" ]; then
  "$PY" -m venv "$BASE/.venv"
fi
"$BASE/.venv/bin/pip" install -q --upgrade pip
"$BASE/.venv/bin/pip" install -q -r "$BASE/requirements.txt"
"$BASE/.venv/bin/python" - <<'EOF'
import vllm, flask, transformers
print(f"  vllm {vllm.__version__} / flask OK / transformers {transformers.__version__}")
EOF

echo "[2/3] 시험대상 모델 다운로드: $MODEL_REPO → model/ (~26GB)"
"$BASE/.venv/bin/python" - <<EOF
import os
from huggingface_hub import snapshot_download, hf_hub_download
tok = os.environ.get("HF_TOKEN") or None
snapshot_download("$MODEL_REPO", token=tok, local_dir="$BASE/model")
# 필수: gemma4_unified 서빙에 processor_config.json 필요 (모델 repo에 없으면 베이스에서)
import shutil
if not os.path.exists("$BASE/model/processor_config.json"):
    p = hf_hub_download("google/gemma-4-12B", "processor_config.json", token=tok)
    shutil.copy(p, "$BASE/model/processor_config.json")
    print("  processor_config.json: google/gemma-4-12B에서 복사")
print("  모델 준비 완료:", "$BASE/model")
EOF

echo "[3/3] 근거문서 DB 다운로드: $DATA_REPO → data/"
"$BASE/.venv/bin/python" - <<EOF
import os
from huggingface_hub import hf_hub_download
import shutil
tok = os.environ.get("HF_TOKEN") or None
p = hf_hub_download("$DATA_REPO", "aimuse_items_enriched_final_v6.json",
                    repo_type="dataset", token=tok)
shutil.copy(p, "$BASE/data/aimuse_items_enriched_final_v6.json")
print("  근거문서 DB 준비 완료 (12,976 유물)")
EOF

echo
echo "설치 완료. 실행:  ./run_gui.sh   (브라우저: http://<서버IP>:7860)"
echo "GPU 장수가 4가 아니면: serving/run_local.sh의 --tensor-parallel-size 조정"
