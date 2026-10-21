#!/usr/bin/env bash
# T-003 검증 GUI 원클릭 실행기 (T-001의 run_gui.bat에 해당)
# - 서빙(vLLM+Flask)이 내려가 있으면 v2 모델로 기동 후 GUI 실행
# - 모든 경로는 이 폴더 기준. 외부 의존(환경변수로 변경 가능):
#     VLLM_BIN (vllm 실행파일), APP_PY (flask·transformers 있는 python)
set -u
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 경로 우선순위: 패키지 내 설치본(.venv, model/ — setup.sh 산출물) → 이 서버의 기존 환경
if [ -x "$BASE/.venv/bin/vllm" ]; then DEF_VLLM="$BASE/.venv/bin/vllm"; else DEF_VLLM="/home/heri/.conda/envs/handbook5/bin/vllm"; fi
if [ -x "$BASE/.venv/bin/python" ]; then DEF_PY="$BASE/.venv/bin/python"; else DEF_PY="/home/heri/test_prepare/serving_local/venv_app/bin/python"; fi
if [ -e "$BASE/model/model.safetensors" ]; then DEF_MODEL="$BASE/model"; else DEF_MODEL="$BASE/v2_model"; fi
MODEL="${MODEL_PATH:-$DEF_MODEL}"
PORT="${GUI_PORT:-7860}"
export VLLM_BIN="${VLLM_BIN:-$DEF_VLLM}"
export APP_PY="${APP_PY:-$DEF_PY}"
IP=$(hostname -I | awk '{print $1}')

echo "=================================================="
echo "  문화유산 해설 Citation Accuracy - 검증 GUI"
echo "=================================================="
echo "검증 대상: 문화유산 648종 (테스트셋 5%, 시드 2026) / 근거문서 DB 12,976건"
echo "시험대상 모델: CESS_NatLang (V 1.0, GPU 4장 TP=4)"
echo

if ! curl -fsS --max-time 3 http://127.0.0.1:5000/health 2>/dev/null | grep -q '"vllm_server_healthy":true'; then
  echo "모델 로딩 중... (해설 생성 서버 기동, 최초 1회 약 3~5분 소요)"
  (cd "$BASE/serving" && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} MODEL_PATH="$MODEL" MAX_NUM_SEQS=24 \
     setsid nohup ./run_local.sh vllm 0 > /tmp/t003_vllm.log 2>&1 < /dev/null &)
  sleep 3
  (cd "$BASE/serving" && MODEL_PATH="$MODEL" APP_THREADED=1 \
     setsid nohup ./run_local.sh app 0 > /tmp/t003_app.log 2>&1 < /dev/null &)
  until curl -fsS --max-time 3 http://127.0.0.1:5000/health 2>/dev/null | grep -q '"vllm_server_healthy":true'; do
    printf "."; sleep 10
  done
  echo
fi
echo "모델 로딩 완료. 해설 생성 서비스 정상 (http://127.0.0.1:5000/health)"
echo
echo "같은 네트워크(서브넷)의 다른 PC에서는 http://$IP:$PORT 로 접속하면 됩니다."
echo "(방화벽에서 이 포트를 막아두었다면 접속이 안 될 수 있습니다.)"
echo "* Running on local URL:  http://0.0.0.0:$PORT"
echo "* 이 창을 닫으면 GUI가 종료됩니다."
echo

exec "$APP_PY" "$BASE/verify_gui.py" \
  --questions "$BASE/testset05_s2026/questions_05.jsonl" \
  --evidence "$BASE/data/aimuse_items_enriched_final_v6.json" \
  --serve-url http://127.0.0.1:5000 \
  --label "테스트셋 5% (648종, 시드 2026) · CESS_NatLang" \
  --port "$PORT"
