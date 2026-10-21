# T-003 자연어 생성모델 근거 기반 신뢰성 지표 — 시연 패키지

문화유산 스토리텔링 AI(CESS_NatLang)의 Citation Accuracy(성능목표 70%)
시연·시험에 필요한 파일 일체. 시험절차서 붙임4의 시험방법 순서 1~4와
1:1 대응한다.

- 공식 지표: 어휘 일치도 기반 Citation Accuracy (문장 단위, 임계값 0.40
  사전 설정, 외부 API·GPU 없이 CPU에서 결정적 재현)
- 테스트셋: 전체 12,968종 중 무작위 5% **648종 (시드 2026)** — 사전
  리허설 실측(배치 1.15건/초)으로 30분 시험 내 현장 전량 생성 규모로 확정
- 사전 측정(2026-10-19, 비공인): **v2 73.29%** (판정 5,799문장, 무관근거
  대조 11.85%) / v1 74.68% — 규모별 일관성 v2 기준 648/2,594/12,968종 =
  73.29/73.30/73.27%. 648종 무작위 재표집 1,000회에서 70% 미만 0회(71.1~75.0%)

## 시연 절차 (캡처 순서)

### 캡처 1 — 검증 데이터셋 확인 (붙임4 순서 1)

```bash
ls testset05_s2026/검증데이터셋 | wc -l    # 648
head -12 testset05_s2026/testset_list.csv  # 번호,유물ID,유물명,관람객유형
```

파일 탐색기 캡처: `testset05_s2026/검증데이터셋/` 폴더를 열면
"648개 항목"(유물별 근거문서 텍스트 파일)이 표시된다. Windows 탐색기
화면이 필요하면 이 폴더를 업무용 PC로 복사해 연다.

### 캡처 2 — 시험대상 실행 (붙임4 순서 2)

```bash
./run_gui.sh
```

터미널에 배너·로딩 상태·접속 주소가 표시된다(캡처). 서빙이 꺼져 있으면
v2 모델을 자동 기동한다(최초 약 3~5분). 이후 업무용 PC 브라우저에서
`http://<서버 IP>:7860` 접속 → **"해설 재생성 포함" 체크** →
**[전체 평가 시작]** → 생성 진행률 화면(N/648건, 건/s, 남은 시간) 캡처.
648종 생성 약 10분.

### 캡처 3 — Citation Accuracy 측정 (붙임4 순서 3)

생성 완료 시 GUI가 공식 측정 스크립트(measure_citation_accuracy.py)를
자동 실행하고 평가 지표 표를 표시한다 — Citation Accuracy 강조행과
"70% 이상 — 통과" 캡처.

### 캡처 4 — 결과 상세 (붙임4 순서 4)

관람객 유형별 정확도 막대(목표 70% 점선) + 문장 일치도 분포
히스토그램(임계치 0.40 표시) 1컷, [문장별 근거 하이라이트 뷰어 열기] →
유물 선택 → 문장 클릭 → 근거 위치 강조 1컷.

### 비상 경로 (현장 생성 장애 시)

사전 수집본으로 즉시 측정만 시연:

```bash
python3 measure_citation_accuracy.py \
    --input testset05_s2026/eval_set_05_v2.jsonl --out-dir output/demo
# 또는 GUI에서 "해설 재생성 포함" 체크 해제 후 [전체 평가 시작]
#   (output/gui_run/eval_set.jsonl 자리에 사전 수집본을 복사해 두면 즉시 측정)
```

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `run_gui.sh` | 원클릭 실행기: 서빙 자동 기동 + 검증 GUI (포트 7860) |
| `verify_gui.py` | 검증 GUI — [전체 평가 시작] → 생성→공식 측정→차트·하이라이트 |
| `citation_metric.py` / `measure_citation_accuracy.py` | **공식 측정** (CPU 단독, GUI도 이 스크립트를 호출) |
| `generate_responses_api.py` | 해설 수집 (REST, `--resume` 중단 재개) |
| `visualize_citation.py` | 해설-근거 하이라이트 뷰어 생성 |
| `prepare_testset.py` | 테스트셋 구성 (무작위 추출·시드 고정) — 데이터 확인 단계 증빙 |
| `testset05_s2026/` | 공식 테스트셋: ids·질문·목록 CSV·검증데이터셋(648 파일)·비상용 사전 수집본 |
| `serving/` | 해설 서비스: run_local.sh(vLLM+Flask 기동)·app_vllm_sep_t003.py·pyshim(flash_attn 우회) |
| `data/` | 근거문서 DB(aimuse v6, 12,976유물)·페르소나 프롬프트·도슨트 RAG |
| `v2_model/` | 시험대상 모델(심링크: /hdd4 체크포인트-2460 + processor_config.json) |

## 신규 서버 설치 (처음부터, 4단계)

```bash
git clone <이 저장소> && cd test_git
export HF_TOKEN=hf_...          # heri2026 비공개 repo 접근 토큰
./setup.sh                      # ① .venv 구성(vllm 0.23 등) ② 모델 heri2026/heri_tts_v2
                                #   다운로드(+processor_config.json 자동 보정) ③ 근거문서 DB 다운로드
./run_gui.sh                    # 서빙 자동 기동 + 검증 GUI (http://<서버IP>:7860)
```

- 요구: Python 3.10+, NVIDIA GPU(80GB급, 기본 4장 TP=4 — 장수가 다르면
  `serving/run_local.sh`의 `--tensor-parallel-size` 조정), 외부망(설치 시에만).
- `setup.sh`가 만든 `.venv`·`model/`이 있으면 run_gui.sh가 자동으로 그걸 쓰고,
  없으면 이 서버(rpower4)의 기존 환경 경로로 폴백한다.
- 테스트셋·질문·프롬프트·도슨트 데이터는 저장소에 포함(소용량). 대용량 2종만
  외부: 모델(HF `heri2026/heri_tts_v2`), 근거문서 DB(HF dataset
  `heri2026/heri_t003_data`).

### 사전 1회 업로드 (저장소 공개 전, 이 서버에서)

```bash
# 모델 v2 업로드 (predetermined: heri2026/heri_tts_v2)
.venv/bin/python -c "from huggingface_hub import HfApi;   HfApi(token='hf_...').upload_folder(folder_path='v2_model', repo_id='heri2026/heri_tts_v2',   repo_type='model', ignore_patterns=['*.pth','global_step*','latest','zero_to_fp32.py'])"
# 근거문서 DB 업로드 (dataset repo 생성 후)
.venv/bin/python -c "from huggingface_hub import HfApi; api=HfApi(token='hf_...');   api.create_repo('heri2026/heri_t003_data', repo_type='dataset', private=True, exist_ok=True);   api.upload_file(path_or_fileobj='data/aimuse_items_enriched_final_v6.json',   path_in_repo='aimuse_items_enriched_final_v6.json', repo_id='heri2026/heri_t003_data', repo_type='dataset')"
```

## 외부 의존 (이 서버 기준 기본값, 환경변수로 변경)

| 항목 | 기본값 | 변경 |
|---|---|---|
| vLLM 실행파일 | `/home/heri/.conda/envs/handbook5/bin/vllm` (0.23) | `VLLM_BIN` |
| GUI/앱 파이썬 | `/home/heri/test_prepare/serving_local/venv_app/bin/python` (flask·transformers≥5.12) | `APP_PY` |
| 모델 가중치 | `v2_model/` → `/hdd4/.../checkpoint-2460` 심링크 | `MODEL_PATH` |
| GPU | 0,1,2,3 (TP=4) | `CUDA_VISIBLE_DEVICES` |

git 주의: 100MB 초과 파일(근거문서 DB, 모델 가중치)과 `.venv/`·`model/`은
`.gitignore`로 제외 — 신규 서버에서는 `setup.sh`가 HF에서 자동 수급한다.

## 측정 근거 요약 (질의 대응)

- 임계값 0.40: 2025 구모델 평가 데이터의 무관근거 대조실험으로 사전 설정,
  금년 시험셋 미변경. 무관근거 통과 11.85%는 지표 한계로 명시.
- 판별력: 자기 근거 73.29% ↔ 무관 근거 11.85% (약 6:1).
- 보조 검증(공식 점수 미사용): GPT-4.1 함의 판정 — 어휘 통과 문장의
  99.5%가 완전/부분함의, 모순(근거 상충) 0.13~0.25%.
- 상세 문안·서지: `/home/heri/test_prepare_final/시험절차서_T003_작성안.md`
