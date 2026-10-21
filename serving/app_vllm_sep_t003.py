#!/usr/bin/env python3
"""
문화유산 해설 생성 REST API
문화유산 ID, 관람객 유형, 질문을 받아서 자연어 해설을 생성하는 API
"""

import copy
import json
import os
import sys

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote
import time
import requests
from flask import Flask, jsonify, request
from transformers import AutoTokenizer

app = Flask(__name__)
# 응답 JSON 필드 순서 보존(태그→답변→tts→참고한 문단/문장) — Flask 기본 키 정렬 해제
try:
    app.json.sort_keys = False          # Flask >= 2.3
except AttributeError:
    app.config["JSON_SORT_KEYS"] = False  # Flask < 2.3

# Configuration
# 2026 재학습 모델: gemma-4-12B SFT, 대화 형식은 ChatML(<|im_start|>/<|im_end|>).
# 서빙 서버의 실제 모델 경로로 맞추세요(이전 gemma-3 모델 경로 아님).
MODEL_PATH = os.getenv("MODEL_PATH", "/home/story/hf_model")
EVIDENCE_FILE = os.getenv("EVIDENCE_FILE", "/home/heri/test_prepare/serving_local/data/aimuse_items_enriched_final_v6.json")
# 관람객 유형별 페르소나 프롬프트(prompt_<type>.txt) 디렉터리.
# 학습(make_train_data.ipynb)에서 쓴 rephrase_prompts/ 파일들을 이 서버에 복사하고 경로를 맞추세요.
PERSONA_PROMPTS_DIR = os.getenv("PERSONA_PROMPTS_DIR", "/home/story/llm/data/prompts/rephrase_prompts")
DEFAULT_SOLR_SCRIPT = "/home/story/ency/solr/query-db.sh"
SOLR_QUERY_SCRIPT = os.getenv("SOLR_QUERY_SCRIPT", DEFAULT_SOLR_SCRIPT)

# ===== 도슨트 테마 RAG (docent_rag_flat_ordered.json) =====
# 테마별 도슨트 서술문을 질문/유물명과 '문자 n-gram 어휘 유사도'로 검색해
# 근거문서에 [도슨트 해설] 블록으로 주입한다. (임베딩/외부 API 불필요 — 110개 규모라 충분)
# 관련도 임계치 미만이면 주입 안 함 → 무관한 유물 프롬프트는 학습과 동일하게 유지.
DOCENT_RAG_FILE = os.getenv("DOCENT_RAG_FILE", "/home/story/llm/data/docent_rag_flat_ordered.json")
ENABLE_DOCENT_RAG = os.getenv("ENABLE_DOCENT_RAG", "1").strip().lower() in ("1", "true", "yes")
DOCENT_RAG_TOP_K = int(os.getenv("DOCENT_RAG_TOP_K", "3"))
DOCENT_RAG_MIN_SCORE = float(os.getenv("DOCENT_RAG_MIN_SCORE", "0.12"))  # n-gram 겹침 임계치
DOCENT_RAG_MAX_CHARS = int(os.getenv("DOCENT_RAG_MAX_CHARS", "1500"))    # 도슨트 블록 총 글자 상한

# 입력 길이 관련 설정
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 2048

# 생성 샘플링 — 평가(test_model_2026_vllm.py)와 동일하게 맞춤(답변/거절 재현성 확보).
# greedy(temperature=0)면 답/거절이 평가와 달라질 수 있어, 평가와 같은 샘플링 + seed 고정으로
# '결정적이면서 평가와 일치'하게 한다.
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.95
GEN_TOP_K = 64
GEN_SEED = 42
# qa 과잉거절 완화: '확인불가' 거절인데 질문 핵심어가 근거문서에 실제로 있으면 seed를 바꿔 재생성(최대 N회, 0이면 비활성).
# 같은 프롬프트도 seed에 따라 답변/거절이 갈리는 확률적 거동을 실측(2026-10-12: 6 seed 중 2건 거절)해 둔 임시책.
REFUSAL_RETRY_MAX = int(os.getenv("REFUSAL_RETRY_MAX", "2"))

# reference_doc 질문 기반 재정렬 설정
# 기본 False: 학습 때 build_evidence_text가 reference_doc 전체를 '원문 그대로' 넣었으므로
# 서빙에서도 동일하게 원문 전체를 넣어 train/serve를 일치시킨다.
# (컨텍스트 초과가 잦으면 ENABLE_REFERENCE_RERANK=1 로 재정렬·트리밍을 켤 수 있다.)
ENABLE_REFERENCE_RERANK = os.getenv("ENABLE_REFERENCE_RERANK", "0").strip().lower() in ("1", "true", "yes")
RERANK_TOP_K = 5
REF_DOC_RESERVE_TOKENS = 500       
REF_DOC_MIN_TOKEN_BUDGET = 0     
REF_DOC_MAX_TOKEN_BUDGET = 2500    
MAX_SINGLE_REF_CHARS = 500   
DEBUG_RERANK = False
# API 응답에 원본/재정렬/모델투입 reference_doc 전체 목록 포함 (용량 큼 — 비활성: INCLUDE_REFERENCE_DOC_TRACE=0)
INCLUDE_REFERENCE_DOC_TRACE = os.getenv("INCLUDE_REFERENCE_DOC_TRACE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)


# Global variables
tokenizer = None
# 토크나이저 로드를 이미 시도했는지(실패 포함). 매 요청마다 재시도하지 않기 위함.
_tokenizer_tried = False
evidence_dict = {}
visitor_prompts = {}
docent_rag = []   # 도슨트 테마 RAG: [{text, theme_title, record_type, relic_ids, _grams}]
docent_relic_ids: Set[str] = set()   # 도슨트 원고가 있는 유물 ID 집합 (주입 게이트)

AUX_PROP_NAMES = {
    "비고",
    "소장품 설명_추가",
    "설명추가",
    "기증자정보_추가",
    "유물크기"
}
MAX_AUX_PROP_ITEMS = 8
MAX_AUX_PROP_CHARS = 320
MAX_MOONYANG_ITEMS = 80
MAX_MOONYANG2_CHARS = 1000

# 2026 재학습 관람객 유형(make_train_data.ipynb VISITOR_PROMPT_FILES 기준).
# light_viewing(가벼운 관람형)은 학습에서 제외되어 여기서도 제외.
VISITOR_PROMPT_FILES = {
    "deep_inquiry": "prompt_deep_inquiry.txt",
    "emotional_engagement": "prompt_emotional-engagement.txt",
    "high_engagement": "prompt_high-engagement.txt",
    "information_centric": "prompt_information-centric.txt",
}

# 시스템 프롬프트에 들어가는 유형명(학습 시 '유형' 값과 동일해야 함)
VISITOR_TYPE_NAMES = {
    "deep_inquiry": "심층 탐구형 관람객",
    "emotional_engagement": "감성 몰입형 관람객",
    "high_engagement": "고강도 참여형 관람객",
    "information_centric": "정보 중심형 관람객",
}

# 미지정/미인식 visitor_type의 기본 유형 (옛 기본값 '탐구형' 포함).
DEFAULT_VISITOR_KEY = "high_engagement"

# API 입력 -> 표준 유형키. 영문 키와 한국어 별칭 모두 허용.
VISITOR_TYPE_MAP = {k: k for k in VISITOR_PROMPT_FILES}
VISITOR_TYPE_MAP.update({
    "심층 탐구형": "deep_inquiry", "심층탐구형": "deep_inquiry",
    "감성 몰입형": "emotional_engagement", "감성몰입형": "emotional_engagement", "몰입형": "emotional_engagement",
    "고강도 참여형": "high_engagement", "고강도참여형": "high_engagement", "참여형": "high_engagement",
    "정보 중심형": "information_centric", "정보중심형": "information_centric", "정보형": "information_centric",
    # 구버전 유형명: 새 모델엔 없음. 옛 기본값 '탐구형'은 새 기본값(고강도 참여형)으로 받는다.
    # (의미상 심층탐구를 원하면 'deep_inquiry' / '심층 탐구형'으로 요청해야 함)
    "탐구형": "high_engagement", "취미형": "high_engagement", "의무형": "high_engagement",
    "휴식형": "high_engagement", "사교형": "high_engagement",
})


def resolve_visitor_key(visitor_type: str) -> str:
    """API 입력 -> 표준 유형키. 미지정/미인식 값은 DEFAULT_VISITOR_KEY(고강도 참여형)로 폴백."""
    vt = (visitor_type or "").strip()
    key = VISITOR_TYPE_MAP.get(vt) or VISITOR_TYPE_MAP.get(re.sub(r"\s+", "", vt))
    if key:
        return key
    print(f"알 수 없는 visitor_type='{visitor_type}' -> 기본값 '{DEFAULT_VISITOR_KEY}'로 처리")
    return DEFAULT_VISITOR_KEY


EXTRA_AIMUSE_KEYS = [
    "summary",
    "국적", "기증관", "문양", "문양2",
    "민족문화대백과 연관정보", "소장기관",
    "소장품 설명_추가", "의미", "의미2", "재질",
]


class TeeStream:
    """Write stream output to both terminal and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_realtime_file_logging() -> Path:
    """
    Mirror stdout/stderr to a log file in this script directory.
    Override file path with APP_LOG_FILE environment variable.
    """
    script_dir = Path(__file__).resolve().parent
    log_path_env = os.getenv("APP_LOG_FILE", "app_vllm_sep.log").strip()
    log_path = Path(log_path_env)
    if not log_path.is_absolute():
        log_path = script_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # line buffering keeps log updates visible almost immediately
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_path


def load_evidence_data(evidence_file_path: str) -> Dict:
    """Load evidence data from JSON file"""
    global evidence_dict

    if evidence_dict:
        return evidence_dict

    print(f"Loading evidence data from {evidence_file_path}...")
    with open(evidence_file_path, 'r', encoding='utf-8') as f:
        evidence_data = json.load(f)

    evidence_dict = {}
    for item in evidence_data:
        item_id = item.get("id") or item.get("소장품번호")
        if item_id is not None:
            evidence_dict[str(item_id)] = item

    print(f"Loaded {len(evidence_dict)} evidence items.")
    return evidence_dict


def load_persona(visitor_key: str, prompts_dir: str) -> str:
    """
    유형별 페르소나 프롬프트 파일에서 '## 유형 ...' 블록만 추출.
    학습(make_train_data.ipynb)의 load_persona와 동일한 규칙: '## ' ~ 다음 '\\n---' 직전까지.
    """
    fn = VISITOR_PROMPT_FILES.get(visitor_key)
    if not fn:
        return ""
    p = os.path.join(prompts_dir, fn)
    if not os.path.exists(p):
        return ""
    with open(p, "r", encoding="utf-8") as f:
        s = f.read()
    i = s.find("## ")
    if i < 0:
        return ""
    j = s.find("\n---", i)
    return s[i:(j if j > 0 else len(s))].strip()


def load_visitor_prompts(prompts_dir: str) -> Dict[str, str]:
    """관람객 유형별 페르소나 로드 (유형키 -> 페르소나 텍스트)."""
    global visitor_prompts

    if visitor_prompts:
        return visitor_prompts

    print(f"Loading visitor personas from {prompts_dir}...")
    loaded = {}
    for vk in VISITOR_PROMPT_FILES:
        persona = load_persona(vk, prompts_dir)
        if persona:
            loaded[vk] = persona
        else:
            print(f"  ⚠️  페르소나 파일을 찾지 못했거나 비어 있음: {vk} "
                  f"({os.path.join(prompts_dir, VISITOR_PROMPT_FILES[vk])})")
            loaded[vk] = ""

    visitor_prompts = loaded
    print(f"Loaded personas for {sum(1 for v in visitor_prompts.values() if v)}/{len(VISITOR_PROMPT_FILES)} visitor types.")
    return visitor_prompts


BASE_MODEL_NAME = "google/gemma-4-12B"

# vLLM OpenAI 호환 서버 (별도 프로세스). 예: vllm serve --host 0.0.0.0 --port 8000
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://127.0.0.1:9000").rstrip("/")
# `vllm serve`에 넘긴 모델 id (--served-model-name 미지정 시 보통 전체 경로)
VLLM_SERVED_MODEL_NAME = os.getenv("VLLM_SERVED_MODEL_NAME", MODEL_PATH)
VLLM_HTTP_TIMEOUT_SEC = float(os.getenv("VLLM_HTTP_TIMEOUT_SEC", "600"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", str(MAX_INPUT_TOKENS + MAX_NEW_TOKENS)))
# 중요: 이 모델은 ChatML(<|im_start|>/<|im_end|>)로 학습됨. assistant 턴은 <|im_end|>로 끝나는데
# tokenizer의 eos_token은 <eos>라서, stop 토큰을 명시하지 않으면 생성이 제대로 멈추지 않는다.
VLLM_STOP_TOKENS = [s for s in os.getenv("VLLM_STOP", "<|im_end|>").split(",") if s]


def load_tokenizer():
    """로컬에서 채팅 템플릿·토큰 길이 검사용 토크나이저만 로드 (GPU/vLLM 엔진 없음)."""
    global tokenizer

    if tokenizer is not None:
        return tokenizer

    print(f"Loading tokenizer from {MODEL_PATH} (for chat template / length checks)...")
    tokenizer_path = (
        MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, "tokenizer_config.json")) else BASE_MODEL_NAME
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded successfully.")
    return tokenizer


def vllm_server_healthy() -> bool:
    try:
        r = requests.get(f"{VLLM_SERVER_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def vllm_completions(
    prompt: str,
    max_tokens: int,
    temperature: float = GEN_TEMPERATURE,
    top_p: float = GEN_TOP_P,
    top_k: int = GEN_TOP_K,
    seed: Optional[int] = GEN_SEED,
) -> Tuple[Optional[str], Optional[str]]:
    url = f"{VLLM_SERVER_URL}/v1/completions"
    payload: Dict[str, Any] = {
        "model": VLLM_SERVED_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,            # vLLM 확장 파라미터(평가와 동일)
    }
    if seed is not None:
        payload["seed"] = seed     # seed 고정 → 결정적, 평가와 동일 샘플링 경로
    if VLLM_STOP_TOKENS:
        payload["stop"] = VLLM_STOP_TOKENS
    try:
        r = requests.post(url, json=payload, timeout=VLLM_HTTP_TIMEOUT_SEC)
    except requests.RequestException as e:
        return None, f"vLLM 서버 연결 실패 ({VLLM_SERVER_URL}): {e}"

    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:500]
        return None, f"vLLM HTTP {r.status_code}: {detail}"

    try:
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return None, "vLLM 응답에 choices가 없습니다."
        text = choices[0].get("text", "")
        return text, None
    except (KeyError, ValueError, TypeError) as e:
        return None, f"vLLM 응답 파싱 오류: {e}"

def run_solr_query(query: str) -> Tuple[int, List[Dict]]:
    """Execute Solr query"""
    if not os.path.exists(SOLR_QUERY_SCRIPT):
        return 0, []

    try:
        script_dir = os.path.dirname(SOLR_QUERY_SCRIPT)
        result = subprocess.run(
            [SOLR_QUERY_SCRIPT, query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            cwd=script_dir
        )
        output = result.stdout

        json_start = output.find('{')
        if json_start == -1:
            return 0, []

        response_json = json.loads(output[json_start:])
        num_found = response_json.get("response", {}).get("numFound", 0)
        docs = response_json.get("response", {}).get("docs", [])
        return num_found, docs
    except Exception as e:
        print(f"Solr query error: {e}")
        return 0, []


def normalize_headword(text: str) -> str:
    """Normalize artifact name (remove brackets)"""
    if not text:
        return ""
    cleaned = re.sub(r"[「」『』《》()\[\]{}<>]", "", text)
    return cleaned.strip()


def extract_metadata(item: Dict) -> Tuple[str, str, str, List[str]]:
    """Extract metadata from item"""
    raw_headword = item.get("유물", "")
    era = item.get("시대", "")
    hanja = ""
    authors = []

    for prop in item.get("data_property", []):
        if prop.get("name") == "한자명" and not hanja:
            hanja = prop.get("value", "").split(",")[0].strip()

    if "작가" in item:
        authors_from_field = [
            a["value"] for a in item["작가"]
            if a.get("name") == "작가명"
        ]
        if authors_from_field:
            authors = authors_from_field
    else:
        for prop in item.get("data_property", []):
            if prop.get("name") == "작가명":
                authors = [prop.get("value", "").strip()]
                break

    return raw_headword, era, hanja, authors


def generate_headword_doc(item: Dict) -> str:
    """Generate headword_doc"""
    raw_headword, era, hanja, authors = extract_metadata(item)
    headword = normalize_headword(raw_headword)

    if not headword:
        return ""

    def build_doc_text(doc):
        parts = []
        for key in ["headword", "type", "definition", "summary", "body"]:
            val = doc.get(key)
            if isinstance(val, list):
                parts.append(f"{key}: {', '.join(val)}")
            elif val:
                parts.append(f"{key}: {val}")
        return "\n".join(parts)

    def headword_phrase(q):
        s = q.replace('"', r'\"')
        if not s.endswith(" "):
            s += " "
        return f'headword:"{s}"'

    base_query = headword_phrase(headword)
    num_found, docs = run_solr_query(base_query)
    
    if num_found == 1:
        return build_doc_text(docs[0])
    elif num_found > 1:
        if hanja:
            q1 = f'{base_query} AND body:"{hanja}"'
            n1, d1 = run_solr_query(q1)
            if n1 == 1:
                return build_doc_text(d1[0])

        for author in authors:
            q2 = f'{base_query} AND body:"{author}"'
            n2, d2 = run_solr_query(q2)
            if n2 == 1:
                return build_doc_text(d2[0])

    return ""


def generate_reference_doc(item: Dict, top_k: int = 3) -> List[str]:
    """Generate reference_doc"""
    raw_headword, era, hanja, authors = extract_metadata(item)
    normalized = normalize_headword(raw_headword)

    if not normalized:
        return []

    base_query = f'body:"{normalized}"'
    num_found, docs = run_solr_query(base_query)

    if num_found == 0:
        return []

    if num_found > top_k:
        filtered_docs = []
        if authors:
            for author in authors:
                filtered_query = f'body:"{normalized}" AND body:"{author}"'
                filtered_num, temp_docs = run_solr_query(filtered_query)
                if 0 < filtered_num <= top_k:
                    filtered_docs = temp_docs
                    break

        if (not filtered_docs or len(filtered_docs) > top_k) and hanja:
            hanja_query = f'body:"{normalized}" AND body:"{hanja}"'
            hanja_num, temp_docs = run_solr_query(hanja_query)
            if 0 < hanja_num <= top_k:
                filtered_docs = temp_docs

        if filtered_docs:
            docs = filtered_docs
        else:
            docs = docs[:top_k]

    docs = docs[:top_k]

    reference_entries = []
    search_terms = [normalized]

    for doc in docs:
        body = doc.get("body", "")
        if not body:
            continue

        paras = re.split(r'\n+\s*', body)
        matched_paras = []

        for para in paras:
            para = para.strip()
            if not para:
                continue
            for term in search_terms:
                if term and term in para:
                    matched_paras.append(para)
                    break

        if matched_paras:
            doc_parts = []
            for key in ["headword", "type", "definition", "summary"]:
                val = doc.get(key)
                if isinstance(val, list):
                    doc_parts.append(f"{key}: {', '.join(val)}")
                elif val:
                    doc_parts.append(f"{key}: {val}")

            combined_text = "\n".join(doc_parts) + "\n\n" + "\n\n".join(matched_paras)
            reference_entries.append(combined_text)

    return reference_entries


# -------------------------
# auxiliary property helpers
# -------------------------
def _shorten_text(text: str, limit: int = MAX_AUX_PROP_CHARS) -> str:
    """Shorten auxiliary property text to avoid prompt bloat."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_top_level_moonjang_blocks(evidence_item: Dict) -> List[str]:
    """
    JSON 최상위 '문양' / '문양2' 필드를 근거문서 블록으로 만든다.
    - 문양: 보통 [{\"name\": \"분류\", \"value\": \"표기\"}, ...]
    - 문양2: 보통 짧은 요약 문자열 (쉼표 구분 등)
    """
    blocks: List[str] = []

    mj = evidence_item.get("문양")
    if isinstance(mj, str) and mj.strip():
        blocks.append(f"[문양]\n{_shorten_text(mj, MAX_MOONYANG2_CHARS)}")
    elif isinstance(mj, list) and mj:
        lines: List[str] = []
        cap = min(len(mj), MAX_MOONYANG_ITEMS)
        for i in range(cap):
            entry = mj[i]
            if isinstance(entry, dict):
                n = str(entry.get("name", "")).strip()
                v = str(entry.get("value", "")).strip()
                if n and v:
                    lines.append(f"{n}: {v}")
                elif v:
                    lines.append(v)
            else:
                s = str(entry).strip()
                if s:
                    lines.append(s)
        if len(mj) > MAX_MOONYANG_ITEMS:
            lines.append(f"(… 외 {len(mj) - MAX_MOONYANG_ITEMS}건 생략)")
        if lines:
            blocks.append("[문양]\n" + "\n".join(lines))

    mj2 = evidence_item.get("문양2")
    if isinstance(mj2, str) and mj2.strip():
        blocks.append(f"[문양2]\n{_shorten_text(mj2, MAX_MOONYANG2_CHARS)}")
    elif isinstance(mj2, list) and mj2:
        joined = ", ".join(
            str(x).strip() for x in mj2 if x is not None and str(x).strip()
        )
        if joined:
            blocks.append(f"[문양2]\n{_shorten_text(joined, MAX_MOONYANG2_CHARS)}")

    return blocks


def extract_aux_property_lines(evidence_item: Dict, max_items: int = MAX_AUX_PROP_ITEMS) -> List[str]:
    """Collect selected data_property entries that should be visible to the model."""
    lines: List[str] = []
    seen = set()

    for prop in evidence_item.get("data_property", []):
        name = str(prop.get("name", "")).strip()
        value = str(prop.get("value", "")).strip()
        if name not in AUX_PROP_NAMES or not value:
            continue

        line = f"{name}: {_shorten_text(value)}"
        if line in seen:
            continue

        seen.add(line)
        lines.append(line)

        if len(lines) >= max_items:
            break

    return lines




def parse_refdoc_fields(ref_text: str) -> Dict[str, str]:
    """
    reference_doc의 key:value 형식을 느슨하게 파싱한다.
    기존 refdoc에는 body 라벨 없이 후행 본문만 있는 경우가 있어
    그런 텍스트는 body로 흡수한다.
    """
    fields: Dict[str, str] = {}
    extras: List[str] = []
    current_key = None

    for raw_line in str(ref_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r"^(headword|type|definition|summary|body|source|matched_keyword)\s*:\s*(.*)$", line, flags=re.I)
        if m:
            key = m.group(1).lower()
            value = m.group(2).strip()
            if fields.get(key):
                fields[key] = f"{fields[key]} {value}".strip()
            else:
                fields[key] = value
            current_key = key
        else:
            if current_key == "body":
                fields["body"] = f"{fields.get('body', '')} {line}".strip()
            else:
                extras.append(line)

    if extras:
        extra_text = " ".join(extras).strip()
        if extra_text:
            if fields.get("body"):
                if normalize_for_rank(extra_text) not in normalize_for_rank(fields["body"]):
                    fields["body"] = f"{fields['body']} {extra_text}".strip()
            else:
                fields["body"] = extra_text

    return fields


def refdoc_same_meaning(a: str, b: str) -> bool:
    na = normalize_for_rank(a)
    nb = normalize_for_rank(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def compact_refdoc_for_prompt(ref_text: str) -> str:
    """
    prompt에 넣기 전에 summary 중복을 제거하고 lean 형태로 압축한다.
    """
    fields = parse_refdoc_fields(ref_text)

    headword = str(fields.get("headword", "")).strip()
    doc_type = str(fields.get("type", "")).strip()
    definition = str(fields.get("definition", "")).strip()
    summary = str(fields.get("summary", "")).strip()
    body = str(fields.get("body", "")).strip()
    source = str(fields.get("source", "")).strip()

    if not definition and summary:
        definition = summary

    if refdoc_same_meaning(body, definition) or refdoc_same_meaning(body, summary):
        body = ""

    if not source:
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", body)
        if m:
            source = m.group(1).strip()
            body = m.group(2).strip()

    pieces: List[str] = []
    if headword:
        pieces.append(f"headword: {headword}")
    if doc_type:
        pieces.append(f"type: {doc_type}")
    if definition:
        pieces.append(f"definition: {_shorten_text(definition, 220)}")
    if body and not refdoc_same_meaning(body, definition):
        pieces.append(f"body: {_shorten_text(body, 420)}")
    if source:
        pieces.append(f"source: {source}")

    return "\n".join(pieces).strip() or str(ref_text).strip()


def prepare_reference_docs_for_prompt(reference_doc) -> List[str]:
    """
    refdoc를 prompt용으로 정제:
    - summary 제거
    - 중복 headword/source 제거
    """
    if not reference_doc:
        return []

    refs = reference_doc if isinstance(reference_doc, list) else [str(reference_doc)]

    cleaned: List[str] = []
    seen = set()

    for ref in refs:
        if not ref:
            continue
        compacted = compact_refdoc_for_prompt(str(ref))
        fields = parse_refdoc_fields(compacted)
        headword = normalize_for_rank(fields.get("headword", ""))
        source = normalize_for_rank(fields.get("source", ""))
        key = (headword, source, normalize_for_rank(fields.get("definition", "")))

        if key in seen:
            continue
        seen.add(key)
        cleaned.append(compacted)

    return cleaned


# -------------------------
# 유물명 정규화 (학습 make_train_data.ipynb의 normalize_name과 동일)
#   - 한글+한자 병기는 한자를 제거, 순수 한자명은 음(reading)으로 변환
#   - hanja 패키지 없으면 음 변환은 비활성(원본 유지)되며, 그 경우 학습과 미세하게 달라짐
# -------------------------
try:
    import hanja as _hanja

    def to_eum(s: str) -> str:
        return _hanja.translate(str(s or ""), "substitution")

    HANJA_OK = True
except Exception:
    def to_eum(s: str) -> str:
        return str(s or "")

    HANJA_OK = False


def _has_hanja(s: str) -> bool:
    return bool(re.search(r"[㐀-鿿]", s or ""))


def clean_hanja_name(s: str) -> str:
    """유물명에서 한자를 제거해 한글 음만 남김(전부 한자면 원본 유지)."""
    s = str(s or "").strip()
    if not s:
        return s
    s2 = re.sub(r"[\(\（]\s*[㐀-鿿\s]+\s*[\)\）]", "", s)   # 괄호 안 한자 병기 제거
    s3 = re.sub(r"[㐀-鿿]+", "", s2)                        # 남은 한자 블록 제거
    s3 = re.sub(r"\s+", " ", s3).strip(" ,·-")
    return s3 if s3 else s


def normalize_name(s: str) -> str:
    """이름: 한글+한자 병기는 한자 제거, 순수 한자는 음(reading)으로 변환."""
    s = clean_hanja_name(s)
    if _has_hanja(s):
        s = to_eum(s)
    return s


def resolve_artifact_label(evidence_item: Dict) -> str:
    """DB의 '유물' 필드에서 표시용 유물명을 가져온다(한자→음 정규화 적용)."""
    v = evidence_item.get("유물")
    label = ""
    if isinstance(v, str) and v.strip():
        label = v.strip()
    elif isinstance(v, list):
        parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
        if parts:
            label = parts[0]
    return normalize_name(label) if label else ""


def _flatten_class(cls):
    """유물_분류(중첩 dict)를 ['문화예술','서화',...] 형태로 평탄화."""
    names = []
    cur = cls
    while isinstance(cur, dict):
        if cur.get("name"):
            names.append(str(cur["name"]))
        cur = cur.get("value")
    if isinstance(cur, str) and cur:
        names.append(cur)
    return names


def extract_evidence_text(evidence_item: Dict) -> str:
    """
    근거문서 텍스트 생성.
    학습 데이터(make_train_data.ipynb의 build_evidence_text)와 동일한 레이아웃으로 맞춰
    train/serve 불일치를 없앤다:
        [유물 설명] / [기본 정보] / [문양] / [의미] / [참고 문헌]
    - 유물명·소장품번호는 학습과 동일하게 근거문서에 넣지 않는다(유물명은 user 메시지로 전달).
    - reference_doc은 학습과 동일하게 '원문 그대로' 렌더한다(compact/rerank 미적용).
    """
    payload = evidence_item or {}
    parts: List[str] = []

    desc = payload.get("description", "")
    if isinstance(desc, list):
        desc = "\n".join(d for d in desc if d)
    desc = str(desc or "").strip()
    if desc:
        parts.append("[유물 설명]\n" + desc)

    info: List[str] = []
    for label, key in (("국적", "국적"), ("시대", "시대")):
        v = payload.get(key)
        if v:
            info.append(f"{label}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}")
    names = _flatten_class(payload.get("유물_분류"))
    if names:
        info.append("분류: " + " > ".join(names))
    mat = payload.get("재질")
    if isinstance(mat, list):
        ms = [m.get("name") for m in mat if isinstance(m, dict) and m.get("name")]
        if ms:
            info.append("재질: " + ", ".join(map(str, ms)))
    dp = payload.get("data_property")
    if isinstance(dp, list):
        for it in dp:
            if isinstance(it, dict) and it.get("name") in ("유물크기", "한자명") and it.get("value"):
                info.append(f"{it['name']}: {it['value']}")
    if info:
        parts.append("[기본 정보]\n" + "\n".join(info))

    pat: List[str] = []
    for k, v in payload.items():
        if k.startswith("문양") and v:
            if isinstance(v, list):
                pat.append(", ".join(
                    f"{x.get('name')}={x.get('value')}" if isinstance(x, dict) else str(x) for x in v))
            else:
                pat.append(str(v))
    if pat:
        parts.append("[문양]\n" + "\n".join(pat))

    mean: List[str] = []
    for k, v in payload.items():
        if k.startswith("의미") and v:
            mean.append(", ".join(map(str, v)) if isinstance(v, list) else str(v))
    if mean:
        parts.append("[의미]\n" + "\n".join(mean))

    rd = payload.get("reference_doc")
    if isinstance(rd, list) and rd:
        parts.append("[참고 문헌]\n" + "\n\n".join(str(r) for r in rd if r))
    elif isinstance(rd, str) and rd.strip():
        parts.append("[참고 문헌]\n" + rd.strip())

    return "\n\n".join(parts) if parts else "[근거문서 정보를 찾을 수 없습니다]"


# -------------------------
# reference_doc reranker
# -------------------------
def normalize_for_rank(text: str) -> str:
    """질문/문서를 정렬하기 쉽게 정규화한다."""
    text = str(text or "")
    text = re.sub(r"[「」『』《》()\[\]{}<>\"'`.,:;!?/\\|_+=~@#$%^&*-]", "", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def make_char_ngrams(text: str, ns=(2, 3)) -> Set[str]:
    """문자 2~3gram 집합 생성"""
    text = normalize_for_rank(text)
    grams: Set[str] = set()

    for n in ns:
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            grams.add(text[i:i + n])

    if not grams and text:
        grams.add(text)

    return grams


def overlap_coeff(a: Set[str], b: Set[str]) -> float:
    """짧은 질문에서 잘 동작하는 겹침 계수"""
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


# -------------------------
# 도슨트 테마 RAG 로드/검색/주입
# -------------------------
def load_docent_rag(path: str = DOCENT_RAG_FILE) -> List[Dict]:
    """도슨트 테마 RAG(JSON)를 로드하고 검색용 n-gram을 미리 만든다(1회)."""
    global docent_rag, docent_relic_ids
    if docent_rag:
        return docent_rag
    if not ENABLE_DOCENT_RAG:
        print("도슨트 RAG 비활성(ENABLE_DOCENT_RAG=0).")
        return []
    if not os.path.exists(path):
        print(f"⚠️ 도슨트 RAG 파일 없음: {path} (도슨트 주입 비활성)")
        return []
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ 도슨트 RAG 로드 실패: {e}")
        return []
    out: List[Dict] = []
    for r in (data if isinstance(data, list) else []):
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        idx_text = (str(r.get("embedding_text") or "") + " " + text).strip()
        rids = r.get("relic_ids") or ([r["relic_id"]] if r.get("relic_id") else [])
        rids = {str(x).strip() for x in rids if str(x).strip()}
        out.append({
            "text": text,
            "theme_title": str(r.get("theme_title") or "").strip(),
            "record_type": str(r.get("record_type") or "").strip(),
            "relic_ids": rids,
            "_grams": make_char_ngrams(idx_text),
        })
    docent_rag = out
    docent_relic_ids = set().union(*(o["relic_ids"] for o in out)) if out else set()
    print(f"도슨트 RAG 로드: {len(out)}개 문단, 유물 ID {len(docent_relic_ids)}개 ({path})")
    return out


def retrieve_docent(query: str, top_k: int = DOCENT_RAG_TOP_K, min_score: float = DOCENT_RAG_MIN_SCORE):
    """질문+유물명과 어휘 유사도(문자 n-gram 겹침)로 도슨트 문단 top-k 검색. [(score, rec)]."""
    if not docent_rag or not str(query or "").strip():
        return []
    qg = make_char_ngrams(query)
    scored = [(overlap_coeff(qg, r["_grams"]), r) for r in docent_rag]
    scored = [(s, r) for s, r in scored if s >= min_score]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(0, top_k)]


# 해설 요청 보일러플레이트("…에 대해 설명해 주세요" 류). 도슨트 문단 선택 쿼리에서 제거한다 —
# 이 문구만으로도 도슨트 문단과 n-gram 겹침이 임계치를 넘기 때문.
_DOCENT_BOILERPLATE_RE = re.compile(
    r"(?:에\s*대해(?:서)?\s*)?(?:자세히\s*|좀\s*)?(?:설명|소개|알려|말씀)\s*"
    r"(?:해\s*주세요|해\s*주실래요|해\s*주시겠어요|해\s*줘|주세요|부탁\s*드립니다|부탁해요|해\s*주십시오)\s*[.?!]*"
)
_DOCENT_QUOTE_CHARS = "'\"‘’“”「」『』《》〈〉"


def clean_question_for_docent(question: str, artifact: str) -> str:
    """도슨트 문단 선택용 질문 정제: 유물명(따옴표 포함)과 해설 보일러플레이트를 제거한 잔여 토큰."""
    q = str(question or "")
    a = str(artifact or "").strip()
    if a:
        q = q.replace(a, " ")
        q = q.replace(clean_hanja_name(a), " ")
    q = _DOCENT_BOILERPLATE_RE.sub(" ", q)
    q = q.translate({ord(c): " " for c in _DOCENT_QUOTE_CHARS})
    q = re.sub(r"\s+", " ", q).strip(" .,?!")
    return q if len(normalize_for_rank(q)) >= 2 else ""


def retrieve_docent_text(artifact: str, question: str, heritage_id: str = "") -> Tuple[str, int]:
    """검색된 도슨트 문단을 학습 rephrase task의 '[기존 해설 원고]' 입력으로 쓸 하나의
    텍스트로 합친다(글자 예산 내). (원고텍스트, 문단수) 반환.

    주입 게이트: 요청 유물 ID가 도슨트 원고의 relic_id에 있을 때만 주입한다.
    (유물명 n-gram 점수는 짧은 이름이 원고에 포함되기만 해도 1.0이 나와 게이트로 쓸 수 없음.)
    문단 선택: 유물명 + 정제된 질문(보일러플레이트·유물명 제거)으로 검색한 결과 중 해당 유물을 담은 문단만 쓰고,
    검색이 비면 해당 유물의 원고를 그대로 쓴다.
    """
    if not ENABLE_DOCENT_RAG or not docent_rag:
        return "", 0
    hid = str(heritage_id or "").strip()
    if not hid or hid not in docent_relic_ids:
        return "", 0
    artifact = str(artifact or "").strip()
    own = [(1.0, r) for r in docent_rag if hid in r["relic_ids"]]
    cleaned = clean_question_for_docent(question, artifact)
    hits = retrieve_docent(f"{artifact} {cleaned}".strip()) if (artifact or cleaned) else []
    hits = [h for h in hits if hid in h[1]["relic_ids"]] or own
    if not hits:
        return "", 0
    segs, used = [], 0
    for _, h in hits:
        seg = h["text"].strip()
        if used + len(seg) > DOCENT_RAG_MAX_CHARS:
            seg = seg[: max(0, DOCENT_RAG_MAX_CHARS - used)].rstrip()
        if not seg:
            break
        segs.append(seg)
        used += len(seg)
        if used >= DOCENT_RAG_MAX_CHARS:
            break
    return ("\n\n".join(segs), len(segs)) if segs else ("", 0)


def extract_ref_headword(ref_text: str) -> str:
    """reference_doc 내부의 headword: ... 줄 추출"""
    text = str(ref_text or "")
    m = re.search(r'(?im)^\s*headword\s*:\s*(.+?)\s*$', text)
    if not m:
        return ""
    return re.split(r"[;,/|]", m.group(1))[0].strip()


INTENT_BONUS_RULES = [
    (("기증", "기증자", "누가줬", "누가기증", "들어온경위", "소장경위"),
     ("기증", "기증자", "기증자명", "대표기증품", "기증동기", "기증유물"),
     0.35),

    (("문양", "무늬", "장식", "패턴"),
     ("문양", "무늬", "pattern", "ornament"),
     0.30),

    (("시대", "언제", "연대", "시기"),
     ("시대", "시기", "연대"),
     0.20),

    (("작가", "제작자", "누가만들", "만든사람"),
     ("작가", "제작", "장인", "제작자"),
     0.20),

    (("재질", "재료", "무엇으로", "무슨재료"),
     ("재질", "재료", "도자", "청동", "금속", "목재", "자기", "토기"),
     0.20),
]


def intent_bonus(question: str, ref_text: str) -> float:
    """질문 의도와 맞는 문서에 약한 보너스"""
    q = normalize_for_rank(question)
    r = normalize_for_rank(ref_text)

    bonus = 0.0
    for q_keys, r_keys, score in INTENT_BONUS_RULES:
        if any(normalize_for_rank(k) in q for k in q_keys) and any(normalize_for_rank(k) in r for k in r_keys):
            bonus += score

    return bonus


def shorten_reference_doc(ref_text: str, limit: int = MAX_SINGLE_REF_CHARS) -> str:
    """문서 1개가 너무 길면 앞부분 위주로 자른다."""
    ref_text = str(ref_text or "").strip()
    if len(ref_text) <= limit:
        return ref_text
    return ref_text[: limit - 1].rstrip() + "…"


def score_reference_doc(question: str, artifact: str, ref_text: str) -> float:
    """문서 1개 점수 계산"""
    ref_text = str(ref_text or "")

    q_norm = normalize_for_rank(question)
    a_norm = normalize_for_rank(artifact)
    r_norm = normalize_for_rank(ref_text)

    q_grams = make_char_ngrams(question)
    r_grams = make_char_ngrams(ref_text)

    ref_headword = extract_ref_headword(ref_text)
    h_norm = normalize_for_rank(ref_headword)
    h_grams = make_char_ngrams(ref_headword)

    score = 0.0

    # 질문과 문서 전체의 겹침
    score += 0.60 * overlap_coeff(q_grams, r_grams)

    # 질문과 문서 headword의 겹침
    if h_grams:
        score += 0.25 * overlap_coeff(q_grams, h_grams)

    # headword가 질문에 직접 등장하면 큰 가산점
    if h_norm and h_norm in q_norm:
        score += 0.35

    # 유물명 자체가 문서에 있으면 약한 가산점
    if a_norm and a_norm in r_norm:
        score += 0.10

    # 질문 의도 보정
    score += intent_bonus(question, ref_text)

    # 아주 긴 문서는 약하게 감점
    score -= min(len(ref_text) / 12000.0, 0.12)

    return score


def rerank_reference_docs(question: str, artifact: str, reference_doc, top_k: int = RERANK_TOP_K) -> List[str]:
    """reference_doc를 질문 기준으로 재정렬"""
    if not reference_doc:
        return []

    refs = prepare_reference_docs_for_prompt(reference_doc)

    scored = []
    for ref in refs:
        if not ref:
            continue
        ref_str = str(ref)
        score = score_reference_doc(question, artifact, ref_str)
        scored.append((score, ref_str))

    scored.sort(key=lambda x: x[0], reverse=True)

    if top_k and top_k > 0:
        scored = scored[:top_k]

    if DEBUG_RERANK:
        print("[RERANK DEBUG]")
        for idx, (score, text) in enumerate(scored, start=1):
            hw = extract_ref_headword(text)
            print(f"  {idx}. score={score:.4f}, headword={hw[:40]}")

    return [ref for _, ref in scored]


def trim_refs_by_token_budget(refs: List[str], tokenizer_obj=None, max_tokens: int = REF_DOC_MAX_TOKEN_BUDGET) -> List[str]:
    """정렬된 ref를 토큰 예산 안에서만 남긴다."""
    kept = []
    used = 0

    for ref in refs:
        ref = shorten_reference_doc(ref)
        if not ref:
            continue

        if tokenizer_obj is not None:
            try:
                tok_count = len(tokenizer_obj(ref, add_special_tokens=False)["input_ids"])
            except Exception:
                tok_count = max(1, int(len(ref) / 3))
        else:
            tok_count = max(1, int(len(ref) / 3))

        if kept and used + tok_count > max_tokens:
            continue

        kept.append(ref)
        used += tok_count

        if used >= max_tokens:
            break

    return kept


# 기증/기증자 질문에서의 과도 거절 완화용 추가 지시.
# 모델이 기증·소장경위를 거절 카테고리로 학습해, 참고 문헌(reference_doc)에 기증자 정보가
# 있어도 거절하는 경향이 있다. 기증 관련 질문일 때만 이 지시를 붙여, 참고 문헌에 실제로 있는
# 기증자 정보를 근거로 답하도록 유도한다. (정보가 없을 땐 여전히 지어내지 않음.)
DONOR_EXTRA_SYSTEM = (
    "근거문서의 '참고 문헌'에는 기증자 정보(기증자 이름·기증 시기·기증 규모·기증 동기·대표 기증품 등)가 "
    "포함될 수 있습니다. 참고 문헌도 근거문서의 일부입니다. 관람객이 기증이나 기증자에 대해 물으면, "
    "근거문서(참고 문헌 포함)에 그 정보가 있을 경우 반드시 그것을 근거로 답하세요. "
    "근거문서에 실제로 있는 내용을 '확인할 수 없다'고 거절하지 마세요. "
    "단, 근거문서에 정보가 없으면 지어내지 말고 없다고 하세요."
)

# 질문이 기증/기증자/소장 경위 관련인지
_DONOR_Q_KEYS = ("기증", "기증자", "기증한", "기증하신", "기증된", "소장 경위", "소장경위",
                 "들여왔", "들여온", "입수", "수집가", "기증품")


def _is_donor_question(question: str) -> bool:
    q = (question or "")
    return any(k in q for k in _DONOR_Q_KEYS)


def _extra_block(extra_system: str) -> str:
    extra = (extra_system or "").strip()
    # 학습 시스템 프롬프트엔 없던 서빙 전용 추가 지시. 비어 있으면 학습과 100% 동일.
    return f"\n\n추가 지시:\n{extra}" if extra else ""


def explanation_system(visitor_name: str, persona: str, evidence_text: str, extra_system: str = "") -> str:
    """해설 생성 task 시스템 프롬프트 (make_train_data_tts.ipynb explanation_system과 동일 —
    TTS 확장 모델용: 작업 선언 flag + {"답변","tts"} JSON 출력)."""
    return (f"당신은 {visitor_name}에게 맞춤형 박물관 해설을 제공하는 AI입니다. "
            f"지금 작업은 '해설 생성'입니다: 주어진 근거문서를 바탕으로 유물 전체를 소개하는 해설문을 해당 관람객 유형의 특성에 맞게 작성하세요. "
            f"응답은 먼저 \"답변\"(화면 표시용 해설)을, 그다음 \"tts\"(화면용을 조금 더 쉽고 간결하게 다듬은 음성 안내 대본)를 제시하는 JSON 형식으로 작성하세요. "
            f"근거문서에 없는 내용은 지어내지 마세요."
            f"{_extra_block(extra_system)}\n\n{persona}\n\n근거문서:\n{evidence_text}")


def qa_system(visitor_name: str, persona: str, evidence_text: str, extra_system: str = "") -> str:
    """질문 답변 task 시스템 프롬프트 (make_train_data_tts.ipynb qa_system과 동일 —
    TTS 확장 모델용: 작업 선언 flag + {"태그","답변","참고한 문단/문장","tts"} JSON 출력)."""
    return (f"당신은 {visitor_name}에게 맞춤형 박물관 가이드를 제공하는 AI입니다. "
            f"지금 작업은 '질의응답'입니다: 주어진 근거문서를 바탕으로 관람객의 질문에만 초점을 맞춰 답하세요. "
            f"질문이 '~에 대해 설명해 주세요' 형태라도 유물 전체 해설문이 아니라 질문 대상에 대한 답변만 합니다. "
            f"응답은 먼저 \"태그\"(답변이 다루는 내용의 분류 — 해당되는 것을 모두 담은 목록)를 제시하고, "
            f"그다음 \"답변\", \"참고한 문단/문장\", 마지막으로 \"tts\"(화면용을 조금 더 쉽고 간결하게 다듬은 음성 안내 대본)를 제시하는 JSON 형식으로 작성하세요. "
            f"근거문서에 없는 내용은 지어내지 마세요."
            f"{_extra_block(extra_system)}\n\n{persona}\n\n근거문서:\n{evidence_text}")


# 학습(make_train_data) rephrase task와 동일한 유형별 리라이팅 지시.
TYPE_CLAUSE = {
    "deep_inquiry": "분석적이고 학술적인 어조로 다시 쓰세요. 원본의 사실, 그리고 제공된 근거문서의 범위 안에서 인과·맥락·양식사적 의미를 더 깊이 연결해 설명해도 됩니다. 단, 원본이나 근거문서에 없는 내용은 추가하지 마세요.",
    "emotional_engagement": "간결하고 감성적인 어조로 다시 쓰세요. 어려운 한자어나 전문용어는 의미를 살린 쉬운 우리말로 풀어 표현하세요(정보 자체는 유지).",
    "high_engagement": "사실·분석·감상을 통합하되, 관람객에게 질문을 던지고 참여를 유도하는 어조로 다시 쓰세요.",
    "information_centric": "전문용어를 피하고 짧고 명료한 문장으로, 핵심 정보 위주로 간결하게 다시 쓰세요.",
}


def rephrase_system(visitor_key: str, visitor_name: str, persona: str, original: str,
                    evidence_text: str = "", extra_system: str = "") -> str:
    """도슨트 원고 리라이팅 task 시스템 프롬프트 (make_train_data.ipynb rephrase_system_train과 동일).
    extra_system 이 비면 학습과 100% 동일."""
    base = (f"당신은 {visitor_name}에게 맞춤형 박물관 해설을 제공하는 AI입니다. "
            f"지금 작업은 '기증관 해설'입니다: 기증관의 기획된 해설 원고를 바탕으로 안내하는 작업으로, "
            f"근거문서만으로 유물을 소개하는 일반 '해설 생성'과는 다른 작업입니다. "
            f"아래 [기존 해설 원고]를 {visitor_name}의 특성에 맞는 문체로 다시 써 설명하세요. "
            f"기존 원고의 사실·수치·인용·등장 요소와 구성 순서는 유지하세요. "
            f"{TYPE_CLAUSE.get(visitor_key, '')} "
            f"기존 원고 또는 [근거문서]에서 확인되지 않는 내용은 지어내지 마세요. "
            f"‘원문’, ‘원본’, ‘근거문서’, ‘자료에 따르면’ 같은 메타 표현은 쓰지 말고, "
            f"응답은 먼저 \"답변\"(화면 표시용 해설 본문)을, 그다음 \"tts\"(화면용을 조금 더 쉽고 간결하게 다듬은 음성 안내 대본)를 제시하는 JSON 형식으로 출력하세요."
            f"{_extra_block(extra_system)}\n\n"
            f"[기존 해설 원고]\n{original}")
    if evidence_text:
        base += f"\n\n[근거문서]\n{evidence_text}"
    return base + f"\n\n{persona}"


# -------------------------------------------------
# 기증관 해설(rephrase) 인사말·맺음말 후처리
# -------------------------------------------------
# 기증관 해설은 주제 순서대로 기획된 투어라 첫 해설에 인사말, 마지막 해설에 맺음말을
# 붙인다. '투어 내 위치'는 클라이언트만 아는 정보이므로 모델 생성이 아닌 요청 파라미터
# (?tour_pos=first|mid|last|single) 기반 후처리. 유물명 기반 결정적 선택(같은 유물=같은 문구).
TOUR_GREETINGS = [
    "안녕하세요, 오늘 기증관 관람을 함께할 안내를 맡았습니다.",
    "안녕하세요, 지금부터 기증관 이야기를 함께 살펴보겠습니다.",
    "반갑습니다. 오늘 기증관의 유물들을 차례로 소개해 드리겠습니다.",
]
TOUR_CLOSINGS = [
    "오늘 준비한 기증관 이야기는 여기까지입니다. 함께해 주셔서 감사합니다.",
    "이것으로 기증관 해설을 마치겠습니다. 즐거운 관람 되시기 바랍니다.",
    "오늘 기증관 관람에 함께해 주셔서 감사합니다. 남은 관람도 즐겁게 이어 가시길 바랍니다.",
]

def apply_tour_frame(text: str, tour_pos: str, artifact: str = "") -> str:
    """tour_pos: first=인사말, last=맺음말, single=둘 다, mid/없음=그대로."""
    pos = (tour_pos or "").strip().lower()
    if not text or pos not in ("first", "last", "single"):
        return text
    idx = sum(ord(c) for c in artifact) if artifact else 0
    out = text
    if pos in ("first", "single"):
        out = f"{TOUR_GREETINGS[idx % len(TOUR_GREETINGS)]} {out}"
    if pos in ("last", "single"):
        out = f"{out} {TOUR_CLOSINGS[idx % len(TOUR_CLOSINGS)]}"
    return out


def is_explanation_request(question: str) -> bool:
    """비어 있거나 '...에 대해 설명해 주세요' 류면 해설(explanation) task로 본다."""
    q = (question or "").strip()
    if not q:
        return True
    norm = re.sub(r"\s+", "", q)
    return norm.endswith("설명해주세요") or norm.endswith("설명해주실래요") or norm.endswith("설명부탁드립니다")


def format_messages(
    visitor_type: str,
    artifact: str,
    question: str,
    evidence_text: str,
    task: str = "qa",
    extra_system: str = "",
    original: str = "",
) -> List[Dict]:
    """학습 데이터 포맷에 맞춘 messages 구성.
    - task='rephrase'   : user = "'{유물명}'에 대해 설명해 주세요.", system = rephrase_system([기존 해설 원고]=도슨트)
    - task='explanation': user = "'{유물명}'에 대해 설명해 주세요.", system = explanation_system
    - task='qa'         : user = 관람객 질문,                         system = qa_system
    """
    visitor_key = VISITOR_TYPE_MAP.get(visitor_type, DEFAULT_VISITOR_KEY)
    visitor_name = VISITOR_TYPE_NAMES.get(visitor_key, visitor_type)
    persona = visitor_prompts.get(visitor_key, "")

    if task == "rephrase":
        system_content = rephrase_system(visitor_key, visitor_name, persona, original, evidence_text, extra_system)
        user_content = f"'{artifact}'에 대해 설명해 주세요."
    elif task == "explanation":
        system_content = explanation_system(visitor_name, persona, evidence_text, extra_system)
        user_content = f"'{artifact}'에 대해 설명해 주세요."
    else:
        system_content = qa_system(visitor_name, persona, evidence_text, extra_system)
        user_content = question

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def format_messages_for_tokenizer(messages: List[Dict]) -> str:
    """apply_chat_template를 쓸 수 없을 때의 ChatML 폴백.
    모델 chat_template.jinja와 동일하게 렌더한다:
      - 첫 메시지가 user/system이면 맨 앞에 <bos>
      - 각 턴: <|im_start|>{role}\n{content}<|im_end|>\n
      - 마지막에 assistant 생성 프롬프트
    """
    formatted = ""
    if messages and messages[0].get("role") in ("user", "system"):
        formatted += "<bos>"
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def render_prompt_text(messages: List[Dict]) -> str:
    """messages -> 모델 입력 프롬프트 문자열.
    HF 토크나이저가 로드돼 있으면 apply_chat_template(모델 chat_template.jinja)을,
    없거나 실패하면 동일 포맷의 ChatML 폴백을 쓴다. (구버전 transformers에서
    gemma4 토크나이저 로드가 실패해도 서빙이 계속되도록 하기 위함.)
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"apply_chat_template 실패, ChatML 폴백 사용: {e}")
    return format_messages_for_tokenizer(messages)


def count_prompt_tokens(text: str) -> int:
    """프롬프트 토큰 수 추정. 토크나이저가 있으면 정확히, 없으면 문자수 기반 근사."""
    if tokenizer is not None:
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    # 한국어는 보통 토큰당 1~3자. 길이 가드용으로 보수적으로 추정.
    return max(1, int(len(text) / 2))

def estimate_reference_token_budget(
    visitor_type: str,
    artifact: str,
    question: str,
    evidence_item: Dict,
    tokenizer_obj=None,
    extra_system: str = "",
) -> int:
    if tokenizer_obj is None:
        return REF_DOC_MAX_TOKEN_BUDGET

    base_item = copy.deepcopy(evidence_item)
    base_item["reference_doc"] = []
    base_evidence_text = extract_evidence_text(base_item)

    base_messages = format_messages(
        visitor_type=visitor_type,
        artifact=artifact,
        question=question,
        evidence_text=base_evidence_text,
        task="qa",
        extra_system=extra_system,
    )

    try:
        if hasattr(tokenizer_obj, "apply_chat_template"):
            base_input_text = tokenizer_obj.apply_chat_template(
                base_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            base_input_text = format_messages_for_tokenizer(base_messages)

        base_tokens = len(tokenizer_obj(base_input_text, add_special_tokens=False)["input_ids"])
    except Exception:
        base_tokens = max(1, int(len(str(base_messages)) / 3))

    budget = MAX_INPUT_TOKENS - base_tokens - REF_DOC_RESERVE_TOKENS
    budget = max(REF_DOC_MIN_TOKEN_BUDGET, budget)
    budget = min(REF_DOC_MAX_TOKEN_BUDGET, budget)
    return budget


def rerank_and_trim_reference_docs(
    evidence_item: Dict,
    visitor_type: str,
    artifact: str,
    question: str,
    tokenizer_obj=None,
    extra_system: str = "",
) -> Tuple[Dict, Dict[str, Any]]:
    """
    질문 기준으로 reference_doc를 재정렬하고,
    현재 prompt 길이에 맞는 예산 안에서만 남긴 새 evidence_item을 반환한다.

    반환 튜플의 두 번째 값은 추적용 메타:
      - 정제_후_원본_목록: dedupe·compact 후 전체
      - 재정렬_상위_k: 질문 기준 상위 RERANK_TOP_K (재정렬 비활성 시 원본과 동일)
      - 모델_투입_목록: 토큰 예산으로 잘린 최종 목록
      - 토큰_예산: reference_doc에 쓴 예산(토큰)
    """
    reference_doc = evidence_item.get("reference_doc", [])
    if not reference_doc:
        empty_meta: Dict[str, Any] = {
            "정제_후_원본_목록": [],
            "재정렬_상위_k": [],
            "모델_투입_목록": [],
            "토큰_예산": None,
        }
        return evidence_item, empty_meta

    prepared_original = prepare_reference_docs_for_prompt(reference_doc)

    ranked_refs = rerank_reference_docs(
        question=question,
        artifact=artifact,
        reference_doc=reference_doc,
        top_k=RERANK_TOP_K,
    )

    budget = estimate_reference_token_budget(
        visitor_type=visitor_type,
        artifact=artifact,
        question=question,
        evidence_item=evidence_item,
        tokenizer_obj=tokenizer_obj,
        extra_system=extra_system,
    )

    trimmed_refs = trim_refs_by_token_budget(
        ranked_refs,
        tokenizer_obj=tokenizer_obj,
        max_tokens=budget,
    )

    new_item = copy.deepcopy(evidence_item)
    new_item["reference_doc"] = trimmed_refs

    meta: Dict[str, Any] = {
        "정제_후_원본_목록": prepared_original,
        "재정렬_상위_k": ranked_refs,
        "모델_투입_목록": trimmed_refs,
        "토큰_예산": budget,
    }

    if DEBUG_RERANK:
        original_count = len(reference_doc) if isinstance(reference_doc, list) else 1
        print(
            f"[RERANK SUMMARY] original_refs={original_count}, "
            f"used_refs={len(trimmed_refs)}, budget={budget}"
        )

    return new_item, meta


def reference_doc_trace_without_rerank(evidence_item: Dict) -> Dict[str, Any]:
    """reference_rerank가 꺼져 있을 때: 정제된 목록이 그대로 모델에 들어간다고 가정한 추적 메타."""
    prepared = prepare_reference_docs_for_prompt(evidence_item.get("reference_doc", []))
    return {
        "정제_후_원본_목록": prepared,
        "재정렬_상위_k": prepared,
        "모델_투입_목록": prepared,
        "토큰_예산": None,
        "reference_rerank_비활성": True,
    }


def _normalize_tags(raw_tag) -> List[str]:
    """태그를 문자열 리스트로 정규화(쉼표 문자열도 수용)."""
    if isinstance(raw_tag, str):
        raw_tag = [t.strip() for t in raw_tag.split(",")]
    if not isinstance(raw_tag, list):
        return []
    return [str(t).strip() for t in raw_tag if str(t).strip()]


def parse_model_answer(raw: str) -> Optional[Tuple[str, List[str], List[str], str]]:
    """
    모델 출력(JSON 문자열)에서 (답변, 참고한 문단/문장, 태그, tts) 추출.
    - TTS 확장 모델: qa={"태그","답변","참고한 문단/문장","tts"}, 해설/기증관={"답변","tts"}.
    - 구형 출력({"답변"(,"참고한 문단/문장")})도 그대로 처리(태그 []/tts "" 반환).
    - 코드펜스(```json ... ```) 제거 후 json.loads 시도, 실패 시 정규식 폴백.
    - '답변' 키를 찾지 못하면 None.
    """
    s = str(raw or "").strip()
    if not s:
        return None

    # ```json ... ``` 펜스 제거
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "답변" in obj:
            refs = obj.get("참고한 문단/문장", []) or []
            if not isinstance(refs, list):
                refs = [str(refs)]
            return (str(obj.get("답변", "")), [str(r) for r in refs],
                    _normalize_tags(obj.get("태그", [])), str(obj.get("tts", "") or ""))
    except Exception:
        pass

    # 정규식 폴백: "답변": "....." (이스케이프 포함; tts 누락·절단 출력도 답변은 살린다)
    m = re.search(r'"답변"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.DOTALL)
    if m:
        try:
            answer = json.loads('"' + m.group(1) + '"')
        except Exception:
            answer = m.group(1).replace('\\"', '"').replace('\\n', '\n')
        tags = []
        mt = re.search(r'"태그"\s*:\s*\[(.*?)\]', s, re.DOTALL)
        if mt:
            tags = _normalize_tags(re.findall(r'"((?:[^"\\]|\\.)*)"', mt.group(1)))
        tts = ""
        mv = re.search(r'"tts"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.DOTALL)
        if mv:
            try:
                tts = json.loads('"' + mv.group(1) + '"')
            except Exception:
                tts = mv.group(1)
        return answer, [], tags, tts

    return None


_REFUSAL_TAG = "확인불가"
_GROUNDING_STOPWORDS = {
    "유물", "작품", "이것", "그것", "저것", "무엇", "무슨", "어떤", "어디", "언제", "누가", "누구", "어떻게", "얼마",
    "대해", "대해서", "설명", "소개", "질문", "내용", "부분", "관련", "정도", "경우", "그리고", "그런데", "혹시",
    "있나요", "인가요", "입니까", "있습니까", "일까요", "주세요", "주실래요", "알려", "말씀", "해주세요",
}
_JOSA_SUFFIXES = ("에서는", "으로는", "에게는", "에서", "으로", "에게", "까지", "부터", "이나", "처럼", "보다", "마다", "조차", "라도",
                  "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만", "랑", "나")


def _strip_josa(tok: str) -> str:
    """어절 끝 조사를 벗긴다(긴 조사 우선). 어간이 2글자 미만이 되면 원형 유지."""
    for suf in _JOSA_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 2:
            return tok[: -len(suf)]
    return tok


def _question_grounded_in_evidence(question: str, evidence_text: str, artifact: str = "") -> bool:
    """질문의 핵심어(조사 제거한 어간, 2글자 이상 한글)가 근거문서에 실제로 있는지.
    유물명에 포함된 단어(와 그 조사 결합형)는 제외 — 유물명 언급만으로 근거 있음으로 보지 않기 위함."""
    ev = str(evidence_text or "")
    if not ev:
        return False
    art_words = set(re.findall(r"[가-힣]{2,}", str(artifact or "")))
    for tok in re.findall(r"[가-힣]{2,}", str(question or "")):
        stem = _strip_josa(tok)
        if len(stem) < 2 or stem in _GROUNDING_STOPWORDS or stem in art_words:
            continue
        if any(tok.startswith(w) for w in art_words):
            continue
        if stem in ev:
            return True
    return False


def generate_response(
    visitor_type: str,
    artifact: str,
    question: str,
    evidence_text: str,
    task: str = "qa",
    extra_system: str = "",
    original: str = "",
    seed: Optional[int] = None,
) -> str:
    global tokenizer

    messages = format_messages(
        visitor_type, artifact, question, evidence_text,
        task=task, extra_system=extra_system, original=original,
    )

    try:
        print(">>> [1] chat template 시작")
        # 토크나이저가 없거나 로드 실패했어도 ChatML 폴백으로 렌더(토크나이저 불필요).
        prompt_text = render_prompt_text(messages)
        print(">>> [1] chat template 완료")

        print(">>> [2] tokenizing 시작")
        prompt_token_count = count_prompt_tokens(prompt_text)
        print(">>> [2] tokenizing 완료")
        print(">>> prompt token count:", prompt_token_count)

        if prompt_token_count >= VLLM_MAX_MODEL_LEN:
            raise ValueError(
                f"Prompt is too long for vLLM max_model_len: prompt_tokens={prompt_token_count}, max_model_len={VLLM_MAX_MODEL_LEN}"
            )

        print(">>> [3] vLLM HTTP completions 시작")
        tic = time.time()
        # 평가와 동일 샘플링(temperature 0.7 / top_p 0.95 / top_k 64 / seed 42) 사용
        decoded, err = vllm_completions(prompt_text, max_tokens=MAX_NEW_TOKENS, seed=seed if seed is not None else GEN_SEED)
        toc = time.time()
        print(">>> [3] vLLM HTTP completions 완료")
        print(f">>> generate 시간: {toc - tic} seconds")

        if err:
            print(f"Generation error: {err}")
            return f"응답 생성 중 오류가 발생했습니다: {err}"

        if not decoded:
            return ""

        print(">>> output text length:", len(decoded))
        return decoded.strip()

    except Exception as e:
        print(f"Generation error: {e}")
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    vllm_ok = vllm_server_healthy()
    return jsonify({
        # 토크나이저는 선택사항(없으면 ChatML 폴백). vLLM 서버만 정상이면 서비스 가능.
        "status": "healthy" if vllm_ok else "degraded",
        "tokenizer_loaded": tokenizer is not None,
        "vllm_server_url": VLLM_SERVER_URL,
        "vllm_server_healthy": vllm_ok,
    })


@app.route('/generate/<heritage_id>/<visitor_type>/<path:question>', methods=['GET'])
def generate_explanation(heritage_id, visitor_type, question):
    """Generate cultural heritage explanation API"""
    try:
        cultural_heritage_id = unquote(heritage_id)
        visitor_type = unquote(visitor_type)
        question = unquote(question)

        if not cultural_heritage_id:
            return jsonify({
                "error": "문화유산_ID가 필요합니다.",
                "status": "error"
            }), 400

        # 관람객 유형 정규화: 옛 기본값 '탐구형' 등 미인식 값은 거부하지 않고
        # 새 기본값(고강도 참여형)으로 받는다.
        visitor_type = resolve_visitor_key(visitor_type)

        global _tokenizer_tried
        if tokenizer is None and not _tokenizer_tried:
            _tokenizer_tried = True
            try:
                load_tokenizer()
            except Exception as e:
                # 구버전 transformers 등으로 gemma4 토크나이저 로드가 실패해도
                # ChatML 폴백(render_prompt_text)으로 계속 진행한다. 503을 내지 않는다.
                print(f"토크나이저 로드 실패 — ChatML 폴백으로 진행: {e}")

        if not vllm_server_healthy():
            return jsonify({
                "error": (
                    f"vLLM 서버에 연결할 수 없습니다 ({VLLM_SERVER_URL}). "
                    "GPU 프로세스에서 `vllm serve`가 떠 있는지, VLLM_SERVER_URL·VLLM_SERVED_MODEL_NAME을 확인하세요."
                ),
                "status": "error"
            }), 503

        original_item = evidence_dict.get(str(cultural_heritage_id))
        evidence_item = copy.deepcopy(original_item) if original_item else None

        if not evidence_item:
            try:
                BASE_DETAIL_URL = "https://platform.aimuse.kr/api/data/{}"
                response = requests.get(BASE_DETAIL_URL.format(cultural_heritage_id), timeout=5)
                if response.status_code == 200:
                    detail_data = response.json()
                    evidence_item = {
                        "소장품번호": detail_data.get("#소장품번호", ""),
                        "id": detail_data.get("#소장품번호", ""),
                        "description": detail_data.get("description", ""),
                        "유물": detail_data.get("유물", ""),
                        "시대": detail_data.get("시대", ""),
                        "유물_분류": detail_data.get("유물_분류", {}),
                        "작가": detail_data.get("작가", []),
                        "data_property": detail_data.get("data_property", [])
                    }
                    for key in EXTRA_AIMUSE_KEYS:
                        if key in detail_data:
                            evidence_item[key] = detail_data[key]
                else:
                    return jsonify({
                        "error": f"문화유산 ID '{cultural_heritage_id}'에 해당하는 정보를 찾을 수 없습니다.",
                        "status": "error"
                    }), 404
            except Exception as e:
                return jsonify({
                    "error": f"문화유산 ID '{cultural_heritage_id}'에 해당하는 정보를 찾을 수 없습니다. (API 오류: {str(e)})",
                    "status": "error"
                }), 404

        if not evidence_item.get('headword_doc') and not evidence_item.get('reference_doc'):
            evidence_item['headword_doc'] = generate_headword_doc(evidence_item)
            evidence_item['reference_doc'] = generate_reference_doc(evidence_item)

        artifact = resolve_artifact_label(evidence_item) or "알 수 없는 유물"

        extra_system = os.getenv("EXTRA_SYSTEM_PROMPT", "").strip()
        if request.args.get("extra_system_prompt") is not None:
            extra_system = (request.args.get("extra_system_prompt") or "").strip()
        # 기증 관련 질문이면(명시 추가지시가 없을 때) 참고문헌 활용 지시를 자동 적용해 과도 거절 완화.
        # 비기증 질문은 extra_system 이 비어 학습과 100% 동일하게 유지된다.
        if not extra_system and _is_donor_question(question):
            extra_system = DONOR_EXTRA_SYSTEM

        # 핵심 추가 부분:
        # 질문(question)과 유물명(artifact)을 기준으로 reference_doc를 재정렬한 뒤,
        # 현재 prompt 길이에 맞게 일부만 남겨서 inference에 넣는다.
        ref_trace: Dict[str, Any] = {}
        if ENABLE_REFERENCE_RERANK and evidence_item.get("reference_doc"):
            evidence_item, ref_trace = rerank_and_trim_reference_docs(
                evidence_item=evidence_item,
                visitor_type=visitor_type,
                artifact=artifact,
                question=question,
                tokenizer_obj=tokenizer,
                extra_system=extra_system,
            )
        elif evidence_item.get("reference_doc"):
            ref_trace = reference_doc_trace_without_rerank(evidence_item)

        # task 판별: ?task= flag 가 있으면 그대로 따른다(결정적 라우팅).
        #   qa | explanation | rephrase(기증관 해설 — curation/docent 별칭 허용)
        # flag 가 없으면 기존 방식: '...설명해 주세요' 류/빈 질문 -> explanation, 그 외 -> qa
        task_override = (request.args.get("task") or "").strip().lower()
        if task_override in ("curation", "docent", "기증관"):
            task_override = "rephrase"
        if task_override in ("explanation", "qa", "rephrase"):
            task = task_override
        else:
            task_override = ""
            task = "explanation" if is_explanation_request(question) else "qa"

        evidence_text = extract_evidence_text(evidence_item)
        # 도슨트 테마 RAG (기존 동작 완전 유지 + rephrase flag 추가):
        # - explanation(지정 여부 무관): 기존처럼 관련 원고가 검색되면 rephrase(기증관)로 자동 전환
        # - ?task=rephrase 강제: 원고 검색, 없으면 explanation 으로 폴백(응답 task 필드로 확인)
        # - qa: 전환 없음(기존과 동일 — 학습에서 도슨트 미사용)
        docent_n = 0
        rephrase_original = ""
        if task in ("explanation", "rephrase"):
            rephrase_original, docent_n = retrieve_docent_text(artifact, question, cultural_heritage_id)
            if rephrase_original:
                task = "rephrase"
            elif task == "rephrase":
                task = "explanation"
        raw_output = generate_response(
            visitor_type, artifact, question, evidence_text,
            task=task, extra_system=extra_system, original=rephrase_original,
        )

        # 모델 출력(TTS 확장): qa={"태그","답변","참고한 문단/문장","tts"}, 해설/기증관={"답변","tts"}.
        response_text = raw_output
        ref_sentences: List[str] = []
        answer_tags: List[str] = []
        tts_text = ""
        parsed_answer = parse_model_answer(raw_output)
        if parsed_answer is not None:
            response_text, ref_sentences, answer_tags, tts_text = parsed_answer
        elif raw_output.strip().startswith("{"):
            print("Warning: JSON-like output without '답변' key")
            response_text = "죄송합니다. 적절한 답변을 생성할 수 없습니다."

        # qa 과잉거절 완화: 근거문서에 질문 핵심어가 있는데 '확인불가'면 seed를 바꿔 재생성(REFUSAL_RETRY_MAX회까지)
        if (task == "qa" and REFUSAL_RETRY_MAX > 0 and _REFUSAL_TAG in answer_tags
                and _question_grounded_in_evidence(question, evidence_text, artifact)):
            adopted_seed = None
            for k in range(1, REFUSAL_RETRY_MAX + 1):
                retry_raw = generate_response(
                    visitor_type, artifact, question, evidence_text,
                    task=task, extra_system=extra_system, original=rephrase_original,
                    seed=GEN_SEED + k,
                )
                retry_parsed = parse_model_answer(retry_raw)
                if retry_parsed and _REFUSAL_TAG not in retry_parsed[2] and retry_parsed[0].strip():
                    response_text, ref_sentences, answer_tags, tts_text = retry_parsed
                    adopted_seed = GEN_SEED + k
                    break
            print(f"[REFUSAL_RETRY] id={cultural_heritage_id} type={visitor_type} q={question!r} "
                  f"tries={k} adopted_seed={adopted_seed} tags={answer_tags}")

        # 기증관 해설(rephrase)일 때만 투어 위치에 따라 인사말/맺음말 후처리 (?tour_pos=first|mid|last|single)
        if task == "rephrase":
            response_text = apply_tour_frame(response_text, request.args.get("tour_pos"), artifact)
            tts_text = apply_tour_frame(tts_text, request.args.get("tour_pos"), artifact)

        ref_in_prompt = len(evidence_item.get("reference_doc", [])) if isinstance(
            evidence_item.get("reference_doc"), list
        ) else (1 if evidence_item.get("reference_doc") else 0)

        evidence_summary = {
            "description": evidence_item.get("description", ""),
            "headword_doc_존재": bool(evidence_item.get("headword_doc")),
            "reference_doc_개수_모델_투입": ref_in_prompt,
            "도슨트원고_문단수": docent_n,        # >0 이면 rephrase 모드로 진행
        }
        if ref_trace:
            evidence_summary["reference_doc_개수_정제_후_원본"] = len(ref_trace.get("정제_후_원본_목록", []))
            evidence_summary["reference_doc_개수_재정렬_상위_k"] = len(ref_trace.get("재정렬_상위_k", []))
            if ref_trace.get("토큰_예산") is not None:
                evidence_summary["reference_doc_토큰_예산"] = ref_trace["토큰_예산"]

        result: Dict[str, Any] = {
            "문화유산_ID": cultural_heritage_id,
            "관람객_유형": VISITOR_TYPE_MAP.get(visitor_type, visitor_type),
            "task": task,
            "유물명": artifact,
            "질문": question,
            "태그": answer_tags,
            "답변": response_text,
            "tts": tts_text,
            "참고한 문단/문장": ref_sentences,
            "근거문서_요약": evidence_summary,
            "status": "success",
        }
        if INCLUDE_REFERENCE_DOC_TRACE and ref_trace:
            result["근거문서_추적"] = ref_trace

        return jsonify(result), 200

    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            "error": f"서버 오류가 발생했습니다: {str(e)}",
            "status": "error"
        }), 500


if __name__ == '__main__':
    log_file_path = setup_realtime_file_logging()
    print(f"실시간 로그 파일: {log_file_path}")
    print("=" * 60)
    print("문화유산 해설 생성 REST API 시작")
    print("=" * 60)

    load_evidence_data(EVIDENCE_FILE)
    load_visitor_prompts(PERSONA_PROMPTS_DIR)
    load_docent_rag(DOCENT_RAG_FILE)

    try:
        load_tokenizer()
    except Exception as e:
        print(f"토크나이저 로드 실패: {e}")
        print("API는 토크나이저 없이 시작됩니다. /generate 엔드포인트는 동작하지 않을 수 있습니다.")

    if not vllm_server_healthy():
        print(
            f"경고: vLLM 서버({VLLM_SERVER_URL}) /health 응답 없음. "
            "별도 터미널에서 GPU로 `vllm serve`를 실행한 뒤 /generate를 사용하세요."
        )

    if not HANJA_OK:
        print("경고: 'hanja' 패키지가 없어 유물명 한자→음(reading) 변환이 비활성화됩니다. "
              "학습 데이터와 정확히 맞추려면 `pip install hanja` 후 재시작하세요.")

    print("=" * 60)
    print("API 준비 완료")
    print("=" * 60)
    print("\n환경 변수: MODEL_PATH, PERSONA_PROMPTS_DIR, VLLM_SERVER_URL, VLLM_SERVED_MODEL_NAME,")
    print("          VLLM_HTTP_TIMEOUT_SEC, VLLM_MAX_MODEL_LEN, VLLM_STOP, ENABLE_REFERENCE_RERANK")
    print("\n관람객 유형(visitor_type): " + ", ".join(VISITOR_PROMPT_FILES.keys()))
    print("\nAPI 엔드포인트:")
    print("  - GET  /health     : 헬스 체크")
    print("  - GET  /generate/<heritage_id>/<visitor_type>/<question>   : 해설/질문답변 생성")
    print("       (질문이 '...설명해 주세요' 류면 해설, 그 외엔 QA. ?task=explanation|qa 로 강제 지정 가능)")
    print("\n예시 요청:")
    print("""
    # 해설:  curl http://localhost:5000/generate/동원1257/information_centric/이%20유물에%20대해%20설명해%20주세요
    # QA:    curl http://localhost:5000/generate/동원1257/high_engagement/이건%20어디에%20쓰던%20물건인가요
    """)
    print("=" * 60)

    app.run(host=os.getenv("APP_HOST", "127.0.0.1"), port=int(os.getenv("APP_PORT", "5000")), debug=False,
            threaded=os.getenv("APP_THREADED", "0").strip() == "1")
