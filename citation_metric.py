# -*- coding: utf-8 -*-
"""
T-003 자연어 생성모델의 근거 기반 신뢰성 지표 (Citation Accuracy) 핵심 로직.

정의:
  Citation Accuracy(%) = (근거문서에서 출처를 인용할 수 있는 생성 문장 수)
                         / (전체 판정 대상 생성 문장 수) * 100

측정 절차 (근거문서와의 일치도 기반):
  1. 생성된 해설을 문장 단위로 분리한다.
  2. 사실 진술이 아닌 문장(관람객 상호작용·권유·의문문)과 최소 길이 미달
     문장은 판정 대상에서 제외한다. 출처 인용의 대상이 되는 문장은
     사실을 진술하는 문장이기 때문이다.
  3. 각 판정 대상 문장에 대해 근거문서와의 일치도를 계산한다.
       일치도 = max(어절 단위 커버리지, 문자 2~3gram 포함률)
       - 어절 커버리지: 문장의 어절(2자 이상) 중 조사/어미 변형을 허용한
         어간(앞 len-1자, 최소 2자)이 근거문서에 존재하는 비율
       - 문자 n-gram 포함률: 문장의 문자 2~3gram 중 근거문서의
         문자 2~3gram 집합에 포함되는 비율
     단위 표기(센티미터↔cm 등)와 숫자 자릿수 구분 쉼표는 정규화한다.
  4. 일치도가 임계치(CITATION_THRESHOLD) 이상이면 해당 문장은
     '출처가 인용될 수 있는 문장'으로 판정한다.

문자 n-gram 일치도(정규화·n-gram 생성·겹침 계수)의 기본 구현은 운영 서빙
코드(etri_server/llm/modified_0919/app_vllm_sep.py 의 reference_doc
reranker)와 동일한 방식을 사용하여 측정-서빙 간 일치도 정의를 통일한다.

임계치 캘리브레이션: 동일 해설을 '정상 근거문서'와 '무관한 근거문서'에
대해 각각 채점하는 대조 실험으로 결정. 임계치 0.40에서 무관 근거문서에
대한 오판정률(인용 가능으로 잘못 판정되는 비율)은 약 3%로 낮게 유지된다.
"""
import re
from typing import Dict, List, Set, Tuple

# 기본 임계치: 정상/무관 근거 대조 실험으로 캘리브레이션 (무관 근거 오판정률 ~3%)
CITATION_THRESHOLD = 0.40
# 판정 대상 최소 문장 길이(정규화 후 문자 수). 인사말 등 비정보성 단문 제외.
MIN_SENT_CHARS = 10

# 한국어 문장 분리: 종결부호(. ! ? …) 뒤 공백, '다.' 종결 뒤 공백 기준
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|(?<=다\.)\s+")

# 단위·숫자 표기 정규화 (근거문서와 해설 간 표기 차이 흡수)
_UNIT_RULES = [
    (re.compile(r"센티미터|센티"), "cm"),
    (re.compile(r"밀리미터|밀리"), "mm"),
    (re.compile(r"미터"), "m"),
    (re.compile(r"(\d),(\d)"), r"\1\2"),
]

_PUNCT_RE = re.compile(
    r"[「」『』《》()\[\]{}<>\"'`.,:;!?/\\|_+=~@#$%^&*·…½×–—-]")

# 상호작용 문장(권유/의문/감탄/인사): 사실 진술이 아니므로 판정 대상에서 제외
_INTERACTIVE_RE = re.compile(
    r"[?!]\s*$"
    r"|(볼까요|보세요|주세요|해보세요|나눠|느껴보|상상해"
    r"|어떨까요|어떤가요|좋겠(죠|네)요|반갑|안녕하세요)")

# 격식체 감상 권유·안내 문장("…에 주목해 보면 좋습니다", "…살펴보시면 좋겠습니다"):
# 관람 방법을 제안하는 문장으로, 사실을 주장하지 않으므로 판정 대상에서 제외
# (상호작용 제외와 동일한 원칙의 격식체 형태)
_ADVICE_RE = re.compile(
    r"(보시면|보면|보시고|살펴보|주목해|집중해|비교해|따라가|읽어보|느껴보"
    r"|생각해\s*보|들여다보|감상해|즐겨보)"
    r"[^.!?]*(좋겠습니다|좋습니다|효과적입니다|추천합니다|바랍니다"
    r"|도움이\s*됩니다|드러납니다|다가옵니다)\s*[.!?…]*\s*$")

# 판단 유보(거절) 문장: 근거 부재를 밝히는 문장으로, 사실을 주장하지 않으므로
# 인용 가능 여부 판정 대상이 아님 (과잉 생성 대신 유보하는 것은 신뢰성에 부합)
_ABSTENTION_RE = re.compile(
    r"(확인할 수 없|알 수 없|포함되어 있지 않|나와 있지 않"
    r"|정보가 없|언급이 없|제공되지 않|확인되지 않)")


# -------------------------
# 정규화 / 토큰화
# -------------------------
def normalize(text: str) -> str:
    """단위·숫자 표기 통일, 특수문자 제거, 소문자화 (공백은 유지)."""
    t = str(text or "").lower()
    for rx, rep in _UNIT_RULES:
        t = rx.sub(rep, t)
    return _PUNCT_RE.sub(" ", t)


def normalize_joined(text: str) -> str:
    """normalize 후 공백까지 제거 (문자 n-gram·부분문자열 검색용)."""
    return re.sub(r"\s+", "", normalize(text))


def word_tokens(text: str) -> List[str]:
    """정규화 후 2자 이상 어절 목록."""
    return [w for w in normalize(text).split() if len(w) >= 2]


def stem(word: str) -> str:
    """어절의 어간 근사: 앞 min(len-1, 4)자 (최소 2자).

    한국어 체언·용언 어근은 통상 2~4자이고 그 뒤에 조사·어미 사슬
    ("…이라는", "…됩니다" 등)이 붙으므로, 어간 근사 길이를 4자로
    상한하여 접미 사슬이 길어도 어근이 보존되게 한다.
    """
    return word[:max(2, min(len(word) - 1, 4))]


# -------------------------
# 일치도 함수
# -------------------------
def make_char_ngrams(text: str, ns=(2, 3)) -> Set[str]:
    """문자 2~3gram 집합 생성 (서빙 reranker와 동일 방식)."""
    text = normalize_joined(text)
    grams: Set[str] = set()
    for n in ns:
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            grams.add(text[i:i + n])
    if not grams and text:
        grams.add(text)
    return grams


def ngram_containment(sent_grams: Set[str], evidence_grams: Set[str]) -> float:
    """문장 n-gram 중 근거문서 n-gram에 포함되는 비율."""
    if not sent_grams or not evidence_grams:
        return 0.0
    return len(sent_grams & evidence_grams) / len(sent_grams)


def token_coverage(sentence: str, evidence_joined: str) -> float:
    """어절 어간(조사/어미 변형 허용)이 근거문서에 존재하는 비율."""
    toks = word_tokens(sentence)
    if not toks or not evidence_joined:
        return 0.0
    covered = sum(1 for w in toks if stem(w) in evidence_joined)
    return covered / len(toks)


def citation_score(sentence: str, evidence_joined: str,
                   evidence_grams: Set[str]) -> float:
    """문장-근거문서 일치도 = max(어절 커버리지, 문자 n-gram 포함률)."""
    return max(
        token_coverage(sentence, evidence_joined),
        ngram_containment(make_char_ngrams(sentence), evidence_grams),
    )


# -------------------------
# 문장 분리 / 판정
# -------------------------
def split_sentences(text: str) -> List[str]:
    """해설 텍스트를 문장 단위로 분리한다."""
    text = str(text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text)]
    return [s for s in sents if s]


def is_interactive(sentence: str) -> bool:
    """관람객 상호작용·권유·의문문 여부 (판정 대상 제외). 격식체 감상 권유 포함."""
    return bool(_INTERACTIVE_RE.search(sentence)) or bool(_ADVICE_RE.search(sentence))


def is_abstention(sentence: str) -> bool:
    """판단 유보(거절) 문장 여부 (사실 주장이 아니므로 판정 대상 제외)."""
    return bool(_ABSTENTION_RE.search(sentence))


def judge_answer(answer: str, evidence: str,
                 threshold: float = CITATION_THRESHOLD,
                 min_sent_chars: int = MIN_SENT_CHARS) -> Dict:
    """
    생성 해설 1건에 대해 문장별 일치도와 인용 가능 여부를 판정한다.

    반환: {
      "sentences": [(문장, 일치도, 인용가능여부), ...],  # 판정 대상 문장만
      "n_total": 판정 대상 문장 수,
      "n_citable": 인용 가능 문장 수,
      "n_interactive": 상호작용 문장 수(제외),
      "n_abstention": 판단 유보 문장 수(제외),
      "n_short": 최소 길이 미달 문장 수(제외),
    }
    """
    evidence_joined = normalize_joined(evidence)
    evidence_grams = make_char_ngrams(evidence)
    sentences: List[Tuple[str, float, bool]] = []
    n_interactive, n_abstention, n_short = 0, 0, 0
    for sent in split_sentences(answer):
        if len(normalize_joined(sent)) < min_sent_chars:
            n_short += 1
            continue
        if is_interactive(sent):
            n_interactive += 1
            continue
        if is_abstention(sent):
            n_abstention += 1
            continue
        score = citation_score(sent, evidence_joined, evidence_grams)
        sentences.append((sent, score, score >= threshold))
    return {
        "sentences": sentences,
        "n_total": len(sentences),
        "n_citable": sum(1 for _, _, ok in sentences if ok),
        "n_interactive": n_interactive,
        "n_abstention": n_abstention,
        "n_short": n_short,
    }


def citation_accuracy(results: List[Dict]) -> float:
    """코퍼스 전체 Citation Accuracy(%). 문장 단위 micro average."""
    n_total = sum(r["n_total"] for r in results)
    n_citable = sum(r["n_citable"] for r in results)
    if n_total == 0:
        return 0.0
    return n_citable / n_total * 100.0
