# -*- coding: utf-8 -*-
"""
운영 해설 서비스(REST)에 질의하여 T-003 측정 입력 JSONL을 수집한다.

서비스 인터페이스 (github_real/04_service/INTERFACE.md):
  GET /generate/{유물ID}/{관람객유형}/{질문}
  응답: {"문화유산_ID", "관람객_유형", "유물명", "질문", "답변", ...}

근거문서는 서빙과 동일한 evidence JSON(aimuse_items_enriched_final_v6.json)에서
유물ID로 찾아 측정 입력에 함께 기록한다. 모델에 투입된 근거와 측정 기준
근거를 일치시키기 위함이다.

질문 목록 파일 형식 (JSONL): {"유물ID": "동원1257", "관람객유형": "탐구형", "질문": "..."}

사용 예:
  python generate_responses_api.py \
      --server http://127.0.0.1:5000 \
      --questions questions.jsonl \
      --evidence /home/heri/etri_server/llm/data/aimuse_items_enriched_final_v6.json \
      --output eval_set.jsonl
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _flatten_class(cls):
    """유물_분류(중첩 dict)를 ['문화예술','서화',...] 형태로 평탄화. (서빙 코드와 동일)"""
    names = []
    cur = cls
    while isinstance(cur, dict):
        if cur.get("name"):
            names.append(str(cur["name"]))
        cur = cur.get("value")
    if isinstance(cur, str) and cur:
        names.append(cur)
    return names


def build_evidence_text(entry: dict) -> str:
    """모델 투입 근거문서 텍스트. 서빙 app_vllm_sep.py의 extract_evidence_text와
    동일 레이아웃([유물 설명]/[기본 정보]/[문양]/[의미]/[참고 문헌])을 포팅해
    측정 기준 근거와 모델 투입 근거를 일치시킨다."""
    payload = entry or {}
    if not isinstance(payload, dict):
        return str(payload or "")
    parts = []

    desc = payload.get("description", "")
    if isinstance(desc, list):
        desc = "\n".join(d for d in desc if d)
    desc = str(desc or "").strip()
    if desc:
        parts.append("[유물 설명]\n" + desc)

    info = []
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

    pat = []
    for k, v in payload.items():
        if k.startswith("문양") and v:
            if isinstance(v, list):
                pat.append(", ".join(
                    f"{x.get('name')}={x.get('value')}" if isinstance(x, dict) else str(x) for x in v))
            else:
                pat.append(str(v))
    if pat:
        parts.append("[문양]\n" + "\n".join(pat))

    mean = []
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

    return "\n\n".join(parts) if parts else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:5000",
                    help="해설 서비스 주소 (쉼표로 여러 개 지정 시 라운드로빈 분배)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="동시 요청 수 (서버가 동시 처리 가능할 때만 >1, 기본 1=순차)")
    ap.add_argument("--questions", required=True, help="질문 목록 JSONL")
    ap.add_argument("--evidence", required=True, help="서빙 evidence JSON 경로")
    ap.add_argument("--output", required=True, help="측정 입력 JSONL 출력 경로")
    ap.add_argument("--sleep", type=float, default=0.0, help="요청 간 대기(초)")
    ap.add_argument("--task", default="", help="task flag (예: qa) — 지정 시 ?task= 로 전달")
    ap.add_argument("--resume", action="store_true",
                    help="출력 파일에 이미 수집된 (유물ID, 질문) 쌍은 건너뛰고 이어서 수집 (전체 유물 장시간 수집용)")
    args = ap.parse_args()

    evidence_db = json.load(open(args.evidence, encoding="utf-8"))
    if isinstance(evidence_db, list):
        evidence_db = {str(e.get("id") or e.get("유물ID") or e.get("소장품번호")): e
                       for e in evidence_db}

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]

    done = set()
    mode = "w"
    if args.resume and os.path.exists(args.output):
        for l in open(args.output, encoding="utf-8"):
            try:
                r = json.loads(l)
                done.add((r.get("유물ID"), r.get("질문")))
            except Exception:
                pass
        mode = "a"
        print(f"[RESUME] 기존 수집 {len(done)}건 건너뜀", file=sys.stderr)

    servers = [s.strip().rstrip("/") for s in args.server.split(",") if s.strip()]
    todo = []
    for i, q in enumerate(questions):
        relic_id = str(q["유물ID"])
        question = q["질문"]
        if (relic_id, question) in done:
            continue
        todo.append((i, relic_id, q.get("관람객유형", "탐구형"), question))

    lock = threading.Lock()
    stats = {"ok": 0, "err": 0}
    t0 = time.time()

    def fetch(item):
        i, relic_id, visitor, question = item
        url = "{}/generate/{}/{}/{}".format(
            servers[i % len(servers)],
            urllib.parse.quote(relic_id, safe=""),
            urllib.parse.quote(visitor, safe=""),
            urllib.parse.quote(question, safe=""),
        )
        if args.task:
            url += "?task=" + urllib.parse.quote(args.task, safe="")
        try:
            with urllib.request.urlopen(url, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            with lock:
                stats["err"] += 1
                print(f"[ERR] {relic_id} {question[:30]}... → {e}", file=sys.stderr)
            return
        row = json.dumps({
            "유물ID": relic_id,
            "관람객유형": visitor,
            "질문": question,
            "근거문서": build_evidence_text(evidence_db.get(relic_id, {})),
            "답변": body.get("답변", ""),
            "참고한_문단문장": body.get("참고한 문단/문장", []),
        }, ensure_ascii=False)
        with lock:
            wf.write(row + "\n")
            wf.flush()
            stats["ok"] += 1
            if stats["ok"] % 100 == 0:
                rate = stats["ok"] / max(1e-9, time.time() - t0)
                eta = (len(todo) - stats["ok"] - stats["err"]) / max(rate, 1e-9)
                print(f"[진행] {stats['ok']}/{len(todo)}건 ({rate:.2f}건/s, 남은 예상 {eta/60:.0f}분)",
                      file=sys.stderr)
        if args.sleep:
            time.sleep(args.sleep)

    with open(args.output, mode, encoding="utf-8") as wf:
        if args.parallel <= 1:
            for item in todo:
                fetch(item)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                list(ex.map(fetch, todo))
    elapsed = time.time() - t0
    print(f"수집 완료: 성공 {stats['ok']}건 / 실패 {stats['err']}건, {elapsed/60:.1f}분 → {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
