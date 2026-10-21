# -*- coding: utf-8 -*-
"""
T-003 공식 시험용 테스트셋 구성 (전체 유물의 20%, 시드 고정 무작위 표본).

생성물 (--out-dir, 기본 testset/):
  - testset_ids.txt          : 표본 유물ID 목록 (한 줄 1개, 정렬)
  - questions_test20.jsonl   : 수집용 질문 목록 {유물ID, 관람객유형, 질문}
  - testset_list.csv         : 데이터 확인 단계 제시용 (유물ID, 유물명, 관람객유형)
  - eval_set_test20.jsonl    : (--from-eval 지정 시) 기수집 응답에서 표본만 필터링

사용:
  python prepare_testset.py \
      --source serving_local/data/answerable_questions_final.json \
      --evidence serving_local/data/aimuse_items_enriched_final_v6.json \
      --from-eval serving_local/eval_set_full_explanation.jsonl \
      --ratio 0.2 --seed 42 --out-dir testset
"""
import argparse
import csv
import json
import os
import random

VISITOR_TYPES = ["탐구형", "취미형", "의무형", "휴식형", "사교형"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="answerable_questions_final.json")
    ap.add_argument("--evidence", default="", help="유물명 보강용 evidence JSON")
    ap.add_argument("--from-eval", default="",
                    help="기수집 평가셋(JSONL) — 표본 유물만 필터링해 eval_set_test20.jsonl 생성")
    ap.add_argument("--ratio", type=float, default=0.2, help="표본 비율 (기본 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="testset")
    args = ap.parse_args()

    data = json.load(open(args.source, encoding="utf-8"))
    artifacts = [a for a in data if a.get("질문들") and a.get("소장품번호")]
    n = round(len(artifacts) * args.ratio)
    random.seed(args.seed)
    picked = random.sample(artifacts, n)
    picked.sort(key=lambda a: str(a["소장품번호"]))
    ids = [str(a["소장품번호"]) for a in picked]

    names = {}
    if args.evidence:
        ev = json.load(open(args.evidence, encoding="utf-8"))
        for e in (ev if isinstance(ev, list) else ev.values()):
            if isinstance(e, dict) and e.get("id"):
                names[str(e["id"])] = str(e.get("유물", "") or "")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "testset_ids.txt"), "w", encoding="utf-8") as wf:
        wf.write("\n".join(ids) + "\n")

    with open(os.path.join(args.out_dir, "questions_test20.jsonl"), "w", encoding="utf-8") as wf, \
         open(os.path.join(args.out_dir, "testset_list.csv"), "w", newline="", encoding="utf-8-sig") as cf:
        cw = csv.writer(cf)
        cw.writerow(["번호", "유물ID", "유물명", "관람객유형"])
        for i, a in enumerate(picked):
            rid = str(a["소장품번호"])
            vt = VISITOR_TYPES[i % len(VISITOR_TYPES)]
            random.seed(args.seed * 100000 + i)  # 유물별 결정적 질문 선택
            q = random.choice(a["질문들"])
            wf.write(json.dumps({"유물ID": rid, "관람객유형": vt, "질문": q},
                                ensure_ascii=False) + "\n")
            cw.writerow([i + 1, rid, names.get(rid, ""), vt])

    print(f"테스트셋: 전체 {len(artifacts)}종 중 {n}종 ({args.ratio*100:.0f}%, 시드 {args.seed})")
    print(f"  {args.out_dir}/testset_ids.txt, questions_test20.jsonl, testset_list.csv")

    if args.from_eval:
        want = set(ids)
        out_path = os.path.join(args.out_dir, "eval_set_test20.jsonl")
        n_hit = 0
        with open(out_path, "w", encoding="utf-8") as wf:
            for l in open(args.from_eval, encoding="utf-8"):
                r = json.loads(l)
                if str(r.get("유물ID")) in want:
                    wf.write(l if l.endswith("\n") else l + "\n")
                    n_hit += 1
        print(f"  {out_path}: 기수집 응답 {n_hit}건 필터링")


if __name__ == "__main__":
    main()
