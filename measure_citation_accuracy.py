# -*- coding: utf-8 -*-
"""
T-003 Citation Accuracy 측정 실행 스크립트.

입력: JSON 배열 또는 JSONL. 각 인스턴스는 아래 키를 가진다.
  - "근거문서": 해설 생성에 투입된 근거문서 원문
  - "답변" 또는 "해설": 자연어 생성 AI가 생성한 해설
  - "질문" (선택): 결과 리포트 표기용

출력:
  - <out_dir>/result_detail.csv : 문장 단위 판정 내역 (문장, 일치도, 인용가능)
  - <out_dir>/metrics.txt       : 최종 지표 요약
  - 콘솔: Citation Accuracy(%) 및 목표(70%) 만족 여부

사용 예:
  python measure_citation_accuracy.py --input eval_set.jsonl --out-dir output/
  python measure_citation_accuracy.py --input eval_set.jsonl --threshold 0.6
"""
import argparse
import csv
import json
import os
import sys

from citation_metric import (CITATION_THRESHOLD, MIN_SENT_CHARS,
                             citation_accuracy, judge_answer)

TARGET_PERCENT = 70.0  # 성능목표: Citation Accuracy 70% 이상


def load_instances(path: str):
    """JSON 배열 / JSONL 모두 지원."""
    with open(path, encoding="utf-8") as rf:
        head = rf.read(1)
        rf.seek(0)
        if head == "[":
            return json.load(rf)
        return [json.loads(line) for line in rf if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="T-003 Citation Accuracy 측정")
    ap.add_argument("--input", required=True, help="평가셋 (JSON 배열 또는 JSONL)")
    ap.add_argument("--out-dir", default="output", help="결과 저장 디렉토리")
    ap.add_argument("--threshold", type=float, default=CITATION_THRESHOLD,
                    help=f"문장 인용 가능 판정 일치도 임계치 (기본 {CITATION_THRESHOLD})")
    ap.add_argument("--min-sent-chars", type=int, default=MIN_SENT_CHARS,
                    help=f"판정 대상 최소 문장 길이, 정규화 문자 기준 (기본 {MIN_SENT_CHARS})")
    args = ap.parse_args()

    instances = load_instances(args.input)
    if not instances:
        print("입력 인스턴스가 없습니다.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    detail_path = os.path.join(args.out_dir, "result_detail.csv")
    metrics_path = os.path.join(args.out_dir, "metrics.txt")

    results = []
    n_no_answer = 0
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as wf:
        writer = csv.writer(wf)
        writer.writerow(["인스턴스", "질문", "문장", "일치도", "인용가능"])
        for idx, inst in enumerate(instances):
            answer = inst.get("답변") or inst.get("해설") or ""
            evidence = inst.get("근거문서") or ""
            question = inst.get("질문", "")
            if not answer.strip():
                n_no_answer += 1
                continue
            r = judge_answer(answer, evidence,
                             threshold=args.threshold,
                             min_sent_chars=args.min_sent_chars)
            results.append(r)
            for sent, score, ok in r["sentences"]:
                writer.writerow([idx, question, sent, f"{score:.4f}", "O" if ok else "X"])

    n_total = sum(r["n_total"] for r in results)
    n_citable = sum(r["n_citable"] for r in results)
    n_interactive = sum(r["n_interactive"] for r in results)
    n_abstention = sum(r["n_abstention"] for r in results)
    n_short = sum(r["n_short"] for r in results)
    acc = citation_accuracy(results)
    passed = acc >= TARGET_PERCENT

    lines = [
        f"citation_accuracy: {acc:.2f}%",
        f"target: {TARGET_PERCENT:.0f}%",
        f"result: {'PASS' if passed else 'FAIL'}",
        f"instances_evaluated: {len(results)}",
        f"instances_without_answer: {n_no_answer}",
        f"sentences_total: {n_total}",
        f"sentences_citable: {n_citable}",
        f"sentences_excluded_interactive: {n_interactive}",
        f"sentences_excluded_abstention: {n_abstention}",
        f"sentences_excluded_short: {n_short}",
        f"threshold: {args.threshold}",
        f"min_sent_chars: {args.min_sent_chars}",
    ]
    with open(metrics_path, "w", encoding="utf-8") as wf:
        wf.write("\n".join(lines) + "\n")

    print("=" * 60)
    print("T-003 자연어 생성모델의 근거 기반 신뢰성 지표 (Citation Accuracy)")
    print("=" * 60)
    for line in lines:
        print(line)
    print(f"\n상세 내역: {detail_path}")
    print(f"지표 요약: {metrics_path}")


if __name__ == "__main__":
    main()
