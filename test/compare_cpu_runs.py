"""比较两个原生 CPU 推理产物的逐题数值与请求耗时。"""

import argparse
import json
import statistics
from pathlib import Path


def load_rows(path, offset, limit):
    """读取指定连续范围的 JSONL 分类结果。"""
    with Path(path).open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    selected = rows[offset:offset + limit] if limit else rows[offset:]
    if not selected:
        raise ValueError(f"没有可比较的结果：{path}")
    return selected


def compare(baseline_path, candidate_path, baseline_offset, candidate_offset, limit):
    """按记录和题目顺序对齐，计算最大差异及加速比。"""
    baseline = load_rows(baseline_path, baseline_offset, limit)
    candidate = load_rows(candidate_path, candidate_offset, limit)
    if len(baseline) != len(candidate):
        raise ValueError("两个产物的请求数量不同")
    max_logit_diff = 0.0
    prediction_flips = []
    questions = 0
    for record_index, (before, after) in enumerate(zip(baseline, candidate)):
        if (before.get("record") is not None and after.get("record") is not None
                and before["record"] != after["record"]) or len(before["logits"]) != len(after["logits"]):
            raise ValueError(f"第 {record_index} 条请求的记录 ID 或题目数量不同")
        for question_index, (left, right) in enumerate(zip(before["logits"], after["logits"])):
            if len(left) != len(right):
                raise ValueError(f"第 {record_index} 条请求第 {question_index} 题候选数量不同")
            max_logit_diff = max(max_logit_diff, *(abs(a - b) for a, b in zip(left, right)))
            left_choice = max(range(len(left)), key=left.__getitem__)
            right_choice = max(range(len(right)), key=right.__getitem__)
            if left_choice != right_choice:
                prediction_flips.append({"record_index": record_index,
                                         "question_index": question_index,
                                         "baseline": left_choice, "candidate": right_choice})
            questions += 1
    baseline_ms = [row["latency_ms"] for row in baseline]
    candidate_ms = [row["latency_ms"] for row in candidate]
    report = {"records": len(baseline), "questions": questions,
              "max_logit_abs_diff": max_logit_diff,
              "prediction_flips": prediction_flips,
              "baseline_ms_total": sum(baseline_ms), "candidate_ms_total": sum(candidate_ms),
              "baseline_ms_median": statistics.median(baseline_ms),
              "candidate_ms_median": statistics.median(candidate_ms),
              "speedup": sum(baseline_ms) / sum(candidate_ms)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    """解析基线与候选产物的路径和对齐范围。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-offset", type=int, default=0)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.baseline_offset,
            args.candidate_offset, args.limit)


if __name__ == "__main__":
    main()
