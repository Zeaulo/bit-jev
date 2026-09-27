"""汇总同权重 I2_S 与 F16 原生 CPU 基准及逐题一致性。"""

import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    """按原生输入顺序读取每条请求的输出。"""
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def summarize(i2s_report_path, f16_report_path, i2s_rows_path, f16_rows_path,
              records_per_round, output_path):
    """核对相同请求的 logits、预测和每轮耗时。"""
    i2s_report = json.loads(Path(i2s_report_path).read_text(encoding="utf-8"))
    f16_report = json.loads(Path(f16_report_path).read_text(encoding="utf-8"))
    i2s_rows = read_jsonl(i2s_rows_path)
    f16_rows = read_jsonl(f16_rows_path)
    if len(i2s_rows) != len(f16_rows) or len(i2s_rows) % records_per_round:
        raise ValueError("两种格式的请求数量或轮次不一致")
    repeats = len(i2s_rows) // records_per_round
    max_logit_diff = 0.0
    flips = []
    questions = 0
    # 每一轮都比较相同顺序的请求和题目，避免只验证最后一轮。
    for record_index, (i2s, f16) in enumerate(zip(i2s_rows, f16_rows)):
        if len(i2s["logits"]) != len(f16["logits"]):
            raise ValueError(f"第 {record_index} 条请求的题目数量不同")
        for question_index, (left, right) in enumerate(zip(i2s["logits"], f16["logits"])):
            if len(left) != len(right):
                raise ValueError(f"第 {record_index} 条请求第 {question_index} 题候选数量不同")
            max_logit_diff = max(max_logit_diff, *(abs(a - b) for a, b in zip(left, right)))
            before = max(range(len(left)), key=left.__getitem__)
            after = max(range(len(right)), key=right.__getitem__)
            if before != after:
                flips.append({"round_index": record_index // records_per_round,
                              "record_index": record_index % records_per_round,
                              "question_index": question_index,
                              "i2s": before, "f16": after})
            questions += 1
    i2s_rounds = [sum(row["latency_ms"] for row in i2s_rows[start:start + records_per_round])
                  for start in range(0, len(i2s_rows), records_per_round)]
    f16_rounds = [sum(row["latency_ms"] for row in f16_rows[start:start + records_per_round])
                  for start in range(0, len(f16_rows), records_per_round)]
    report = {
        "hardware": "AMD Ryzen 7 4800H, 8 physical cores / 16 logical threads",
        "configuration": {"threads": i2s_report["threads"], "batch": i2s_report["batch"],
                          "records_per_round": records_per_round, "repeats": repeats,
                          "questions_per_round": questions // repeats},
        "i2s": {"model_bytes": i2s_report["model_bytes"],
                "peak_rss_mib": i2s_report["peak_rss_mib"],
                "round_ms": i2s_rounds, "round_mean_ms": statistics.mean(i2s_rounds),
                "process_wall_seconds": i2s_report["process_wall_seconds"]},
        "f16": {"model_bytes": f16_report["model_bytes"],
                "peak_rss_mib": f16_report["peak_rss_mib"],
                "round_ms": f16_rounds, "round_mean_ms": statistics.mean(f16_rounds),
                "process_wall_seconds": f16_report["process_wall_seconds"]},
        "i2s_inference_speedup_over_f16": statistics.mean(f16_rounds) / statistics.mean(i2s_rounds),
        "f16_file_size_multiple_of_i2s": f16_report["model_bytes"] / i2s_report["model_bytes"],
        "f16_peak_rss_multiple_of_i2s": f16_report["peak_rss_mib"] / i2s_report["peak_rss_mib"],
        "max_logit_abs_diff": max_logit_diff,
        "prediction_flips": flips,
    }
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    """解析两种格式的原始报告、JSONL 和汇总路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2s-report", required=True)
    parser.add_argument("--f16-report", required=True)
    parser.add_argument("--i2s-rows", required=True)
    parser.add_argument("--f16-rows", required=True)
    parser.add_argument("--records-per-round", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summarize(args.i2s_report, args.f16_report, args.i2s_rows, args.f16_rows,
              args.records_per_round, args.out)


if __name__ == "__main__":
    main()
