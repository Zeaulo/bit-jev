"""逐题核对完整 development 的原生 CPU 与已保存 PyTorch 评估结果。"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    """按文件顺序读取 UTF-8 JSONL，并拒绝空白记录。"""
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 是空行")
            yield json.loads(line)


def best_index(values):
    """与训练评估相同：并列最大值时选择最早候选项。"""
    return max(range(len(values)), key=lambda index: values[index])


def compare(input_path, cpu_path, reference_path, report_path, rows_path, allow_partial=False):
    """校验记录和题目一一对应，保存逐题差异与汇总指标。"""
    requests = list(read_jsonl(input_path))
    cpu_records = list(read_jsonl(cpu_path))
    reference_rows = list(read_jsonl(reference_path))
    if allow_partial and len(cpu_records) <= len(requests):
        requests = requests[:len(cpu_records)]
    if len(requests) != len(cpu_records):
        raise ValueError(f"输入 {len(requests)} 条，CPU 输出 {len(cpu_records)} 条，数量不一致")
    # 引用行按原评估顺序排列；迭代器逐题消费并在末尾检查是否完全用尽。
    reference_iter = iter(reference_rows)
    rows = []
    latencies = []
    for request, cpu_record in zip(requests, cpu_records):
        record_id = request["_meta"]["id"]
        question_ids = list(request["questions"])
        if cpu_record["record"] != record_id:
            raise ValueError(f"记录 ID 不匹配：{record_id} / {cpu_record['record']}")
        if len(question_ids) != len(cpu_record["logits"]) or len(question_ids) != len(cpu_record["probabilities"]):
            raise ValueError(f"题目数量不匹配：{record_id}")
        latency = float(cpu_record["latency_ms"])
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"推理耗时无效：{record_id}")
        latencies.append(latency)
        # 同一请求中的问题顺序由 JSONL 保存的插入顺序决定。
        for question_index, question_id in enumerate(question_ids):
            reference = next(reference_iter, None)
            if reference is None or reference["record"] != record_id or reference["qid"] != question_id:
                raise ValueError(f"PyTorch 参考行错位：{record_id}/{question_id}")
            logits = [float(value) for value in cpu_record["logits"][question_index]]
            probabilities = [float(value) for value in cpu_record["probabilities"][question_index]]
            if len(logits) != len(reference["logits"]) or len(probabilities) != len(logits):
                raise ValueError(f"候选数量不匹配：{record_id}/{question_id}")
            if not logits or not all(math.isfinite(value) for value in logits + probabilities):
                raise ValueError(f"CPU 概率或 logit 非有限：{record_id}/{question_id}")
            if abs(sum(probabilities) - 1.0) > 1e-4:
                raise ValueError(f"CPU 概率和不为 1：{record_id}/{question_id}")
            cpu_pred = best_index(logits)
            ref_pred = int(reference["pred"])
            label = int(reference["label"])
            # 原始概率还需与 TypeSafe 答案字段一致，覆盖最终对外输出层。
            answer = cpu_record["answers"][question_id]
            if answer["type"] != reference["qtype"]:
                raise ValueError(f"答案题型不匹配：{record_id}/{question_id}")
            if reference["qtype"] == "choice":
                keys = reference["keys"]
                if answer["choice"] != keys[cpu_pred] or any(
                    answer["probabilities"][key] != round(probability, 4)
                    for key, probability in zip(keys, probabilities)
                ):
                    raise ValueError(f"Choice 答案与概率不一致：{record_id}/{question_id}")
            elif reference["qtype"] == "noul":
                if answer["noul"] != round(probabilities[1], 4):
                    raise ValueError(f"Noul 答案与概率不一致：{record_id}/{question_id}")
            elif reference["qtype"] == "score":
                score = sum(index * probability for index, probability in enumerate(probabilities))
                if answer["score"] != round(score, 4):
                    raise ValueError(f"Score 答案与概率不一致：{record_id}/{question_id}")
            # 保存每题可复查的预测、真实标签和最大数值差。
            rows.append({
                "record": record_id,
                "qid": question_id,
                "qtype": reference["qtype"],
                "label": label,
                "cpu_pred": cpu_pred,
                "reference_pred": ref_pred,
                "cpu_correct": int(cpu_pred == label),
                "reference_correct": int(ref_pred == label),
                "argmax_flip": int(cpu_pred != ref_pred),
                "max_abs_logit_diff": max(abs(a - b) for a, b in zip(logits, reference["logits"])),
                "max_abs_probability_diff": max(abs(a - b) for a, b in zip(probabilities, reference["probs"])),
            })
    if not allow_partial and next(reference_iter, None) is not None:
        raise ValueError("PyTorch 参考结果含额外题目")
    # 汇总全局与题型准确率；计数使用整数，避免四舍五入影响最终验收。
    counts = defaultdict(Counter)
    for row in rows:
        for kind in ("all", row["qtype"]):
            counts[kind]["questions"] += 1
            counts[kind]["cpu_correct"] += row["cpu_correct"]
            counts[kind]["reference_correct"] += row["reference_correct"]
            counts[kind]["argmax_flips"] += row["argmax_flip"]
    by_type = {}
    for kind, values in counts.items():
        total = values["questions"]
        by_type[kind] = {
            "questions": total,
            "cpu_correct": values["cpu_correct"],
            "reference_correct": values["reference_correct"],
            "argmax_flips": values["argmax_flips"],
            "cpu_accuracy": values["cpu_correct"] / total,
            "reference_accuracy": values["reference_correct"] / total,
        }
    ordered_latency = sorted(latencies)
    report = {
        "input_file": str(Path(input_path).resolve()),
        "cpu_file": str(Path(cpu_path).resolve()),
        "reference_file": str(Path(reference_path).resolve()),
        "records": len(requests),
        "questions": len(rows),
        "by_type": by_type,
        "cpu_only_correct": sum(row["cpu_correct"] and not row["reference_correct"] for row in rows),
        "reference_only_correct": sum(row["reference_correct"] and not row["cpu_correct"] for row in rows),
        "both_wrong_argmax_flips": sum(row["argmax_flip"] and not row["cpu_correct"]
                                        and not row["reference_correct"] for row in rows),
        "max_abs_logit_diff": max(row["max_abs_logit_diff"] for row in rows),
        "mean_row_max_abs_logit_diff": sum(row["max_abs_logit_diff"] for row in rows) / len(rows),
        "max_abs_probability_diff": max(row["max_abs_probability_diff"] for row in rows),
        "native_latency_total_s": sum(latencies) / 1000,
        "native_latency_mean_ms_per_record": sum(latencies) / len(latencies),
        "native_latency_p50_ms_per_record": ordered_latency[len(latencies) // 2],
        "native_latency_p95_ms_per_record": ordered_latency[int((len(latencies) - 1) * 0.95)],
        "flips": [row for row in rows if row["argmax_flip"]],
    }
    # 报告与逐题结果均采用 UTF-8；验收失败时仍保留证据供定位。
    Path(rows_path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "flips"}, ensure_ascii=False, indent=2))
    return report


def main():
    """解析 development 输入、CPU 输出与参考评估产物路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    compare(args.input, args.cpu, args.reference, args.report, args.rows,
            allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
