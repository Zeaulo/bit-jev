"""汇总同一批请求的 CPU 与 GPU 耗时，并逐题核对预测。"""

import argparse
import json
import statistics
from pathlib import Path


def read_cpu(path, records, repeats):
    """读取一条预热结果及按轮次排列的 CPU 分类结果。"""
    with Path(path).open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    expected = 1 + records * repeats
    if len(rows) != expected:
        raise ValueError(f"CPU 结果应有 {expected} 条，实际 {len(rows)} 条")
    warmup = rows[0]
    rounds = [rows[1 + index * records:1 + (index + 1) * records]
              for index in range(repeats)]
    return warmup, rounds


def compare(cpu_path, gpu_path, cpu_wall_path, gpu_wall_path, output_path):
    """计算稳定推理速度、进程总耗时及预测翻转。"""
    gpu = json.loads(Path(gpu_path).read_text(encoding="utf-8"))
    records = gpu["records"]
    repeats = gpu["repeats"]
    warmup, rounds = read_cpu(cpu_path, records, repeats)
    cpu_round_ms = [sum(row["latency_ms"] for row in round_rows)
                    for round_rows in rounds]
    cpu_predictions = [[max(range(len(logits)), key=logits.__getitem__)
                        for logits in row["logits"]]
                       for row in rounds[-1]]
    gpu_predictions = gpu["predictions_last_repeat"]
    if [len(item) for item in cpu_predictions] != [len(item) for item in gpu_predictions]:
        raise ValueError("CPU 与 GPU 的逐条请求题目数不同")
    flips = [{"record_index": record_index, "question_index": question_index,
              "cpu": cpu_answer, "gpu": gpu_answer}
             for record_index, (cpu_row, gpu_row) in enumerate(zip(cpu_predictions, gpu_predictions))
             for question_index, (cpu_answer, gpu_answer) in enumerate(zip(cpu_row, gpu_row))
             if cpu_answer != gpu_answer]
    cpu_mean_ms = statistics.mean(cpu_round_ms)
    gpu_mean_ms = gpu["inference_ms_mean_per_repeat"]
    cpu_wall_seconds = float(Path(cpu_wall_path).read_text(encoding="utf-8-sig").strip())
    gpu_wall_seconds = float(Path(gpu_wall_path).read_text(encoding="utf-8-sig").strip())
    report = {
        "records": records,
        "questions": gpu["questions"],
        "repeats": repeats,
        "cpu": {"format": "I2_S GGUF", "threads": 8, "batch": 128,
                "warmup_ms": warmup["latency_ms"], "round_ms": cpu_round_ms,
                "round_mean_ms": cpu_mean_ms, "process_wall_seconds": cpu_wall_seconds},
        "gpu": {"device": gpu["device"], "format": gpu["numerics"],
                "load_seconds": gpu["load_seconds"], "warmup_ms": gpu["warmup_ms"],
                "round_ms": gpu["inference_ms_by_repeat"], "round_mean_ms": gpu_mean_ms,
                "process_wall_seconds": gpu_wall_seconds,
                "model_memory_mib": gpu["model_memory_mib"],
                "peak_memory_mib": gpu["peak_memory_mib"]},
        "steady_inference_speedup_gpu_over_cpu": cpu_mean_ms / gpu_mean_ms,
        "process_wall_speedup_gpu_over_cpu": cpu_wall_seconds / gpu_wall_seconds,
        "prediction_flips": flips,
        "prediction_agreement": (gpu["questions"] - len(flips)) / gpu["questions"],
    }
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    """解析已保存的测速文件并写出比较报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpu-wall", required=True)
    parser.add_argument("--gpu-wall", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    compare(args.cpu, args.gpu, args.cpu_wall, args.gpu_wall, args.out)


if __name__ == "__main__":
    main()
