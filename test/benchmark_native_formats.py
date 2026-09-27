"""对固定原生 JSONL 输入测量 CPU 推理、启动和进程峰值内存。"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import psutil


def benchmark(binary, model, head, input_path, output_path, report_path,
              stderr_path, threads, batch, limit):
    """保持模型进程常驻，逐条收集原生耗时与进程工作集峰值。"""
    with Path(input_path).open(encoding="utf-8") as source:
        requests = [line for line in source if line.strip()]
    if limit:
        requests = requests[:limit]
    if not requests:
        raise ValueError("原生测速输入为空")
    command = [str(binary), "--model", str(model), "--head", str(head),
               "--threads", str(threads), "--batch", str(batch)]
    started = time.perf_counter()
    peak_rss = 0
    stop_monitor = threading.Event()
    with Path(stderr_path).open("w", encoding="utf-8") as errors:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=errors, text=True, encoding="utf-8", bufsize=1) as process:
            observed = psutil.Process(process.pid)

            def monitor_memory():
                """在原生进程存活期间采样工作集常驻字节数。"""
                nonlocal peak_rss
                while not stop_monitor.wait(0.05):
                    try:
                        peak_rss = max(peak_rss, observed.memory_info().rss)
                    except psutil.NoSuchProcess:
                        break

            monitor = threading.Thread(target=monitor_memory, daemon=True)
            monitor.start()
            native_latencies = []
            request_wall_ms = []
            responses = []
            first_response_seconds = None
            # 每条请求写入后立即读取输出，确保进程内只同时处理一条请求。
            for request in requests:
                request_started = time.perf_counter()
                process.stdin.write(request if request.endswith("\n") else request + "\n")
                process.stdin.flush()
                response = process.stdout.readline()
                if not response:
                    raise RuntimeError(f"原生进程提前退出，退出码：{process.poll()}")
                request_wall_ms.append((time.perf_counter() - request_started) * 1000)
                result = json.loads(response)
                responses.append(result)
                native_latencies.append(result["latency_ms"])
                if first_response_seconds is None:
                    first_response_seconds = time.perf_counter() - started
            process.stdin.close()
            # 避免 Popen 上下文在退出时重复关闭已关闭的 Windows 管道。
            process.stdin = None
            exit_code = process.wait()
            stop_monitor.set()
            monitor.join()
    if exit_code:
        raise RuntimeError(f"原生进程退出码 {exit_code}，请查看 {stderr_path}")
    Path(output_path).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n"
                                         for item in responses), encoding="utf-8")
    report = {
        "model": str(model), "model_bytes": Path(model).stat().st_size,
        "threads": threads, "batch": batch, "records": len(requests),
        "questions": sum(len(item["logits"]) for item in responses),
        "native_latency_ms": native_latencies,
        "native_latency_ms_total": sum(native_latencies),
        "request_wall_ms": request_wall_ms,
        "first_response_seconds": first_response_seconds,
        "process_wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": peak_rss / (1024 * 1024),
    }
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    """解析原生二进制、模型、指针头和报告路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    benchmark(args.binary, args.model, args.head, args.input, args.out, args.report,
              args.stderr, args.threads, args.batch, args.limit)


if __name__ == "__main__":
    main()
