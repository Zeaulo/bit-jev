"""对蒸馏模型的 GPU 逐题因果行推理进行可复现测速。"""

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

# 本机 Triton 与 PyTorch 编译器版本不匹配，参考模型保持 eager 计算。
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import torch


# 根目录测试脚本直接使用正式检查点、请求规范化和编码实现。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bit_jev.api import to_record  # noqa: E402
from bit_jev.checkpoint import BitJevCheckpoint  # noqa: E402


def read_requests(path, limit):
    """按输入顺序读取固定数量的 TypeSafe JSONL 请求。"""
    requests = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                requests.append(json.loads(line))
                if limit and len(requests) >= limit:
                    break
    if not requests:
        raise ValueError("测速输入没有请求")
    return requests


def stable_mlp_forward(module, hidden):
    """保持 FP16 权重，以 FP32 计算 relu2 与门控乘积，再交回 FP16 线性层。"""
    gate = module.gate_proj(hidden).float()
    up = module.up_proj(hidden).float()
    fused = module.ffn_sub_norm(module.act_fn(gate) * up)
    return module.down_proj(fused.to(dtype=hidden.dtype))


def stable_decoder_forward(module, hidden_states, attention_mask=None, position_ids=None,
                           past_key_values=None, use_cache=False, cache_position=None,
                           position_embeddings=None, **kwargs):
    """残差保持 FP32，归一化后再转 FP16 供注意力和线性层计算。"""
    residual = hidden_states.float()
    attention_input = module.input_layernorm(residual).to(dtype=torch.float16)
    attention_output, _ = module.self_attn(
        hidden_states=attention_input,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + attention_output.float()
    residual = hidden_states
    mlp_input = module.post_attention_layernorm(hidden_states).to(dtype=torch.float16)
    return residual + module.mlp(mlp_input).float()


def benchmark(run, input_path, limit, repeats, output_path, safe_relu2):
    """分开测量模型加载、预编码和每次 GPU forward 的同步耗时。"""
    if not torch.cuda.is_available():
        raise RuntimeError("本机未检测到 CUDA GPU")
    if repeats < 1:
        raise ValueError("重复次数必须至少为 1")
    requests = read_requests(input_path, limit)
    # RTX 2060 的显存无法容纳 FP32 2B 骨干，直接以 FP16 加载完整模型。
    load_started = time.perf_counter()
    tokenizer, model = BitJevCheckpoint(run).load("cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    model.eval()
    if safe_relu2:
        # 原 FP16 的 relu2 在首层平方时溢出；仅在测试进程中提高这段计算精度。
        for layer in model.lm.layers:
            layer.mlp.forward = types.MethodType(stable_mlp_forward, layer.mlp)
            layer.forward = types.MethodType(stable_decoder_forward, layer)
    model_memory_mib = torch.cuda.memory_allocated() / (1024 * 1024)
    # 编码计时独立于 GPU forward，避免把 tokenizer 开销算进设备推理速度。
    encode_started = time.perf_counter()
    encoded = []
    for request in requests:
        record, _ = to_record(request, labelled=False)
        encoded.append(model.encode(tokenizer, record))
    encode_seconds = time.perf_counter() - encode_started
    question_count = sum(len(item["decide_idx"]) for item in encoded)
    # 预热同一条请求，排除首次 CUDA kernel 初始化对稳定推理时间的影响。
    diagnostics = []
    sub_diagnostics = []
    hooks = []
    # 预热时检查每层输出，FP16 在 6 GB 显存设备上可能发生中间值溢出。
    for layer_index, layer in enumerate(model.lm.layers):
        def inspect_layer(_module, _inputs, output, index=layer_index):
            """记录第一条请求经过该解码层后是否仍为有限值。"""
            hidden = output[0] if isinstance(output, tuple) else output
            diagnostics.append((index, bool(torch.isfinite(hidden).all().item())))
        hooks.append(layer.register_forward_hook(inspect_layer))
    # 依据是否稳定首层 relu2，继续检查首个可能失稳的后续解码层。
    diagnostic_layer = 7 if safe_relu2 else 0
    for module_name, module in model.lm.layers[diagnostic_layer].named_modules():
        if not module_name or list(module.children()):
            continue
        def inspect_submodule(_module, _inputs, output, name=module_name):
            """记录第一层各叶子模块输出是否有限。"""
            hidden = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden, torch.Tensor):
                sub_diagnostics.append((name, bool(torch.isfinite(hidden).all().item())))
        hooks.append(module.register_forward_hook(inspect_submodule))
    with torch.inference_mode():
        torch.cuda.synchronize()
        warm_started = time.perf_counter()
        warmup_outputs = model.forward_rows_batch([encoded[0]])[0]
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - warm_started) * 1000
        for hook in hooks:
            hook.remove()
        invalid_layers = [index for index, valid in diagnostics if not valid]
        invalid_questions = [index for index, logits in enumerate(warmup_outputs)
                             if not bool(torch.isfinite(logits).all().item())]
        if invalid_layers or invalid_questions:
            raise RuntimeError(f"GPU FP16 输出非有限：首个异常层 {invalid_layers[:1]}，"
                               f"异常题目 {invalid_questions}，检查层 {diagnostic_layer} 的模块 "
                               f"{sub_diagnostics[:30]}")
        torch.cuda.reset_peak_memory_stats()
        run_totals = []
        per_record = []
        last_outputs = []
        # 每轮保持模型常驻，并对每条请求的 forward 单独同步计时。
        for repeat_index in range(repeats):
            round_latencies = []
            for item in encoded:
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = model.forward_rows_batch([item])[0]
                torch.cuda.synchronize()
                round_latencies.append((time.perf_counter() - started) * 1000)
                if repeat_index == repeats - 1:
                    last_outputs.append(outputs)
            run_totals.append(sum(round_latencies))
            per_record.extend(round_latencies)
    # 计时完成后才读取预测，避免 .item() 的同步影响逐条 forward 耗时。
    predictions = [[int(logits.argmax().item()) for logits in outputs]
                   for outputs in last_outputs]
    if any(not bool(torch.isfinite(logits).all().item())
           for outputs in last_outputs for logits in outputs):
        raise RuntimeError("GPU 某些请求的 logits 非有限，速度结果无效")
    ordered_latencies = sorted(per_record)
    report = {
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "numerics": "fp16 weights and linear; fp32 relu2, gate product, FFN norm, and residual" if safe_relu2
                    else "unmodified fp16",
        "input_file": str(Path(input_path).resolve()),
        "records": len(requests),
        "questions": question_count,
        "repeats": repeats,
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "warmup_ms": warmup_ms,
        "inference_ms_by_repeat": run_totals,
        "inference_ms_mean_per_repeat": statistics.mean(run_totals),
        "inference_ms_median_per_record": statistics.median(per_record),
        "inference_ms_p95_per_record": ordered_latencies[int((len(per_record) - 1) * 0.95)],
        "model_memory_mib": model_memory_mib,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "predictions_last_repeat": predictions,
    }
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    """解析检查点、输入请求、重复次数及报告路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--safe-relu2", action="store_true")
    args = parser.parse_args()
    benchmark(args.run, args.input, args.limit, args.repeats, args.out, args.safe_relu2)


if __name__ == "__main__":
    main()
