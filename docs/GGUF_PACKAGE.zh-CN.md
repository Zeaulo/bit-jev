# `bit-jev` pip 包：GGUF 加载与推理

[返回 README](../README.md) · [English](GGUF_PACKAGE.md) · [Hugging Face 模型卡](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)

## 安装

```bash
pip install bit-jev
```

默认安装包含请求编码、Hugging Face 下载器和原生程序管理接口。约 1.19 GB 的 I2_S GGUF 不在 PyPI wheel 中；首次调用 `from_pretrained()` 时下载到 Hugging Face 缓存。构建原生 runner 需要 Git、CMake 3.28+ 和 C++17 编译器。Vulkan GPU 模式需要 Vulkan SDK，CUDA 模式需要 CUDA Toolkit。安装命令本身不下载模型或编译程序。

## Python API

```python
from bit_jev.gguf import BitJev

request = {
    "state": "A customer reports a duplicate charge.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payment and refund issues", "shipping": "Delivery issues"},
        }
    },
}

with BitJev.from_pretrained(device="cpu", threads=8, batch=128) as model:
    first = model.infer(request)
    second = model.infer(request)
    print(first["answers"], first["probabilities"], first["latency_ms"])
```

`from_pretrained()` 等待原生程序完成模型加载后才返回。`infer()` 复用常驻模型，返回 `answers`、`logits`、`probabilities`、`latency_ms` 和 `device`。`latency_ms` 仅记录原生计算，不包含下载、原生编译、tokenizer 编码或进程间通信。`infer_many()` 按请求顺序复用同一模型；`close()` 或 `with` 结束后释放资源。

设备选项：`cpu` 不卸载层；`gpu` 是 `vulkan` 的便捷别名；`cuda` 使用 NVIDIA CUDA 构建。多 GPU 主机可传入 `gpu_index=1` 选择 Vulkan/CUDA 可见设备序号；它只修改原生子进程的环境。GPU 模式在没有可用 GPU 后端或可见 GPU 时返回错误，不自动退回 CPU。I2_S Vulkan 后端可能保留部分权重在 CPU 内存，因此 GPU 路径不等于全部权重位于显存。不同硬件和输入长度的性能需要自行测量；原有 AutoDL RTX 5090 数据来自 FP16 PyTorch 路径，不能当作此 GGUF GPU 路径的速度。

## 已下载模型与自有二进制

```python
from bit_jev.gguf import BitJev

with BitJev.from_pretrained(
    "./models/bit-jev-2b-distilled",
    device="cpu",
    binary="./core/build/bit-jev-cpu/bit-jev-cpu.exe",
) as model:
    result = model.infer(request)
```

本地模型目录必须包含 `backbone-i2_s.gguf`、`head.f32`、配置与 tokenizer 文件。`binary` 适合自行编译或已经验证的原生程序。首次自动构建缓存默认位于 `~/.cache/bit-jev`；可设置 `BIT_JEV_CACHE` 或传入 `native_cache` 改变位置。构建源码固定为 BitNet `0b341e58` 与 llama.cpp `390c3077`，运行时应用 ReLU² 补丁以匹配公开蒸馏检查点。

## CLI

将每条结构化请求保存为 JSONL 一行，再运行：

```bash
bit-jev --model jinghao1632/bit-jev-2b-distilled --device cpu --input requests.jsonl --output results.jsonl --threads 8 --batch 128
```

如要训练或蒸馏，另装 `pip install 'bit-jev[train]'`。本包的 Apache-2.0 许可证只覆盖源码；模型权重没有单独的开放许可证，训练数据包含 Yelp 评论，权利方书面答复仍待收到。使用前请阅读 Hugging Face 模型卡。
