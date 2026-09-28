# `bit-jev` pip 包：GGUF 加载与推理

[返回 README](../README.md) · [English](GGUF_PACKAGE.md) · [Hugging Face 模型卡](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)

## 安装

```bash
pip install bit-jev
```

默认安装包含请求编码、Hugging Face 下载器和原生程序管理接口。约 1.19 GB 的 I2_S GGUF 不在 PyPI wheel 中；首次调用 `from_pretrained()` 时下载到 Hugging Face 缓存。构建原生 runner 需要 [Git](https://git-scm.com/install/)、[CMake 3.28+](https://cmake.org/download/) 和 C++17 编译器。Vulkan GPU 模式需要 Vulkan SDK，CUDA 模式需要 CUDA Toolkit。安装命令本身不下载模型或编译程序。

Git 只用于首次自动构建：获取微软 BitNet 的固定提交、初始化其中的 llama.cpp 子模块、核对提交并应用本项目的 ReLU² 兼容补丁。正常推理只启动已编译的原生程序，不调用 Git。若使用经过验证的自有原生程序，可通过 `binary=` 跳过自动构建；若传入已准备并打好补丁的 `native_source=`，自动构建无需从网络检出源码。bit-jev 0.8.9 在下载大模型前检查 Git、CMake 及其版本，并为 Windows MSVC 编译目标启用 UTF-8 源码和字符串编码。

### Windows 首次加载前检查

在准备运行 bit-jev 的同一个 CMD 或 PowerShell 终端执行：

```bat
where.exe git
git --version
where.exe cmake
cmake --version
where.exe cl
where.exe g++
```

若 Git 或 CMake 不存在，从上面的官方链接安装；CMake 版本至少 3.28。若 `cl` 和 `g++` 均不可用，安装 [Visual Studio C++ Build Tools](https://learn.microsoft.com/cpp/build/vscpp-step-0-installation) 的 **Desktop development with C++** 组件。使用 MSVC 命令行工具时，打开 **x64 Native Tools Command Prompt**，再激活原来的 Python 环境。安装工具后重新打开终端；`pip install bit-jev` 成功不代表首次原生构建已完成。Python 导入名为 `bit_jev`，不是带连字符的 PyPI 包名 `bit-jev`。

如果 0.8.8 在 MSVC 编译 `main.cpp` 时出现 `error C2001: 常量中有换行符` 和大量后续语法错误，更新到 0.8.9：`python -m pip install --upgrade bit-jev==0.8.9 -i https://pypi.org/simple`。0.8.9 使用新的构建缓存目录，原先失败的 0.8.8 构建不会被复用；已下载的 Hugging Face 模型缓存仍可复用。本机在 MSVC 下完成固定源码编译，并对公开 GGUF 连续运行两次结构化请求。

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
