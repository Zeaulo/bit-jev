# `bit-jev` pip 包：GGUF 加载与推理

[返回 README](../README.md) · [English](GGUF_PACKAGE.md) · [Hugging Face 模型卡](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)

## 安装

```bash
pip install bit-jev
```

默认安装包含请求编码、Hugging Face 下载器和原生程序管理接口。约 1.19 GB 的 I2_S GGUF 不在 PyPI wheel 中；首次调用 `from_pretrained()` 时下载到 Hugging Face 缓存。Windows x64 且 CPU 支持 AVX2 时，0.9.10 平台 wheel 已携带 CPU 推理程序，推理无需 Git、CMake 或 C++ 编译器。其他平台及 GPU 后端首次构建需要 [Git](https://git-scm.com/install/)、[CMake 3.28+](https://cmake.org/download/) 和 C++17 编译器；Vulkan GPU 模式还需要 Vulkan SDK，CUDA 模式还需要 CUDA Toolkit。安装命令本身不下载模型或编译程序。

Git 只用于源码自动构建：获取微软 BitNet 的固定提交、初始化其中的 llama.cpp 子模块、核对提交并应用本项目的 ReLU² 兼容补丁。Windows x64 AVX2 CPU 直接启动 wheel 内程序，不调用 Git 或 CMake。若使用经过验证的自有原生程序，可通过 `binary=` 指定；若传入已准备并打好补丁的 `native_source=`，则强制走本地源码构建。预编译程序由固定上游提交构建，已关闭针对构建机的原生 CPU 指令特化，并静态链接 MSVC 运行库；当前仍要求 AVX2。源码构建路径继续在下载大模型前检查工具。

### Windows 源码构建前检查

在准备运行 bit-jev 的同一个 CMD 或 PowerShell 终端执行：

```bat
where.exe git
git --version
where.exe cmake
cmake --version
where.exe cl
where.exe g++
```

仅在其他平台、GPU 后端或明确传入 `native_source=` 时执行上面的检查。若 Git 或 CMake 不存在，从上面的官方链接安装；CMake 版本至少 3.28。若 `cl` 和 `g++` 均不可用，安装 [Visual Studio C++ Build Tools](https://learn.microsoft.com/cpp/build/vscpp-step-0-installation) 的 **Desktop development with C++** 组件。使用 MSVC 命令行工具时，打开 **x64 Native Tools Command Prompt**，再激活原来的 Python 环境。Python 导入名为 `bit_jev`，不是带连字符的 PyPI 包名 `bit-jev`。

如果旧版在 MSVC 编译 `main.cpp` 时出现 `error C2001: 常量中有换行符`，在 Windows x64 AVX2 机器上更新到 0.9.10：`python -m pip install --upgrade bit-jev==0.9.10 -i https://pypi.org/simple`。新版本会直接选择 wheel 内程序；已下载的 Hugging Face 模型缓存仍可复用。确认 pip 安装日志下载的是 `win_amd64.whl`，而不是源码包。没有 AVX2 的 CPU 会在加载前收到明确错误，需要自备适配该 CPU 的原生 `binary=`。

如果终端显示 `Looking in indexes: https://mirrors.aliyun.com/pypi/simple/` 并继续提示旧版 `Requirement already satisfied`，则镜像还没有提供新版本。使用 `python -m pip install --upgrade --no-cache-dir --index-url https://pypi.org/simple bit-jev==0.9.10`，再用 `python -c "import bit_jev; print(bit_jev.__version__)"` 核对运行时版本。请使用当前环境的 `python -m pip`，避免 `pip` 和 `python` 指向不同 Conda 环境。

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

Windows 上的 `device="gpu"` 首次运行会编译 Vulkan 程序，需要安装[完整的 LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)，仅安装显卡驱动或 Vulkan Runtime 不够。安装后重开终端，执行 `echo %VULKAN_SDK%`、`where.exe glslc`、`glslc --version`；CMake 还需要能找到 SDK 的 `Include/vulkan/vulkan.h` 与 `Lib/vulkan-1.lib`。若 CMake 提示 `Could NOT find Vulkan (missing: Vulkan_LIBRARY Vulkan_INCLUDE_DIR glslc)`，问题在 SDK 配置，CPU 预编译推理不受影响。

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
