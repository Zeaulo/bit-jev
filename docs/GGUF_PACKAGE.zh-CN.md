# `bit-jev` pip 包：GGUF 加载与推理

[返回 README](../README.md) · [English](GGUF_PACKAGE.md) · [Hugging Face 模型卡](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) · [ModelScope 模型卡](https://www.modelscope.cn/models/JingHao9616/bit-jev-2b-distilled)

## 安装

```bash
pip install bit-jev -i https://pypi.org/simple --upgrade
python -m bit_jev.demo
python -c "from importlib.metadata import version; import bit_jev; print(version('bit-jev'), bit_jev.__file__)"
```

第二条命令会启动本机 Gradio 页面并打开 `http://127.0.0.1:7860`；第三条命令应显示 `0.13.11` 和当前环境的 `site-packages/bit_jev/__init__.py`。页面提供 Choice、Noul、Score 三个题型，首次提交才下载和加载模型。使用 `python -m bit_jev.demo --source modelscope` 可直接从 ModelScope 下载；`--device gpu` 选择 Vulkan。需要旧式一次性 JSON 结果时加 `--once`。`bit-jev-demo` 是终端命令，不是 Python 模块，因此 `python -m bit-jev-demo` 会报 `No module named bit-jev-demo`。

默认安装包含请求编码、Hugging Face 和 ModelScope 下载器，以及原生程序管理接口。约 1.19 GB 的 I2_S GGUF 不在 PyPI wheel 中；首次调用 `from_pretrained()` 时按需下载。默认先尝试 Hugging Face，连接失败后回退 ModelScope；可传 `source="modelscope"` 或运行 `python -m bit_jev.demo --source modelscope`，直接从 ModelScope 下载。Windows x64 且 CPU 支持 AVX2 时，0.13.11 平台 wheel 已携带 CPU 与 Vulkan GPU 推理程序，正常推理无需 Git、CMake、C++ 编译器或 Vulkan SDK。GPU 仍需支持 Vulkan 的显卡驱动与 `vulkan-1.dll`。其他平台及 CUDA 后端首次源码构建需要 [Git](https://git-scm.com/install/)、[CMake 3.28+](https://cmake.org/download/) 和 C++17 编译器；自行编译 Vulkan 需要 SDK，CUDA 需要 CUDA Toolkit。安装命令本身不下载模型或编译程序。

Git 只用于源码自动构建：获取微软 BitNet 的固定提交、初始化其中的 llama.cpp 子模块、核对提交并应用本项目的 ReLU² 兼容补丁。Windows x64 AVX2 的 CPU/Vulkan 推理直接启动 wheel 内对应程序，不调用 Git 或 CMake。若使用经过验证的自有原生程序，可通过 `binary=` 指定；若传入已准备并打好补丁的 `native_source=`，则强制走本地源码构建。两个预编译程序均由固定上游提交构建，关闭针对构建机的原生 CPU 指令特化，仍要求 AVX2；CPU EXE 静态链接 MSVC 运行库，Vulkan EXE 静态链接 MinGW 运行库、动态使用系统的 `vulkan-1.dll`。源码构建路径继续在下载大模型前检查工具。

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

仅在其他平台、CUDA 后端或明确传入 `native_source=` 时执行上面的检查。若 Git 或 CMake 不存在，从上面的官方链接安装；CMake 版本至少 3.28。若 `cl` 和 `g++` 均不可用，安装 [Visual Studio C++ Build Tools](https://learn.microsoft.com/cpp/build/vscpp-step-0-installation) 的 **Desktop development with C++** 组件。使用 MSVC 命令行工具时，打开 **x64 Native Tools Command Prompt**，再激活原来的 Python 环境。Python 导入名为 `bit_jev`，不是带连字符的 PyPI 包名 `bit-jev`。

如果旧版在 MSVC 编译 `main.cpp` 时出现 `error C2001: 常量中有换行符`，在 Windows x64 AVX2 机器上更新到 0.13.11：`python -m pip install --upgrade bit-jev==0.13.11 -i https://pypi.org/simple`。新版本会直接选择 wheel 内程序；已下载的模型缓存仍可复用。确认 pip 安装日志下载的是 `win_amd64.whl`，而不是源码包。没有 AVX2 的 CPU 会在加载前收到明确错误，需要自备适配该 CPU 的原生 `binary=`。

如果终端显示 `Looking in indexes: https://mirrors.aliyun.com/pypi/simple/` 并继续提示旧版 `Requirement already satisfied`，则镜像可能还没有提供新版本；使用本节开头的官方 PyPI 索引命令，并用 `importlib.metadata.version('bit-jev')` 和 `bit_jev.__file__` 同时核对**安装包元数据**与**实际导入路径**。仅打印 `bit_jev.__version__` 可能读到当前源码目录里的文件，不能证明当前 Python 环境已安装对应 wheel。请使用当前环境的 `python -m pip`，避免 `pip` 和 `python` 指向不同 Conda 环境。

若 `python -m pip show bit-jev` 同时显示 `Version: 0.4.3` 和 `Editable project location:`，说明当前环境保留了旧版可编辑安装。导入可能读取工作区中已更新为 0.13.11 的源码，但发行版元数据和控制台入口仍属于 0.4.3。这正是 `bit_jev.__version__` 看似正确、`bit-jev-demo` 却不存在的原因。若要保留训练/Unsloth 环境，建议另建干净的推理环境安装官方 wheel；若确定要把当前环境改为发行版安装，再卸载旧可编辑版并安装本节的 wheel。

如果安装日志还显示 `peft==0.14.0`、`unsloth ... requires peft>=0.18.0`，说明旧版 bit-jev 的依赖已经影响了这个 Conda 环境；0.13.11 的默认推理安装不会依赖 `peft`。建议先在新的 Python 3.11/3.12 环境中按上面的命令安装并运行示例，再单独处理原环境的 Unsloth 依赖。若出现 `WARNING: Ignoring invalid distribution ~umpy`，这是该环境已有的 NumPy 安装残留警告，与模型推理接口不是同一故障。

如果官方索引临时无法访问，可稍后重试或在可联网机器下载 [PyPI 发行文件](https://pypi.org/project/bit-jev/#files) 后离线安装。不要使用 0.11.10 的旧 wheel，因为它尚无 ModelScope 下载回退。


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

Windows x64 AVX2 上的 `device="gpu"` 直接启动 wheel 内的 Vulkan EXE，不需要 Vulkan SDK；显卡驱动必须提供 `vulkan-1.dll` 和可用设备。若系统报告缺少 `vulkan-1.dll`，请更新显卡厂商的驱动。只有操作者显式传入 `native_source=`，或在没有对应 wheel 的平台自行构建 Vulkan 后端时，才需要 SDK 的库、头文件和 `glslc`；原先报 `Could NOT find Vulkan` 的机器安装新 wheel 后无需重试该构建步骤。GPU 即使正常运行，也不保证比 CPU 快：I2_S Vulkan 路径可能保留部分权重和计算在 CPU。

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
