# CPU 源码构建与使用

[English](CPU_QUICKSTART.md) · [返回中文 README](../README.md)

本指南面向需要手动构建原生 I2_S CPU 程序的开发者。普通安装、自动下载与 GPU 选项请看 [GGUF pip 包指南](GGUF_PACKAGE.zh-CN.md)。bit-jev-2b-distilled 模型包位于 [Hugging Face](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)；本仓库不存放检查点。[微软公开基础模型基准](BENCHMARKS.zh-CN.md#微软公开-bitnet-基础模型独立实测)测的是另一个原版骨干，不是 bit-jev 分类结果。下载前请阅读 Hugging Face 模型卡中的 Yelp 数据说明、指标边界和许可证状态。

Python 启动器负责编码请求并管理常驻原生进程；原生程序完成骨干推理和指针头评分。这条路径提供命令行 JSONL 接口，不是 HTTP 服务。

## 1. 克隆、安装并取得固定版本上游源码

准备 Python 3.11 或 3.12、Git、CMake 3.28+ 和 C++17 编译器。先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择适用于平台的 PyTorch，再从 PowerShell 执行：

```powershell
git clone https://github.com/Zeaulo/bit-jev.git
Set-Location bit-jev
python -m pip install -e './core'
python test/bootstrap_bitnet.py
python test/bootstrap_bitnet.py --check
```

首次运行 `bootstrap_bitnet.py` 需要网络下载固定版本的 BitNet 与 llama.cpp；脚本应用仓库自带的转换器和 ReLU² 兼容补丁。`--check` 检查源码版本和补丁状态。公开网络下载能否完成取决于当时的 GitHub 连接情况。

## 2. 编译原生 CPU 程序

Windows + MinGW Makefiles：

```powershell
cmake -S core/native -B core/build/bit-jev-cpu -G 'MinGW Makefiles' -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build core/build/bit-jev-cpu --target bit-jev-cpu -j 4
```

Linux + 默认 CMake 生成器：

```bash
python3 test/bootstrap_bitnet.py
python3 test/bootstrap_bitnet.py --check
cmake -S core/native -B core/build/bit-jev-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build core/build/bit-jev-cpu --target bit-jev-cpu -j 4
```

`GGML_NATIVE=ON` 会针对**编译机 CPU** 优化，换目标处理器时建议重新编译。Windows 可执行文件通常位于 `core/build/bit-jev-cpu/bit-jev-cpu.exe`；Linux 文件名没有 `.exe`。只编译程序不需要模型权重。

## 3. 准备请求

在本项目根目录的 `test/my-request.jsonl` 中保存 UTF-8 文本，每行一个完整 JSON 对象。例如：

```json
{"state":"A customer reports a duplicate charge.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payment and refund issues","shipping":"Delivery issues"}}}}
```

`questions` 不能为空。请求可同时包含 `choice`、`noul` 和 `score`。上面只有输入，没有“预期输出”；公开 I2_S 检查点可以从 Hugging Face 下载，实际答案需要执行模型后读取。

## 4. 下载并运行公开 I2_S 模型

先从 Hugging Face 下载模型文件。`--run` 指向 tokenizer/config 目录；`--artifact` 指向含 `backbone-i2_s.gguf` 和 `head.f32` 的导出目录。本模型包把两类文件放在同一目录，因此两个参数使用相同路径。

```powershell
hf download jinghao1632/bit-jev-2b-distilled --local-dir './models/bit-jev-2b-distilled'
python -m bit_jev.cpu `
  --run './models/bit-jev-2b-distilled' `
  --artifact './models/bit-jev-2b-distilled' `
  --binary './core/build/bit-jev-cpu/bit-jev-cpu.exe' `
  --input './test/my-request.jsonl' `
  --out './test/my-result.jsonl' `
  --threads 8 --batch 128
```

Linux 用户需换成不带 `.exe` 的程序路径，并使用 shell 对应的续行语法。输出包含结构化答案、选项分数与概率以及原生计算耗时。该模型的 Yelp 数据许可申请仍待权利方书面答复，模型卡没有为权重指定开放许可证；代码 Apache-2.0 不覆盖模型权重。请将自建输入、运行结果和临时脚本放在根目录 `test/` 下。

当前原生程序**每个问题运行一条因果序列**。它不逐 token 生成答案，但多个问题会重复读取共享内容，长候选项也会增加工作量。加载耗时、常驻推理耗时和峰值内存请按[性能测量规范](BENCHMARKS.zh-CN.md)分开记录。
