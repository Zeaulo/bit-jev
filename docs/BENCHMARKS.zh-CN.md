# 性能测量规范：用于未来可合法公开的模型文件

[English](BENCHMARKS.md) · [返回中文 README](../README.zh-CN.md)

当前纯源码发布**没有公开的 bit-jev 训练检查点实测值**。对应的权重、模型输出以及由该检查点得到的准确率、速度和内存报告暂缓公开。仓库中 [`test/`](../test/) 的脚本是测量工具；缺少兼容且有权使用的模型和输入集时，不能据此复现结果。

bit-jev 对明确列出的候选项打分，不逐 token 解码答案文本。因此，“生成输出 tokens/s”不适合表示其吞吐量。优先报告**单请求延迟**；必要时可报告**输入 tokens/s**，但要说明输入 token 的计数方法。

## 微软公开 BitNet 基础模型独立实测

这组数据只检验公开的 **[Microsoft BitNet b1.58 2B 原版 I2_S GGUF](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/tree/9f43072f69492cbd5bc5d5ebb085fec519686a93)** 在两条部署路径上的表现。两次测试使用同一文件：1,844,472,032 字节，SHA-256 为 `13939ce5030319a35db346e5dba7a3a3bd599dfc18b113a2a97446ff964714c5`。模型文件不收入本仓库。使用此固定旧修订，是因为[官方仓库对较新权重的异常已有公开反馈](https://github.com/microsoft/BitNet/issues/608)；不能把两种修订的结果混在一起。

![128-token 预填充与 32-token 解码的中位吞吐](figures/public-base-speed.svg)

![同一 GGUF 在两种路径下的进程内存与显存观测](figures/public-base-memory.svg)

| 测量项 | Ryzen 7 4800H，8 线程，CPU-only 二进制 | RTX 2060，Vulkan1 混合执行 |
| --- | ---: | ---: |
| 128 输入 token 预填充中位耗时 | 2,276.20 ms | 2,831.68 ms |
| 预填充中位吞吐 | 56.23 token/s | 45.20 token/s |
| 32 输出 token 解码中位耗时 | 7,008.26 ms | 9,000.01 ms |
| 解码中位吞吐 | 4.57 token/s | 3.56 token/s |
| 进程峰值 RSS，含加载和测试 | 1,230.19 MiB | 1,955.56 MiB |
| 整卡显存相对运行前基线的峰值增量 | 0 MiB | 722 MiB |
| 模型张量缓冲区的后端分配日志 | CPU_Mapped 1,751.06 MiB | CPU_Mapped 1,124.80 MiB；Vulkan1 627.93 MiB |

采用 `llama-bench -p 128 -n 32 -r 5 -t 8 -b 128 -ub 128`，各阶段预热，顺序运行 CPU 和 GPU。预填充与解码是 **llama-bench 的两种合成工作负载**，不能将两项耗时直接相加充作“一道题”的端到端耗时。吞吐排除模型加载；RSS 每 50 ms 采样，覆盖加载与测试；显存由 `nvidia-smi` 每 250 ms 采样。Windows WDDM 未返回 Vulkan 进程显存，因此仅报告**整卡**增量，可能受其他进程影响。31/31 层设置为 GPU 卸载，但 I2_S 部分运算仍由 CPU 映射权重执行。此路径不是纯 GPU，也不是 bit-jev 分类推理。CPU 在本机此负载上更快，不能外推到别的设备。

[逐次样本与硬件、二进制、模型哈希](benchmark-data/public-bitnet-base-2026-09-27.json)可直接下载。原始本地报告不含模型输出，图表由 [`test/render_public_base_figures.py`](../test/render_public_base_figures.py) 生成。复现时先运行 [`test/bootstrap_bitnet.py`](../test/bootstrap_bitnet.py) 取得固定 BitNet/llama.cpp 版本及 ReLU² 补丁，再在 `learning/bitnet/3rdparty/llama.cpp` 应用 [公开模型兼容补丁](../test/patches/public-bitnet25-compat.patch)。该补丁将原版 GGUF 的 `bitnet-25` 架构名映射到运行时，并读取独立 `output.weight`；它只服务于这项基础模型测量。分别以 `GGML_VULKAN=OFF` 与 `GGML_VULKAN=ON` 构建 `llama-bench`，再执行：

```powershell
python test/bootstrap_bitnet.py
git -C learning/bitnet/3rdparty/llama.cpp apply "$(Resolve-Path test/patches/public-bitnet25-compat.patch)"
cmake -S learning/bitnet/3rdparty/llama.cpp -B test/build-public-base -G "MinGW Makefiles" -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00" -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00"
cmake --build test/build-public-base --target llama-bench -j 8
cmake -S learning/bitnet/3rdparty/llama.cpp -B test/build-public-vulkan -G "MinGW Makefiles" -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00" -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00"
cmake --build test/build-public-vulkan --target llama-bench -j 8
python test/benchmark_public_base.py --model test/ggml-model-i2_s-original.gguf --cpu-binary test/build-public-base/bin/llama-bench.exe --gpu-binary test/build-public-vulkan/bin/llama-bench.exe --gpu-device Vulkan1 --threads 8 --prompt-tokens 128 --decode-tokens 32 --repetitions 5 --output test/public_base_llama_bench_final.json
python test/render_public_base_figures.py
```

上例构建命令对应 Windows MSYS2 UCRT 工具链；Vulkan 版还需可被 CMake 找到的 Vulkan 头文件和导入库。原版 GGUF 可以从上方固定修订下载到 `test/`；`--gpu-device` 应按本机 `llama-bench --list-devices` 的结果调整。复现依赖完整模型文件和同等补丁；两张图不支持 bit-jev 端到端速度、准确率或与 Kev 的倍数比较。

## CPU：把加载和常驻推理分开

1. 记录源码提交号、骨干与指针头文件的 SHA-256、分词器版本、量化格式、编译选项及原生程序哈希。
2. 记录 CPU 型号、物理核心数、容器允许的核心数或配额、NUMA 位置、实际线程数、批次大小、操作系统及可用内存。
3. 描述输入：请求数、总问题数、每题选项数、编码后的序列长度，以及共享 `state` 是否在多个问题中重复计算。
4. 区分进程启动、模型加载、分词、预热、原生计算和完整请求墙钟时间。模型常驻后的延迟与冷启动另报。
5. 保留多轮原始计时，并写明**进程峰值 RSS** 的采样方法。模型磁盘大小、进程 RSS 和整机内存占用不是同一数值。
6. 比较量化格式前检查分数是否有限、获胜选项是否一致、数值误差多大。单个答案一致不能证明准确率一致。

可用 [`prepare_native_benchmark.py`](../test/prepare_native_benchmark.py) 编码有权使用的 JSONL 请求，用 [`benchmark_native_formats.py`](../test/benchmark_native_formats.py)记录逐请求时间与 RSS，再用 [`summarize_cpu_format_benchmark.py`](../test/summarize_cpu_format_benchmark.py)在相同输入顺序下比较格式。生成的输入和报告应保存在根目录 `test/` 下；分享前先核查原始数据条款。

## CPU 与 GPU：说明对比的边界

两条路径应分别标注模型格式和计算精度、预热次数、GPU 同步方式、输入批次形状、内存或显存口径，以及计时起止点。**CPU I2_S 与 GPU 混合精度**的比较同时改变硬件和数值格式，它反映两套部署配置，不能独立证明硬件带来了多少加速。[`benchmark_gpu.py`](../test/benchmark_gpu.py)是实验性前向测试脚本，不代表已经验证的 GPU 服务端点。

CPU 核心数可能影响延迟，但内存带宽、主频、NUMA、输入长度、候选项数和问题序列条数同样重要。要判断线程扩展性，应在**同一台机器和同一批输入**上测试多个线程数。当前原生实现会为多问题请求重复计算共享 `state`。

## 内存图应该标明什么

| 量 | 计量对象 | 不能替代什么 |
| --- | --- | --- |
| 模型文件大小 | 存在磁盘上的权重文件字节数 | 不能代表进程运行时内存。 |
| CPU 峰值 RSS | 测量区间内 CPU 推理进程的常驻内存峰值 | 不能与文件大小直接等同，也不包括所有系统占用。 |
| GPU 显存分配峰值 | 指定 GPU 框架或监控接口统计的分配峰值 | 不能直接与 CPU RSS 作严格同口径比较。 |

若一张图同时画这三种量，必须在图例中写明计量对象和单位，并区分加载期间峰值与模型已加载后的请求峰值。不同模型大小、格式或输入形状要分组展示，不应省去实验条件。

## 准确率与概率校准

评估前固定模型和协议。写清数据集、使用权限、训练/验证/测试划分、去重方法、选项顺序、问题类型和每类样本数。开发集、锁定测试集和新来源数据应分别报告。权利条件允许公开时，再发布按类型拆分的准确率、CPU 与参考路径的数值一致性以及概率校准结果。少量冒烟输入不能支持总体可靠性判断。

## 将来公开结果的最小记录表

| 字段 | 需要提供的内容 |
| --- | --- |
| 模型文件 | 模型、分词器、指针头与源码哈希；公开分发依据。 |
| 输入 | 数据集与切分、数据来源权限、请求/问题数、候选项数、编码后长度。 |
| CPU | 硬件配额、量化格式、线程数、批次大小、编译选项。 |
| GPU | 硬件、数值精度、同步方式、显存统计口径。 |
| 时间 | 冷/热运行的原始记录、均值/中位数/尾延迟（样本数足够时）、准确计时区间。 |
| 内存 | 文件大小、CPU 峰值 RSS、GPU 分配峰值及采样方式。 |
| 正确性 | 数值有效性、候选项一致性，以及获准数据上的准确率和校准。 |

未来的对比图应链接到生成它的确切输入与报告。当前纯源码发布中的架构示意图不代表已经公开分发的 bit-jev 检查点跑分。
