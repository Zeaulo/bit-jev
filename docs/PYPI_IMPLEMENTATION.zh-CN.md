# PyPI 与 GGUF 推理封装实施文档

记录日期：2026-09-28。执行范围：`pip install bit-jev`、公开 GGUF 模型下载、CPU/GPU 推理入口、GitHub 与 Hugging Face 文档。

## 当前事实

- 模型由 `backbone-i2_s.gguf`、`head.f32`、tokenizer/config 组成，位于 `jinghao1632/bit-jev-2b-distilled`。GGUF 文件约 1.19 GB，因此 PyPI 发行包只放代码、构建所需的小文件和下载入口；模型按需下载到 Hugging Face 缓存。
- 已有 `core/native/main.cpp` 使用固定版本 bitnet.cpp/llama.cpp，对每道题执行因果前向并读取候选项隐藏状态；模型在进程中常驻。CPU 模式不卸载层；Vulkan 与 CUDA 模式按选定设备申请 GPU 层。
- 公开模型权重没有单独的开放许可证；Yelp 训练数据许可申请尚无书面答复。包许可证只覆盖代码，下载入口须提示阅读模型卡。

## 实施步骤与验收

1. 增加独立的 GGUF Python API：下载模型、校验必需文件、寻找或构建原生 runner、加载后常驻、接受结构化请求、返回答案和耗时；提供关闭和上下文管理方法。
2. 原生 runner 增加 `cpu`/`cuda` 设备选项。CPU 不申请 GPU 层；CUDA 构建显式开启 `GGML_CUDA` 并申请全层卸载。缺少 CUDA 构建依赖或实际 GPU 时明确报错，不能静默退回 CPU。
3. 使 PyPI wheel 包含原生构建入口和固定上游补丁，自动构建只在首次加载时执行。安装本身不启动下载或编译。构建所需 Git、CMake、C++17 编译器，以及 CUDA 模式的 CUDA Toolkit，必须写明。
4. 在本机执行 CPU 结果一致性和常驻两次请求测试；GPU 仅在 CUDA 编译及运行均通过后标为已验证。构建 wheel、从 wheel 干净安装、检查归档内容与元数据，再上传 PyPI。
5. 中文默认 README 参考 bitnet.cpp 的标题、导航、新闻/状态、快速开始、性能、方法和文档结构，保留原始测量口径。同步更新英文 README、Hugging Face 模型卡、`versions/update.log` 和 `versions/project_overall/`。

## 发布约束

- 不把 PyPI 凭证、模型权重、训练原文、临时结果放进源码或发行包。
- 不把历史 AutoDL 的 FP16 GPU 数字标成 GGUF/CUDA 的实测速度。
- 如果 CUDA 环境无法实测，PyPI 与 README 必须准确说明支持构建方式和验证状态。

## 实施结果

- PyPI 已发布 `bit-jev 0.8.7`；wheel 58,929 字节，源码包 52,759 字节。公开索引、`twine check`、归档内容和 wheel 安装检查通过。
- Windows 本机隔离源码构建的 CPU 与 Vulkan runner 均加载公开 GGUF 并在同一进程连续处理两次手写请求。Vulkan 在 RTX 2060 推理时产生可见显存占用；两条路径最终答案一致，概率有小幅浮点差异。
- Hugging Face 双语模型卡已同步。CUDA 构建未实测，因为本机没有 CUDA Toolkit；历史 RTX 5090 FP16 数字仍独立标注。
- PyPI 0.8.8 修复首次加载体验：在下载 GGUF 前检查 Git、CMake 3.28+；缺少工具时给出官方安装链接和 Git 在固定源码、子模块、补丁流程中的用途。已缓存原生程序或传入 `binary=` 时跳过构建工具检查。Windows 排错步骤见 [GGUF 包指南](GGUF_PACKAGE.zh-CN.md#windows-首次加载前检查)。
- PyPI 0.8.9 为 MSVC 编译 bit-jev 原生目标添加 `/utf-8`，修复简体中文 Windows 上无 BOM UTF-8 源码被系统代码页误读的问题。本机使用固定 BitNet 与 llama.cpp 提交、仅 ReLU² 补丁通过 MSVC 编译，并对公开 GGUF 连续执行两次真实请求；发行 wheel 和源码包均通过归档与元数据检查。
- PyPI 0.9.9 增加 Windows x64 AVX2 CPU 预编译 wheel。固定上游与 ReLU² 补丁在 MSVC Release 下构建，关闭构建机 CPU 特化和 OpenMP，静态链接 MSVC 运行库；PE 依赖只剩 KERNEL32.dll 与 ADVAPI32.dll。预编译程序本体 3,881,472 字节，wheel 1,628,730 字节，源码包 1,617,019 字节。包内附 BitNet 和 llama.cpp 的 MIT 许可证文本。
- `native_build.py` 在已有缓存后查找 wheel 内程序，核对 Windows x64/AVX2 并直接返回；指定 `native_source=` 时继续源码构建，传入 `binary=` 时优先使用自备程序。非 Windows、GPU 后端仍按需构建。实际从 wheel 安装后屏蔽 Git/CMake 工具发现，公开 GGUF 能加载并连续两次返回同一结构化答案，原生单次计时分别为 4187.7684 ms 和 3813.8056 ms；这只是本机手写请求验收，不是跨机器性能基准。
- PyPI 0.10.10 新增 Windows x64 AVX2 Vulkan GPU 预编译 EXE。固定 BitNet 0b341e58、llama.cpp 390c3077 与 ReLU² 补丁，MinGW Release 开启 Vulkan/AVX2，关闭 GGML_NATIVE、BMI2 与 OpenMP，静态链接 GCC 运行库。EXE 为 45,801,472 字节；动态依赖 Windows 系统/UCRT DLL 与驱动提供的 `vulkan-1.dll`，不依赖 MinGW 或 Vulkan SDK DLL。
- `native_build.py` 现在在 Windows x64 AVX2 上按 `cpu` 和 `gpu`/`vulkan` 选择对应包内 EXE；GPU 在模型下载前检查系统 Vulkan loader。普通 CPU/GPU 推理无需 Git、CMake、C++ 编译器或 Vulkan SDK；只有显式源码编译 Vulkan 才要求 SDK 头文件、库和 `glslc`。原先无法访问的 SDK 页面不再作为故障提示，缺 SDK 时指向项目构建指南。CUDA 与其他平台仍使用源码路径。
- 0.10.10 wheel 为 16,388,760 字节、sdist 为 16,133,285 字节，均通过发行包内容检查和 `twine check`。wheel 隔离安装后在 RTX 2060 上限制 PATH 为 Windows 系统目录、屏蔽 Git/CMake 发现，公开 GGUF 连续两次 Vulkan 推理返回一致结构化答案。该结果验证无需 SDK 的运行路径，不代表 GPU 一定比 CPU 快。
