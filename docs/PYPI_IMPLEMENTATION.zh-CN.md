# PyPI 与 GGUF 推理封装实施文档

记录日期：2026-09-28。执行范围：`pip install bit-jev`、公开 GGUF 模型下载、CPU/GPU 推理入口、GitHub 与 Hugging Face 文档。

## 当前事实

- 模型由 `backbone-i2_s.gguf`、`head.f32`、tokenizer/config 组成，位于 `jinghao1632/bit-jev-2b-distilled`。GGUF 文件约 1.19 GB，因此 PyPI 发行包只放代码、构建所需的小文件和下载入口；模型按需下载到 Hugging Face 缓存。
- 已有 `core/native/main.cpp` 使用固定版本 bitnet.cpp/llama.cpp，对每道题执行因果前向并读取候选项隐藏状态；模型在进程中常驻。当前构建是 CPU，GPU 层数固定为零。
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
