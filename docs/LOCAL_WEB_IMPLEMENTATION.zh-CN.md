# pip 安装后启动本地 Gradio 页面：实施文档

编写时间：2026-09-29。对应项目版本目标：v0.19.9；PyPI 包目标：0.13.11。

## 目标与入口

GitHub 中文和英文首页的置顶安装命令统一改为 `pip install bit-jev -i https://pypi.org/simple --upgrade`。紧接着运行 `python -m bit_jev.demo`，在本机 `127.0.0.1:7860` 启动 Gradio 页面并打开浏览器。页面提供与公开 ModelScope Space 同源的 Choice、Noul、Score 三个任务 Tab、中英语言切换、可折叠背景与说明，以及 Choice 和 Score 可增删的二至四项输入。默认使用 CPU；启动参数可选择 Vulkan GPU、线程数、模型来源及本地模型目录。

## 代码安排

1. 将表单和请求转换逻辑放进可安装包的 `core/bit_jev/web/`。ModelScope Space 的薄入口与本地入口共用表单和校验逻辑，避免两套页面逐渐分叉。
2. 新增本地 Web 启动模块，在首次提交题目时才调用 `BitJev.from_pretrained()`，之后复用常驻进程；页面启动本身不下载约 1.19 GB 模型。服务器仅绑定本机回环地址，不对局域网开放。
3. `python -m bit_jev.demo` 默认启动页面；保留 `--once` 作为原有的单题 JSON 命令模式。`bit-jev-demo` 使用相同入口。原有 Python `BitJev` API 不改变。
4. PyPI 默认依赖加入 Gradio 5，更新包版本和安装文档。Windows x64 wheel 继续携带 CPU 与 Vulkan 预编译程序；其他平台仍按现有源码构建边界说明。
5. ModelScope Docker Space 改为从新包调用共用页面，并重新部署。Hugging Face 静态 Space 继续嵌入该页面；更新模型卡与首页的本地启动说明。

## 页面体验修正

现有公开页面截图显示：选项说明在过窄的右栏中逐字换行，原始 JSON 占据首屏，表单与结果并排产生大块空白。共用页面改为单列阅读顺序：问题 → 背景 → 选项／等级 → 运行按钮 → 可读答案与概率条。每项右侧的“＋ 说明”只负责展开当前项目下方的整行说明输入；原始 JSON 放进折叠的高级信息。Choice 与 Score 都支持添加与删除任意项目，允许二至四项；删除后将后续项目文本与说明一同前移，达到四项后禁用添加按钮，只剩两项时隐藏删除按钮。中英两种语言均沿用同一布局，优先保证窄屏不出现横向溢出或单字竖排。

## 验证与发布

在项目根目录 `test/` 更新命令入口、表单与发行包测试。验证页面无需模型即可启动，浏览器 API 暴露六种中英文推理入口，旧 `--once` 仍调用真实 API 契约。构建 Windows wheel 和源码包，检查入口、Gradio 依赖、原生二进制与许可证，再在隔离环境从 wheel 启动页面。发布 PyPI 0.13.11 后，用官方 PyPI 索引安装验证；之后部署 ModelScope Space，检查其三题型推理和 Hugging Face 嵌入。所有代码变更写入 `versions/update.log`，同步 `versions/project_overall/index.html`。

## 兼容与边界

旧版 `python -m bit_jev.demo` 会直接打印单题结果；新默认行为是本地页面，因此需要脚本化单题调用的用户改用 `--once`。页面报告的 `latency_ms` 仅为原生计算耗时，首次模型下载与加载另计。服务器默认只监听 `127.0.0.1`；任何公开网络访问都需要用户自行明确配置和保护。
