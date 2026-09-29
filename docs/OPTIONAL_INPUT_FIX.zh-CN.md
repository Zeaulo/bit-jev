# 可选说明为空时推理失败：实施文档

## 现象与原因

用户在 Choice 页面直接点击“运行真实推理”时，返回 `ValueError: 选项 1 说明 必须是文本 / must be text`。已发布 Gradio 5 页面 `/config` 显示未展开的说明 Textbox 初始值为 JSON `null`。`web/logic.py` 的 `clean_text()` 对所有非字符串都报错，因此可选说明的 `None` 在模型加载前被拒绝。背景和 Noul、Score 的可选说明使用同一清理函数，存在相同风险。

## 修改范围与顺序

1. 先在不下载模型的回归测试中覆盖 `None`、空字符串和非法数字：可选背景与说明接受 `None`，继续采用“常见问题”或复制对应选项文本；必填问题与选项依旧拒绝 `None`，数字依旧报类型错误。
2. 在 `core/bit_jev/web/logic.py` 的统一文本入口处理可选 `None`，保持校验、长度限制和各题型调用顺序。给 `core/bit_jev/web/forms.py` 的可选 Textbox 显式空字符串初值，使网页默认提交稳定；将中文 Choice 默认问题改为“2027年，如果拿出1000美金，我应该买比特币还是黄金”，选项 A/B 改为“比特币”“黄金”。英文 Choice 示例同步为同义问题和 Bitcoin/Gold，避免语言切换后出现旧示例。
3. 测试默认表单构造的真实输入契约，再从独立安装的 wheel 验证本机页面；使用公开 Space API 验证未填写说明的默认请求和新示例。模型输出只展示实验性候选分数，不作为投资建议。
4. 按仓库规则把项目版本从 v0.19.9 升至 v0.19.10，把 PyPI 包从 0.13.11 升至 0.13.12；同步 `versions/update.log`、`versions/project_overall/index.html`、GitHub/PyPI 快速开始、双站点模型卡和 ModelScope Space Dockerfile。构建并发布 PyPI 后再重建在线空间，最后推送 GitHub。

## 验收标准

- 默认中文 Choice 不展开背景和说明，直接点击推理，不再出现“选项 1 说明必须是文本”。
- Choice、Noul、Score 的空可选字段都能进入模型；必填问题/选项仍有明确错误。
- 默认中文 Choice 在网页中显示指定的 2027 年比特币／黄金问题与两项候选，英文页面显示对应译文。
- PyPI 0.13.12、ModelScope 空间与 GitHub 中的代码和版本一致；Hugging Face 静态 Space 继续嵌入同一 ModelScope 页面。
