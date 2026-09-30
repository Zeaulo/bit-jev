"""本机与在线页面共用的主题样式。"""

# 使用 Gradio 在当前主题下提供的颜色；样式限定在项目组件内。
PAGE_CSS = """
.gradio-container { max-width: 1020px !important; margin: 0 auto !important; }
.decision-form { padding: 18px !important; }
.decision-summary { margin-top: 14px; font-size: 1.05rem; }
.jev-results {
  padding: 18px 20px;
  border: 1px solid var(--border-color-primary, #dce5f2);
  border-radius: 16px;
  background: var(--background-fill-secondary, #f8fbff);
  color: var(--body-text-color, #273951);
}
/* 显式设置子元素颜色，避免 Markdown 排版规则覆盖概率文字。 */
.jev-results h3, .jev-results .jev-prob-heading,
.jev-results .jev-prob-heading span, .jev-results .jev-prob-heading strong {
  color: var(--body-text-color, #273951);
}
.jev-results h3 { margin: 0 0 14px; font-size: 1rem; }
.jev-prob-row + .jev-prob-row { margin-top: 16px; }
.jev-prob-heading { display: flex; justify-content: space-between; gap: 12px;
  margin-bottom: 7px; overflow-wrap: anywhere; }
.jev-prob-heading span { min-width: 0; }
.jev-prob-heading strong { white-space: nowrap; flex-shrink: 0; }
.jev-prob-track { height: 12px; border-radius: 999px;
  background: var(--border-color-primary, #e5ebf5); overflow: hidden; }
.jev-prob-fill { height: 100%; border-radius: inherit; background: #5865e8; }
/* Gradio 将 dark 类加在主题祖先节点；暗色底上提高概率条亮度。 */
.dark .jev-results .jev-prob-fill { background: #a5b4fc; }
@media (max-width: 640px) { .decision-form { padding: 10px !important; }
  .jev-results { padding: 14px; } }
"""
