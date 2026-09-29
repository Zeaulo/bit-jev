// 静态 Space 仅负责界面翻译；推理请求由嵌入的 ModelScope Gradio 页面处理。
const copy = {
  zh: {
    eyebrow: "真实模型 · 免费 CPU 演示",
    title: "普通电脑，CPU 就能跑 JEV，而且很快！",
    description: "选择选择题、是非题或等级题，填写问题后运行真实模型。",
    open: "打开在线测试 ↗",
    note: "说明 / Note: 免费空间使用 2 vCPU 进行推理。",
  },
  en: {
    eyebrow: "Real model · free CPU demo",
    title: "Run structured JEV decisions on an everyday CPU.",
    description: "Choose Choice, Noul, or Score, enter a question, and run the real model.",
    open: "Open the live test ↗",
    note: "Note: The free Space uses 2 vCPUs for inference.",
  },
};

// 仅替换带 data-i18n 的文本节点，避免影响外部模型测试 iframe。
function setLanguage(language) {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = copy[language][node.dataset.i18n];
  }
  for (const button of document.querySelectorAll("[data-lang]")) {
    button.setAttribute("aria-pressed", String(button.dataset.lang === language));
  }
}

// 按钮只切换页面引导语言；嵌入页面有独立的中英文语言开关。
for (const button of document.querySelectorAll("[data-lang]")) {
  button.addEventListener("click", () => setLanguage(button.dataset.lang));
}
