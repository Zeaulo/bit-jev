// 静态 Space 仅负责界面翻译；推理请求由嵌入的 ModelScope Gradio 页面处理。
const copy = {
  zh: {
    eyebrow: "真实模型 · 免费 CPU 演示",
    title: "普通电脑，CPU 就能跑 JEV，而且很快！",
    description: "在下方修改客服分流案例并运行。答案、概率和耗时来自真实模型；首次请求需下载约 1.19 GB 权重。",
    open: "打开在线测试 ↗",
    note: "此 Hugging Face 页面是免费静态入口。实际推理在 ModelScope 免费 2 vCPU 空间执行；若下方嵌入被浏览器阻止，请点击“打开在线测试”。页面推理时间不含首次下载与加载。",
  },
  en: {
    eyebrow: "Real model · free CPU demo",
    title: "Run structured JEV decisions on an everyday CPU.",
    description: "Edit the customer support example below and run it. The answer, probabilities, and compute time come from the real model. The first request downloads about 1.19 GB.",
    open: "Open the live test ↗",
    note: "This Hugging Face page is a free static entry point. Inference runs in a free 2-vCPU ModelScope Space. If your browser blocks the embed, choose “Open the live test.” Displayed compute time excludes download and loading.",
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

// 按钮只切换页面引导语言；嵌入页面自身包含中英两个选项卡。
for (const button of document.querySelectorAll("[data-lang]")) {
  button.addEventListener("click", () => setLanguage(button.dataset.lang));
}
