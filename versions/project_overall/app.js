/* 导航高亮跟随当前阅读位置，方便在长篇内部手册中定位。 */
const sections = [...document.querySelectorAll("main section[id]")];
const links = [...document.querySelectorAll(".nav-pill")];

/* 观察各章节进入视口的时机，更新对应导航项。 */
const observer = new IntersectionObserver((entries) => {
  /* 只处理实际进入视口的章节，避免滚动时重复切换。 */
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    /* 章节 id 与导航锚点保持一一对应。 */
    for (const link of links) {
      link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
    }
  }
}, { rootMargin: "-22% 0px -62% 0px" });

/* 每个带 id 的主章节都参加导航观察。 */
for (const section of sections) observer.observe(section);
