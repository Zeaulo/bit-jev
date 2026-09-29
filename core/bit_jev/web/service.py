"""Web 表单与常驻 GGUF 模型之间的请求处理。"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..gguf import BitJev, DEFAULT_MODEL
from .logic import build_choice_request, build_noul_request, build_score_request, present_result
from .presentation import render_probabilities


@dataclass(frozen=True)
class WebConfig:
    """本地或公开空间启动时固定的模型与页面设置。"""

    model: str = DEFAULT_MODEL
    source: str = "auto"
    device: str = "cpu"
    threads: int = 4
    gpu_index: int | None = None
    binary: str | None = None
    public_space: bool = False


class ModelSession:
    """按需下载模型，并为六个页面回调复用同一原生进程。"""

    def __init__(self, config: WebConfig) -> None:
        """保存启动设置；此时不下载权重。"""
        self.config = config
        self._model: BitJev | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def get_model(self) -> BitJev:
        """在第一条有效请求出现时加载一次模型。"""
        with self._load_lock:
            if self._model is None:
                # 预编译程序选择由 gguf 模块完成；本地页面不重复实现构建逻辑。
                self._model = BitJev.from_pretrained(
                    self.config.model, source=self.config.source,
                    device=self.config.device, threads=self.config.threads,
                    gpu_index=self.config.gpu_index, binary=self.config.binary,
                )
            return self._model

    def infer(self, kind: str, language: str, *values: object) -> tuple[str, str, dict]:
        """校验表单，串行调用常驻模型并展示真实返回值。"""
        try:
            if kind == "noul":
                state, question, no_description, yes_description = values
                request = build_noul_request(state, question, language,
                                             no_description, yes_description)
            else:
                count, state, question, *entries = values
                # 隐藏行也会传给回调；只有当前已添加的二至四项进入请求。
                active_count = int(count)
                options = entries[:4][:active_count]
                descriptions = entries[4:][:active_count]
                request = (build_choice_request(state, question, options, descriptions)
                           if kind == "choice" else
                           build_score_request(state, question, options, descriptions))
            with self._infer_lock:
                result = self.get_model().infer(request)
            summary, detail = present_result(result, language)
            return summary, render_probabilities(detail, language), detail
        except Exception as error:
            label = "运行失败" if language == "zh" else "Inference failed"
            return f"{label}: `{type(error).__name__}: {error}`", "", {}

    def close(self) -> None:
        """退出页面进程时释放原生模型。"""
        with self._load_lock:
            if self._model is not None:
                self._model.close()
                self._model = None
