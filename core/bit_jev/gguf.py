"""公开 I2_S GGUF 模型的下载、常驻加载与 CPU/GPU 结构化推理接口。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .api import to_answers
from .cpu import prepare_request
from .encoding import load_tokenizer
from .native_build import build_native, preflight_native


# 模型权重不属于 PyPI wheel；默认只按需获取运行所需文件。
DEFAULT_MODEL = "jinghao1632/bit-jev-2b-distilled"
DEFAULT_MODELSCOPE_MODEL = "JingHao9616/bit-jev-2b-distilled"
MODEL_FILES = ("backbone-i2_s.gguf", "head.f32", "config.json", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "pointer.json")


def download_model(repo_id: str = DEFAULT_MODEL, *, revision: str | None = None,
                   local_dir: str | Path | None = None, source: str | None = None) -> Path:
    """优先复用本地文件，再从指定模型站点下载推理所需文件。"""
    # 现有本地目录直接使用，适合离线部署和用户自建兼容模型。
    local_path = Path(repo_id).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    # 显式参数优先于环境变量；默认先尝试 Hugging Face，再回退至 ModelScope。
    selected_source = (source or os.environ.get("BIT_JEV_MODEL_SOURCE", "auto")).lower()
    if selected_source not in {"auto", "huggingface", "modelscope"}:
        raise ValueError("source 必须为 auto、huggingface 或 modelscope")
    # 文件筛选阻止意外下载仓库中的训练分片或其他大文件。
    allowed_files = [*MODEL_FILES, "SHA256SUMS.json", "README.md"]
    errors: list[tuple[str, Exception]] = []
    for current_source in (("huggingface", "modelscope") if selected_source == "auto"
                           else (selected_source,)):
        try:
            if current_source == "huggingface":
                from huggingface_hub import snapshot_download

                downloaded = snapshot_download(
                    repo_id=repo_id, revision=revision, allow_patterns=allowed_files,
                    local_dir=str(local_dir) if local_dir else None,
                    etag_timeout=5,
                )
            else:
                from modelscope_hub import HubApi

                # 默认公开检查点使用 ModelScope 账号下的同名仓库。
                model_id = DEFAULT_MODELSCOPE_MODEL if repo_id == DEFAULT_MODEL else repo_id
                downloaded = HubApi().download_repo(
                    model_id, "model", revision=revision, allow_patterns=allowed_files,
                    local_dir=str(local_dir) if local_dir else None,
                )
            return Path(downloaded).resolve()
        except Exception as error:
            errors.append((current_source, error))
            if selected_source != "auto":
                raise
    # 两站点均不可用时保留各自错误，便于判断网络或仓库问题。
    details = "; ".join(f"{name}: {error}" for name, error in errors)
    raise RuntimeError(f"模型下载失败，已尝试 Hugging Face 和 ModelScope：{details}") from errors[-1][1]


class BitJev:
    """管理一个常驻原生进程，并逐次返回 TypeSafe 风格的决策结果。"""

    def __init__(self, model_dir: str | Path, *, device: str = "cpu", threads: int = 4,
                 batch: int = 256, binary: str | Path | None = None,
                 native_cache: str | Path | None = None,
                 native_source: str | Path | None = None,
                 gpu_index: int | None = None):
        """载入 GGUF 与指针头，并等待原生进程确认加载完成。"""
        # gpu 是 Vulkan 的便捷别名；cuda 必须显式选择并由 CUDA Toolkit 编译。
        backend = "vulkan" if device == "gpu" else device
        if backend not in {"cpu", "vulkan", "cuda"}:
            raise ValueError("device 必须为 cpu、gpu、vulkan 或 cuda")
        if threads < 1 or batch < 1:
            raise ValueError("threads 和 batch 必须大于零")
        if gpu_index is not None and (backend == "cpu" or gpu_index < 0):
            raise ValueError("gpu_index 仅适用于 GPU 模式，且必须是非负整数")
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = backend
        self.threads = threads
        self.batch = batch
        # 在启动耗时较高的原生程序前校验文件，错误能直接定位到模型包。
        missing = [name for name in MODEL_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"模型目录缺少文件：{', '.join(missing)}")
        self.tokenizer = load_tokenizer(str(self.model_dir))
        self.binary = Path(binary).expanduser().resolve() if binary else build_native(
            backend, cache_dir=native_cache, source_dir=native_source)
        if not self.binary.is_file():
            raise FileNotFoundError(f"找不到原生程序：{self.binary}")
        # 临时日志文件避免 stderr 管道填满阻塞模型加载；关闭时自动清理。
        self._error_log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        command = [str(self.binary), "--model", str(self.model_dir / "backbone-i2_s.gguf"),
                   "--head", str(self.model_dir / "head.f32"), "--device", backend,
                   "--threads", str(threads), "--batch", str(batch), "--ready"]
        # 设备序号只作用于子进程，避免改变调用者进程的全局 GPU 可见性。
        native_environment = os.environ.copy()
        if gpu_index is not None:
            variable = "GGML_VK_VISIBLE_DEVICES" if backend == "vulkan" else "CUDA_VISIBLE_DEVICES"
            native_environment[variable] = str(gpu_index)
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=self._error_log,
                                             text=True, encoding="utf-8", bufsize=1,
                                             env=native_environment)
            ready_line = self._process.stdout.readline()
            if not ready_line or json.loads(ready_line).get("ready") is not True:
                raise RuntimeError(self._diagnostic("模型未返回加载完成握手"))
        except Exception:
            self.close()
            raise

    @classmethod
    def from_pretrained(cls, model: str = DEFAULT_MODEL, *, revision: str | None = None,
                        local_dir: str | Path | None = None, source: str | None = None,
                        **kwargs: Any) -> "BitJev":
        """按需从可用站点下载模型，再构建或复用原生程序并加载。"""
        # 缺少构建工具时先给出安装链接，避免下载大模型后才失败。
        if kwargs.get("binary") is None:
            preflight_native(kwargs.get("device", "cpu"),
                             cache_dir=kwargs.get("native_cache"),
                             source_dir=kwargs.get("native_source"))
        # model 可以是公开仓库 ID，也可以是已下载的本地模型目录。
        model_dir = download_model(model, revision=revision, local_dir=local_dir,
                                   source=source)
        return cls(model_dir, **kwargs)

    def _diagnostic(self, message: str) -> str:
        """在原生进程退出后附加最后的 stderr 诊断。"""
        self._error_log.flush()
        self._error_log.seek(0)
        detail = self._error_log.read()[-3000:].strip()
        return f"{message}：{detail}" if detail else message

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """对一条结构化请求进行推理；模型加载时间不计入 latency_ms。"""
        if self._process is None:
            raise RuntimeError("模型已关闭；请重新调用 from_pretrained 加载")
        if self._process.poll() is not None:
            raise RuntimeError(self._diagnostic("原生进程已退出"))
        # 与既有 JSONL CPU 路径共用编码契约和答案格式。
        prepared, metadata = prepare_request(request, self.tokenizer)
        self._process.stdin.write(json.dumps(prepared, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError(self._diagnostic("原生进程未返回推理结果"))
        response = json.loads(response_line)
        probabilities = response["probabilities"]
        logits = response["logits"]
        if len(probabilities) != len(metadata) or len(logits) != len(metadata):
            raise RuntimeError("原生结果的问题数与输入不一致")
        return {"answers": to_answers(probabilities, metadata), "logits": logits,
                "probabilities": probabilities, "latency_ms": response["latency_ms"],
                "device": self.device}

    def infer_many(self, requests: Iterable[dict[str, Any]]):
        """复用同一常驻模型，按传入顺序逐题返回结果。"""
        for request in requests:
            yield self.infer(request)

    def close(self) -> None:
        """关闭输入并释放原生模型、进程和诊断文件。"""
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            self._process = None
        error_log = getattr(self, "_error_log", None)
        if error_log is not None:
            error_log.close()
            self._error_log = None

    def __enter__(self) -> "BitJev":
        """允许用 with 语句管理模型生命周期。"""
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        """离开上下文时关闭模型。"""
        self.close()


def main(argv: list[str] | None = None) -> None:
    """提供 pip 安装后的 bit-jev 命令行 JSONL 入口。"""
    parser = argparse.ArgumentParser(description="bit-jev GGUF 结构化推理")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型仓库 ID 或本地模型目录")
    parser.add_argument("--source", choices=("auto", "huggingface", "modelscope"),
                        default="auto", help="模型下载来源；auto 在 Hugging Face 失败时回退 ModelScope")
    parser.add_argument("--device", choices=("cpu", "gpu", "vulkan", "cuda"), default="cpu")
    parser.add_argument("--input", required=True, help="UTF-8 JSONL 输入路径")
    parser.add_argument("--output", required=True, help="UTF-8 JSONL 输出路径")
    parser.add_argument("--binary", help="可选：已编译的 bit-jev 原生程序")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gpu-index", type=int, help="Vulkan 或 CUDA 可见设备序号")
    args = parser.parse_args(argv)
    # 一次打开模型，整个输入文件共用同一个原生进程。
    with BitJev.from_pretrained(args.model, source=args.source, device=args.device, binary=args.binary,
                                threads=args.threads, batch=args.batch,
                                gpu_index=args.gpu_index) as model:
        with open(args.input, encoding="utf-8") as source, open(args.output, "w", encoding="utf-8") as target:
            for line in source:
                if line.strip():
                    result = model.infer(json.loads(line))
                    target.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
