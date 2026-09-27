"""Model and dataset download backend: Hugging Face or ModelScope, picked per call or globally.

Why this exists: AutoDL instances (and mainland users generally) cannot reach huggingface.co but
can reach modelscope.cn, which mirrors every checkpoint bit-jev needs. Callers use snapshot,
load_tokenizer, load_config, load_model; --hub or BIT_JEV_HUB picks:

  hf           huggingface.co (HF_ENDPOINT mirror if set)
  modelscope   modelscope.cn                                [default on AutoDL]
  auto         modelscope when BIT_JEV_ON_AUTODL / MODELSCOPE_CACHE is set, else hf

bit_jev's default repos (e.g. microsoft/bitnet-b1.58-2B-4T-bf16) are translated to their
ModelScope mirrors (AI-ModelScope/...) when ModelScope is active; explicit ids pass through.
"""
import os

MS_MIRRORS = {
    "microsoft/bitnet-b1.58-2B-4T-bf16": "AI-ModelScope/bitnet-b1.58-2B-4T-bf16",
    "microsoft/BitNet-b1.58-2B-4T": "AI-ModelScope/bitnet-b1.58-2B-4T",
    "microsoft/bitnet-embedding-0.6b": "AI-ModelScope/bitnet-embedding-0.6b",
    "microsoft/bitnet-embedding-270m": "AI-ModelScope/bitnet-embedding-270m",
}

_HUBS = ("hf", "modelscope", "auto")


def default_hub(env=os.environ):
    explicit = env.get("BIT_JEV_HUB", "").lower()
    if explicit in _HUBS and explicit != "auto":
        return explicit
    if env.get("BIT_JEV_ON_AUTODL") or env.get("MODELSCOPE_CACHE"):
        return "modelscope"
    if "autodl" in env.get("HOSTNAME", ""):
        return "modelscope"
    return "hf"


def ms_repo(repo):
    """Translate a HF repo id to its ModelScope mirror when we know one."""
    return MS_MIRRORS.get(repo, repo)


def snapshot(repo, revision=None, allow_patterns=None, hub=None):
    """Download repo at revision to a local dir and return its path."""
    hub = hub or default_hub()
    if hub == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
        return ms_download(ms_repo(repo), revision=revision,
                           allow_file_pattern=allow_patterns,
                           cache_dir=os.environ.get("MODELSCOPE_CACHE"))
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, allow_patterns=allow_patterns)


def load_tokenizer(repo, revision=None, hub=None):
    """本地目录直接加载 tokenizer；远程仓库继续按配置选择下载源。"""
    from transformers import AutoTokenizer
    # 发布包与本地检查点目录应直接读取，避免 AutoDL 默认的 ModelScope 解析本地路径。
    if os.path.isdir(repo):
        return AutoTokenizer.from_pretrained(repo)
    hub = hub or default_hub()
    if hub == "modelscope":
        return AutoTokenizer.from_pretrained(snapshot(repo, revision, hub="modelscope"))
    return AutoTokenizer.from_pretrained(repo, revision=revision)


def load_config(repo, revision=None, hub=None):
    from transformers import AutoConfig
    hub = hub or default_hub()
    if hub == "modelscope":
        return AutoConfig.from_pretrained(snapshot(repo, revision, hub="modelscope"))
    return AutoConfig.from_pretrained(repo, revision=revision)


def load_model(name, revision=None, hub=None, **kwargs):
    """AutoModelForCausalLM via the chosen hub (local snapshot + transformers' loader)."""
    from transformers import AutoModelForCausalLM
    hub = hub or default_hub()
    if hub == "modelscope":
        return AutoModelForCausalLM.from_pretrained(snapshot(name, revision, hub="modelscope"), **kwargs)
    return AutoModelForCausalLM.from_pretrained(name, revision=revision, **kwargs)


def dataset_file(repo_id, filename, hub=None):
    """One file out of a dataset repo (kev-suites; no ModelScope mirror today)."""
    hub = hub or default_hub()
    if hub == "modelscope":
        from modelscope.hub.file_download import dataset_file_download
        return dataset_file_download(ms_repo(repo_id), filename,
                                     cache_dir=os.environ.get("MODELSCOPE_CACHE"))
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)


CHECKPOINT_PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja", "*.model"]
