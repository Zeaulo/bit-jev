"""核对官方 PyPI 上当前版本产物与本地发行文件的 SHA-256。"""

import hashlib
import tomllib
from pathlib import Path

import requests

# 从发行元数据取得版本，避免升级后继续校验旧产物。
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """官方 JSON 必须同时列出与本地完全一致的 wheel 和源码包。"""
    version = tomllib.loads((ROOT / "core/pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    response = requests.get(f"https://pypi.org/pypi/bit-jev/{version}/json", timeout=30)
    response.raise_for_status()
    artifacts = response.json()["urls"]
    if len(artifacts) != 2:
        raise AssertionError(f"预期两个发行文件，实际 {len(artifacts)} 个")
    for artifact in artifacts:
        # 仅读取明确的当前产物，不把未发布的其他文件纳入检查。
        path = ROOT / "test/pypi-dist" / artifact["filename"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact["digests"]["sha256"] or path.stat().st_size != artifact["size"]:
            raise AssertionError(f"远端发行文件与本地不一致：{path.name}")
        print(path.name, artifact["size"], digest)


if __name__ == "__main__":
    main()
