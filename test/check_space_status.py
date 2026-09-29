"""读取 ModelScope 演示空间的公开构建状态，不执行发布。"""

import argparse
import os

from modelscope_hub import HubApi


def main() -> None:
    """使用环境令牌读取 Studio 元信息，避免在日志中输出令牌。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-logs", action="store_true")
    parser.add_argument("--run-logs", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError("缺少 MODELSCOPE_API_TOKEN")
    # 只打印部署状态字段；完整返回值可能带有无关账户信息。
    api = HubApi(token=token)
    data = api.get_repo("JingHao9616/bit-jev-demo", "studio")
    print("runtime:", data.runtime)
    if args.build_logs:
        # 构建失败可能发生在 pip 依赖解析，保留足够行数查看真正错误。
        result = api.get_repo_logs("JingHao9616/bit-jev-demo", log_type="build", page_size=100)
        lines = "".join(result.get("logs", [])).splitlines()
        print("last build lines:", *lines[-50:], sep="\n")
    if args.run_logs:
        # 运行日志只保留最近行，避免公开接口验收期间打印大段依赖输出。
        result = api.get_repo_logs("JingHao9616/bit-jev-demo", log_type="run", page_size=20)
        lines = "".join(result.get("logs", [])).splitlines()
        print("last run lines:", *lines[-12:], sep="\n")


if __name__ == "__main__":
    main()
