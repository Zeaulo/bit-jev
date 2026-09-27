"""安全清理本次导出留下的两个已确认无效的诊断目录。"""

from pathlib import Path


# 只允许清理项目导出目录下这两个已确认的失败产物。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = (PROJECT_ROOT / "export").resolve()
FAILED_NAMES = ("failed-metadata-only-20260926", "failed-unmapped-subnorm-20260926")
ALLOWED_FILES = {
    "backbone-i2_s.gguf", "head.pt", "manifest.json", "pointer.json",
    "config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
}


def main():
    """先核对完整目录结构，再逐个解除文件链接与空目录。"""
    # 第一轮只校验，不在发现意外文件后进行部分清理。
    folders = []
    files = []
    nested_dirs = []
    for name in FAILED_NAMES:
        folder = (EXPORT_ROOT / name).resolve(strict=True)
        if folder.parent != EXPORT_ROOT or not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"拒绝清理工作区外或符号链接目录：{folder}")
        folders.append(folder)
        # 失败目录最多包含一个由导出器创建的临时输入目录。
        for entry in folder.iterdir():
            if entry.is_dir() and entry.name == "converter-input" and not entry.is_symlink():
                nested_dirs.append(entry)
                for nested_file in entry.iterdir():
                    if not nested_file.is_file() or nested_file.is_symlink() or nested_file.name not in ALLOWED_FILES:
                        raise ValueError(f"发现不属于本次导出的文件：{nested_file}")
                    files.append(nested_file)
            elif entry.is_file() and not entry.is_symlink() and entry.name in ALLOWED_FILES:
                files.append(entry)
            else:
                raise ValueError(f"发现不属于本次导出的路径：{entry}")
    # 第二轮逐个解除文件硬链接，不调用递归删除。
    for failed_file in files:
        failed_file.unlink()
    for nested_dir in nested_dirs:
        nested_dir.rmdir()
    for folder in folders:
        folder.rmdir()
    print(f"已清理 {len(folders)} 个失败目录、{len(files)} 个诊断文件或硬链接")


if __name__ == "__main__":
    main()
