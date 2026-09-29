FROM python:3.11-slim-bookworm

# Linux runner 在镜像构建阶段编译；访客首次推理无需安装 Git、CMake 或编译器。
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir "cmake>=3.28,<4" "bit-jev==0.12.10" "gradio>=5,<6"
RUN python -c "from bit_jev.native_build import build_native; import shutil; source = build_native('cpu', cache_dir='/opt/bit-jev-build', jobs=2); shutil.copy2(source, '/opt/bit-jev-cpu')" \
    && chmod 755 /opt/bit-jev-cpu

# 模型缓存属于非特权运行账户；镜像只包含程序，权重由运行时按需下载。
RUN useradd -m -u 1000 user
ENV HOME=/home/user BIT_JEV_BINARY=/opt/bit-jev-cpu PYTHONUNBUFFERED=1
WORKDIR /app
COPY --chown=user:user app.py forms.py logic.py ./
USER user
EXPOSE 7860
CMD ["python", "app.py"]
