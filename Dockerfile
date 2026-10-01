# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=0.34.0
ARG CUDA_VERSION_FOR_COMFY=12.8
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Harden pip against slow/flaky PyPI during long dependency installs: the
# `comfy install` step below pulls hundreds of MB, and pip's default 15s
# read timeout aborts slow-but-progressing downloads (build failed on this 2026-10-01).
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
# Route pip/uv through the Tsinghua PyPI mirror: the self-hosted Mac builder's
# route to files.pythonhosted.org measured ~50-160 kB/s vs ~15 MB/s via this
# mirror (2026-10-01). Applies to `comfy install` (pip) and `uv pip` steps.
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
# comfy-cli is pinned: its install/torch-index behavior decides what lands in
# the workspace venv, so an unpinned version makes builds non-reproducible.
RUN uv pip install comfy-cli==1.13.0 pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Install custom nodes required by the IPAdapter FaceID workflow.
# Cloned BEFORE the dependency-install step below so their requirements.txt
# files are picked up by the loop that installs every custom node's deps.
# - ComfyUI_IPAdapter_plus: IPAdapterModelLoader, IPAdapterFaceID,
#   IPAdapterInsightFaceLoader nodes (+ no extra pip requirements)
# - rgthree-comfy: "Image Comparer (rgthree)" node used by the workflow
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /comfyui/custom_nodes/ComfyUI_IPAdapter_plus \
    && git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy

# insightface + onnxruntime are required at runtime by the FaceID nodes
# (IPAdapterInsightFaceLoader extracts the face embedding with insightface).
RUN uv pip install insightface onnxruntime

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# comfy-cli installs ComfyUI into its own workspace venv (/comfyui/.venv), but
# start.sh launches ComfyUI with /opt/venv's python. That mismatch leaves the
# launch venv missing ComfyUI's runtime deps (e.g. sqlalchemy, pulled in by
# ComfyUI's asset DB), so ComfyUI crashes at startup and surfaces as the
# misleading "ComfyUI server (127.0.0.1:8188) not reachable" error. Mirror
# ComfyUI's full dependency set (core + custom nodes) into /opt/venv so the
# launch venv is complete. Root-cause fix for DR-1170.
#
# The transformers/huggingface-hub pin is part of the SAME step on purpose:
# ComfyUI declares transformers>=4.50.3 and huggingface-hub with NO upper bound,
# so a fresh install can pull transformers 5.x / huggingface-hub 1.x whose
# breaking API changes also crash ComfyUI at startup. Pinning them in the same
# RUN downgrades within one layer, so the unwanted versions aren't left behind
# bloating the image.
#
# torch is installed FIRST, pinned to +cu128 builds: ComfyUI's requirements.txt
# declares a bare `torch`, and default PyPI serves CUDA 13 builds (torch's PyPI
# wheels depend on nvidia-*-cu13 since 2.11) that require driver >= 580. Hosts
# allowed in .runpod/hub.json advertise CUDA 12.8/12.9 (driver 570/575), where
# a cu13 torch fails CUDA init at startup. cu128 builds run on driver >= 570,
# i.e. every allowed host. Installing torch first satisfies the bare `torch`
# requirement so the PyPI pass doesn't touch it.
RUN uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install -r /comfyui/requirements.txt \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && uv pip install -r "$r" || true; \
       done \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# Build-time smoke test: actually start ComfyUI (imports the full node graph) so
# a startup-breaking dependency is caught HERE, at build time, instead of as a
# runtime "server not reachable" failure on a live worker. Runs on CPU — no GPU
# needed to exercise the import graph.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=flux1-dev-fp8

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/text_encoders models/diffusion_models models/model_patches models/clip_vision models/ipadapter models/insightface/models models/loras

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors "https://civitai.com/api/download/models/782002" && \
      wget -q -O models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors && \
      wget -q -O models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors && \
      wget -q -O /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
      python3 -c "import zipfile; zipfile.ZipFile('/tmp/buffalo_l.zip').extractall('models/insightface/models/buffalo_l')" && \
      rm /tmp/buffalo_l.zip; \
    fi

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "z-image-turbo" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/text_encoders/qwen_3_4b.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/diffusion_models/z_image_turbo_bf16.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors; \
    fi

# NOTE: minimaxh3 models (~44GB) are intentionally NOT downloaded here.
# They are downloaded directly in the final stage below: keeping them in the
# downloader stage and COPYing them into the final image stores the 44GB
# twice during the build, which exhausts the GitHub Actions runner disk
# ("no space left on device" at COPY --from=downloader, 2026-10-01).

# Stage 3: Final image
FROM base AS final

# MODEL_TYPE must be re-declared after FROM for this stage to see the build arg
ARG MODEL_TYPE=flux1-dev-fp8

# Copy models from stage 2 to the final image.
# For minimaxh3 this only copies the (empty) model directories created above;
# the actual ~44GB MiniMax H3 model files are downloaded directly in this
# stage (see below) so they are stored exactly once.
COPY --from=downloader /comfyui/models /comfyui/models

# MiniMax H3 reference-to-video model set (~44GB total): video/audio VAEs,
# pruned int8 DiT, Qwen3-VL-32B NVFP4 text encoder, 4-step turbo LoRA.
RUN if [ "$MODEL_TYPE" = "minimaxh3" ]; then \
      mkdir -p /comfyui/models/vae /comfyui/models/diffusion_models /comfyui/models/text_encoders /comfyui/models/loras && \
      wget -q -O /comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors && \
      wget -q -O /comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors && \
      wget -q -O /comfyui/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors && \
      wget -q -O /comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors && \
      wget -q -O /comfyui/models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors; \
    fi
