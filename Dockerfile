FROM nvidia/cuda:13.4.2-cudnn-runtime-ubuntu24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (git for VCS dependencies, python3 for venv compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    sudo \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create default non-root user 'user' (UID 1000, GID 1000) with passwordless sudo
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN if id -u ${USER_UID} >/dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd ${USER_UID} | cut -d: -f1); \
        if [ "$EXISTING_USER" != "${USERNAME}" ]; then \
            userdel -r "$EXISTING_USER" 2>/dev/null || true; \
        fi; \
    fi && \
    if getent group ${USER_GID} >/dev/null 2>&1; then \
        EXISTING_GROUP=$(getent group ${USER_GID} | cut -d: -f1); \
        if [ "$EXISTING_GROUP" != "${USERNAME}" ]; then \
            groupdel "$EXISTING_GROUP" 2>/dev/null || true; \
        fi; \
    fi && \
    groupadd -g ${USER_GID} ${USERNAME} 2>/dev/null || true && \
    useradd -u ${USER_UID} -g ${USER_GID} -m -s /bin/bash ${USERNAME} 2>/dev/null || true && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create workspace and cache directories with proper permissions
RUN mkdir -p /home/user/workspaces/neuralnet /home/user/.cache/uv && \
    chown -R user:user /home/user && \
    chmod -R a+rwX /home/user

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=tensorflow \
    UV_CACHE_DIR=/home/user/.cache/uv \
    PATH="/home/user/workspaces/neuralnet/.venv/bin:${PATH}"

# Set working directory to the mounted workspace
WORKDIR /home/user/workspaces/neuralnet

USER user
