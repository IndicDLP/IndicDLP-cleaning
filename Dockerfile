FROM nvcr.io/nvidia/pytorch:22.03-py3

# Define build arguments
ARG UID
ARG GID
ARG USERNAME=sahithi_kukkala


# Ensure UID, GID, and USERNAME are provided
RUN if [ -z "$UID" ] || [ -z "$GID" ] || [ -z "$USERNAME" ]; then \
    echo "Error: UID, GID, and USERNAME build arguments must be provided." >&2; \
    exit 1; \
    fi

# Set timezone and disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 poppler-utils git sudo vim screen \
    software-properties-common tzdata python3.8-venv python3.8-dev && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 && \
    update-alternatives --set python3 /usr/bin/python3.12

ENV PATH="/usr/bin:$PATH"

# Verify Python version
RUN python3 --version

# Create a group and user with passwordless sudo
RUN groupadd --gid ${GID} ${USERNAME} && \
    useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd && \
    chmod 0440 /etc/sudoers.d/nopasswd

# Switch to the new user
USER ${USERNAME}

# Set working directory
WORKDIR /home/${USERNAME}/

# Create a Python 3.12 virtual environment
RUN python3 -m venv .yolo

# Install packages inside the virtual environment
RUN /bin/bash -c ". /home/${USERNAME}/.yolo/bin/activate && \
    pip install --upgrade pip && \
    pip install ultralytics pytest"

# Verify installation
RUN /bin/bash -c ". /home/${USERNAME}/.yolo/bin/activate && \
    echo 'Python version:' && python --version && \
    echo 'Installed packages:' && pip list"

# Set working directory for your project
WORKDIR /home/${USERNAME}/indicDLP

# Default command
CMD ["/bin/bash"]