FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y \
        curl \
        git \
        unzip; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    unzip awscliv2.zip; \
    ./aws/install; \
    rm -rf aws/ awscliv2.zip

COPY requirements.txt /workspace/
RUN set -eux; \
    pip install -r requirements.txt; \
    pip cache purge; \
    rm requirements.txt

COPY . /workspace/lm_evaluation_harness
WORKDIR /workspace/lm_evaluation_harness

ENTRYPOINT ["/workspace/lm_evaluation_harness/entrypoint.sh"]
