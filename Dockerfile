FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git

RUN git clone -b feat-docker \
    https://github.com/aisingapore/lm-evaluation-harness.git \
    lm_evaluation_harness
WORKDIR /workspace/lm_evaluation_harness
RUN pip install -r requirements.txt

ENTRYPOINT ["/workspace/lm_evaluation_harness/entrypoint.sh"]
