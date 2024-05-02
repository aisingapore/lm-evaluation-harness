FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git

COPY . lm_evaluation_harness
RUN git clone -b feat-docker-open-llm-leaderboard \
    https://github.com/aisingapore/lm-evaluation-harness.git \
    lm_evaluation_harness
RUN cd /workspace/lm_evaluation_harness && \
    pip install -r requirements.txt

ENTRYPOINT ["/workspace/lm_evaluation_harness/entrypoint.sh"]
