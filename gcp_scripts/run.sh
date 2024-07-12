#!/usr/bin/env bash

set -euo pipefail

task_idx=${1:--1}

export SHARED_FS_DIR=/mnt/fs-arf-01
export MY_CACHE_DIR="${SHARED_FS_DIR}/gcp5_cache/ob1"
export HF_HOME="${MY_CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${MY_CACHE_DIR}/pip"
export CONDA_PKGS_DIRS="${MY_CACHE_DIR}/conda_pkgs"

# . "${HOME}/hf_token"
. /opt/conda/etc/profile.d/conda.sh
conda activate "${SHARED_FS_DIR}/envs/lm-evaluation-harness"

#organization_name='google'
##model_name='gemma-2b'
#model_name='gemma-2-9b'
#model_path="${organization_name}/${model_name}"
##model_revision='2ac59a5d7bf4e1425010f0d457dde7d146658953'
#model_revision='7305b337e801768dc5c40319c421052afac25b77'
#model_dtype='bfloat16'
#output_folder="results/${organization_name}/${model_name}"

tasks=(
	leaderboard_bbh
	leaderboard_arc_challenge
	leaderboard_gsm8k
	aisg_internal_hellaswag
	mmlu
)
if [[ ${task_idx} == -1 ]]; then 
	tasks_str=$(IFS=,; echo "${tasks[*]}")
else
	tasks_str=${tasks[${task_idx}]}
fi
model_args=(
	"pretrained=/home/ob1/eval/s3_sync/90-10-5e7/ba1526"
	"parallelize=False"
	"dtype=bfloat16"
)
model_args_str=$(IFS=,; echo "${model_args[*]}")
printf "INFO: Evaluating %s on tasks %s\n" "${model_args_str}" "${tasks_str}"

time accelerate launch -m lm_eval \
	--model_args="${model_args_str}" \
	--tasks="${tasks_str}" \
	--batch_size=auto \
	--max_batch_size=16 \
	--output_path="/home/ob1/eval/english_results"
