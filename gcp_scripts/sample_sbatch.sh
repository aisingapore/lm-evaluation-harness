#!/usr/bin/env bash

set -euo pipefail

# Expects the following ENV VAR as input
# LM_EVAL_MODEL_PATH: Path to the downloaded Hugging Face weights
# LM_EVAL_MODEL_DTYPE: Data type, typically bfloat16
# LM_EVAL_OUTPUT_DIR: Output directory to store the evaluation results.
#                     Should be unique to each checkpoint/evaluation run.

lm_eval_model_path='/mnt/fs-arf-01/eval/models/a_meta_llama_3_ckpt'
lm_eval_model_dtype='bfloat16'
model_name=$(basename "${lm_eval_model_path}")
lm_eval_output_dir="/mnt/fs-arf-01/eval/results/eval_english/${model_name}"
sbatch \
	--export=ALL,LM_EVAL_MODEL_PATH=${lm_eval_model_path},LM_EVAL_MODEL_DTYPE=${lm_eval_model_dtype},LM_EVAL_OUTPUT_DIR=${lm_eval_output_dir} \
	launch.slurm

# Output
# lm-evaluation will create the result at
# ${lm_eval_output_dir}/<${lm_eval_model_path} with '/' replaced by '__'>/results_<timestamp>.json
#
# Specify a unique output directory per checkpoint makes the look up command slightly easier since
# there will only be one file
jq '.results | {aisg_internal_hellaswag, mmlu, leaderboard_arc_challenge, leaderboard_bbh, leaderboard_gsm8k}' ${lm_eval_output_dir}/*${model_name}/*.json
