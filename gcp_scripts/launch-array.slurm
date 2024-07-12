#!/usr/bin/env bash

#SBATCH --job-name=eval_english
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem=320G
#SBATCH --time=1440:00:00
#SBATCH --output=log/%x-%j.log
#SBATCH --array=0-7

set -eou pipefail

[ -d log ] || mkdir log

LM_EVAL_MODEL_PATHS=(
    "/mnt/fs-arf-01/cpt_checkpoints/hf_checkpoints/61-llama3-4node-replay10/ep0-ba8000"
    "/mnt/fs-arf-01/cpt_checkpoints/hf_checkpoints/61-llama3-4node-replay10/ep0-ba8500"
    "/mnt/fs-arf-01/cpt_checkpoints/hf_checkpoints/61-llama3-4node-replay10/ep0-ba9000"
    "/mnt/fs-arf-01/cpt_checkpoints/hf_checkpoints/61-llama3-4node-replay10/ep1-ba9347"
    "/mnt/fs-arf-01/cpt_checkpoints/hf_checkpoints/62-llama3-4node-replay30/ep0-ba8000"
    "/mnt/fs-arf-01/cpt_checkpoints/hf_checkpoints/62-llama3-4node-replay30/ep0-ba8500"
    "/mnt/fs-arf-01/cpt_checkpoints/hf_checkpoints/62-llama3-4node-replay30/ep0-ba9000"
    "/mnt/fs-arf-01/cpt_checkpoints/meta-llama/Meta-Llama-3-8B"
)

export LM_EVAL_MODEL_PATH=${LM_EVAL_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}

export LM_EVAL_MODEL_DTYPE=bfloat16

export LM_EVAL_OUTPUT_DIR=/mnt/fs-arf-01/eval/results/eval_english

printf 'INFO: Model path=%s\n' "${LM_EVAL_MODEL_PATH}"
printf 'INFO: Model dtype=%s\n' "${LM_EVAL_MODEL_DTYPE}"
printf 'INFO: Ouput dir=%s\n' "${LM_EVAL_OUTPUT_DIR}"

srun bash run.sh
