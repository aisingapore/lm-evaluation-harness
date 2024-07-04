#!/usr/bin/env bash

#SBATCH --job-name=lm-evaluation-harness
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=1440:00:00
#SBATCH --output=log/%x-%j.log

set -eou pipefail

[ -d log ] || mkdir log

printf 'INFO: Model path=%s\n' "${LM_EVAL_MODEL_PATH}"
printf 'INFO: Model dtype=%s\n' "${LM_EVAL_MODEL_DTYPE}"
printf 'INFO: Ouput dir=%s\n' "${LM_EVAL_OUTPUT_DIR}"

srun bash run.sh
