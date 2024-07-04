#!/usr/bin/env bash

set -euo pipefail

bucket_name=${1:-'s3://checkpoint-ohio-test/hf_checkpoints'}
ckpt_name=${2:-'gemma-2-9b-unique-name'}
output_dir=${3:-'/mnt/fs-arf-01/gcp2_cache/gcp_user/checkpoints'}

src="${bucket_name}/${ckpt_name}"
dst="${output_dir}/${ckpt_name}"

[[ -d "${output_dir}" ]] || {
	printf 'WARN: Creating %s as it does not exist\n' "${output_dir}"
	mkdir -p "${output_dir}"
}

printf 'INFO: Downloading from %s to %s\n' "${src}" "${dst}"

aws s3 cp --recursive "${src}" "${dst}"
