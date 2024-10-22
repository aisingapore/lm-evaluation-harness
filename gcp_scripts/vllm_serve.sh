export CUDA_VISIBLE_DEVICES=0,1
CONDA_SH="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
. ${CONDA_SH}

random_port=$(( ( RANDOM % 1000 )  + 8000 ))

conda activate lm-eval
lm_eval_path="/home/ubuntu/lm-evaluation-harness"
vllm_path="/home/ubuntu/source_files/vllm"
export PYTHONPATH="/home/ubuntu/lm-evaluation-harness"
export VLLM_LOGGING_LEVEL=WARNING
export model="aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct"
vllm_args=(
    --model $model
    --dtype bfloat16
    # --enable-prefix-caching
    --gpu-memory-utilization 0.35
    --tensor-parallel-size 1
    --port $random_port
    --uvicorn-log-level warning
)

vllm serve $model ${vllm_args[@]} > $HOME/vllm.log 2>&1 &
vllm_process_id=$!
# check the status of the server
echo "vLLM process id: $vllm_process_id"

check_server_health() {
  curl -s -o /dev/null http://0.0.0.0:${random_port}/health
  return $?
}
wait_for_server() {
  echo "Waiting for vLLM..."
  while ! check_server_health; do
    sleep 1
  done
  echo "vLLM is up"
}
wait_for_server

tasks=(
    medqa_4options
    # medmcqa
)

tasks_str=$(IFS=,; echo "${tasks[*]}")

lm_eval_args=(
    --model local-completions
    --model_args model=$model,base_url=http://localhost:${random_port}/v1/completions,num_concurrent=256,max_retries=0,tokenized_requests=False
    --tasks $tasks_str
    --log_samples
    --output_path results
    # --limit 100
)

lm_eval ${lm_eval_args[@]}

kill $vllm_process_id