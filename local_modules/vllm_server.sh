CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /home/ljl/Data/LLM/GLM-4-32B-0414 \
  --host 127.0.0.1 \
  --port 8080 \
  --dtype auto \
  --tensor-parallel-size 1
  