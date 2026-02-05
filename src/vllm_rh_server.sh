CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /home/ljl/zhipu/checkpoints/glm4-9b-0414-rhetoric-use \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype auto \
  --tensor-parallel-size 1 
