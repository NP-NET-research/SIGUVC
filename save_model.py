import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL_PATH = "../glm-4-9b-chat-hf"  
CHECKPOINT_PATH = "/home/zhipu_fund/Debug/output/checkpoint-450"  
FINAL_MODEL_DIR = "model_modern_new"  

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
print(f"基础模型已成功加载")

if not os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_model.safetensors")):
    raise FileNotFoundError("找不到 LoRA 适配器文件 adapter_model.safetensors")
peft_model = PeftModel.from_pretrained(
    base_model,
    CHECKPOINT_PATH,
    device_map="auto"
)
print(f"LoRA 适配器已成功注入")
merged_model = peft_model.merge_and_unload()
print(f"LoRA 已成功合并到基础模型")


ppo_weight_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
if os.path.exists(ppo_weight_path):
    print("检测到 PPO Value Head 权重")
    ppo_state_dict = torch.load(ppo_weight_path, map_location="cuda:0")
    merged_model.load_state_dict(ppo_state_dict, strict=False)
    print("Value Head 权重已合并")
else:
    print("未检测到 PPO Value Head 权重")

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

merged_model = merged_model.cpu()
torch.save(merged_model.state_dict(), os.path.join(FINAL_MODEL_DIR, "pytorch_model.bin"))
merged_model.config.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print(f"最终模型已保存至 {FINAL_MODEL_DIR}")
