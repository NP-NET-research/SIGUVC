import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CheckpointManager:
    def __init__(self, output_dir: str, model_name_or_path: str):
        self.output_dir = output_dir
        self.model_name_or_path = model_name_or_path

    def load_checkpoint(self, model, tokenizer):

        checkpoint_dirs = [f for f in os.listdir(self.output_dir) if "checkpoint-" in f]
        if not checkpoint_dirs:
            print("未找到检查点，从预训练模型加载……")
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            return model, tokenizer, 0  
        else:

            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = checkpoint_dirs[-1]
            checkpoint_path = os.path.join(self.output_dir, latest_checkpoint)

            print(f"成功从此处加载检查点：{checkpoint_path}")

            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), weights_only=True)
            model.load_state_dict(state_dict, strict=False)

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            step_count = int(latest_checkpoint.split("-")[-1])
            return model, tokenizer, step_count

    def save_checkpoint(self, model, tokenizer, step_count):
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step_count}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"将检查点保存到：{checkpoint_dir}")
