from concurrent.futures import ThreadPoolExecutor
import torch
from dataclasses import dataclass
from typing import List, Optional, Dict

from transformers import AutoTokenizer
from src.inference.prompt import PromptRegistry, PromptTemplate
from .backend import InferenceBackend, HFBackend, VLLMBackend

# ==============================
# config
# ==============================
@dataclass
class GenerationConfig:
    model_name_or_path: str
    adapter_paths: Optional[Dict[str, str]] = None      # 支持多适配器
    use_vllm: bool = False
    max_new_tokens: int = 2048
    do_sample: bool = True
    top_p: float = 0.95
    top_k: Optional[int] = 0
    temperature: float = 0.95
    repetition_penalty: float = 1.0
    vllm_config: Optional[Dict] = None
    num_sequences: int = 1
    batch_size: int = 8  # 新增: 批推理微批大小


class BaseGenerator:
    def __init__(self, config: GenerationConfig=None):
        default_config = GenerationConfig(
            model_name_or_path="/home/ljl/Data/LLM/GLM-4-9B-0414",
            adapter_paths=None,
            use_vllm=False,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            temperature=0.95,
            repetition_penalty=1.0,
            vllm_config=None,
            num_sequences=1,
            batch_size=8,  # 新增默认微批
        )
        if config is None:
            self.config = default_config
        else:
            if isinstance(config, dict):
                self.config = GenerationConfig(**config)
            else:
                self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, trust_remote_code=True)

        if self.config.use_vllm:
            self.backend = VLLMBackend(self.tokenizer, self.config)
        else:
            self.backend = HFBackend(self.config, self.tokenizer)

    def generate(self, queries: List[str], prompt_template: Optional[PromptTemplate] = None, adapter_name: Optional[str] = None, **gen_kwargs) -> List[List[str]]:
        # 构建 prompt
        if prompt_template:
            prompts = [prompt_template.build(self.tokenizer, q) for q in queries]
        else:
            prompts = [self.tokenizer.apply_chat_template(
                query,
                add_generation_prompt=True,
                tokenize=False,
            ) for query in queries]

        # 合并参数
        backend_kwargs = self._merge_generation_params(**gen_kwargs)
        backend_kwargs["adapter_name"] = adapter_name

        # 调用 backend
        return self.backend.generate(prompts, **backend_kwargs)

    def _merge_generation_params(self, **overrides) -> dict:
        params = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": self.config.do_sample,
            "num_sequences": self.config.num_sequences,
            "batch_size": self.config.batch_size,
        }
        params.update(overrides)
        return params
    
    

if __name__ == "__main__":
    
    # Example usage
    config = {
        "model_name_or_path": "/home/ljl/Disk/ljl/GLM-4-9B-0414",
        "adapter_path": None,
        "use_vllm": False
    }
    prompt_template = PromptRegistry.get("segmentation")
    gen = BaseGenerator(config, prompt_template)

    queries = ["奶牛是什么动物?", "量子计算的原理是什么?"]
    print(gen.generate(queries))
       