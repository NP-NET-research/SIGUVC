from dataclasses import dataclass
from typing import Optional, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
from threading import Lock 
import gc

# ---------------- 配置 ----------------
@dataclass
class GenerationConfig:
    model_name_or_path: str
    adapter_paths: Optional[Dict[str, str]] = None
    use_vllm: bool = False
    max_new_tokens: int = 256
    do_sample: bool = True
    top_p: float = 0.7
    top_k: int = 50
    temperature: float = 0.95
    repetition_penalty: float = 1.0
    num_sequences: int = 1
    vllm_config: Optional[Dict] = None  # 保留,用于 VLLM
    batch_size: int = 8

# ---------------- 后端抽象 ----------------
class InferenceBackend:
    def generate(self, prompts: List[str], **kwargs):
        raise NotImplementedError

# ---------------- HF 后端(单卡/多卡统一) ----------------
class HFBackend(InferenceBackend):
    def __init__(self, config: GenerationConfig, tokenizer: Optional[AutoTokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

        # 确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 获取设备列表
        self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not self.devices:
            self.devices = ["cpu"]

        # 每个设备创建一个模型实例
        self.models = []
        for device in self.devices:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=True,
                device_map=None,
                torch_dtype="auto"
            ).to(device).eval()
            self.models.append(model)

        # 加载 adapters
        self.adapters = {}
        if config.adapter_paths:
            for name, path in config.adapter_paths.items():
                self.adapters[name] = path
                for model in self.models:
                    model.load_adapter(path, adapter_name=name)
                    model.disable_adapters()

    @torch.no_grad()
    def generate(self, prompts: List[str], adapter_name: Optional[str] = None, **kwargs) -> List[List[str]]:
        
        num_sequences = int(kwargs.get("num_sequences", self.config.num_sequences))
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        top_p = kwargs.get("top_p", self.config.top_p)
        top_k = kwargs.get("top_k", self.config.top_k)
        temperature = kwargs.get("temperature", self.config.temperature)
        repetition_penalty = kwargs.get("repetition_penalty", self.config.repetition_penalty)
        do_sample = kwargs.get("do_sample", self.config.do_sample)
        batch_size = int(kwargs.get("batch_size", getattr(self.config, "batch_size", 8)))

        from tqdm.auto import tqdm
        pbar = tqdm(total=len(prompts), desc="Generating", unit="prompt", dynamic_ncols=True, leave=False)
        pbar_lock = Lock()
        
        # 设置 adapter
        def set_adapter(model):
            if adapter_name:
                if adapter_name in self.adapters:
                    model.set_adapter(adapter_name)
                else:
                    print(f"Adapter {adapter_name} not found, using no adapter.")
            else:
                if self.adapters:
                    model.disable_adapters()

        # 切分 batch 给每张卡
        chunk_size = (len(prompts) + len(self.models) - 1) // len(self.models)
        chunks = [prompts[i:i+chunk_size] for i in range(0, len(prompts), chunk_size)]

        results = []

        def worker(model, chunk):
            set_adapter(model)
            grouped_all = []
            # 进一步按微批跑，避免 OOM
            with torch.inference_mode():
                for start in range(0, len(chunk), max(1, batch_size)):
                    sub_prompts = chunk[start:start+batch_size]
                    inputs = self.tokenizer(sub_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                    prompt_len = inputs.input_ids.shape[1]

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        do_sample=(do_sample or num_sequences > 1),
                        num_return_sequences=num_sequences,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    decoded_all = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
                    bsz = len(sub_prompts)
                    grouped = [
                        [decoded_all[i*num_sequences + j].strip() for j in range(num_sequences)]
                        for i in range(bsz)
                    ]
                    grouped_all.extend(grouped)

                    with pbar_lock:
                        pbar.update(len(sub_prompts))
            
            return grouped_all

        # 多线程并行
        with ThreadPoolExecutor(max_workers=len(self.models)) as ex:
            futures = [ex.submit(worker, model, chunk) for model, chunk in zip(self.models, chunks)]
            for f in futures:
                results.extend(f.result())

        pbar.close()
        return results


class VLLMBackend(InferenceBackend):
    def __init__(self, tokenizer, config):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.LLM = LLM
        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest
        self.config = config
        self.tokenizer = tokenizer

        num_gpus = torch.cuda.device_count()
        engine_args = {
            "model": config.model_name_or_path,
            "enable_lora": config.adapter_paths is not None,
            "dtype": "auto",
            "trust_remote_code": True,
            "disable_log_stats": True,
            "tensor_parallel_size": num_gpus,
            "gpu_memory_utilization": 0.9,
        }
        if config.vllm_config:
            engine_args.update(config.vllm_config)
        self.vllm_model = self.LLM(**engine_args)

        # 注册多个 LoRA
        self.adapters = {}
        if config.adapter_paths:
            for idx, (name, path) in enumerate(config.adapter_paths.items(), start=1):
                self.adapters[name] = self.LoRARequest(name, idx, path)

    def generate(self, prompts: List[str], adapter_name: Optional[str] = None, **kwargs) -> List[List[str]]:
        import gc
        
        num_sequences = int(kwargs.get("num_sequences", self.config.num_sequences))
        stop_ids = []
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        stop_ids += getattr(self.tokenizer, "additional_special_tokens_ids", []) or []

        sampling_params = self.SamplingParams(
            n=num_sequences,
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            stop_token_ids=stop_ids,
            max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            skip_special_tokens=True,
        )

        lora_req = None
        if adapter_name and adapter_name in self.adapters:
            print(f"Using adapter: {adapter_name}")
            lora_req = self.adapters[adapter_name]

        results = self.vllm_model.generate(prompts, sampling_params, lora_request=lora_req)
        
        grouped_results = []
        for res in results:
            if res.outputs:
                texts = [o.text.strip() for o in res.outputs]
                grouped_results.append(texts)
            else:
                grouped_results.append([""])
        
        # 清理中间结果
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return grouped_results
