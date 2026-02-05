from typing import List, Dict
import gc
from openai import APIConnectionError, OpenAI
from openai.pagination import SyncPage
from openai.types.model import Model
from openai.types.chat.chat_completion import ChatCompletion
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


# 默认生成器配置
DEFAULT_MODEL_PATH = "/home/ljl/Data/LLM/GLM-4-32B-0414"
DEFAULT_VLLM_CONFIG = {
    "gpu_memory_utilization": 0.95,
    "max_model_len": 32768,
}


def create_generator(
    model_name_or_path: str = None,
    temperature: float = 0.7,
    max_new_tokens: int = 16384,
):
    """
    创建 vLLM 生成器
    
    Args:
        model_name_or_path: 模型路径，如果为None则使用默认路径
        temperature: 生成温度
        max_new_tokens: 最大生成token数
        
    Returns:
        BaseGenerator 实例
    """
    from src.inference.generator import BaseGenerator, GenerationConfig

    if model_name_or_path is None:
        model_name_or_path = DEFAULT_MODEL_PATH
    
    config = GenerationConfig(
        model_name_or_path=model_name_or_path,
        use_vllm=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_sequences=1,
        vllm_config=DEFAULT_VLLM_CONFIG,
    )
    return BaseGenerator(config)


def get_api_first_model_name(client: OpenAI) -> str:
    """获取 OpenAI API 服务器上的第一个模型名称"""
    try:
        models: SyncPage[Model] = client.models.list()
    except APIConnectionError as e:
        raise RuntimeError("APIConnectionError") from e

    if len(models.data) == 0:
        raise RuntimeError(f"No models found on the vLLM server at {client.base_url}")

    return models.data[0].id


def call_api(client: OpenAI, msgs, model_name, temperature, max_tokens):
    """调用 OpenAI API 生成单个响应"""
    resp: ChatCompletion = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60
    )
    return resp.choices[0].message.content


def generate_with_api_parallel(
    client: OpenAI,
    conversations: List[List[Dict[str, str]]],
    workers: int = 64,
    prefix: str = 'processing',
    model_name: str = '',
    temperature: float = 0.2,
    max_tokens: int = 32768,
) -> List[str]:
    """
    使用 OpenAI API 并行生成多个对话的响应
    
    Args:
        client: OpenAI 客户端
        conversations: 对话列表
        workers: 并行工作线程数
        prefix: 进度条前缀
        model_name: 模型名称，为空则自动获取
        temperature: 生成温度
        max_tokens: 最大token数
        
    Returns:
        生成的响应列表
    """
    if not model_name:
        model_name = get_api_first_model_name(client)
    
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(call_api, client, msgs, model_name, temperature, max_tokens): idx
            for idx, msgs in enumerate(conversations)
        }
        
        results = [None] * len(conversations)  # 按原顺序存储结果
        with tqdm(total=len(conversations), desc=prefix) as pbar:
            for future in as_completed(futures):
                idx = futures[future]  # 获取原索引
                try:
                    results[idx] = future.result()  # 按原顺序保存结果
                except Exception as e:
                    results[idx] = f"Error: {str(e)}"  # 捕获异常，避免中断
                pbar.update(1)  # 进度条+1
    
    # TODO: 失败重试机制
    # 提取内容, 失败的返回空字符串
    results = [res if res and not res.startswith("Error:") else "" for res in results]

    return results


def generate_responses(
    conversations: List[List[Dict]],
    generator = None,
    temperature: float = 0.1,
    max_tokens: int = 16384,
) -> List[str]:
    """
    通用的响应生成函数
    
    Args:
        conversations: 对话列表
        generator: BaseGenerator 实例或 OpenAI 客户端，如果为None则创建临时生成器
        temperature: 生成温度
        max_tokens: 最大token数
        use_parallel: 是否使用并行生成（仅对 OpenAI 客户端有效）
        workers: 并行工作线程数
        
    Returns:
        生成的响应列表
    """
    need_cleanup = False
    
    if generator is None:
        generator = create_generator(temperature=temperature, max_new_tokens=max_tokens)
        need_cleanup = True
    
    # 检测是否为 OpenAI 客户端
    is_openai = hasattr(generator, 'chat') and hasattr(generator.chat, 'completions')
    
    if is_openai:
        # 使用 OpenAI API 方式
        responses = generate_with_api_parallel(
            client=generator,
            conversations=conversations,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        # 使用 vLLM 方式
        outputs = generator.generate(
            conversations,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        responses = [output_list[0].strip() if output_list else "" for output_list in outputs]

    if need_cleanup:
        del generator
        gc.collect()
    
    return responses
