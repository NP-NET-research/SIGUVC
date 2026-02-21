import os
from typing import Dict, List
from openai import APIConnectionError, OpenAI
from openai.pagination import SyncPage
from openai.types.model import Model
from openai.types.chat.chat_completion import ChatCompletion
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def get_api_first_model_name(client: OpenAI) -> str:
    try:
        models: SyncPage[Model] = client.models.list()
    except APIConnectionError as e:
        raise RuntimeError("APIConnectionError") from e

    if len(models.data) == 0:
        raise RuntimeError(f"No models found on the vLLM server at {client.base_url}")

    return models.data[0].id


def call_api(client: OpenAI, msgs, model_name, temperature):
    resp: ChatCompletion = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=temperature
    )
    return resp.choices[0].message.content


def generate_with_api_parallel(
    client: OpenAI,
    conversations: List[List[Dict[str, str]]],
    workers: int = 16,
    prefix = 'processing',
    model_name: str = '',
    temperature: float = 0.2,
    max_tokens=512,
):
    """
    Generate responses for multiple conversations in parallel using the OpenAI API.
    """

    if not model_name:
        model_name = get_api_first_model_name(client)
    
    # 初始化future任务列表
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # 提交所有任务并记录future与索引的映射（保证结果顺序）
        futures = {
            pool.submit(call_api, client, msgs, model_name, temperature): idx
            for idx, msgs in enumerate(conversations)
        }
        
        # 初始化进度条（总数为对话数量）
        results = [None] * len(conversations)  # 按原顺序存储结果
        with tqdm(total=len(conversations), desc=prefix) as pbar:
            # 遍历已完成的future，更新进度
            for future in as_completed(futures):
                idx = futures[future]  # 获取原索引
                try:
                    results[idx] = future.result()  # 按原顺序保存结果
                except Exception as e:
                    results[idx] = f"Error: {str(e)}"  # 捕获异常，避免中断
                pbar.update(1)  # 进度条+1
    
    return results
