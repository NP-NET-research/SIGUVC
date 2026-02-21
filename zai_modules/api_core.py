import os
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime
from zai import ZhipuAiClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

# 日志文件配置
datestamp = datetime.now().strftime("%m%d_%H%M")
GLOBAL_LOG_FILE = f"logs/zai_responses_{datestamp}.log"
os.makedirs(os.path.dirname(GLOBAL_LOG_FILE) if os.path.dirname(GLOBAL_LOG_FILE) else "logs", exist_ok=True)

# 并发配置
MODEL_CONCURRENCY_CONFIG = {
    "GLM-4.6": 5,
    "GLM-4.5-Air": 3,
    "GLM-4.5": 10,
    "GLM-4.5-Flash": 2,
}
# 超时配置
TIMEOUT_CONFIG = {
    'Default': 60,
    'Thinking': 300,
}

def _get_model_concurrency(model_name: str, default_workers: int = 2) -> int:
    return MODEL_CONCURRENCY_CONFIG.get(model_name, default_workers)

def _calculate_token_usage(responses):
    """统计responses中的总token使用量"""
    total_tokens = sum(response.usage.total_tokens for response in responses if hasattr(response, 'usage') and response.usage)
    print(f"总tokens使用量: {total_tokens}")
    return total_tokens

def _log_zai_responses(responses: List[Any]):
    """将ZAI API响应记录到全局日志文件中"""
    with open(GLOBAL_LOG_FILE, 'a', encoding='utf-8') as f:
        for response in responses:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] ")
            if hasattr(response, 'to_dict'):
                f.write(json.dumps(response.to_dict(), ensure_ascii=False))
            else:
                f.write(str(response))
            f.write('\n')

def _call_zai_api(client: ZhipuAiClient, model, messages, temperature, max_tokens, thinking):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        thinking=thinking,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=TIMEOUT_CONFIG['Thinking'] if thinking.get("type") == "enabled" else TIMEOUT_CONFIG['Default']
    )

def zai_generate(
    messages_list: List[List[Dict[str, str]]], 
    api_key: Optional[str] = None,
    model: str = "glm-4.5",
    temperature: float = 0.2,
    max_tokens: int = 32768,
    thinking: bool = False,
    desc: str = "Generating"
) -> Dict[str, Any]:
    """
    ZAI生成的核心函数，处理批量消息的API调用。
    
    Args:
        messages_list: 消息列表的列表，每个元素是一个对话历史
        api_key: ZhipuAI API密钥，如果为None则从环境变量读取
        model: 使用的模型名称
        temperature: 生成温度
        max_tokens: 最大token数
        thinking: 是否启用思考模式，默认为False
        desc: 进度条描述
        
    Returns:
        包含 'content', 'reasoning_content' 的字典
    """
    
    if api_key is None:
        api_key = os.getenv("ZAI_API_KEY_XX")
    client = ZhipuAiClient(api_key=api_key)
    
    thinking_config = {"type": "enabled"} if thinking else {"type": "disabled"}
    timeout_limit = TIMEOUT_CONFIG['Thinking'] if thinking else TIMEOUT_CONFIG['Default']
    
    workers = _get_model_concurrency(model)
    
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_call_zai_api, client, model, msg, temperature, max_tokens, thinking_config): idx
            for idx, msg in enumerate(messages_list)
        }
        
        responses = [None] * len(messages_list)
        with tqdm(total=len(messages_list), desc=desc) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    responses[idx] = future.result(timeout=timeout_limit)
                except Exception as e:
                    responses[idx] = str(e)
                pbar.update(1)

    _calculate_token_usage([resp for resp in responses if not isinstance(resp, str)])
    _log_zai_responses(responses)

    # 提取内容, 失败的返回空字符串
    content = [
        response.choices[0].message.content.strip() if not isinstance(response, str) else ""
        for response in responses
    ]
    reasoning_content = [None] * len(responses)
    if thinking:
        reasoning_content = [
            response.choices[0].message.reasoning_content.strip() if not isinstance(response, str) else ""
            for response in responses
        ]
    
    return {
        'content': content,
        'reasoning_content': reasoning_content
    }

