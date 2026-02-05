from zai import ZhipuAiClient
import time
import json
import os
import tempfile
import requests
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm


def execute_batch_api_job(
    client: ZhipuAiClient,
    messages_list: List[List[Dict[str, str]]], 
    model: str = "glm-4-plus",
    generate_config: Optional[Dict[str, Any]] = None,
    parser_func: callable = None,
    description: str = "批量处理任务",
    max_retries: int = 300,
    poll_interval: int = 300
) -> Tuple[List[Any], bool]:
    """
    执行完整的批量API任务流程
    支持自动分批，每批最大6000条。所有batch一次性提交，最后合并结果。
    """
    MAX_BATCH_SIZE = 6000

    def create_batch_requests(
        messages_list: List[List[Dict[str, str]]], 
        model: str = "glm-4-plus",
        generate_config: Optional[Dict[str, Any]] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        default_config = {
            "do_sample": False,
            "temperature": 0.95,
            "top_p": 0.7,
            "max_new_tokens": 2058,
            "repetitive_penalty": 1.2,
        }
        config = generate_config or default_config
        batch_requests = []
        for i, messages in enumerate(messages_list):
            batch_requests.append({
                "custom_id": f"request-{offset + i}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    **config
                }
            })
        return batch_requests

    def save_batch_requests_to_file(batch_requests: List[Dict[str, Any]]) -> str:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        temp_file.write('\n'.join([json.dumps(req, ensure_ascii=False) for req in batch_requests]))
        temp_file.close()
        return temp_file.name

    def submit_batch_job(client: ZhipuAiClient, file_path: str, description: str = "批量处理任务") -> Tuple[Any, Any]:
        batch_file = client.files.create(
            file=open(file_path, "rb"), 
            purpose="batch"
        )
        print(f"batch文件id: {batch_file.id}")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v4/chat/completions",
            auto_delete_input_file=True,
            metadata={"description": description}
        )
        print(f"创建Batch成功：{batch}")
        return batch_file, batch

    def poll_batch_status(client: ZhipuAiClient, batch_id: str, max_retries: int = 1500, poll_interval: int = 60) -> Optional[Any]:
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            batch_status = client.batches.retrieve(batch_id)
            print(f"[{attempt}] 任务状态: {batch_status.status}")
            if batch_status.status == "completed":
                print("API推理任务完成！")
                return batch_status
            elif batch_status.status in ["failed", "expired", "cancelled"]:
                print(f"任务失败，状态: {batch_status.status}")
                return None
            time.sleep(poll_interval)
        print("达到最大轮询次数，任务未完成")
        return None

    def download_batch_results(client: ZhipuAiClient, output_file_id: str) -> str:
        result_content = client.files.content(output_file_id)
        temp_result_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False, encoding='utf-8')
        result_content.write_to_file(temp_result_file.name)
        return temp_result_file.name

    def parse_batch_results(result_file_path: str, parser_func: callable = None, batch_size: int = None, offset: int = 0) -> List[Any]:
        from src.utils import load_jsonl_file
        results = [None] * (batch_size if batch_size is not None else 0)
        for res in load_jsonl_file(result_file_path):
            custom_id = res.get('custom_id', '')
            if custom_id.startswith('request-'):
                index = int(custom_id.split('-')[1]) - offset
                if 0 <= index < len(results):
                    if parser_func:
                        parsed_result = parser_func(res)
                    else:
                        if 'response' in res and 'body' in res['response']:
                            parsed_result = res['response']['body']['choices'][0]['message']['content']
                        else:
                            parsed_result = None
                    results[index] = parsed_result
        return results

    def cancel_batch_job(client: ZhipuAiClient, batch_id: str) -> bool:
        try:
            client.batches.cancel(batch_id)
            print("已取消远端 Batch 任务")
            return True
        except Exception as e:
            print(f"取消批量任务失败: {e}")
            return False
    
    total = len(messages_list)
    if total == 0:
        return [], True
    
    num_batches = (total + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
    print(f"统一分批处理: 总请求 {total}，批大小 {MAX_BATCH_SIZE}，批次数 {num_batches}")
    
    batch_infos = []
    temp_files = []
    all_results = [None] * total
    all_success = True

    try:
        # 提交所有批
        for i in range(num_batches):
            start = i * MAX_BATCH_SIZE
            end = min((i + 1) * MAX_BATCH_SIZE, total)
            sub_messages = messages_list[start:end]
            batch_requests = create_batch_requests(sub_messages, model, generate_config, offset=start)
            temp_file_path = save_batch_requests_to_file(batch_requests)
            temp_files.append(temp_file_path)
            batch_file, batch = submit_batch_job(
                client, temp_file_path, f"{description} (批 {i+1}/{num_batches})"
            )
            batch_infos.append({
                "batch_file": batch_file,
                "batch": batch,
                "start": start,
                "end": end,
                "size": end - start
            })

        # 轮询并收集结果
        for i, info in enumerate(batch_infos):
            print(f"等待第 {i+1}/{num_batches} 批完成...")
            batch_status = poll_batch_status(client, info["batch"].id, max_retries, poll_interval)
            if batch_status and batch_status.status == "completed":
                result_file_path = download_batch_results(client, batch_status.output_file_id)
                sub_results = parse_batch_results(
                    result_file_path, parser_func, batch_size=info["size"], offset=info["start"]
                )
                for j, res in enumerate(sub_results):
                    all_results[info["start"] + j] = res
                if result_file_path and os.path.exists(result_file_path):
                    os.unlink(result_file_path)
            else:
                print(f"第 {i+1} 批失败，填充 None")
                for idx in range(info["start"], info["end"]):
                    all_results[idx] = None
                all_success = False

        return all_results, all_success
    
    except Exception as e:
        print(f"批量任务执行出错: {e}")
        all_success = False
        # 尝试取消仍在运行的批
        for info in batch_infos:
            try:
                cancel_batch_job(client, info["batch"].id)
            except Exception:
                pass
        return all_results, all_success
    finally:
        for temp_file_path in temp_files:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                

def create_messages_list(system_prompt: str,user_prompts: List[str],) -> List[List[Dict[str, str]]]:
    messages_list = []
    for user_prompt in user_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        messages_list.append(messages)
    return messages_list


class GLMAPI:
    def __init__(self, api_key="", model='glm-4-plus'):
        self.client = ZhipuAiClient(api_key=api_key)
        self.api_key = api_key
        self.model = model
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.async_max_retries = 120  # 异步任务最大重试次数，60次，每次等待1秒，总计120秒
        self.max_concurrent_tasks = 5  # 最大并发任务数
        
    def async_process(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        """异步批处理数据，限制并发数在5以内
        
        Args:
            conversations: 消息列表的列表，每个元素是一个完整的对话消息列表
        """
        total_data = len(conversations)
        results = [""] * total_data
        total_batches = (total_data + self.max_concurrent_tasks - 1) // self.max_concurrent_tasks

        with tqdm(total=total_batches, desc="批次处理进度", unit="batch") as pbar:
            for batch_idx, batch_start in enumerate(range(0, total_data, self.max_concurrent_tasks), 1):
                batch_end = min(batch_start + self.max_concurrent_tasks, total_data)
                batch_data = conversations[batch_start:batch_end]
                
                pbar.set_postfix({"任务": f"{batch_start+1}-{batch_end}/{total_data}"})

                # 提交当前批次的任务
                task_ids = self._submit_async_tasks(batch_data)

                # 轮询当前批次的结果
                batch_results = self._poll_async_tasks(task_ids)

                # 将结果填入总结果列表
                for i, result in enumerate(batch_results):
                    results[batch_start + i] = result
                
                pbar.update(1)

        print(f"所有任务完成，共处理 {total_data} 个任务")
        return results
    
    def _submit_async_tasks(self, batch_data: List[List[Dict[str, str]]]) -> List[str]:
        """提交异步任务
        
        Args:
            batch_data: 批次消息列表，每个元素是一个完整的对话消息列表
        """
        task_ids = []
        
        for messages in batch_data:
            try:
                response = self.client.chat.asyncCompletions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=64,
                    temperature=0.2
                )
                task_ids.append(response.id)
            except Exception as e:
                print(f"提交任务失败: {str(e)}")
                task_ids.append(None)
        
        return task_ids
    
    def _poll_async_tasks(self, task_ids: List[str]) -> List[str]:
        """轮询异步任务结果"""
        results = [""] * len(task_ids)
        
        for attempt in range(self.async_max_retries):
            all_done = True
            
            for idx, task_id in enumerate(task_ids):
                if results[idx] or task_id is None:  # 已完成或任务ID为空
                    continue
                    
                try:
                    resp = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                    
                    if resp.task_status == "SUCCESS":
                        results[idx] = resp.choices[0].message.content.strip()
                    elif resp.task_status == "FAILED":
                        results[idx] = "Task failed"
                    else:
                        all_done = False
                        
                except Exception as e:
                    print(f"轮询任务 {task_id} 时出错: {str(e)}")
                    results[idx] = f"API Error: {str(e)}"
            
            if all_done:
                break
                
            time.sleep(1)  # 等待1秒后重试
        
        # 处理未完成的任务
        for idx, result in enumerate(results):
            if not result and task_ids[idx] is not None:
                results[idx] = "Task timeout"
        
        return results

    def http_call(self, messages: List[Dict], temperature: float = 0.6, max_tokens: int = 1024) -> str:
        """HTTP方式调用智谱AI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API调用失败: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"HTTP调用出错: {str(e)}")
            return f"HTTP Error: {str(e)}"

    def http_process(self, infer_data: List[dict], temperature: float = 0.6) -> List[str]:
        """使用HTTP方式批量处理数据"""
        results = []
        total_data = len(infer_data)
        
        for i, item in enumerate(infer_data):
            print(f"处理任务 {i+1}/{total_data}")
            
            system = item.get('system', '')
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            
            # 构建完整的prompt
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"}
            ]
            
            result = self.http_call(messages, temperature)
            results.append(result)
            
            # 避免请求过于频繁
            time.sleep(0.1)
        
        print(f"HTTP批量处理完成，共处理 {total_data} 个任务")
        return results


if __name__ == "__main__":
    api_key = "c03f99aa01c14d9b8456c552b995c2e8.dusLhmOVIGJn9PDH"
    api_client = GLMAPI(api_key=api_key, model="glm-4-plus")
    test_data = [
        {
            "system": "你是一个有帮助的助手。",
            "instruction": "请总结以下内容的主要观点。",
            "input": "人工智能正在迅速发展，影响着各个行业。它有潜力提高生产力，但也带来了伦理和隐私方面的挑战。"
        },
        {
            "system": "你是一个有帮助的助手。",
            "instruction": "请翻译以下句子成英文。",
            "input": "今天天气很好，我们去公园散步吧。"
        }
    ]
    results = api_client.async_process(test_data)
    for res in results:
        print("结果:", res)