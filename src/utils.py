
from collections import defaultdict
from datetime import datetime
import json
import os
import pickle
import re
from typing import List, Dict, Set, Tuple
from tqdm import tqdm


def setup_save_directory(base_dir="./saves", prefix="run", input_file="data.jsonl") -> str:
    timestamp = datetime.now().strftime("%m%d_%H%M")
    filename = os.path.basename(input_file).replace('.jsonl', '').replace('.json', '')
    save_dir = os.path.join(base_dir, f"{prefix}_{filename}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def load_sts_file(path: str) -> List[Tuple[str, str, float]]:
    """读取STS文件，返回句子对和相似度分数"""
    sts_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            sent1, sent2, score = parts
            sts_data.append((sent1, sent2, float(score)))
    return sts_data


def pku_seg(texts: List[str]) -> List[List[str]]:
    import pkuseg
    seg = pkuseg.pkuseg()           # 以默认配置加载模型
    return [seg.cut(text) for text in tqdm(texts, desc="Segmenting texts")]

def copy_file(src: str, dst: str):
    with open(src, "rb") as f_src:
        with open(dst, "wb") as f_dst:
            f_dst.write(f_src.read())
    print(f"Copied file from {src} to {dst}")

def parse_glm4_chat_template(text: str) -> List[Dict[str, str]]:
    """
    解析 GLM4 chat 模板输出内容，提取原始的对话 messages 列表。
    支持 role: system, user, assistant（含 metadata）, observation。
    """

    # 匹配 role 标签，例如 <|user|> 或 <|assistant|>{"foo":"bar"}
    role_pattern = re.compile(
        r"<\|(?P<role>system|user|assistant|observation)\|>(?P<meta>\{.*?\})?",
        re.DOTALL
    )

    messages = []
    last_end = 0
    current_role = None
    current_meta = None

    for match in role_pattern.finditer(text):
        # 匹配到新 role，先把之前的内容作为前一条消息保存
        if current_role is not None:
            content = text[last_end:match.start()].strip()
            if content:
                # assistant 可能带 metadata
                if current_role == "assistant" and current_meta:
                    messages.append({
                        "role": current_role,
                        "content": content,
                        "metadata": current_meta
                    })
                else:
                    messages.append({
                        "role": current_role,
                        "content": content
                    })

        # 更新当前 role
        current_role = match.group("role")
        meta_raw = match.group("meta")

        # 提取 metadata（如果存在）
        if meta_raw:
            try:
                # 尝试安全解析 JSON 格式的 meta
                import json
                current_meta = json.loads(meta_raw)
            except:
                # meta 格式可能不是完整 JSON，这里保留原文
                current_meta = meta_raw
        else:
            current_meta = None

        last_end = match.end()

    # 最后一段内容（如果有）
    if current_role is not None:
        content = text[last_end:].strip()
        if content:
            if current_role == "assistant" and current_meta:
                messages.append({
                    "role": current_role,
                    "content": content,
                    "metadata": current_meta
                })
            else:
                messages.append({
                    "role": current_role,
                    "content": content
                })

    return messages


def safe_parse_json(text: str) -> tuple[str, dict]:
    """
    Robust JSON parsing from LLM outputs.
    
    输入: 模型输出 (str)
    输出: (解析后的json字符串, dict对象)
    """
    from json_repair import repair_json
    import ast
    raw_input = text.strip()
    result = None

    # 1. 直接尝试 json.loads
    try:
        result = json.loads(raw_input)
        return raw_input, result
    except json.JSONDecodeError:
        pass

    # 2. 清理 Markdown 包裹
    if raw_input.startswith("```json"):
        raw_input = raw_input[len("```json"):]
    if raw_input.startswith("```"):
        raw_input = raw_input[len("```"):]
    if raw_input.endswith("```"):
        raw_input = raw_input[:-len("```")]

    raw_input = raw_input.strip()

    # 3. 尝试正则抽取 {...}
    match = re.search(r"\{.*\}", raw_input, re.DOTALL)
    if match:
        raw_input = match.group(0)

    # 4. 替换常见错误字符
    raw_input = (
        raw_input.replace("{{", "{")
        .replace("}}", "}")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\\", "")
        .strip()
    )

    # 5. 再次尝试 json.loads
    try:
        result = json.loads(raw_input)
        return raw_input, result
    except json.JSONDecodeError:
        pass

    # 6. 使用 json_repair 尝试修复
    try:
        repaired = repair_json(raw_input, return_objects=False)
        result = json.loads(repaired)
        return repaired, result
    except Exception:
        pass

    # 7. 最后尝试 Python AST 解析 (兼容单引号 / True/False/None)
    try:
        result = ast.literal_eval(raw_input)
        if isinstance(result, dict):
            return str(result), result
    except Exception:
        pass

    # 8. 全部失败，返回空 dict
    print("Failed to parse JSON from input: %s", text[:200])
    return raw_input, {}


def load_lines_file(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

# 允许空行的load_lines_file
def load_lines_file_allow_empty(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.rstrip('\n') for line in lines]

def save_lines_file(lines: list, path: str, mode='w'):
    with open(path, mode=mode, encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")
    print(f"Successfully saved lines to {path}")

def load_json_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json_file(data, path: str):
    if not path.lower().endswith(".json"):
        path = os.path.splitext(path)[0] + ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved data to {path}")

def load_jsonl_file(file_path: str) -> list:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total = len(lines)
        for line in tqdm(lines, total=total, desc="Loading JSONL"):
            data.append(json.loads(line.strip()))
    return data

def save_jsonl_file(data: list, path: str, mode="w"):
    with open(path, mode=mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # print(f"Successfully saved data to {path}")

def save_tsv_file(data, path):
    assert isinstance(data, list), "data must be a list"
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            assert isinstance(item, (list, tuple)), "Each item in data must be a list or tuple"
            f.write('\t'.join(str(x) for x in item) + "\n")
    print(f"Successfully saved data to {path}")

def load_tsv_file(path, skip_header=False):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        if skip_header:
            next(f)  # Skip the header line
        for line in f:
            data.append(line.strip().split('\t'))
    return data



def load_config_from_yaml(yaml_file: str)-> dict:
    import yaml
    with open(yaml_file, 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)
    return params

def save_config_to_yaml(params, yaml_file):
    import yaml
    with open(yaml_file, 'w', encoding='utf-8') as file:
        yaml.dump(params, file, default_flow_style=False, allow_unicode=True)
    print(f"Successfully saved config to {yaml_file}")


def update_dataset_info(dataset_info_name, dataset_path, save_dir=None):
    """
    Update the file path of a specific dataset in the dataset_info.json file.
    Args:
        dataset_name (str): dataset_info 的注册名称.
        dataset_path (str): 指向的新的infer数据名称位置.
        save_path (str, optional): dataset_info保存位置.
    """
    config_path = "LLaMA-Factory/data/dataset_info.json"
    
    data = load_json_file(config_path)
    if dataset_info_name in data and "file_name" in data[dataset_info_name]:
        data[dataset_info_name]["file_name"] = dataset_path
        
        if save_dir is not None:
            save_json_file(data, os.path.join(save_dir, 'dataset_info.json'))
    else:
        print(f"Dataset '{dataset_info_name}' not found or 'file_name' key not present.")


def save_pickle_file(data, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle_file(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
   pass
