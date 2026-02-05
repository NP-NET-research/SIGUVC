from datetime import datetime
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import subprocess
from src.utils import copy_file, load_config_from_yaml, load_json_file, load_jsonl_file, save_config_to_yaml, save_json_file, setup_save_directory, update_dataset_info
from src.inference import GLMAPI
from src.inference.vllm_infer import vllm_infer

DEFAULT_VLLM_CONFIG = {
        "model_name_or_path": "/home/ljl/Data/LLM/GLM-4-9B-0414",
        "adapter_name_or_path": None,
        "template": "glm4",
        "cutoff_len": 2048,
        "max_samples": 1000000,
        "vllm_config": {
            "max_model_len": 15424,
            "enforce_eager": False,
        },
        "temperature": 0.95,
        "top_p": 0.8,
        "max_new_tokens": 512,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "num_sequences": 1,
        "dataset": "test_infer",
        "dataset_dir": "data",
        "save_name": "generated_predictions.jsonl",
    }

def process_inference_results(work_dir, test_data_path, predictions_source: list|str, save_frame="llm_pred"):
    """    
    Args:
        work_dir: 工作目录
        test_data_path: 测试数据路径
        predictions_source: 预测结果源（jsonl文件路径或直接的预测结果列表）
        save_frame: 保存结果的字段名
    """
    all_test_data = load_json_file(test_data_path)
    
    # 根据源类型获取预测结果
    if isinstance(predictions_source, str):
        assert predictions_source.endswith('.jsonl'), "predictions_source must be a jsonl file path"
        # 从jsonl文件中提取预测结果
        jsonl_data = load_jsonl_file(predictions_source)
        predictions = []
        for item in jsonl_data:
            if 'predict' in item:
                pred = item['predict']
                # 判断predict格式并提取
                if isinstance(pred, list):  # directly_infer格式是一个list
                    if len(pred) == 1:
                        predictions.append(pred[0])  # 直接取第一个元素
                    else:
                        predictions.append(pred)  
                elif isinstance(pred, str):
                    predictions.append(pred)     # llamafy_adapter_infer格式
                else:
                    predictions.append(str(pred))
            else:
                predictions.append("")
    elif isinstance(predictions_source, list):
        # API结果格式
        predictions = predictions_source
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    
    # 验证长度匹配
    assert len(predictions) == len(all_test_data), f"Length mismatch: {len(predictions)} vs {len(all_test_data)}"
    
    for pred, record in zip(predictions, all_test_data):
        if "data_frame" not in record:
            record["data_frame"] = {}
        record["data_frame"][save_frame] = [pred.strip()] if isinstance(pred, str) else pred

    results_file = os.path.join(work_dir, "results.json")
    save_json_file(data=all_test_data, path=results_file)
    
    return results_file

def api_infer(model="glm-4-plus", test_data_path=None, task_description="修辞检测", base_dir="./saves"):
    """使用GLM API进行批处理推理"""
    
    assert test_data_path, "test_data_path must be provided for API inference"
    
    work_dir = setup_save_directory(base_dir, "api_infer", test_data_path)
    
    # 初始化API客户端
    api_key = "c03f99aa01c14d9b8456c552b995c2e8.dusLhmOVIGJn9PDH"
    api_client = GLMAPI(api_key=api_key, model=model)

    test_data = load_json_file(test_data_path)
    
    # 进行批处理推理
    # print("开始API批处理推理...")
    # results = api_client.batch_process(
    #     infer_data=load_json_file(test_data_path),
    #     description=task_description
    # )
    print("API异步推理")
    results:list = api_client.async_process(
        infer_data=test_data,
    )
    
    process_inference_results(work_dir, test_data_path, results)
    return work_dir

def train_sft(model_name_or_path="/home/ljl/Data/LLM/GLM-4-9B-0414", train_dataset_path="/home/ljl/zhipu/corpus/train_sft.json", base_dir="./saves"):
    
    work_dir = setup_save_directory(base_dir, "train_sft", train_dataset_path)
    copy_file(train_dataset_path, os.path.join(work_dir, "train_sft.json"))
    update_dataset_info(dataset_info_name="train_sft", dataset_path='train_sft.json', save_dir=work_dir)
    
    config = load_config_from_yaml("configs/train_sft.yaml")
    config.update({
        "model_name_or_path": model_name_or_path,
        "output_dir": work_dir,
        "adapter_name_or_path": None,  
        "dataset_dir": work_dir
    })

    updated_config_file = os.path.join(work_dir, "updated_train_config.yaml")
    save_config_to_yaml(config, updated_config_file)
    subprocess.run(["llamafactory-cli", "train", updated_config_file])
    return work_dir

def train_ppo(
        train_dataset_path="corpus/train_ppo.json", 
        train_config_path="configs/train_ppo.yaml",
        base_dir="./saves"
    ):
    
    work_dir = setup_save_directory(base_dir, "train_ppo", train_dataset_path)
    copy_file(train_dataset_path, os.path.join(work_dir, "train_ppo.json"))
    update_dataset_info(dataset_info_name="train_ppo", dataset_path='train_ppo.json', save_dir=work_dir)
    
    config = load_config_from_yaml(train_config_path)
    config.update({
        "model_name_or_path": config.get("model_name_or_path", "/home/ljl/Data/LLM/GLM-4-9B-0414"),
        "output_dir": work_dir,
        "adapter_name_or_path": None,  
        "dataset_dir": work_dir,
        "dataset": "train_ppo"
    })

    updated_config_file = os.path.join(work_dir, "updated_train_config.yaml")
    save_config_to_yaml(config, updated_config_file)
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore:Trainer.tokenizer is now deprecated"
    subprocess.run(
        ["llamafactory-cli", "train", updated_config_file],
        env=env
    )
    return work_dir

def train_dpo(
        model_name_or_path="/home/ljl/Data/LLM/GLM-4-9B-0414", 
        train_dataset_path="corpus/train_dpo.json", 
        dev_dataset_path="corpus/dev_dpo.json",
        base_dir="./saves",
        config_path="configs/train_dpo.yaml"
    ):
    """
    DPO训练
    
    Args:
        model_name_or_path: 基础模型路径
        train_dataset_path: 训练数据路径
        base_dir: 保存目录
        config_path: 配置文件路径
    
    Returns:
        adapter保存路径
    """
    work_dir = setup_save_directory(base_dir, "train_dpo", train_dataset_path)
    # copy train dataset to work_dir and update dataset info
    copy_file(train_dataset_path, os.path.join(work_dir, "train_dpo.json"))
    copy_file(dev_dataset_path, os.path.join(work_dir, "dev_dpo.json"))
    update_dataset_info(dataset_info_name="train_dpo", dataset_path='train_dpo.json', save_dir=work_dir)
    update_dataset_info(dataset_info_name="dev_dpo", dataset_path='dev_dpo.json', save_dir=work_dir)
    
    config = load_config_from_yaml(config_path)
    config.update({
        "model_name_or_path": model_name_or_path,
        "output_dir": work_dir,
        "adapter_name_or_path": None,  
        "dataset_dir": work_dir,
        "dataset": "train_dpo",
        "eval_dataset": "dev_dpo",
    })

    updated_config_file = os.path.join(work_dir, "updated_train_config.yaml")
    save_config_to_yaml(config, updated_config_file)
    subprocess.run(["llamafactory-cli", "train", updated_config_file])
    
    # 返回adapter路径
    return work_dir

def merge(
        model_name_or_path="/home/ljl/Data/LLM/GLM-4-9B-0414", 
        adapter_name_or_path=None, 
        save_model_name="merged_model", 
        base_dir="./saves",
        config_path="configs/merge_adapter.yaml"
    ):
    """
    合并模型和adapter
    
    Args:
        model_name_or_path: 基础模型路径
        adapter_name_or_path: adapter路径
        save_model_name: 合并后模型名称
        base_dir: 保存目录
        config_path: 配置文件路径
    
    Returns:
        合并后模型路径
    """
    assert adapter_name_or_path is not None, "adapter_name_or_path must be provided for merging"

    work_dir = setup_save_directory(base_dir, "merge", save_model_name)
    merged_model_path = os.path.join(base_dir, save_model_name)
    
    config = load_config_from_yaml(config_path)
    config.update({
        "model_name_or_path": model_name_or_path,
        "adapter_name_or_path": adapter_name_or_path,
        "export_dir": merged_model_path,
    })

    updated_config_file = os.path.join(work_dir, "updated_merge_config.yaml")
    save_config_to_yaml(config, updated_config_file)
    subprocess.run(["llamafactory-cli", "export", updated_config_file])
    
    return merged_model_path

def infer_with_adapter(
        model_name_or_path="/home/ljl/Data/LLM/GLM-4-9B-0414", 
        adapter_name_or_path=None, 
        data_path=None,
        save_frame="llm_pred",
        base_dir="./saves"
    ):

    assert data_path, "test_data_path must be provided for LLaMA-Factory inference"
    work_dir = setup_save_directory(base_dir, "adapter_infer", data_path)
    
    copy_file(data_path, os.path.join(work_dir, "test_infer.json"))
    update_dataset_info(dataset_info_name="test_infer", dataset_path='test_infer.json', save_dir=work_dir)
    
    config = load_config_from_yaml("configs/test_infer.yaml")
    config.update({
        "model_name_or_path": model_name_or_path,
        "output_dir": work_dir,
        "adapter_name_or_path": adapter_name_or_path,  
        "dataset_dir": work_dir
    })
    updated_config_file = os.path.join(work_dir, "updated_llamafy_config.yaml")
    save_config_to_yaml(config, updated_config_file)
    
    subprocess.run(["llamafactory-cli", "train", updated_config_file])

    llm_pred_file = os.path.join(work_dir, "generated_predictions.jsonl")
    result_file_path = process_inference_results(work_dir, data_path, llm_pred_file, save_frame=save_frame)

    return result_file_path

def infer_by_vllm(config=None, data_path=None, save_frame="llm_pred",base_dir="./saves"):
    """使用VLLM进行推理
    Args:
        data_path: 数据文件路径
        save_frame: 保存结果的字段名
        base_dir: 保存目录
        config: 推理配置字典，包含所有推理参数
    """
    assert data_path, "data_path must be provided for direct inference"
    assert config, "config must be provided for VLLM inference"

    work_dir = setup_save_directory(base_dir, "vllm_infer", data_path)
    copy_file(data_path, os.path.join(work_dir, "test_infer.json"))
    update_dataset_info(dataset_info_name="test_infer", dataset_path='test_infer.json', save_dir=work_dir)

    llm_pred_file = os.path.join(work_dir, 'generated_predictions.jsonl')
    
    final_config = DEFAULT_VLLM_CONFIG.copy()
    final_config.update(config)
    if "vllm_config" in config:
        final_config["vllm_config"] = {**DEFAULT_VLLM_CONFIG["vllm_config"], **config["vllm_config"]}
    final_config.update({
        "dataset": "test_infer",
        "dataset_dir": work_dir,
        "save_name": llm_pred_file,
    })

    vllm_infer(**final_config)

    result_file_path = process_inference_results(work_dir, data_path, llm_pred_file, save_frame=save_frame)
    return result_file_path


if __name__ == "__main__":
    model_name_or_path = "/data/ljl/Data/LLM/GLM-4-9B-0414"
    adapter_name_or_path = "/data/ljl/zhipu/checkpoints/rhetoric_use_adapter/train-ppo-300"

    merge(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        save_model_name="glm4-9b-rhetoric-use",
        base_dir="/data/ljl/zhipu/saves"
    )


