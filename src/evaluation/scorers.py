import os
import gc
from typing import Any, Dict, List, Tuple

from openai import OpenAI
import torch
from src.inference.generator import BaseGenerator
from src.inference.prompt import PromptRegistry
from src.utils import safe_parse_json

# ====================== 词汇覆盖率评估 ======================
def evaluate_vocab_coverage(
        tokens: List[List[str]], 
        mini_vocab: List[set],
    ) -> List[float]:
    """评估词汇覆盖率，基于小词表
    返回:
    - coverages: 每条的覆盖率
    """

    if not tokens or not mini_vocab or len(tokens) != len(mini_vocab):
        n = len(tokens) if tokens else 0
        return [0.0] * n
    
    coverages: List[float] = []
    
    for tok_list, vocab in zip(tokens, mini_vocab):
        total_tokens = len(tok_list)
        if total_tokens == 0:
            coverages.append(0.0)
        else:
            matched = [t for t in tok_list if t in vocab]
            unmatched = [t for t in tok_list if t not in vocab]
            coverage = len(matched) / total_tokens
            coverages.append(coverage)
            
    return coverages

# ====================== 相似度评估 ======================

def llm_evaluate_similarity(sentence_pairs: List[Tuple[str, str]], client: OpenAI) -> List[float]:
    """
    使用 LLM 进行相似度评估。
    
    Args:
        sentence_pairs: 待评估的句子对列表（原句, 重写句）
        client: OpenAI 客户端实例
        
    Returns:
        评估结果列表
    """
    from src.inference.prompt import PromptRegistry
    from src.inference.openai_utils import generate_with_api_parallel
    
    similarity_template = PromptRegistry.get("similarity_evaluation")

    convs = [
        [
            {"role": "system", "content": similarity_template.system},
            {"role": "user", "content": similarity_template.instruction.format(original=orig, rewritten=rew)},
        ] for orig, rew in sentence_pairs
    ]

    similarity_responses = generate_with_api_parallel(
        client=client,
        conversations=convs,
        temperature=0.0,
        max_tokens=128,
        prefix="similarity evaluation: ",
    )

    evaluation_results = []
    for resp in similarity_responses:
        _, parsed = safe_parse_json(resp)
        if parsed is not None and "score" in parsed:
            try:
                score = float(parsed["score"])
                evaluation_results.append(score)
            except ValueError:
                evaluation_results.append(-1.0)  # 无效分数
        else:
            evaluation_results.append(-1.0)  # 解析失败或缺失字段
    
    return evaluation_results

def vllm_evaluate_similarity(sentence_pairs: List[Tuple[str, str]], generator=None) -> List[float]:
    """
    使用本地 vLLM 进行相似度评估。
    
    Args:
        sentence_pairs: 待评估的句子对列表（原句, 重写句）
        generator: BaseGenerator 实例（已配置使用 vLLM），如果为 None 则使用默认配置
        
    Returns:
        评估分数列表，每个元素为一个句子对的相似度分数（0-10），解析失败返回 -1.0
    """
    from src.inference.prompt import PromptRegistry
    
    similarity_template = PromptRegistry.get("similarity_evaluation")
    
    convs = [
        [
            {"role": "system", "content": similarity_template.system},
            {"role": "user", "content": similarity_template.instruction.format(original=orig, rewritten=rew)},
        ] for orig, rew in sentence_pairs
    ]

    need_cleanup = False    # 回收标记
    if generator is None:
        from src.inference.generator import BaseGenerator, GenerationConfig
        
        config = GenerationConfig(
            model_name_or_path="/home/ljl/Data/LLM/GLM-4-32B-0414",
            use_vllm=True,
            max_new_tokens=128,
            temperature=0.0,
            top_p=0.95,
            do_sample=False,
            num_sequences=1,
            vllm_config={
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            }
        )
        generator = BaseGenerator(config)
        need_cleanup = True
    
    # 使用本地生成器生成结果
    outputs = generator.generate(
        convs,
        temperature=0.0,
        max_new_tokens=128
    )
    
    # 处理输出结果并解析分数
    evaluation_results = []
    for output_list in outputs:
        resp = output_list[0].strip() if output_list else ""
        _, parsed = safe_parse_json(resp)
        if parsed is not None and "score" in parsed:
            try:
                score = float(parsed["score"])
                evaluation_results.append(score)
            except ValueError:
                evaluation_results.append(-1.0)  # 无效分数
        else:
            evaluation_results.append(-1.0)  # 解析失败或缺失字段

    if need_cleanup:
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    return evaluation_results

# ====================== 流畅性评估 ======================

def llm_evaluate_fluency(sentences: List[str], client: OpenAI) -> List[float]:
    """
    使用 LLM 评估文本是否流畅。
    
    Args:
        sentences: 待评估的文本列表
        client: OpenAI 客户端实例
        
    Returns:
        是否流畅的分数列表（0或1）
    """
    from src.inference.prompt import PromptRegistry
    from src.inference.openai_utils import generate_with_api_parallel

    fluency_template = PromptRegistry.get("fluency_evaluation")

    convs = [
        [
            {"role": "system", "content": fluency_template.system},
            {"role": "user", "content": fluency_template.instruction + sent},
        ] for sent in sentences
    ]

    fluency_responses = generate_with_api_parallel(
        client=client,
        conversations=convs,
        temperature=0.2,
        max_tokens=64,
        prefix="fluency evaluation: "
    )

    def _is_fluent(response: str) -> bool:
        """判断流畅性检测结果是否为正面（流畅）"""
        if not response:
            return False
        return response.strip().startswith("流畅")
    
    evaluation_results = [
        1.0 if _is_fluent(resp) else 0.0 for resp in fluency_responses  
    ]

    return evaluation_results

def vllm_evaluate_fluency(sentences: List[str], generator=None) -> List[float]:
    """
    使用本地 vLLM 评估文本是否流畅。
    
    Args:
        sentences: 待评估的文本列表
        generator: BaseGenerator 实例（已配置使用 vLLM），如果为 None 则使用默认配置
        
    Returns:
        是否流畅的分数列表（0或1）
    """
    from src.inference.prompt import PromptRegistry

    fluency_template = PromptRegistry.get("fluency_evaluation")

    convs = [
        [
            {"role": "system", "content": fluency_template.system},
            {"role": "user", "content": fluency_template.instruction + sent},
        ] for sent in sentences
    ]

    need_cleanup = False    # 回收标记
    if generator is None:
        from src.inference.generator import BaseGenerator, GenerationConfig
        
        config = GenerationConfig(
            model_name_or_path="/home/ljl/Data/LLM/GLM-4-32B-0414",
            use_vllm=True,
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            num_sequences=1,
            vllm_config={
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            }
        )
        generator = BaseGenerator(config)
        need_cleanup = True
    
    # 使用本地生成器生成结果
    outputs = generator.generate(
        convs,
        temperature=0.2,
        max_new_tokens=64
    )
    
    # 处理输出结果并解析分数
    def _is_fluent(response: str) -> bool:
        """判断流畅性检测结果是否为正面（流畅）"""
        if not response:
            return False
        return response.strip().startswith("流畅")
    
    evaluation_results = []
    for output_list in outputs:
        resp = output_list[0].strip() if output_list else ""
        evaluation_results.append(1.0 if _is_fluent(resp) else 0.0)

    if need_cleanup:
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    return evaluation_results

# ====================== 修辞使用评估 ======================
def llm_evaluate_rhetoric(sentences: List[str], client: OpenAI) -> List[float]:
    """
    使用 LLM 评估文本中的修辞使用情况。
    
    Args:
        sentences: 待评估的文本列表
        client: OpenAI 客户端实例
        
    Returns:
        每个文本的修辞使用分数列表（0 或 1）
    """
    from src.inference.prompt import PromptRegistry
    from src.inference.openai_utils import generate_with_api_parallel

    rhetoric_template = PromptRegistry.get("rhetoric_detection")

    convs = [
        [
            {"role": "system", "content": rhetoric_template.system},
            {"role": "user", "content": rhetoric_template.instruction + sent},
        ] for sent in sentences
    ]

    rhetoric_responses = generate_with_api_parallel(
        client=client,
        conversations=convs,
        temperature=0.0,
        max_tokens=64,
        prefix="rhetoric usage evaluation: ",
    )

    def _is_positive(response: str) -> bool:
        """判断修辞检测结果是否为正面（有修辞）"""
        if not response:
            return False
        return response.strip().startswith("是")

    evaluation_results = [
        1.0 if _is_positive(resp) else 0.0 for resp in rhetoric_responses
    ]

    return evaluation_results

def evaluate_rhetoric_usage(
        texts: List[str],
        model_path: str = 'checkpoints/glm4-9b-0414-rhetoric-use'
    ) -> List[float]:
    """评估文本修辞使用情况，返回每个文本的修辞使用分数（0或1）
    """

    def _is_positive(response: str) -> bool:
        """判断修辞检测结果是否为正面（有修辞）"""
        if not response:
            return False
        return response.strip().startswith("是")
    
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
    _generator = BaseGenerator(config={
        "model_name_or_path": model_path,
        "adapter_paths": {},
        "use_vllm": True,
        "do_sample": False,
    })
    
    try:
        rhetoric_judgements = _generator.generate(
            queries=texts,
            prompt_template=PromptRegistry.get("rhetoric_detection")
        )
        return [1.0 if _is_positive(judgement[0]) else 0.0 for judgement in rhetoric_judgements]
    finally:
        del _generator
        torch.cuda.empty_cache()
        gc.collect()

# ====================== 其他 ======================
def evaluate_semantic_local(
        candidates_flat: List[str],
        references_flat: List[str],
        model_path: str = '/home/ljl/Data/LLM/GLM-4-32B-0414'
    ) -> Tuple[List[Dict[str, float]], List[Any]]:
    """本地语义评估函数，输入扁平列表，输出扁平三分与原始响应"""
    assert len(candidates_flat) == len(references_flat), "候选和参考数量不匹配"

    # 构造逐候选查询
    queries: List[str] = []
    for ref, cand in zip(references_flat, candidates_flat):
        input_text = f"【原句】\n {ref}\n\n【改写句子】\n{cand}"
        queries.append(input_text)

    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
    _generator = BaseGenerator(config={
        "model_name_or_path": model_path,
        "adapter_paths": {},
        "use_vllm": True,
        "do_sample": False,
    })

    try:
        prompt_template = PromptRegistry.get("eval_fluency")
        outputs = _generator.generate(queries=queries, prompt_template=prompt_template)

        scores_flat: List[Dict[str, float]] = []
        raw_outputs_flat: List[Any] = []

        for out in outputs:
            resp = out[0]
            sem_for_cand: Dict[str, float] = {"fluency": 0.0, "similarity": 0.0, "simplicity": 0.0}
            raw_obj: Any = resp
            try:
                _, parsed = safe_parse_json(resp)
                if isinstance(parsed, dict):
                    sem_for_cand = {
                        "fluency": float(parsed.get("fluency", 0.0)),
                        "similarity": float(parsed.get("similarity", 0.0)),
                        "simplicity": float(parsed.get("simplicity", 0.0)),
                    }
                raw_obj = parsed
            except Exception:
                sem_for_cand = {"fluency": 0.0, "similarity": 0.0, "simplicity": 0.0}
            scores_flat.append(sem_for_cand)
            raw_outputs_flat.append(raw_obj)

        return scores_flat, raw_outputs_flat

    finally:
        del _generator
        torch.cuda.empty_cache()
        gc.collect()

