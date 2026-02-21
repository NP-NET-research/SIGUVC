from typing import Dict, List, Set, Tuple
from openai import OpenAI
import gc
from src.utils import load_lines_file

# 默认生成器配置
DEFAULT_MODEL_PATH = "/home/ljl/Data/LLM/GLM-4-32B-0414"
DEFAULT_VLLM_CONFIG = {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
}


def _create_default_generator(temperature: float = 0.1, max_new_tokens: int = 512):
    """创建默认的 vLLM 生成器"""
    from src.inference.generator import BaseGenerator, GenerationConfig
    
    config = GenerationConfig(
        model_name_or_path=DEFAULT_MODEL_PATH,
        use_vllm=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_sequences=1,
        vllm_config=DEFAULT_VLLM_CONFIG,
    )
    return BaseGenerator(config)


def _generate_responses(
    conversations: List[List[Dict]],
    client: OpenAI = None,
    generator = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    prefix: str = ""
) -> List[str]:
    """
    通用的响应生成函数，处理 client 或 generator 的选择和资源清理。
    
    Args:
        conversations: 对话列表
        client: OpenAI 客户端实例
        generator: BaseGenerator 实例
        temperature: 生成温度
        max_tokens: 最大token数
        prefix: API调用的前缀（仅用于client模式）
        
    Returns:
        生成的响应列表
    """
    need_cleanup = False
    
    if client is not None:
        from src.inference.openai_utils import generate_with_api_parallel
        responses = generate_with_api_parallel(
            client=client,
            conversations=conversations,
            temperature=temperature,
            max_tokens=max_tokens,
            prefix=prefix,
        )
    else:
        if generator is None:
            generator = _create_default_generator(temperature=temperature, max_new_tokens=max_tokens)
            need_cleanup = True
        
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


def llm_segment_queries(queries: List[str], client: OpenAI = None, generator = None) -> List[List[str]]:
    """
    对查询进行分词处理。
    
    Args:
        queries: 待分词的查询列表
        client: OpenAI 客户端实例（与 generator 二选一）
        generator: BaseGenerator 实例（与 client 二选一）
        
    Returns:
        分词后的结果列表，每个元素为一个词列表
    """
    from src.inference.prompt import PromptRegistry
    
    seg_template = PromptRegistry.get('seg')
    convs = [
        [
            {"role": "system", "content": seg_template.system},
            {"role": "user", "content": seg_template.instruction + q},
        ] for q in queries
    ]

    seg_responses = _generate_responses(
        convs, client, generator,
        temperature=0.2,
        max_tokens=512,
        prefix="segmentation: "
    )
    
    segmented_queries = [resp.strip().split(' ') for resp in seg_responses]
    return segmented_queries


def llm_extract_entities(queries: List[str], client: OpenAI = None, generator = None) -> List[List[str]]:
    """
    进行实体识别。
    
    Args:
        queries: 待识别的查询列表
        client: OpenAI 客户端实例（与 generator 二选一）
        generator: BaseGenerator 实例（与 client 二选一）
        
    Returns:
        实体集合列表，每个元素为一个查询对应的实体集合
    """
    
    SYSTEM = """你是一个专业的信息抽取系统，请严格按照要求从句子中提取信息，不要进行任何推断或补充。
【抽取目标】
1. 命名实体（NER）
   - 包括：人名、地名、机构名、时间表达、国家、公司名称、事件名称等明确指代实体的词语。
   - 必须是句子中出现的文本，不能根据常识或语境自行补全。

2. 术语（Term）
   - 专业领域内有明确概念定义的词语，如技术术语、学科术语、政策术语等。
   - 只能抽取句子中出现的术语，不得扩展或改写。

3. 数字信息（Number）
   - 包括：具体数量、百分比、时间点、日期、统计数值、序号等。
   - 保留数字原文表达（如“3万”“2024年”“50%”）。

【输出格式要求】
- 将抽取到的所有信息用顿号“、”连接。
- 按出现顺序输出，不分组。
- 如果没有任何可抽取信息，返回“无”。
- 不允许返回解释、分析或额外文本。"""
    
    INSTRUCTION = """按照上述要求抽取句子中的实体。
【待处理句子】
句子：{sentence}"""

    convs = [
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.format(sentence=q)},
        ] for q in queries
    ]

    ner_responses = _generate_responses(
        convs, client, generator,
        temperature=0.1,
        max_tokens=512,
        prefix="entity extraction: "
    )
    
    # 将结果为"无"的替换为空字符串
    ner_responses = [resp if resp.strip() != "无" else "" for resp in ner_responses]
    
    # 解析实体，返回格式为空格分隔的实体列表
    entities = [
        resp.strip().split('、') if resp.strip() else []
        for resp in ner_responses
    ]
    
    return entities


def llm_build_vocabs(
    queries: List[str], 
    client: OpenAI = None, 
    generator = None,
    wordbank_path: str = 'corpus/wordbank/Words-List.txt'
) -> List[List[str]]:
    """
    批量为多个句子构建词汇表。
    
    Args:
        queries: 待改写的原句列表
        client: OpenAI 客户端实例（与 generator 二选一）
        generator: BaseGenerator 实例（与 client 二选一）
        wordbank_path: 词库文件路径
        
    Returns:
        词汇列表的列表，每个元素对应一个查询的词汇列表
    """
    SYSTEM_PROMPT = """你的任务是判断给定候选词是否适合用于改写原句。
其中改写的目标是将原句改写为更直白、简单、不含修辞但语义相近的表达

评分标准：
- 合适：该词能自然用于改写，并能保持或增强原句的核心语义。
- 一般：该词可用于改写，但和原句语义的贴合度一般。
- 不合适：该词无法用于改写，或会导致语义偏离原句。

直接返回评分结果：合适 / 一般 / 不合适。
不要提供解释。
"""
    INSTRUCTION_TEMPLATE = """原句: "{sentence}" 
候选词: "{word}"
评分:"""
    
    word_list = load_lines_file(wordbank_path)
    word_list = [w.strip() for w in word_list if w.strip()]
    
    # 构建所有对话和元数据
    conversations = []
    metadata = []
    for query in queries:
        for word in word_list:
            instruction = INSTRUCTION_TEMPLATE.format(
                sentence=query,
                word=word
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction}
            ]
            conversations.append(messages)
            metadata.append({"query": query, "word": word})
    
    # 生成评分结果
    scores = _generate_responses(
        conversations, client, generator,
        temperature=0.2,
        max_tokens=128
    )
    
    # 构建结果列表
    query_to_vocab = {query: [] for query in queries}
    for i, score in enumerate(scores):
        query = metadata[i]["query"]
        word = metadata[i]["word"]
        if score.strip() in ["合适", "一般"]:
            query_to_vocab[query].append(word)
    
    # 按原始查询顺序返回结果
    results = [query_to_vocab[query] for query in queries]
    
    return results


def llm_judge_entity(words: List[str], client: OpenAI = None, generator = None) -> List[bool]:
    """
    判断词语是否为应该被抽取的实体（命名实体、术语或数字信息）。
    
    Args:
        words: 待判断的词语列表
        client: OpenAI 客户端实例（与 generator 二选一）
        generator: BaseGenerator 实例（与 client 二选一）
        
    Returns:
        布尔值列表，True 表示是实体，False 表示不是
    """
    SYSTEM = """任务: 判断给定词语是否为应该被抽取的实体（命名实体、术语或数字信息）。
【实体定义】
1. 命名实体（NER）
   - 包括：人名、地名、机构名、时间表达、国家、公司名称、事件名称等明确指代实体的词语。
2. 术语（Term）
   - 专业领域内有明确概念定义的词语，如技术术语、学科术语、政策术语等。
3. 数字信息（Number）
   - 包括：具体数量、百分比、时间点、日期、统计数值、序号等。
【输出要求】
- 如果词语是上述定义的实体，返回"是"。
- 如果不是，返回"否"。
- 不允许返回解释、分析或额外文本。"""
    
    INSTRUCTION = """请判断以下词语是否为应该被抽取的实体。
词语: {word}"""
    
    convs = [
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.format(word=word)},
        ] for word in words
    ]
    
    responses = _generate_responses(
        convs, client, generator,
        temperature=0.1,
        max_tokens=10,
        prefix="entity判断: "
    )
    
    # 解析响应，"是"为True，其他为False
    results = [resp.strip() == "是" for resp in responses]
    
    return results
