import re
from typing import List, Dict, Any

from src.utils import load_json_file, save_json_file, save_lines_file
from .api_core import zai_generate
from . import zai_prompts as prompts

# ======================== 预处理相关 ========================

def zai_write_plain_sentences( 
    queries: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    thinking: bool = False,
) -> Dict[str, Any]:
    """
    将输入句子改写为直白、不含修辞的句子（去除比喻、夸张、修辞手法等）。
    要求：
    - 保持原句的核心意思（主体、动作、客体等），尽量保留事实信息和时态；
    - 去除比喻、拟人、夸张、成语/典故、修辞性修饰等，使表达直接明了；
    返回：
        DICT: 包含 'content', 'reasoning_content' 的字典
    """
    convs = [
        [
            {"role": "system", "content": prompts.PLAIN_SENTENCES_SYSTEM},
            {"role": "user", "content": prompts.PLAIN_SENTENCES_INSTRUCTION + query}
        ]
        for query in queries
    ]

    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        desc="Simplifying to plain sentences"
    )

    results['content'] = [res.strip() if isinstance(res, str) else "" for res in results['content']]
    return results

def zai_write_simple_sentences(
    queries: List[str], 
    model: str = "glm-4.5-air",
    temperature: float = 0.2,
    max_tokens: int = 16384,
    thinking: bool = False
) -> List[List[str]]:
    
    """
    将输入句子改写为一系列连续通顺的简单句。
    
    Args:
        queries: 待改写的句子列表
        api_key: ZhipuAI API密钥，如果为None则从环境变量读取
        
    Returns:
        简单句列表的列表，每个元素对应一个查询的简单句列表
    """
    convs = [
        [
            {"role": "system", "content": prompts.SIMPLE_SENTENCES_SYSTEM},
            {"role": "user", "content": prompts.SIMPLE_SENTENCES_INSTRUCTION + query}
        ]
        for query in queries
    ]
    
    result = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        desc="Writing simple sentences"
    )

    result["content"] = [
        [s.strip() for s in rewritten.split('。') if s.strip()]
        for rewritten in result['content']
    ]
    
    return result

def zai_segment_sentences(
    queries: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    thinking: bool = False
) -> Dict[str, Any]:
    """
    使用ZAI对句子进行分词
    Args:
        queries: 待分词的句子列表
        temperature: 生成温度
        thinking: 是否启用思考模式
    Returns:
        包含 'content', 'reasoning_content' 的字典
    """
    convs = [
        [
            {"role": "system", "content": prompts.SEGMENT_SYSTEM},
            {"role": "user", "content": prompts.SEGMENT_INSTRUCTION.format(sentence=queries[i])},
        ]
        for i in range(len(queries))
    ]
    return zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        desc="Segmenting sentences"
    )

def zai_extract_entities(
    queries: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.1,
    max_tokens: int = 2048,  
    thinking: bool = False                   
) -> Dict[str, Any]:
    """使用 ZhipuAI API 抽取句子中的命名实体、术语和数字信息。"""
    convs = [
        [
            {"role": "system", "content": prompts.EXTRACT_ENTITIES_SYSTEM},
            {"role": "user", "content": prompts.EXTRACT_ENTITIES_INSTRUCTION + query}
        ]
        for query in queries
    ]
    
    result = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        desc="Extracting entities"
    )

    result['content'] = [
        [] if ent == "无" else [e.strip() for e in re.split("、", ent) if e.strip()]
        for ent in result['content']
    ]
    return result

def zai_extract_entities_cot(
    queries: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """使用 ZhipuAI API 抽取句子中的命名实体、术语和数字信息，带有思考链。"""
    convs = [
        [
            {"role": "system", "content": prompts.EXTRACT_ENTITIES_SYSTEM},
            {"role": "user", "content": prompts.EXTRACT_ENTITIES_COT_INSTRUCTION.format(sentence=query)},
        ]
        for query in queries
    ]
    
    result = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        desc="Extracting entities with CoT"
    )

    final_contents = []
    final_reasonings = []
    for response_content in result['content']:
        try:
            reasoning = response_content.split("【思维过程】")[-1].split("【抽取结果】")[0].strip()
            final_reasonings.append(reasoning)
        except:
            final_reasonings.append("无思维过程")
        
        try:
            entities_str = response_content.split("【抽取结果】")[-1].strip()
            if entities_str == "无":
                final_contents.append([])
            else:
                entities = [e.strip() for e in re.split("、", entities_str) if e.strip()]
                final_contents.append(entities)
        except:
            final_contents.append([])
    
    result['content'] = final_contents
    result['reasoning_content'] = final_reasonings
    return result

# ======================= 词表重写相关 ========================

def zai_rewrite_with_vocab(
    queries: List[str],
    vocabs_strs: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.1,
    thinking: bool = True,
) -> Dict[str, Any]:
    """
    使用词表重写句子
    
    Args:
        queries: 待重写的句子列表
        vocabs_strs: 词表字符串列表（用顿号分隔）
        temperature: 生成温度
        thinking: 是否启用思考模式
        
    Returns:
        包含 'content', 'reasoning_content', 'conversations' 的字典
    """
    convs = [
        [
            {"role": "system", "content": prompts.REWRITE_SYSTEM},
            {"role": "user", "content": prompts.REWRITE_INSTRUCTION.format(
                vocabs=vocabs_strs[i],
                sentence=queries[i]
            )},
        ]
        for i in range(len(queries))
    ]
    
    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        thinking=thinking,
        desc="Rewriting with vocab"
    )
    
    # 新增：为每个对话添加assistant的回复，构建完整对话历史
    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": results['content'][i]
        })
        conversations.append(conversation)
    
    results['conversations'] = conversations
    return results

def zai_rewrite_with_feedback(
    previous_conversations: List[List[Dict[str, str]]],
    wrong_words_list: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.1,
    thinking: bool = True
) -> Dict[str, Any]:
    """
    根据反馈修正重写结果（通过拼接对话历史）
    
    Args:
        conversations_list: 第一轮的完整对话历史列表
        wrong_words_list: 错误词汇列表（用顿号分隔）
        temperature: 生成温度
        thinking: 是否启用思考模式
        
    Returns:
        包含 'content', 'reasoning_content' 的字典
    """
    # 基于对话历史，追加反馈消息
    convs = [
        previous_conversations[i] + [
            {"role": "user", "content": prompts.REWRITE_FEEDBACK_INSTRUCTION.format(
                wrong_words=wrong_words_list[i]
            )}
        ]
        for i in range(len(previous_conversations))
    ]

    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        thinking=thinking,
        desc="Rewriting with feedback"
    )

    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": results['content'][i]
        })
        conversations.append(conversation)
    
    results['conversations'] = conversations
    return results

def zai_merge_sentences(
    queries: List[List[str]],
    ori_sentences: List[str],
    vocab_strs: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.1,
    thinking: bool = True
) -> Dict[str, Any]:
    """
    将简单句重写结果合并为一个句子
    
    Args:
        queries: 待合并的句子列表的列表
        temperature: 生成温度
        thinking: 是否启用思考模式
        
    Returns:
        包含 'content', 'reasoning_content' 的字典
    """
    convs = [
        [
            {"role": "system", "content": prompts.MERGE_SYSTEM},
            {"role": "user", "content": prompts.MERGE_INSTRUCTION.format(
                vocab=vocab_strs[i],
                ori_sentence=ori_sentences[i],
                simple_sentences='\n'.join(queries[i])
            )},
        ]
        for i in range(len(queries))
    ]
    
    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        thinking=thinking,
        desc="Merging sentences"
    )
    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": results['content'][i]
        })
        conversations.append(conversation)
    results['conversations'] = conversations
    return results


def zai_rewrite_with_vocab_cot(
    queries: List[str],
    vocabs_strs: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    使用词表重写句子，带有思考链
    
    Args:
        queries: 待重写的句子列表
        vocabs_strs: 词表字符串列表（用顿号分隔）
        temperature: 生成温度
    """
    convs = [
        [
            {"role": "system", "content": prompts.REWRITE_SYSTEM},
            {"role": "user", "content": prompts.REWRITE_COT_INSTRUCTION.format(
                vocabs=vocabs_strs[i],
                sentence=queries[i]
            )},
        ]
        for i in range(len(queries))
    ]

    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=16384,
        desc="Rewriting with vocab and CoT"
    )

    result_contents = results['content']
    final_results = []
    final_reasonings = []
    for response_content in result_contents:
        try:
            reasoning = response_content.split("【思维过程】")[-1].split("【重写结果】")[0].strip()
            final_reasonings.append(reasoning)
        except:
            final_reasonings.append("无思维过程")
        
        if "无法重写" in response_content and "【重写结果】\n无法重写" in response_content:
            final_results.append("无法重写")
        else:
            try:
                final_result = response_content.split("【重写结果】")[-1].strip()
            except:
                final_result = '无法重写'
            final_results.append(final_result)
    
    results['content'] = final_results
    results['reasoning_content'] = final_reasonings

    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": results['content'][i]
        })
        conversations.append(conversation)
    results['conversations'] = conversations

    return results

def zai_rewrite_with_feedback_cot(
    previous_conversations: List[List[Dict[str, str]]],
    wrong_words_list: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    根据反馈修正重写结果（通过拼接对话历史），带有思考链
    
    Args:
        conversations_list: 第一轮的完整对话历史列表
        wrong_words_list: 错误词汇列表（用顿号分隔）
        temperature: 生成温度
    Returns:
        包含 'content', 'reasoning_content' 的字典
    """
    convs = [
        previous_conversations[i] + [
            {"role": "user", "content": prompts.REWRITE_FEEDBACK_COT_INSTRUCTION.format(
                wrong_words=wrong_words_list[i]
            )}
        ]
        for i in range(len(previous_conversations))
    ]
    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=16384,   
        desc="Rewriting with feedback and CoT"
    )
    result_contents = results['content']
    final_results = []
    final_reasonings = []
    for response_content in result_contents:
        try:
            reasoning = response_content.split("【思维过程】")[-1].strip()
            final_reasonings.append(reasoning)
        except:
            final_reasonings.append("无思维过程")
        
        if "无法重写" in response_content and "【重写结果】\n无法重写" in response_content:
            final_results.append("无法重写")
        else:
            try:
                final_result = response_content.split("【重写结果】")[-1].strip()
            except:
                final_result = '无法重写'
            final_results.append(final_result)
    results['content'] = final_results
    results['reasoning_content'] = final_reasonings
    
    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": results['content'][i]
        })
        conversations.append(conversation)
    results['conversations'] = conversations
    
    return results

def zai_rewrite_merge_sentences_cot(
    queries: List[List[str]],
    ori_sentences: List[str],
    vocab_strs: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    将简单句重写结果合并为一个句子，带有思考链
    
    Args:
        queries: 待合并的句子列表的列表
        ori_sentences: 原始句子列表
        vocab_strs: 词表字符串列表（用顿号分隔）
        temperature: 生成温度
    Returns:
        包含 'content', 'reasoning_content' 的字典
    """
    convs = [
        [
            {"role": "system", "content": prompts.MERGE_COT_SYSTEM},
            {"role": "user", "content": prompts.MERGE_COT_INSTRUCTION.format(
                vocab=vocab_strs[i],
                ori_sentence=ori_sentences[i],
                simple_sentences='\n'.join(queries[i])
            )},
        ]
        for i in range(len(queries))
    ]
    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=16384,
        desc="Merging sentences with CoT"
    )
    result_contents = results['content']
    final_results = []
    final_reasonings = []
    for response_content in result_contents:
        try:
            reasoning = response_content.split("【思维过程】")[-1].strip()
            final_reasonings.append(reasoning)
        except:
            final_reasonings.append("无思维过程")
        
        try:
            final_result = response_content.split("【合并结果】")[-1].strip()
        except:
            final_result = '无法合并'
        final_results.append(final_result)
    results['content'] = final_results
    results['reasoning_content'] = final_reasonings
    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": results['content'][i]
        })
        conversations.append(conversation)
    results['conversations'] = conversations
    return results

def zai_rewrite_merge_with_feedback_cot(
    previous_conversations: List[List[Dict[str, str]]],
    wrong_words_list: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    根据反馈修正合并结果（通过拼接对话历史），带有思考链
    
    Args:
        conversations_list: 第一轮的完整对话历史列表
        wrong_words_list: 错误词汇列表（用顿号分隔）
        temperature: 生成温度
    Returns:
        包含 'content', 'reasoning_content' 的字典
    """
    convs = [
        previous_conversations[i] + [
            {"role": "user", "content": prompts.MERGE_FEEDBACK_COT_INSTRUCTION.format(
                wrong_words=wrong_words_list[i]
            )}
        ]
        for i in range(len(previous_conversations))
    ]
    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=16384,
        desc="Merging with feedback and CoT"
    )
    result_contents = results['content']
    final_results = []
    final_reasonings = []
    for response_content in result_contents:
        try:
            reasoning = response_content.split("【思维过程】")[-1].strip()
            final_reasonings.append(reasoning)
        except:
            final_reasonings.append("无思维过程")
        
        try:
            final_result = response_content.split("【合并结果】")[-1].strip()
        except:
            final_result = '无法合并'
        final_results.append(final_result)
    results['content'] = final_results
    results['reasoning_content'] = final_reasonings
    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": results['content'][i]
        })
        conversations.append(conversation)
    results['conversations'] = conversations
    return results

# ======================= 评测相关 ========================

def zai_dependency_parsing(
    queries: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2,
    max_tokens: int = 32768,
    thinking: bool = False,
    lang: str = "zh"
) -> Dict[str, Any]:
    """
    使用ZAI对句子进行依存句法分析
    Args:
        queries: 待分析的句子列表
        temperature: 生成温度
        thinking: 是否启用思考模式
    Returns:
        Dict: 一个包含以下字段的字典：
            - content (List[str]): 依存关系输出列表，每个元素为该句子的依存结构。
            - reasoning_content (List[str]): 推理内容的列表（若启用思考模式）。
    """
    if lang == "en":
        PARSING_SYSTEM = prompts.PARSING_SYSTEM_ENG
        PARSING_INSTRUCTION = prompts.PARSING_INSTRUCTION_ENG
    else:
        PARSING_SYSTEM = prompts.PARSING_SYSTEM
        PARSING_INSTRUCTION = prompts.PARSING_INSTRUCTION
        
    convs = [
        [
            {"role": "system", "content": PARSING_SYSTEM},
            {"role": "user", "content": PARSING_INSTRUCTION.format(sentence=queries[i])},
        ]
        for i in range(len(queries))
    ]

    def _extract_dependency_relations(llm_output: str) -> List[str]:
        relations = []
        for line in llm_output.split('\n'):
            if line.strip():
                relations.append(' '.join(line.strip().split(' ')[:3])) # 只取前三列：词、依存关系、父节点
        return relations
    
    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        desc="Dependency parsing"
    )
    
    results['content'] = [_extract_dependency_relations(res) for res in results['content']]
    return results

def zai_match_dependency_relations(
    gold_sentences: List[str],
    test_sentences: List[str],
    dependency_relations_list: List[str],
    model: str = "glm-4.5-air",
    temperature: float = 0.2,
    max_tokens: int = 32768,
    thinking: bool = False,
    lang: str = "zh"
) -> Dict[str, Any]:
    """
    使用ZAI匹配测试句子中的依存关系
    Args:
        gold_sentences: 参考句子列表
        test_sentences: 待测试的句子列表
        dependency_relations_list: 依存关系列表
    Returns:
        content: 包含匹配结果的列表，每个元素是一个浮点数列表（1表示匹配，0表示不匹配） 
        reasoning_content: 包含推理内容的列表
    """
    # 参考句子、测试句子、参考句子中的依存关系
    if lang == "en":
        MATCH_SYSTEM = prompts.MATCH_SYSTEM_ENG
        MATCH_INSTRUCTION = prompts.MATCH_INSTRUCTION_ENG
    else:
        MATCH_SYSTEM = prompts.MATCH_SYSTEM
        MATCH_INSTRUCTION = prompts.MATCH_INSTRUCTION
        
    convs = [
        [
            {"role": "system", "content": MATCH_SYSTEM},
            {"role": "user", "content": MATCH_INSTRUCTION.format(
                gold_sentence=gold_sentences[i],
                test_sentence=test_sentences[i],
                dependency_relations_str="\n".join(dependency_relations_list[i])
            )},
        ]
        for i in range(len(test_sentences))
    ]
     
    def _parse_match_result(response_text, input_relations):
        expected_count = len(input_relations)
        results = [False] * expected_count
        reasons = ["解析失败或未找到对应行"] * expected_count

        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

        current_idx = 0
    
        for line in lines:
            if current_idx >= expected_count:
                break # 防止模型多输出了
            parts = line.split('|')
            if len(parts) >= 2:
                # 第一列：依存关系 (用于人工核对，代码里主要依靠顺序)
                relation_echo = parts[0].strip()
                
                # 第二列：结论
                result_str = parts[1].strip()
                is_match = "是" in result_str
                
                # 第三列：原因
                reason_str = parts[2].strip() if len(parts) > 2 else ""
                
                # 填入结果
                results[current_idx] = is_match
                reasons[current_idx] = reason_str
                
                current_idx += 1
        
        if current_idx != expected_count:
            print(f"Warning: Expected {expected_count} lines, got {current_idx}")
        
        return results, reasons

    results = zai_generate(
        messages_list=convs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        desc="Matching dependency relations"
    )

    final_contents = []
    final_reasonings = []
    for i in range(len(convs)):
        match_results, reasons = _parse_match_result(results['content'][i], dependency_relations_list[i])
        final_contents.append([1.0 if res else 0.0 for res in match_results])
        final_reasonings.append(reasons)
    
    results['content'] = final_contents
    results['reasoning_content'] = final_reasonings
    
    return results


def eval_sts_data():

    # 计算皮尔逊系数
    def caculate_pearson_correlation(gold_scores: List[float], pred_scores: List[float]) -> float:
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(gold_scores, pred_scores)
        return correlation
    
    # score_data = load_json_file("debug_sts_evaluation_results.json")
    # pred_scores = [item['pred_score'] for item in score_data]
    # gold_scores = [item['gold_score'] for item in score_data]
    # pearson_correlation = caculate_pearson_correlation(gold_scores, pred_scores)
    # print(f"Sample Size: {len(score_data)}")
    # print(f"Pearson Correlation: {pearson_correlation}")
    # return 
    from src.utils import load_sts_file
    input_file = "corpus/sts2016-english-with-gs-v1.0/processed/STS2016.processed.answer-answer.txt"
    output_file = "debug_sts_evaluation_results.json"
    
    # 读取STS数据
    sts_data = load_sts_file(input_file)[:50]
    sentences1 = [item[0] for item in sts_data]
    sentences2 = [item[1] for item in sts_data]
    gold_scores = [float(item[2]) for item in sts_data]

    # 使用LLM计算相似度分数（基于依存句法分析）
    from zai_modules.api_core import zai_compute_sentence_similarity
    model = 'glm-4.5'
    similarity_results = zai_compute_sentence_similarity(
        sentences1, 
        sentences2, 
        model=model, 
        thinking=True
    )
    pred_scores = similarity_results['content']
    
    # 将0-1分数转换为0-5分数以匹配STS标准
    pred_scores_scaled = [score * 5.0 for score in pred_scores]
    # pred_scores_scaled = pred_scores

    # 保存结果
    save_content = []
    for i in range(len(sts_data)):
        item = {
            "sentence1": sentences1[i],
            "sentence2": sentences2[i],
            "gold_score": gold_scores[i],
            "pred_score": pred_scores_scaled[i],
            "pred_score_raw": pred_scores[i],
            "segments": similarity_results['segments'][i],
            "dependency_relations": similarity_results['dependency_relations'][i],
            "matched_relations": similarity_results['matched_relations'][i]
        }
        save_content.append(item)
    save_json_file(save_content, output_file)
    
    # 单独把得分保存为一个文件
    save_scores = [pred_scores_scaled[i] for i in range(len(sts_data))]
    save_lines_file([str(score) for score in save_scores], "debug_sts_pred_scores.txt")

    # 计算皮尔逊相关系数
    pearson_correlation = caculate_pearson_correlation(gold_scores, pred_scores_scaled)
    print(f"Pearson Correlation: {pearson_correlation}")

def eval_wmt_data():
    from zai_modules.api_core import zai_dependency_parsing, zai_segment_sentences
    from zai_modules.api_core import zai_match_dependency_relations

    output_score_file = "auto_scores.en-zh.txt"
    data = load_json_file("corpus/wmt19/wmt_rr_extracted_translations.json")
    translations_data = data[0:5]

    with open(output_score_file, "a", encoding="utf-8") as fout:
        
        save_content = []

        for item in translations_data:
            sid = item["sid"]
            source = item['source']
            reference = item['reference']
            system_outputs = item['translations']

            # 1. 计算ref的依存句法关系
            segmented = zai_segment_sentences(
                queries=[source]
            )['content']
            dependency = zai_dependency_parsing(
                queries=segmented,
                model='glm-4.5',
                thinking=True,
                lang='en'
            )['content'][0]

            # 2. 计算每个translation与ref的相似度
            translations = list(system_outputs.values())
            system_names = list(system_outputs.keys())

            match_results = zai_match_dependency_relations(
                gold_sentences=[source] * len(system_outputs),
                test_sentences=translations,
                dependency_relations_list=[dependency] * len(system_outputs),
                model='glm-4.5',
                thinking=True
            )['content']

            similarity_scores = []
            for i in range(len(translations)):
                total_rels = len(dependency)
                if total_rels == 0:
                    similarity_scores.append(0.0)
                    continue
                matched_rels = sum(match_results[i])
                similarity = matched_rels / total_rels
                similarity_scores.append(similarity)
            
            # 3. 记录结果
            lp = 'en-zh'
            for sys_name, score in zip(system_names, similarity_scores):
                fout.write(f"{lp}\t{sid}\t{sys_name}\t{score}\n")

            save_content.append({
                "source": source,
                "reference": reference,
                "dependency": dependency,
                "translations": system_outputs,
                "scores": {system_names[i]: similarity_scores[i] for i in range(len(system_names))}
            })
    save_json_file(save_content, "debug_wmt_dependency_matching_results.json")
