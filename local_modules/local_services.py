import json
from typing import Dict, List, Any, Tuple
import re
from . import local_prompts as prompts
from .local_core import generate_responses

# ======================== 预处理相关 ========================

def llm_write_plain_sentences(
    queries: List[str],
    generator = None,
    temperature: float = 0.2
) -> List[str]:
    """
    将输入句子改写为直白、不含修辞的句子
    """
    convs = [
        [
            {"role": "system", "content": prompts.PLAIN_SENTENCES_SYSTEM},
            {"role": "user", "content": prompts.PLAIN_SENTENCES_INSTRUCTION + query}
        ]
        for query in queries
    ]

    results = generate_responses(convs, generator, temperature)

    return [res.strip() if isinstance(res, str) else "" for res in results]

def llm_write_simple_sentences(
    queries: List[str],
    generator = None,
    temperature: float = 0.2
) -> List[List[str]]:
    """
    将输入句子改写为一系列连续通顺的简单句
    """
    convs = [
        [
            {"role": "system", "content": prompts.SIMPLE_SENTENCES_SYSTEM},
            {"role": "user", "content": prompts.SIMPLE_SENTENCES_INSTRUCTION + query}
        ]
        for query in queries
    ]
    
    results = generate_responses(convs, generator, temperature)

    return [
        [s.strip() for s in rewritten.split('。') if s.strip()]
        for rewritten in results
    ]

def llm_write_simple_sentences_cot(
    queries: List[str],
    generator = None,
    temperature: float = 0.2
) -> List[List[str]]:
    """
    将输入句子改写为一系列连续通顺的简单句（CoT版本）
    """
    convs = [
        [
            {"role": "system", "content": prompts.SIMPLE_SENTENCES_COT_SYSTEM},
            {"role": "user", "content": prompts.SIMPLE_SENTENCES_COT_INSTRUCTION.format(sentence=query)},
        ]
        for query in queries
    ]

    def extract_simple_sentences(response_text):
        
        if not response_text:
            return []
        pattern = r"(?:【|\[|###\s*|##\s*)?改写结果(?:】|\]|:|：|\n)?"
    
        parts = re.split(pattern, response_text)
        
        if len(parts) > 1:
            result_section = parts[-1].strip()
        else:
            result_section = response_text.strip()

        result_section = result_section.replace("</演示示例>", "").strip()

        sentences = [
            s.strip() 
            for s in re.split(r'[。.]', result_section) 
            if s.strip()
        ]
        
        return sentences
    
    results = generate_responses(convs, generator, temperature, max_tokens=16384)

    results = [extract_simple_sentences(res) for res in results]
    
    return results


def llm_segment_sentences(
    queries: List[str],
    generator = None,
    temperature: float = 0.2
) -> List[str]:
    """
    对句子进行分词
    """
    convs = [
        [
            {"role": "system", "content": prompts.SEGMENT_SYSTEM},
            {"role": "user", "content": prompts.SEGMENT_INSTRUCTION.format(sentence=query)},
        ]
        for query in queries
    ]
    results = generate_responses(convs, generator, temperature)
    return results

def llm_extract_entities(
    queries: List[str],
    generator = None,
    temperature: float = 0.1
) -> List[List[str]]:
    """
    抽取句子中的命名实体、术语和数字信息
    """
    convs = [
        [
            {"role": "system", "content": prompts.EXTRACT_ENTITIES_SYSTEM},
            {"role": "user", "content": prompts.EXTRACT_ENTITIES_INSTRUCTION + query}
        ]
        for query in queries
    ]
    
    results = generate_responses(convs, generator, temperature)

    return [
        [] if ent == "无" else [e.strip() for e in re.split("、", ent) if e.strip()]
        for ent in results
    ]

# ======================= 词表重写相关 ========================

def llm_rewrite_with_vocab(
    queries: List[str],
    vocabs_strs: List[str],
    generator = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    使用词表重写句子（CoT版本）
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
    
    results = generate_responses(convs, generator, temperature, max_tokens=16384)
    
    # 提取思维过程和重写结果
    final_results = []
    final_reasonings = []
    for response_content in results:
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
    
    # 构建完整对话历史
    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": final_results[i]
        })
        conversations.append(conversation)
    
    return {
        'content': final_results,
        'reasoning_content': final_reasonings,
        'conversations': conversations
    }

def llm_rewrite_with_feedback(
    previous_conversations: List[List[Dict[str, str]]],
    feedback_content_list: List[str],
    generator = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    根据反馈修正重写结果（CoT版本）
    """
    convs = [
        previous_conversations[i] + [
            {"role": "user", "content": prompts.REWRITE_FEEDBACK_COT_INSTRUCTION.format(
                feedback_content=feedback_content_list[i]
            )}
        ]
        for i in range(len(previous_conversations))
    ]

    results = generate_responses(convs, generator, temperature, max_tokens=16384)

    # 提取思维过程和重写结果
    final_results = []
    final_reasonings = []
    for response_content in results:
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

    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": final_results[i]
        })
        conversations.append(conversation)
    
    return {
        'content': final_results,
        'reasoning_content': final_reasonings,
        'conversations': conversations
    }

def llm_merge_sentences(
    queries: List[List[str]],
    ori_sentences: List[str],
    vocab_strs: List[str],
    generator = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    将简单句重写结果合并为一个句子（CoT版本）
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
    
    results = generate_responses(convs, generator, temperature, max_tokens=16384)

    # 提取思维过程和合并结果
    final_results = []
    final_reasonings = []
    for response_content in results:
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

    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": final_results[i]
        })
        conversations.append(conversation)
    
    return {
        'content': final_results,
        'reasoning_content': final_reasonings,
        'conversations': conversations
    }

def llm_merge_with_feedback(
    previous_conversations: List[List[Dict[str, str]]],
    wrong_words_list: List[str],
    generator = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    根据反馈修正合并结果（CoT版本）
    """
    convs = [
        previous_conversations[i] + [
            {"role": "user", "content": prompts.MERGE_FEEDBACK_COT_INSTRUCTION.format(
                wrong_words=wrong_words_list[i]
            )}
        ]
        for i in range(len(previous_conversations))
    ]

    results = generate_responses(convs, generator, temperature, max_tokens=16384)

    # 提取思维过程和合并结果
    final_results = []
    final_reasonings = []
    for response_content in results:
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

    conversations = []
    for i in range(len(convs)):
        conversation = convs[i].copy()
        conversation.append({
            "role": "assistant",
            "content": final_results[i]
        })
        conversations.append(conversation)
    
    return {
        'content': final_results,
        'reasoning_content': final_reasonings,
        'conversations': conversations
    }


# ======================= 评测相关 ========================

def llm_dependency_parsing(
    queries: List[List[str]],
    generator = None,
    temperature: float = 0.2,
    max_tokens: int = 16384,
    lang: str = "zh"
) -> List[List[str]]:
    """
    对句子进行依存句法分析
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
            {"role": "user", "content": PARSING_INSTRUCTION.format(sentence=" ".join(query))},
        ]
        for query in queries
    ]

    def _extract_dependency_relations(llm_output: str) -> List[str]:
        relations = []
        for line in llm_output.split('\n'):
            if line.strip():
                relations.append(' '.join(line.strip().split(' ')[:3]))
        return relations
    
    results = generate_responses(convs, generator, temperature, max_tokens)
    
    return {
        'content': [_extract_dependency_relations(res) for res in results]
    }

def llm_match_dependency_relations(
    gold_sentences: List[str],
    test_sentences: List[str],
    dependency_relations_list: List[List[str]],
    generator = None,
    temperature: float = 0.4,
    max_tokens: int = 8192,
    lang: str = "zh"
) -> Dict[str, Any]:
    """
    匹配测试句子中的依存关系
    
    Returns:
        Dict: 包含以下字段的字典：
            - content (List[List[float]]): 匹配结果列表
            - reasoning_content (List[List[str]]): 推理内容列表
    """
    system_prompt = prompts.MATCH_SYSTEM_JSON
    user_prompt_template = prompts.MATCH_INSTRUCTION_JSON
    
    # 分批处理的阈值
    BATCH_SIZE = 5
    
    all_convs = []
    batch_info = []  # 存储元组: (原始句子索引, 本批次包含的关系数量)
    
    for i in range(len(test_sentences)):
        relations = dependency_relations_list[i]
        
        # 处理空列表的情况
        if not relations:
            batch_info.append((i, 0)) # 占位，防止索引对不上
            continue

        # 计算分块
        num_batches = (len(relations) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(relations))
            batch_chunk = relations[start_idx:end_idx]
            
            # 格式： "1 | 核心词A SBV 核心词B"
            batch_relations_str = "\n".join(
                [f"{idx+1} | {rel}" for idx, rel in enumerate(batch_chunk)]
            )
            
            conv = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(
                    gold_sentence=gold_sentences[i],
                    test_sentence=test_sentences[i],
                    dependency_relations_with_ids=batch_relations_str
                )},
            ]
            all_convs.append(conv)
            
            # 记录信息以便后续还原：(原始句子index, 当前batch的大小)
            batch_info.append((i, len(batch_chunk)))
    
    def _parse_json_result(response_text: str, expected_count: int) -> Tuple[List[bool], List[str]]:
        """
        解析 LLM 返回的 JSON 列表，并根据 ID 映射回顺序列表
        """
        # 初始化结果
        results = [False] * expected_count
        reasons = ["解析失败或未返回"] * expected_count
        
        try:
            # 清洗 Markdown 标记 (```json ... ```)
            text = response_text.strip()
            # 提取 JSON 部分 (寻找最外层的 [])
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = text # 尝试直接解析

            data = json.loads(json_str)
            
            if not isinstance(data, list):
                return results, ["返回格式不是List"] * expected_count

            # 建立 ID 映射: {id: item}
            # 注意：Prompt 中 ID 是从 1 开始的
            res_map = {item.get('id'): item for item in data if isinstance(item, dict)}
            
            # 按顺序回填结果
            for idx in range(expected_count):
                curr_id = idx + 1
                if curr_id in res_map:
                    item = res_map[curr_id]
                    # 确保 match 是布尔值
                    is_match = item.get('match')
                    if isinstance(is_match, str):
                        is_match = str(is_match).lower() in ['true', 'yes', '是']
                    
                    results[idx] = bool(is_match)
                    reasons[idx] = str(item.get('reason', ''))
                else:
                    reasons[idx] = "模型未返回该ID的结果"
                    
        except json.JSONDecodeError:
            reasons = [f"JSON解码失败: {response_text[:50]}..."] * expected_count
        except Exception as e:
            reasons = [f"未知错误: {str(e)}"] * expected_count
            
        return results, reasons
    
    if not all_convs:
        return {'content': [[] for _ in range(len(test_sentences))], 'reasoning_content': [[] for _ in range(len(test_sentences))]}
    
    llm_responses = generate_responses(all_convs, generator, temperature=temperature, max_tokens=max_tokens)
    
    # 6. 结果重组
    final_contents = [[] for _ in range(len(test_sentences))]
    final_reasonings = [[] for _ in range(len(test_sentences))]
    
    current_response_idx = 0
    
    for sent_idx, batch_size in batch_info:
        if batch_size == 0:
            continue
            
        response_text = llm_responses[current_response_idx]
        current_response_idx += 1
        
        # 解析当前 Batch
        batch_matches, batch_reasons = _parse_json_result(response_text, batch_size)
        
        # 转换为 float 分数 (1.0 / 0.0)
        batch_scores = [1.0 if m else 0.0 for m in batch_matches]
        
        # 追加到对应句子的结果列表中
        final_contents[sent_idx].extend(batch_scores)
        final_reasonings[sent_idx].extend(batch_reasons)

    return {
        'content': final_contents,
        'reasoning_content': final_reasonings
    }
    

# ======================= 词语替换判断相关 ========================

def llm_judge_word_replacement(
    contexts: List[str],
    target_words: List[str],
    candidate_words: List[str],
    generator = None,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> List[bool]:
    """
    判断候选词是否可以在给定上下文中替换目标词
    
    Args:
        contexts: 上下文句子列表
        target_words: 待替换的目标词列表
        candidate_words: 候选替换词列表
        generator: BaseGenerator 实例
        temperature: 生成温度
        max_tokens: 最大token数
        
    Returns:
        判断结果列表
    """
    SYSTEM_PROMPT = """#角色：你是一个语言学专家，擅长词汇语义分析和同义词替换判断。"""
    
    INSTRUCTION = """#任务：判断候选词是否可以在给定上下文中替换目标词。

你将获得：
1. 一个完整句子
2. 句子中被标注的【目标词】
3. 一个【候选替换词】

你的任务是判断：  
在不改变句子原有核心语义、不引入额外信息、不造成歧义的前提下，  
【候选替换词】是否可以替换句子中的【目标词】，并使句子保持自然、通顺、符合常规用法。

#判断标准（必须同时满足）：
1. **语义等价性**：替换后句子的整体语义与原句基本一致（允许轻微风格差异，但不允许意义变化）。
2. **上下文适配性**：候选词在该具体语境中用法正确，不突兀、不违背常识。
3. **语法与搭配**：替换后句子在语法、固定搭配、语气上均自然。
4. **指代与范围一致**：不改变原词的指代对象、范围、强弱程度或抽象层级。

输出要求：
- 只输出以下两种结论之一：
  - 可以替换
  - 不可以替换
- 并在下一行用一句话简要说明原因（不超过30字，不要复述句子）。


#示例1：
句子：他对这项研究产生了浓厚的兴趣。
目标词：兴趣
候选词：爱好
输出：
不可以替换
原因：爱好更偏长期稳定，语义不完全一致

#示例2：
句子：该方法在实际应用中表现良好。
目标词：表现
候选词：效果
输出：
可以替换
原因：效果在此语境中与表现语义相近，替换后句子通顺自然。

#示例3：
句子：每个人都应该尊重他人的隐私。
目标词：每
候选词：各个
输出：
不可以替换
原因：各个强调个体差异，改变了原意。

#正式输入：
句子：{context}
目标词：{target_word}
候选词：{candidate_word}

#正式输出："""

    convs = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INSTRUCTION.format(
                context=contexts[i],
                target_word=target_words[i],
                candidate_word=candidate_words[i]
            )},
        ]
        for i in range(len(contexts))
    ]
    
    results = generate_responses(convs, generator, temperature, max_tokens)
    
    def _parse_judgment(llm_output: str) -> bool:
        """解析LLM输出的判断结果"""
        first_line = llm_output.strip().split('\n')[0]
        if first_line == "可以替换":
            return True
        elif first_line == "不可以替换":
            return False
        else:
            return False  # 默认不可以替换
    
    return [_parse_judgment(res) for res in results]


def llm_pos_tagging(
    queries: List[str],
    generator = None,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    对句子进行词性标注
    """
    convs = [
        [
            {"role": "system", "content": prompts.POS_TAGGING_SYSTEM},
            {"role": "user", "content": prompts.POS_TAGGING_INSTRUCTION.format(sentence=query)},
        ]
        for query in queries
    ]

    def parse_pos_result(result_text: str) -> List[tuple]:
        # 提取结果字符串
        # return format: List of (word, pos) tuples
        if "【标注结果】" in result_text:
            content = result_text.split("【标注结果】")[-1].strip()
        else:
            content = result_text.strip()
        
        # 解析 "词/性" 格式
        pairs = []
        tokens = content.split()
        for token in tokens:
            if '/' in token:
                # rsplit 确保只从最后一个斜杠分割（防止词本身包含斜杠）
                word, tag = token.rsplit('/', 1)
                pairs.append((word, tag))
                
        return pairs
    
    results = generate_responses(convs, generator, temperature, max_tokens=16384)

    final_results = []
    final_reasonings = []
    for res in results:
        try:
            reasoning = res.split("【思维过程】")[-1].split("【标注结果】")[0].strip()
            final_reasonings.append(reasoning)
        except:
            final_reasonings.append("无思维过程")
        parsed_result = parse_pos_result(res)
        final_results.append(parsed_result)


    return {
        'content': final_results,
        'reasoning_content': final_reasonings
    }


# def _parse_match_result(response_text, input_relations):
#         # 解析</演示示例>之后的内容
#         if "</演示示例>" in response_text:
#             content = response_text.split("</演示示例>")[-1].strip()
#         else:
#             content = response_text.strip()
#         expected_count = len(input_relations)
#         results = [False] * expected_count
#         reasons = ["解析失败或未找到对应行"] * expected_count

#         lines = [line.strip() for line in content.strip().split('\n') if line.strip()]

#         current_idx = 0
    
#         for line in lines:
#             if current_idx >= expected_count:
#                 break
#             parts = line.split('|')
#             if len(parts) >= 2:
#                 result_str = parts[1].strip()
#                 is_match = "是" in result_str
                
#                 reason_str = parts[2].strip() if len(parts) > 2 else ""
                
#                 results[current_idx] = is_match
#                 reasons[current_idx] = reason_str
                
#                 current_idx += 1
        
#         # if current_idx != expected_count:
#         #     print(f"Warning: Expected {expected_count} lines, got {current_idx}")
        
#         return results, reasons