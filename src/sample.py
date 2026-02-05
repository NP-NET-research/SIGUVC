import gc
import json
import os
import re
from typing import List, Dict, Set, Tuple

from openai import OpenAI
import torch
from tqdm import tqdm
from src.inference.generator import BaseGenerator
from src.inference.prompt import PromptRegistry
from src.data.wordbank import WordBank
from src.utils import load_json_file, load_lines_file, safe_parse_json, save_json_file, setup_save_directory

def align_tokens_and_entities(seg_tokens_list: List[List[str]], ner_texts_list: List[List[str]]) -> List[List[str]]:
    """对齐分词和NER结果，返回对齐后的分词列表
    """
    aligned_lists = []

    for seg_tokens, ner_texts in zip(seg_tokens_list, ner_texts_list):
        if not ner_texts:
            aligned_lists.append(seg_tokens)
            continue

        aligned_words = []
        i = 0
        while i < len(seg_tokens):
            merged = None
            for ent_text in ner_texts:
                candidate = ""
                j = i
                while j < len(seg_tokens) and len(candidate) < len(ent_text):
                    candidate += seg_tokens[j]
                    j += 1
                if candidate == ent_text:
                    merged = ent_text
                    break
            if merged:
                aligned_words.append(merged)
                # 跳过已合并的分词
                consumed = 0
                char_count = 0
                for tok in seg_tokens[i:]:
                    char_count += len(tok)
                    consumed += 1
                    if char_count >= len(merged):
                        break
                i += consumed
            else:
                aligned_words.append(seg_tokens[i])
                i += 1

        aligned_lists.append(aligned_words)

    return aligned_lists

def build_strict_rewrite_input(original_text: str, miniwb: List[str], ner_entities: list[str]) -> str:
    """
    构建单样本 input 部分
    original_text: 原始文本
    candidates: 每句检索到的候选词列表（已按句聚合）
    """
    lines = [
        '原句：',
        original_text,
        "",
    ]
    mini_wb = miniwb + ner_entities
    lines.append(f"检索词库（共{len(mini_wb)}个词）：")
    lines.append("，".join(mini_wb) if mini_wb else "无")
    lines.append("")

    return "\n".join(lines)

def build_segmentation_input(original_text: str, ner_entities: list[str]) -> str:
    """构建分词输入"""
    lines = [original_text]
    
    if ner_entities:
        lines.append("")
        lines.append("命名实体保护名单：")
        lines.append(", ".join(ner_entities))
    
    return "\n".join(lines)


def retrieve_generate_sample(
        texts: List[str], 
        
        vocab_file: str,
        vectorizer_model_path: str,
        retrieval_k: int = 100,

        model_name_or_path: str = None,
        temperature: float = 0.95,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
        num_sequences : int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        use_vllm: bool = True,
        
    ) -> dict:
    
    generator = BaseGenerator(
        config={
            "model_name_or_path": model_name_or_path,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "num_sequences": num_sequences,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "use_vllm": use_vllm,
        }
    )
    wordbank = WordBank(
        vocab_file=vocab_file,
        vectorizer_model_path=vectorizer_model_path,
    )
    
    data = [{"data_frame": {"text": text}} for text in texts]
    all_texts = texts
    
    # ====================== NER ======================
    ner_prompt_template = PromptRegistry.get("ner")
    ner_results = generator.generate(
        queries=all_texts,
        prompt_template=ner_prompt_template,
        do_sample=False,
        num_sequences=1,
    )
    ner_results = [res[0] for res in ner_results]
    ner_texts_list = []
    for ner_text in ner_results:
        _, entities = safe_parse_json(ner_text)
        if not isinstance(entities, list):
            entities = []
        ner_texts = [ent.get("text") for ent in entities if "text" in ent]
        ner_texts_list.append(ner_texts)

    # ====================== 分词 ======================
    seg_prompt_template = PromptRegistry.get("seg")
    seg_inputs = [
        build_segmentation_input(text, ner_texts)
        for text, ner_texts in zip(all_texts, ner_texts_list)
    ]
    seg_tokens_list = generator.generate(
        queries=seg_inputs,
        prompt_template=seg_prompt_template,
        do_sample=False,
        num_sequences=1,
    )
    seg_tokens_list = [tokens[0].split() for tokens in seg_tokens_list]

    # ====================== 对齐 & 过滤 ======================
    tokens_lists = align_tokens_and_entities(seg_tokens_list, ner_texts_list)
    filtered_tokens_lists  = [
        [tok for tok in tokens if tok not in ner_texts]
        for tokens, ner_texts in zip(tokens_lists, ner_texts_list)
    ]

    # ====================== 检索 ======================
    mini_wordbanks = wordbank.retrieve(
        batch_query_words=filtered_tokens_lists, 
        return_k=retrieval_k    # 固定返回数量
    )

    del wordbank
    torch.cuda.empty_cache()
    gc.collect()

    # ====================== 改写 ======================
    rewrite_prompt_template = PromptRegistry.get("strict_rewrite_with_candidates")
    rewrite_inputs = [
        build_strict_rewrite_input(text, miniwb, ner_texts)
        for text, miniwb, ner_texts in zip(all_texts, mini_wordbanks, ner_texts_list)
    ]
    rewrite_results = generator.generate(
        queries=rewrite_inputs,
        prompt_template=rewrite_prompt_template,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        num_sequences=num_sequences,    # 控制生成个数
    )

    # ===================== 改写结果分词 ======================
    rewrite_flat_inputs = []
    rewrite_indices = []  
    for i, seqs in enumerate(rewrite_results):
        for j, rewritten_text in enumerate(seqs):
            rewrite_flat_inputs.append(build_segmentation_input(rewritten_text, ner_texts_list[i]))
            rewrite_indices.append((i, j))
    # 运行分词
    rewrite_seg_results_flat = generator.generate(
        queries=rewrite_flat_inputs,
        prompt_template=seg_prompt_template,
        do_sample=False,
        num_sequences=1,
    )
    rewrite_seg_tokens_flat = [res[0].split() for res in rewrite_seg_results_flat]
    # 重新组合回原结构
    rewrite_seg_tokens_grouped = [[] for _ in rewrite_results]
    for (i, j), toks in zip(rewrite_indices, rewrite_seg_tokens_flat):
        rewrite_seg_tokens_grouped[i].append(' '.join(toks))

    # ====================== 保存结果 ======================
    for item, seg_tokens, ner_texts, words, inputs, rewrites, rewrite_segs in zip(
        data, tokens_lists, ner_texts_list, mini_wordbanks, rewrite_inputs, rewrite_results, rewrite_seg_tokens_grouped
    ):
        item['system'] = rewrite_prompt_template.system
        item['instruction'] = rewrite_prompt_template.instruction
        item['input'] = inputs
        item['data_frame'].update({
            "seg_tokens": ' '.join(seg_tokens),
            "ner_texts": ' '.join(ner_texts),
            "retrieve_words": ' '.join(words),
            "rewrites": rewrites,
            "rewrites_seg": rewrite_segs,
        })
    
    # 清理
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    
    return data
        

def retrieve_for_texts(
        data: List[dict],
        
        vocab_file: str,
        vectorizer_model_path: str,
        retrieval_k: int = 1000,

        extra_words: set = None,
    ) -> List[Dict]:
    """
    仅进行检索，返回检索结果
    """
    from tqdm import tqdm
    import pkuseg
    
    wordbank = WordBank(
        vocab_file=vocab_file,
        vectorizer_model_path=vectorizer_model_path,
    )

    tmpfile_path = None
    if extra_words:
        # 临时文件保存额外词汇
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmpfile:
            for word in extra_words:
                tmpfile.write(word + '\n')
            tmpfile_path = tmpfile.name
    
    seg = pkuseg.pkuseg()
    texts = [item.get("data_frame", {}).get("text", "") for item in data]
    seg_tokens_lists = [seg.cut(text) for text in texts]

    filtered_tokens_lists = []
    for item, seg_tokens in zip(data, seg_tokens_lists):
        df = item.get("data_frame", {})
        entities = df.get("entity", [])
        df['seg_tokens'] = ' '.join(seg_tokens)
        # 过滤实体
        filtered_tokens = [tok for tok in seg_tokens if tok not in entities]
        filtered_tokens_lists.append(filtered_tokens)
    
    # filtered_tokens_lists = []
    # for item in tqdm(data):
    #     df = item.get("data_frame", {})
    #     text = df.get("text", "")
    #     entities = df.get("entity", [])
    #     seg_tokens = segment_with_entities(text, tmpfile_path)
    #     df['seg_tokens'] = ' '.join(seg_tokens)
    #     # 过滤实体
    #     filtered_tokens = [tok for tok in seg_tokens if tok not in entities]
    #     filtered_tokens_lists.append(filtered_tokens)

    mini_wordbanks = wordbank.retrieve(
        batch_query_words=filtered_tokens_lists, 
        return_k=retrieval_k    # 固定返回数量
    )
    del wordbank
    torch.cuda.empty_cache()
    gc.collect()

    for item, words in zip(data, mini_wordbanks):
        df = item.get("data_frame", {})
        df["retrieve_words"] = ' '.join(words)
        df['entity'] = ' '.join(item.get("data_frame", {}).get("entity", []))

    return data


def rewrite_with_wordbank(
        data: List[dict],

        model_name_or_path: str = None,
        temperature: float = 0.95,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
        num_sequences : int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        use_vllm: bool = True,
    ) -> List[Dict]:
    """
    使用检索词库进行改写
    """
    generator = BaseGenerator(
        config={
            "model_name_or_path": model_name_or_path,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "num_sequences": num_sequences,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "use_vllm": use_vllm,
        }
    )
    def build_input(text, wordbank_str, entity_str):
        return f'原始句子：\n{text}\n\n子词表（共{len(wordbank_str.split())}个词）：\n{wordbank_str}\n\n实体列表：\n{entity_str}\n'

    rewrite_prompt_template = PromptRegistry.get("rewrite_with_wordbank")
    rewrite_inputs = [
        build_input(
            item.get("data_frame", {}).get("text", ""),
            item.get("data_frame", {}).get("retrieve_words", ""),
            item.get("data_frame", {}).get("entity", ""),
        )
        for item in data
    ]
    rewrite_results = generator.generate(
        queries=rewrite_inputs,
        prompt_template=rewrite_prompt_template,
        do_sample=do_sample,
        num_sequences=num_sequences,    # 控制生成个数
    )

    for item, inputs, rewrites in zip(data, rewrite_inputs, rewrite_results):
        item['system'] = rewrite_prompt_template.system
        item['instruction'] = rewrite_prompt_template.instruction
        item['input'] = inputs
        item['data_frame']['rewrites'] = rewrites

    del generator
    torch.cuda.empty_cache()
    gc.collect()

    return data


def filter_samples_for_sft():

    from src.utils import load_json_file, save_json_file, setup_save_directory
    model_name_or_path = "/home/ljl/Data/LLM/GLM-4-9B-0414"
    fluency_model_path = '/home/ljl/Data/LLM/GLM-4-32B-0414'

    work_dir = setup_save_directory("saves/debug")

    train_data = load_json_file('corpus/train.json')
    train_data = train_data[:20000]

    train_data_sample = rewrite_with_wordbank(
        data=train_data,
        model_name_or_path=fluency_model_path,
        num_sequences=5,
        do_sample=True,
    )
    save_json_file(train_data_sample, f"{work_dir}/debug_rewrite_with_wordbank_train.json")

    
    for item in train_data_sample:
        df = item['data_frame']
        rewrites = df['rewrites']   # list of str
        wordbank_str = df['retrieve_words']
        entities_str = df['entity']
        wordbank_words = set(wordbank_str.split(' '))
        entities = set(entities_str.split(' '))
        
        # 计算在词表中的词占比
        def wordbank_coverage(wordbank_words: set, entities: set, rewrites_str: str) -> float:
            rewrite_words = rewrites_str.split(' ')

            wordbank_words = wordbank_words.union(entities)
            if not wordbank_words:
                return 0.0
            
            covered_words = {word for word in rewrite_words if word in wordbank_words}
            coverage = len(covered_words) / len(rewrite_words)
            return coverage
        
        coverages = []
        for rewrite in rewrites:
            coverage = wordbank_coverage(wordbank_words, entities, rewrite)
            coverages.append(coverage)
        
        # 选择最大覆盖率的作为该样本的覆盖率
        max_coverage = max(coverages) if coverages else 0.0
        df['wordbank_coverage'] = max_coverage
        chosen_for_sft = rewrites[coverages.index(max_coverage)] if coverages else ""
        # 清理不在词表中的词
        chosen_for_sft = ' '.join([word for word in chosen_for_sft.split(' ') if word in wordbank_words.union(entities)])

        df['chosen_for_sft'] = chosen_for_sft
        item['output'] = chosen_for_sft
        item['history'] = []

    save_json_file(train_data_sample, f"{work_dir}/debug_rewrite_with_wordbank_train_processed.json")

    # 构造用于SFT微调的数据
    sft_data = []
    for item in train_data_sample:
        if item['data_frame'].get('wordbank_coverage', 0.0) < 0.8:
            continue
        sft_data.append({
            'system': item.get('system', ''),
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
            'history': item.get('history', []),
            'output': item.get('output', ''),
        })

    save_json_file(sft_data, f"{work_dir}/debug_rewrite_with_wordbank_sft_data.json")

    from llamafy import train_sft

    train_sft(
        model_name_or_path=model_name_or_path,
        train_dataset_path=f"{work_dir}/debug_rewrite_with_wordbank_sft_data.json",
        base_dir=work_dir
    )


def interleave_from_ordered(
    ordered_lists: List[List[str]],
    sep: str = "、",
    dedup: bool = True,
    per_list_limit: int = None,
    total_limit: int = None,
    ) -> str:
    """
    以交替方式从多个候选列表中取词并拼接。
    - dedup: 是否去重
    - per_list_limit: 每个候选列表最多选取的数量（None 表示不限制）
    - total_limit: 全部结果的最大总数（None 表示不限制）
    """
    if not ordered_lists:
        return ""
    if per_list_limit is not None and per_list_limit <= 0:
        return ""
    if total_limit is not None and total_limit <= 0:
        return ""

    max_len = max((len(lst) for lst in ordered_lists), default=0)
    result: List[str] = []
    seen = set()
    picked_per_list = [0] * len(ordered_lists)

    for i in range(max_len):
        for li, lst in enumerate(ordered_lists):
            # 每列表上限检查
            if per_list_limit is not None and picked_per_list[li] >= per_list_limit:
                continue
            if i >= len(lst):
                continue

            token = (lst[i] or "").strip()
            if not token:
                continue
            if dedup and token in seen:
                # 未加入则不计数
                continue

            # 全局上限检查（加入前）
            if total_limit is not None and len(result) >= total_limit:
                return sep.join(result)

            result.append(token)
            seen.add(token)
            picked_per_list[li] += 1

            # 全局上限检查（加入后）
            if total_limit is not None and len(result) >= total_limit:
                return sep.join(result)

    return sep.join(result)

def align_tokens_and_entities(
        tokens_list: List[List[str]], 
        entities_list: List[Set[str]]
    ) -> Tuple[List[List[str]], List[List[bool]]]:
    """
    基于 token 的对齐（贪心最长匹配，按 token 顺序扫描）：
    - 对每句构建实体集合与前缀集合，避免全量字符级搜索与正则。
    - 从当前位置起向后拼接 token，若拼接串仍是某实体前缀则继续，记录最近一次完整命中的实体；
        最终选择最长命中并合并为一个 token，标记为 True；否则保留原 token 并标记为 False。
    - 返回：对齐后的 tokens 及对应位置是否为实体的布尔掩码。
    """
    aligned_tokens_all = []
    aligned_is_entity_all = []

    for tokens, ents in zip(tokens_list, entities_list):
        # 清洗实体集合
        ent_set = {e.strip() for e in ents if e and e.strip()}
        if not ent_set:
            aligned_tokens_all.append(tokens[:])
            aligned_is_entity_all.append([False] * len(tokens))
            continue

        # 构建前缀集合与最大长度（字符数）用于剪枝
        prefix_set = set()
        max_ent_len = 0
        for e in ent_set:
            max_ent_len = max(max_ent_len, len(e))
            # 将实体所有前缀加入前缀集合
            # 注意：包含完整实体本身，方便 "in ent_set" 判断
            for k in range(1, len(e) + 1):
                prefix_set.add(e[:k])

        merged_tokens: List[str] = []
        merged_mask: List[bool] = []

        i = 0
        while i < len(tokens):
            # 贪心尝试从 i 起扩展
            s_parts = []
            s_len = 0
            last_match_end = -1
            last_match_text = None

            j = i
            while j < len(tokens):
                tok = tokens[j]
                s_parts.append(tok)
                s_len += len(tok)
                # 剪枝：超过任一实体最大长度则停止
                if s_len > max_ent_len:
                    break
                s = "".join(s_parts)
                if s not in prefix_set:
                    break
                if s in ent_set:
                    last_match_end = j
                    last_match_text = s
                j += 1

            if last_match_end >= i and last_match_text is not None:
                # 命中最长实体
                merged_tokens.append(last_match_text)
                merged_mask.append(True)
                i = last_match_end + 1
            else:
                # 无命中则保留原 token
                merged_tokens.append(tokens[i])
                merged_mask.append(False)
                i += 1

        aligned_tokens_all.append(merged_tokens)
        aligned_is_entity_all.append(merged_mask)

    return aligned_tokens_all, aligned_is_entity_all

def filter_punctuations_and_align(
    aligned_tokens: List[List[str]], 
    aligned_entity_masks: List[List[bool]]
) -> Tuple[List[List[str]], List[List[bool]]]:
    """
    过滤标点符号并记录标点位置。
    
    Args:
        aligned_tokens: 对齐后的 token 列表
        aligned_entity_masks: 实体掩码列表
        
    Returns:
        (filtered_tokens, punctuation_masks): 过滤后的 token 列表和标点掩码列表
    """
    # 定义标点正则
    common_punctuations = r"""[，。！？、：；（）《》【】〃·…—\.,!?;:'"”“‘’(){}\[\]<>/\s]+"""
    PUNCTUATION_RE = re.compile(common_punctuations)

    # 筛掉实体与标点（用于检索），并记录标点位置以便对齐
    filtered_aligned_tokens = []
    punctuation_masks: List[List[bool]] = []
    
    for token_list, mask in zip(aligned_tokens, aligned_entity_masks):
        punc_mask = [bool(PUNCTUATION_RE.fullmatch(tok)) for tok in token_list]
        punctuation_masks.append(punc_mask)
        filtered_tokens = [
            tok for tok, is_ent, is_punc in zip(token_list, mask, punc_mask)
            if not is_ent and not is_punc
        ]
        filtered_aligned_tokens.append(filtered_tokens)
    
    return filtered_aligned_tokens, punctuation_masks


def prepare_rewrite_candidates(queries: List[str], client: OpenAI, return_k: int =300) -> List[str]:
    """
    准备重写候选词列表。
    
    Args:
        queries: 待处理的查询列表
        client: OpenAI 客户端实例
        return_k: 每个查询返回的候选词数量
        
    Returns:
        候选词字符串列表，每个元素为一个查询对应的候选词字符串
    """
    import torch
    import gc
    from src.data.wordbank import WordBank
    from src.nlp_utils import pku_segment_queries, llm_extract_entities
    
    # seg
    queries_segmented = pku_segment_queries(queries)
    
    # ner
    queries_entities = llm_extract_entities(
        [' '.join(seg) for seg in queries_segmented],
        client
    )
    
    # align
    queries_aligned_tokens, aligned_entity_masks = align_tokens_and_entities(queries_segmented, queries_entities)

    # filter punctuations
    filtered_aligned_tokens, punctuation_masks = filter_punctuations_and_align(
        queries_aligned_tokens, aligned_entity_masks
    )

    # retrieve
    word_bank = WordBank()
    queries_retrieve_results = word_bank.retrieve_distributed(
        batch_query_words=filtered_aligned_tokens,
        return_k=return_k,
    )
    # queries_candidates_list = [list(cand.values()) for cand in queries_retrieve_results]
    
    
    ordered_retrieved = []
    for token_list, mask, punc_mask, cands_map in zip(
        queries_aligned_tokens, aligned_entity_masks, punctuation_masks, queries_retrieve_results
    ):
        ordered = []
        for tok, is_ent, is_punc in zip(token_list, mask, punc_mask):
            if is_ent:
                ordered.append([tok])  # 实体存放自身
            elif is_punc:
                ordered.append([])     # 标点不参与检索，保持空
            else:
                ordered.append(cands_map.get(tok, []))  # 使用词->候选映射
        ordered_retrieved.append(ordered)
    
    candidates_strs = [
        interleave_from_ordered(ordered, sep="、", dedup=True)
        for ordered in ordered_retrieved
    ]

    # 清理
    del word_bank
    torch.cuda.empty_cache()
    gc.collect()

    return candidates_strs

def plain_rewrite_queries(queries: List[str], client: OpenAI) -> List[str]:
    """
    使用 LLM 进行简单重写。
    
    Args:
        queries: 待重写的查询列表
        client: OpenAI 客户端实例
        
    Returns:
        重写后的查询列表
    """
    from src.inference.prompt import PromptRegistry
    from src.inference.openai_utils import generate_with_api_parallel
    
    rewrite_template = PromptRegistry.get("plain_rewrite")
    convs = [
        [
            {"role": "system", "content": rewrite_template.system},
            {"role": "user", "content": rewrite_template.instruction + q},
        ] for q in queries
    ]
    rewrite_responses = generate_with_api_parallel(
        client=client,
        conversations=convs,
        temperature=0.7,
        max_tokens=512,
        prefix="plain rewrite: ",
    )
    rewritten_queries = [resp.strip() for resp in rewrite_responses]
    return rewritten_queries


def rag_muti_round_rewrite():
    # api_key = '4d6e2af37fd14d4fa0e6126d40826804.d2bRZmTVx54JU45M'
    api_key = 'ENMPT'
    model_name_or_path = "/home/ljl/Data/LLM/GLM-4-9B-0414"
    fluency_model_path = '/home/ljl/Data/LLM/GLM-4-32B-0414'

    vectorizer_model_path="checkpoints/bge-large-zh-v1.5"
    vocab_file = 'corpus/wordbank/Words-List.txt'
    rhetoric_model_path = 'checkpoints/glm4-9b-0414-rhetoric-use'

    train_path = "corpus/giga/train_27000.txt"
    test_path = "corpus/giga/test_3000.txt"

    work_dir = setup_save_directory("saves/debug")

    # test_data = load_json_file("corpus/test.json")
    # train_data = load_json_file("corpus/train.json")
    # ================================ 删减区 ================================
    return_k = 300
    max_rounds = 10
    queries = load_lines_file(train_path)
    queries = queries

    from src.rewrite_service import SentenceRewriteService    
    client = OpenAI(api_key=api_key, base_url="http://localhost:8080/v1")

    # plain rewrite
    queries_plain = plain_rewrite_queries(queries, client)
    candidates_strs = prepare_rewrite_candidates(queries_plain, client, return_k=return_k)
    
    save_json_file(
        [{"query": q, "query_plain": qp, "candidates": c} for q, qp, c in zip(queries, queries_plain, candidates_strs)],
        os.path.join(work_dir, "prepared_candidates.json")
    )

    # candidates_strs = load_json_file('saves/debug/run_data_1126_1915/prepared_candidates.json')
    # candidates_strs = [item["candidates"] for item in candidates_strs]
    # 多轮重写
    sentence_rewriter = SentenceRewriteService(max_rounds=max_rounds)
    
    batch_size = 256
    save_path = os.path.join(work_dir, "multi_round_rewrite_results.json")
    all_results = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Multi-round rewriting"):
        # batch_queries = queries[i:i+batch_size]
        batch_queries = queries_plain[i:i+batch_size]
        batch_candidates = candidates_strs[i:i+batch_size]
        batch_results = sentence_rewriter.batch_rewrite(
            texts=batch_queries,
            retrieve_words_strs=batch_candidates,
        )
        all_results.extend(batch_results)
        
        save_json_file(all_results, save_path)
    
    #  ================================ 删减区 ================================


def analyze_logs(rewrite_file_path: str):
    data = load_json_file(rewrite_file_path)

    logs = [item['logs'] for item in data]
    logs_round_10 = [l for l in logs if len(l) ==10]
    logs_failed = [l for l in logs if l[-1]['feedback_type'] != 'done']
    save_json_file(logs_round_10, rewrite_file_path.replace('.json', '_logs_round_10.json'))
    save_json_file(logs_failed, rewrite_file_path.replace('.json', '_logs_failed.json'))
    # 统计失败反馈类型
    from collections import Counter
    failed_types = [l[-1]['feedback_type'] for l in logs_failed]
    counter = Counter(failed_types)
    print(counter)
    save_json_file(counter, rewrite_file_path.replace('.json', '_failed_types_counter.json'))

if __name__ == "__main__":

    pass