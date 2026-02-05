import os
import re
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from FlagEmbedding import FlagModel
from src.data.wordbank_utils import VocabVectorizer, SubWordBank


class WordBank:

    def __init__(
            self,
            vocab_file: str = 'corpus/wordbank/Words-List.txt',
            serialized_vocab_file: str = "corpus/wordbank/cache/vocab.pkl",
            vectorizer_model_path: str = "checkpoints/bge-large-zh-v1.5",
            faiss_index_file: str = "corpus/wordbank/cache/vocab.faiss",
            entity_cache_file: str = "corpus/wordbank/cache/entity.pkl",
            force_rebuild: bool = False,
            use_fp16: bool = True,
            query_instruction: str = ""
    ):
        
        # 词汇库 - 只读，从文件加载
        self.vocab_bank = SubWordBank(
            cache_file=serialized_vocab_file,
            source_file=vocab_file,
            read_only=True,
            force_rebuild=force_rebuild
        )
        
        # 实体库 - 可读写，初始为空，仅从缓存加载
        # 使用固定的默认分类 "#entities" 存储所有实体
        self.entity_bank = SubWordBank(
            cache_file=entity_cache_file,
            source_file=None,  # 不从源文件加载
            read_only=False,
            force_rebuild=False
        )
        
        # 初始化编码模型
        print(f"加载编码模型: {vectorizer_model_path}")
        self.encoder_model = FlagModel(
            vectorizer_model_path,
            query_instruction_for_retrieval=query_instruction,
            use_fp16=use_fp16
        )
        
        # 向量化器 (FAISS CPU) - 传入编码模型
        self.vectorizer = VocabVectorizer(
            encoder_model=self.encoder_model,
            faiss_index_file=faiss_index_file,
            force_rebuild=force_rebuild
        )

        if force_rebuild or not os.path.exists(faiss_index_file):
            # 构建向量和 FAISS 索引
            words = self.vocab_bank.words | self.entity_bank.words
            self.vectorizer.vectorize_and_build_index(words)
            self.vectorizer.save_faiss_index()
        else:
            # 直接加载 FAISS 索引
            self.vectorizer.load_faiss_index()

        # 数字单位正则
        self._compile_number_unit_regex()

    def _compile_number_unit_regex(self):
        """编译数字单位匹配的正则表达式"""

        # 定义数字、单位、前缀
        ordinals = ["第","连续第","上","下","前","后","本","这","上个","下个","前个","后个"]
        digit_chars = "零一二三四五六七八九十百千万亿0-9０-９壹贰叁肆伍陆柒捌玖拾佰仟两俩几多半双成"
        multi_num_words = ["分之","百分之","千分之","万分之","0.25","0.5","0.75","1/2","1/4","3/4"]
        
        units = []
        units = ['年', '月', '日', '号', '天', '周', '星期', '世纪', '年代', '季度', '上午', '下午', '晚上', '凌晨', '点', '分', '秒', '毫秒', '公里', '米', '厘米', '毫米', '微米', '纳米', '英寸', '尺', '码', '吨', '公斤', '克', '毫克', '升', '毫升',
                 '度', '摄氏度', '华氏度', '元']
        # for class_name in unit_classes_name:
        #     units.extend(self.vocab_bank.get_words_by_category(class_name))
        # print(f"Loaded {len(units)} units for number-unit matching.")
        
        # 拼接正则
        num_pattern = f"[{digit_chars}]+"
        multi_num_pattern = "|".join(multi_num_words)
        full_num_pattern = f"(?:{num_pattern}|{multi_num_pattern})"
        
        unit_pattern = "|".join(sorted(units, key=lambda x: -len(x)))
        ordinal_pattern = "|".join(sorted(ordinals, key=lambda x: -len(x)))

        self.number_unit_regex = re.compile(
            f"(?:{ordinal_pattern})?({full_num_pattern})(?:{unit_pattern})?"
        )
        
    def match_number_unit_tokens(self, batch_segmented_texts: List[List[str]]) -> List[List[int]]:
        batch_matched = []
        for tokens in batch_segmented_texts:
            matched = [i for i, tok in enumerate(tokens) if tok and self.number_unit_regex.fullmatch(tok)]
            batch_matched.append(matched)
        return batch_matched
   
    def contains_word(self, word: str) -> bool:
        if not isinstance(word, str) or not word.strip():
            return False
        word = word.strip()
        # 先检查词汇库，再检查实体库
        return self.vocab_bank.contains_word(word) or self.entity_bank.contains_word(word)

    def match(self, batch_segmented_texts: List[List[str]]) -> List[List[int]]:
        matched_word_ids = []
        matched_numerical_ids = self.match_number_unit_tokens(batch_segmented_texts)

        for idx, token_list in enumerate(batch_segmented_texts):
            words = token_list
            matched = [i for i, w in enumerate(words) if w and self.contains_word(w)]
            combined = list(set(matched + matched_numerical_ids[idx]))
            matched_word_ids.append(combined)
        return matched_word_ids

    def retrieve(
            self,
            batch_query_words: List[List[str]],
            top_k: int = 5,
            threshold: Optional[float] = None,
            return_k: Optional[int] = None
        ) -> List[List[str]]:
        """
        按句返回候选：
        - 返回 List[List[str]]，每句返回至多 return_k（若为 None 则使用 top_k）个候选词
        - 对同一句中的所有查询词的检索结果进行合并去重，按相似度（取最大值）排序
        - 当提供 return_k 时，忽略阈值 threshold
        """
        if not batch_query_words:
            return []

        batch_query_words = self.remove_punctuations(batch_query_words)
        batch_size = len(batch_query_words)

        # 词统一检索（跨批次去重），记录词属于哪些句子
        words_to_vectorize = []
        word_to_batch = {}  # word -> list of batch idx
        for batch_idx, query_words in enumerate(batch_query_words):
            for word in query_words:
                clean_word = word.strip()
                if not clean_word:
                    continue
                if clean_word not in word_to_batch:
                    words_to_vectorize.append(clean_word)
                    word_to_batch[clean_word] = []
                word_to_batch[clean_word].append(batch_idx)

        if not words_to_vectorize:
            return [[] for _ in range(batch_size)]

        # 有 return_k 时固定返回 return_k 个，否则使用 top_k
        effective_k = return_k if return_k is not None else top_k
        search_k = max(top_k, effective_k or 0, 1)

        # 批量编码并归一化
        query_vectors = self.encoder_model.encode_queries(
            words_to_vectorize, batch_size=256, convert_to_numpy=True
        )
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors.astype(np.float32)

        # FAISS 查询
        similarities, indices = self.vectorizer.index.search(query_vectors, search_k)

        # 按句聚合候选：candidate_word -> best_score
        batch_candidate_scores = [dict() for _ in range(batch_size)]  # List[Dict[str, float]]

        for i, word in enumerate(words_to_vectorize):
            idxs = indices[i]
            sims = similarities[i]
            for batch_idx in word_to_batch[word]:
                cand_map = batch_candidate_scores[batch_idx]
                for score, idx in zip(sims, idxs):
                    if idx < 0:
                        continue
                    cand_word = self.vectorizer.words[idx]
                    prev = cand_map.get(cand_word)
                    if prev is None or score > prev:
                        cand_map[cand_word] = score

        # 生成每句的 top-K 候选（忽略阈值以满足“每句返回 return_k 个词”的要求）
        results: List[List[str]] = []
        for cand_map in batch_candidate_scores:
            if not cand_map:
                results.append([])
                continue
            sorted_items = sorted(cand_map.items(), key=lambda x: x[1], reverse=True)
            results.append([w for w, _ in sorted_items][:effective_k])

        return results
    

    def retrieve_distributed(
        self,
        batch_query_words: List[List[str]],
        return_k: int = 5
    ) -> List[Dict[str, List[str]]]:
        """
        按句、按词均分返回候选（改进：优先精确命中，返回 dict）
        - 输入：batch_query_words 为按句的词列表
        - 清洗：先去标点
        - 逻辑：
          1) 对每句的唯一词集合做"精确命中"判断（词库命中或纯数字），命中者候选固定为 [词本身]；
          2) 若精确命中唯一词数 < return_k，则将剩余份额在未命中的唯一词之间平均分配，余数从前往后补 1；
          3) 仅对份额>0 的未命中唯一词进行向量检索，按份额截断候选；份额=0 的未命中词返回 []。
        - 返回：List[Dict[str, List[str]]]（每句一个 dict：词 -> 候选列表）
        """
        if not batch_query_words:
            return []

        # 清洗标点
        cleaned_batches = self.remove_punctuations(batch_query_words)

        # 工具：均分 k 到 n 个槽位，余数从前往后补 1
        def split_k(n: int, k: int) -> List[int]:
            if n <= 0 or k <= 0:
                return [0] * max(n, 0)
            base, rem = divmod(k, n)
            return [base + (1 if i < rem else 0) for i in range(n)]
        
        # 工具：判断是否为纯数字（包括中文数字和阿拉伯数字）
        def is_pure_number(word: str) -> bool:
            if not word:
                return False
            # 匹配纯阿拉伯数字（包括小数）
            if re.fullmatch(r'\d+\.?\d*', word):
                return True
            # 匹配纯中文数字
            chinese_digits = '零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億兩两'
            if all(c in chinese_digits for c in word):
                return True
            return False

        # 第一遍：为每句确定精确命中集合、未命中集合，并计算每个未命中唯一词的份额
        results: List[Dict[str, List[str]]] = []
        # 需要进行向量检索的唯一词（跨句去重）
        words_to_vectorize: List[str] = []
        # 记录每个唯一词出现在哪些句子及其份额：word -> [(b_idx, share), ...]
        word_to_positions: Dict[str, List[Tuple[int, int]]] = {}
        k_max = 0  # 未命中词的最大份额（用于一次性检索 top-k）

        for b_idx, tokens in enumerate(cleaned_batches):
            # 句内唯一词
            unique_words_in_sent = []
            seen = set()
            for w in tokens:
                w = w.strip()
                if not w or w in seen:
                    continue
                seen.add(w)
                unique_words_in_sent.append(w)

            # 精确命中（词库命中或纯数字）
            exact_words = [w for w in unique_words_in_sent if self.contains_word(w) or is_pure_number(w)]
            non_exact_words = [w for w in unique_words_in_sent if w not in exact_words]
            exact_count = len(exact_words)

            # 初始化本句结果：精确词 -> [自身]；未命中词先置空
            sent_map: Dict[str, List[str]] = {w: [w] for w in exact_words}
            for w in non_exact_words:
                if w not in sent_map:
                    sent_map[w] = []

            # 将剩余份额均分到未命中唯一词
            remaining_k = max(0, return_k - exact_count)
            shares = split_k(len(non_exact_words), remaining_k)

            # 记录需检索的词及份额
            for w, share in zip(non_exact_words, shares):
                if share > 0:
                    if w not in word_to_positions:
                        word_to_positions[w] = []
                        words_to_vectorize.append(w)
                    word_to_positions[w].append((b_idx, share))
                    if share > k_max:
                        k_max = share

            results.append(sent_map)

        # 若没有需要向量检索的项（全部精确命中或剩余额度为0），直接返回
        if not words_to_vectorize or k_max <= 0:
            return results

        # 向量化并归一化（跨句唯一词一次性处理）
        query_vectors = self.encoder_model.encode_queries(
            words_to_vectorize, batch_size=256, convert_to_numpy=True
        )
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors.astype(np.float32)

        # FAISS 检索：一次取足 k_max
        similarities, indices = self.vectorizer.index.search(query_vectors, k_max)

        # 为每个唯一词准备候选（同候选保留最大相似度），并截断到 k_max
        word_to_candidates: Dict[str, List[str]] = {}
        for i, w in enumerate(words_to_vectorize):
            idxs = indices[i]
            sims = similarities[i]
            best_scores: Dict[str, float] = {}
            for idx, sim in zip(idxs, sims):
                if idx < 0:
                    continue
                cand = self.vectorizer.words[idx]
                prev = best_scores.get(cand)
                if prev is None or sim > prev:
                    best_scores[cand] = float(sim)
            sorted_items = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
            word_to_candidates[w] = [cw for cw, _ in sorted_items][:k_max]

        # 将检索结果按份额填回对应句子的 dict（未命中词）
        for w, positions in word_to_positions.items():
            cands = word_to_candidates.get(w, [])
            for b_idx, share in positions:
                # 注意：本句中该词已初始化为 []（若非精确）；按份额截断
                results[b_idx][w] = cands[:share]

        return results
    
    
    # ----------------- 命名实体库管理接口 -----------------
    
    def add_entities(self, entities: List[str]) -> int:
        """
        添加命名实体
        :param entities: 实体列表
        :return: 成功添加的实体数量
        """
        if not entities:
            return 0
        # 使用固定分类 "#entities" 存储所有实体
        added_count = self.entity_bank.add_words("#entities", entities)
        return added_count
    
    def remove_entities(self, entities: List[str]) -> int:
        """
        从实体库中删除指定实体
        :param entities: 要删除的实体列表
        :return: 成功删除的实体数量
        """
        if not entities:
            return 0
        return self.entity_bank.remove_words(entities)
    
    def get_all_entities(self) -> Set[str]:
        """
        获取所有命名实体
        :return: 所有实体的集合
        """
        return self.entity_bank.words
    
    def contains_entity(self, entity: str) -> bool:
        """
        检查实体是否存在于实体库中
        :param entity: 实体名称
        :return: 是否存在
        """
        if not isinstance(entity, str) or not entity.strip():
            return False
        return self.entity_bank.contains_word(entity.strip())
    
    def export_entities(self, output_file: str) -> None:
        """
        导出实体库到文本文件
        :param output_file: 输出文件路径
        """
        self.entity_bank.export_to_txt(output_file)
    
    @property
    def entity_size(self) -> int:
        """实体库大小"""
        return len(self.entity_bank.words)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_bank.words)
    
    @property
    def vocab(self) -> Set[str]:
        return self.vocab_bank.words

    @staticmethod
    def remove_punctuations(batch_segmented_texts: List[List[str]]) -> List[List[str]]:

        common_punctuations = r"""[，。！？、：；（）《》【】〃·…—\.,!?;:'"”“‘’(){}\[\]<>/\s]+"""
        PUNCTUATION_RE = re.compile(common_punctuations)
    
        cleaned_batches = []
        for tokens in batch_segmented_texts:
            cleaned_tokens = []
            for token in tokens:
                if token and not PUNCTUATION_RE.match(token):
                    cleaned_token = token.strip(common_punctuations)
                    if cleaned_token: 
                        cleaned_tokens.append(cleaned_token)
            cleaned_batches.append(cleaned_tokens)
        return cleaned_batches

