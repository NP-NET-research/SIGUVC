from typing import List, Dict

from .local_services import (
    llm_rewrite_with_vocab, 
    llm_rewrite_with_feedback, 
    llm_merge_sentences,
    llm_merge_with_feedback
)

from src.utils import load_lines_file


class SentenceRewriter:
    """
    句子重写器：将原始句子改写为使用指定词表的句子
    使用本地 vLLM 实现完整的重写流程

    使用预处理后的记录列表（rewrite_from_preprocessed），
       预处理记录需包含：sentence / plain_sentence、sub_sentences、entities
    """
    
    def __init__(
        self, 
        wordbank_path: str = 'saves/local_rewrite/selected_dict_words.txt',
        generator = None,
        temperature: float = 0.2
    ):
        """
        初始化句子重写器
        
        Args:
            wordbank_path: 词库文件路径（每行一个词）
            generator: BaseGenerator 实例
            temperature: 温度参数
        """
        
        self.generator = generator
        self.temperature = temperature

        self.wordbank_path = wordbank_path
        self.unified_vocab = None  # 使用列表保持顺序
        self.unified_vocab_str = None
    
    def _load_unified_vocab(self) -> List[str]:
        """从词库文件加载统一词表，保持文件读取顺序"""
        if self.unified_vocab is None:
            print(f"Loading unified vocabulary from {self.wordbank_path}...")
            lines = load_lines_file(self.wordbank_path)
            vocab_list = []
            for line in lines:
                word = line.strip()
                if word:
                    vocab_list.append(word)
            self.unified_vocab = vocab_list
            self.unified_vocab_str = '、'.join(self.unified_vocab)
            print(f"Loaded {len(self.unified_vocab)} unique words into the unified vocabulary")
        return self.unified_vocab
     
    def _eval_vocab_coverage(self, rewritten: str, vocab: List[str]) -> float:
        """
        计算词汇覆盖率
        
        Args:
            rewritten: 重写后的句子（词之间用'、'分隔）
            vocab: 词汇表
            
        Returns:
            覆盖率（0.0-1.0）
        """
        tokens = rewritten.split('、')
        if not tokens:
            return 0.0
        covered_tokens = sum(1 for token in tokens if token in vocab)
        return covered_tokens / len(tokens)
    
    def build_vocab(self, dict_data: List[Dict], selected_words: List = None) -> List[Dict]:
        """
        使用预处理后的数据构建词表：
        - 所有子句的 vocab = 主句实体列表 + 统一词表
        """
        if selected_words:
            self.unified_vocab = selected_words
            self.unified_vocab_str = '、'.join(self.unified_vocab)
        else:
            self._load_unified_vocab()

        for item in dict_data:
            entities = item.get('entities')
            combined_vocab = entities + self.unified_vocab
            vocab_str = '、'.join(combined_vocab)
            item['vocab_str'] = vocab_str

        return dict_data
    
    def rewrite_with_iterations(
        self,
        vocab_data: List[Dict],
        max_iterations: int = 2
    ) -> List[Dict]:
        """
        使用迭代式方法进行词表重写（CoT版本）
        
        Args:
            vocab_data: 包含词表的数据
            max_iterations: 最大迭代次数
            
        Returns:
            重写结果
        """
        # 准备数据
        queries = []
        vocabs = []
        num_sentences_list = []
        
        for item in vocab_data:
            sub_sentences = item['sub_sentences']
            vocab_str = item['vocab_str']
            num_sentences_list.append(len(sub_sentences))
            queries.extend(sub_sentences)
            vocabs.extend([vocab_str] * len(sub_sentences))
        
        # 第一轮重写（CoT版本）
        result = llm_rewrite_with_vocab(
            queries=queries,
            vocabs_strs=vocabs,
            generator=self.generator,
            temperature=self.temperature,
        )
        
        current_results = result['content']
        current_conversations = result['conversations']
        
        # 计算覆盖率
        current_coverages = []
        for i in range(len(queries)):
            vocab_set = set(vocabs[i].split('、'))
            current_coverages.append(self._eval_vocab_coverage(current_results[i], list(vocab_set)))
        
        # 迭代修正
        for _ in range(2, max_iterations + 1):
            
            needs_correction = []
            correction_conversations = []
            feedback_content_list = []
            
            for i in range(len(queries)):
                
                # 需要修正的情况：结果为"无法重写"
                if current_results[i] == "无法重写":
                    feedback_content_list.append("上次结果：无法重写（请尝试放宽语义标准，深度挖掘词表中的词汇）")

                # 需要修正的情况：结果非"无法重写"且覆盖率不足
                elif current_coverages[i] < 1.0:
                    vocab_set = set(vocabs[i].split('、'))
                    tokens = current_results[i].split('、')
                    wrong_words = [t for t in tokens if t not in vocab_set]
                    
                    if wrong_words:
                        needs_correction.append(i)
                        correction_conversations.append(current_conversations[i])
                        feedback_content_list.append('、'.join(wrong_words))
            
            if not needs_correction:
                break
            
            # 反馈重写（CoT版本）
            feedback_result = llm_rewrite_with_feedback(
                previous_conversations=correction_conversations,
                feedback_content_list=feedback_content_list,
                generator=self.generator,
                temperature=self.temperature
            )
            
            # 更新结果
            for idx, original_idx in enumerate(needs_correction):
                current_results[original_idx] = feedback_result['content'][idx]
                current_conversations[original_idx] = feedback_result['conversations'][idx]
                
                vocab_set = set(vocabs[original_idx].split('、'))
                current_coverages[original_idx] = self._eval_vocab_coverage(
                    current_results[original_idx],
                    list(vocab_set)
                )
        
        # 最后检查：仍然不满足覆盖率的标记为"无法重写"
        for i in range(len(queries)):
            if current_results[i] != "无法重写" and current_coverages[i] < 1.0:
                current_results[i] = "无法重写"
                current_coverages[i] = 0.0
        
        # 重新组织结果
        results = []
        index = 0
        for i, item in enumerate(vocab_data):
            num_sentences = num_sentences_list[i]
            new_item = item.copy()
            new_item['rewritten_sentences'] = current_results[index:index + num_sentences]
            new_item['coverages'] = current_coverages[index:index + num_sentences]
            results.append(new_item)
            index += num_sentences
        
        return results
    
    def merge_sentences(self, rewrite_data: List[Dict], max_merge_iterations: int = 2) -> List[Dict]:
        """
        合并简单句重写结果为完整句子（CoT版本）
        
        Args:
            rewrite_data: 重写后的数据
            max_merge_iterations: 合并阶段的最大迭代次数
            
        Returns:
            合并后的结果
        """

        self._load_unified_vocab()
        
        sentences = []
        sub_rewritten_sentences = []
        vocab_strs = []
        valid_indices = []
        
        for i, item in enumerate(rewrite_data):

            sentences.append(item['sentence'])
            vocab_strs.append(item['vocab_str'])
            
            # 过滤无法重写的句子
            rewrittens = [
                s for s in item['rewritten_sentences']
                if s != "无法重写" and s.strip()
            ]
            sub_rewritten_sentences.append(rewrittens)
            
            # 如果有有效的重写子句，记录索引
            if rewrittens:
                valid_indices.append(i)
        
        # 初始化结果列表
        merged_results = [""] * len(rewrite_data)
        merge_coverages = [0.0] * len(rewrite_data)
        merge_conversations = [None] * len(rewrite_data)
        
        # 第一轮合并（CoT版本）
        if valid_indices:
            
            valid_sentences = [sentences[i] for i in valid_indices]
            valid_sub_rewritten = [sub_rewritten_sentences[i] for i in valid_indices]
            valid_vocab_strs = [vocab_strs[i] for i in valid_indices]
            
            result = llm_merge_sentences(
                queries=valid_sub_rewritten,
                ori_sentences=valid_sentences,
                vocab_strs=valid_vocab_strs,
                generator=self.generator,
                temperature=self.temperature
            )
            
            api_results = result['content']
            conversations = result['conversations']
            
            for idx, valid_idx in enumerate(valid_indices):
                merged_results[valid_idx] = api_results[idx]
                merge_conversations[valid_idx] = conversations[idx]
                if api_results[idx] and api_results[idx] != "无法重写":
                    vocab_list = vocab_strs[valid_idx].split('、')
                    merge_coverages[valid_idx] = self._eval_vocab_coverage(
                        api_results[idx], 
                        vocab_list
                    )
        
        # 迭代修正合并结果（CoT版本）
        for _ in range(2, max_merge_iterations + 1):
            
            needs_correction = []
            correction_conversations = []
            correction_wrong_words = []
            correction_valid_indices = []
            
            for valid_idx in valid_indices:
                if (merged_results[valid_idx] and 
                    merged_results[valid_idx] != "无法重写" and 
                    merge_coverages[valid_idx] < 1.0):
                    
                    vocab_set = set(vocab_strs[valid_idx].split('、'))
                    tokens = merged_results[valid_idx].split('、')
                    wrong_words = [t for t in tokens if t not in vocab_set]
                    
                    if wrong_words:
                        needs_correction.append(valid_idx)
                        correction_conversations.append(merge_conversations[valid_idx])
                        correction_wrong_words.append('、'.join(wrong_words))
                        correction_valid_indices.append(valid_idx)
            
            if not needs_correction:
                break

            # 反馈重写（CoT版本）
            feedback_result = llm_merge_with_feedback(
                previous_conversations=correction_conversations,
                wrong_words_list=correction_wrong_words,
                generator=self.generator,
                temperature=self.temperature
            )
            
            # 更新结果
            for idx, valid_idx in enumerate(correction_valid_indices):
                merged_results[valid_idx] = feedback_result['content'][idx]
                merge_conversations[valid_idx] = feedback_result['conversations'][idx]
                
                if feedback_result['content'][idx] and feedback_result['content'][idx] != "无法重写":
                    vocab_list = vocab_strs[valid_idx].split('、')
                    merge_coverages[valid_idx] = self._eval_vocab_coverage(
                        feedback_result['content'][idx],
                        vocab_list
                    )
        
        # 对仍然失败的句子进行简单拼接
        failed_indices = []
        for valid_idx in valid_indices:
            if (not merged_results[valid_idx] or 
                merged_results[valid_idx] == "无法重写" or
                merge_coverages[valid_idx] < 1.0):
                failed_indices.append(valid_idx)
        
        if failed_indices:
            for failed_idx in failed_indices:
                simple_concat = '、'.join(sub_rewritten_sentences[failed_idx])
                merged_results[failed_idx] = simple_concat
                vocab_list = vocab_strs[failed_idx].split('、')
                merge_coverages[failed_idx] = self._eval_vocab_coverage(
                    simple_concat,
                    vocab_list
                )
          
        # 添加结果
        results = []
        for i, item in enumerate(rewrite_data):
            new_item = item.copy()
            new_item['final_rewrite'] = merged_results[i].replace('、', ' ')  # 最终结果用空格分隔
            new_item['merge_coverage'] = merge_coverages[i]
            results.append(new_item)
        
        return results
    
    def rewrite(self, preprocessed_data: List[Dict], selected_words: List = None, max_iterations: int = 3, max_merge_iterations: int = 2) -> List[Dict]:
        """
        完整的重写流程
        
        Args:
            preprocessed_data: 预处理后的数据
            selected_words: 可选的词表列表，如果提供则使用该词表而不是从文件加载
            max_iterations: 重写阶段最大迭代次数
            max_merge_iterations: 合并阶段最大迭代次数
            
        Returns:
            最终重写结果
        """
        
        # 1. 基于预处理数据构建「共享实体库」词表
        data_with_vocab = self.build_vocab(preprocessed_data, selected_words)

        # 2. 迭代重写（CoT版本）
        rewrite_data = self.rewrite_with_iterations(data_with_vocab, max_iterations)

        # 3. 合并句子（CoT版本）
        final_results = self.merge_sentences(rewrite_data, max_merge_iterations)

        # 4. 统一结果字段顺序
        key_order = [
            "id", 
            "sentence", "final_rewrite", "merge_coverage", 
            "sub_sentences", "rewritten_sentences", "coverages", 
            "segment", "dependency_relations", 
            "entities", "vocab_str", 
        ]
        final_results = [{k: r[k] for k in key_order if k in r} for r in final_results]

        return final_results
