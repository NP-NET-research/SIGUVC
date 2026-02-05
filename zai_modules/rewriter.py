import os
import json
from typing import List, Dict, Set, Tuple

from .zai_services import (
    zai_rewrite_with_vocab, 
    zai_rewrite_with_feedback, 
    zai_merge_sentences,
    zai_rewrite_with_vocab_cot,
    zai_rewrite_with_feedback_cot,
    zai_rewrite_merge_sentences_cot,
    zai_rewrite_merge_with_feedback_cot
)

from src.utils import (
    load_lines_file,
    load_jsonl_file
)

class SentenceRewriter:
    """
    句子重写器：将原始句子改写为使用指定词表的句子
    使用智谱 API 实现完整的重写流程

    使用预处理后的记录列表（rewrite_from_preprocessed），
       预处理记录需包含：sentence / plain_sentence、sub_sentences、entities
    """
    
    def __init__(
        self, 
        wordbank_path: str = 'saves/zai_rewrite/selected_dict_words.txt',
        model: str = 'glm-4.5',
        temperature: float = 0.2,
        thinking: bool = False
    ):
        """
        初始化句子重写器
        
        Args:
            wordbank_path: 词库文件路径（每行一个词）
            model: 使用的模型名称
            temperature: 温度参数
            thinking: 是否启用思考模式
        """
        
        self.model = model
        self.temperature = temperature
        self.thinking = thinking

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
    
    def build_vocab(self, dict_data: List[Dict], selected_words: List) -> List[Dict]:
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
        使用迭代式方法进行词表重写
        
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
        
        # 第一轮重写
        result = zai_rewrite_with_vocab_cot(
            queries=queries,
            vocabs_strs=vocabs,
            model=self.model,
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
            correction_wrong_words = []
            
            for i in range(len(queries)):
                if current_results[i] != "无法重写" and current_coverages[i] < 1.0:
                    vocab_set = set(vocabs[i].split('、'))
                    tokens = current_results[i].split('、')
                    wrong_words = [t for t in tokens if t not in vocab_set]
                    
                    if wrong_words:
                        needs_correction.append(i)
                        correction_conversations.append(current_conversations[i])
                        correction_wrong_words.append('、'.join(wrong_words))
            
            if not needs_correction:
                break
            
            # 反馈重写
            feedback_result = zai_rewrite_with_feedback_cot(
                previous_conversations=correction_conversations,
                wrong_words_list=correction_wrong_words,
                model=self.model,
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
        合并简单句重写结果为完整句子
        
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
        
        # 第一轮合并
        if valid_indices:
            
            valid_sentences = [sentences[i] for i in valid_indices]
            valid_sub_rewritten = [sub_rewritten_sentences[i] for i in valid_indices]
            valid_vocab_strs = [vocab_strs[i] for i in valid_indices]
            
            result = zai_rewrite_merge_sentences_cot(
                queries=valid_sub_rewritten,
                ori_sentences=valid_sentences,
                vocab_strs=valid_vocab_strs,
                model=self.model,
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
        
        # 迭代修正合并结果
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

            # 反馈重写
            feedback_result = zai_rewrite_merge_with_feedback_cot(
                previous_conversations=correction_conversations,
                wrong_words_list=correction_wrong_words,
                model=self.model,
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
        
        # 1. 基于预处理数据构建「共享实体库」词表
        data_with_vocab = self.build_vocab(preprocessed_data, selected_words)

        # 2. 迭代重写
        rewrite_data = self.rewrite_with_iterations(data_with_vocab, max_iterations)

        # 3. 合并句子
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
    
    def rewrite_partial(
        self, 
        preprocessed_data: List[Dict], 
        selected_words: List = None,
        affected_subsentences: Dict = None,
        max_iterations: int = 3, 
        max_merge_iterations: int = 2
    ) -> List[Dict]:
        """
        部分重写：只重写受影响的子句
    
        Args:
            preprocessed_data: 预处理数据
            selected_words: 新词表
            affected_subsentences: 受影响的子句映射
            max_iterations: 迭代次数
            max_merge_iterations: 合并迭代次数
        """
        # 1. 构建词表
        data_with_vocab = self.build_vocab(preprocessed_data, selected_words)
        
        # 2. 准备重写数据
        rewrite_tasks = []
        task_to_item_map = []
        
        for item in data_with_vocab:
            sent_id = item['id']
            sub_sentences = item['sub_sentences']
            vocab_str = item['vocab_str']
            
            # 检查是否有受影响的子句
            if affected_subsentences and sent_id in affected_subsentences:
                affected_info = affected_subsentences[sent_id]
                affected_indices = affected_info['affected_indices']
                
                if affected_indices:
                    # 只重写受影响的子句
                    for idx in affected_indices:
                        if 0 <= idx < len(sub_sentences):
                            rewrite_tasks.append({
                                'query': sub_sentences[idx],
                                'vocab_str': vocab_str
                            })
                            task_to_item_map.append({
                                'item_idx': sent_id,
                                'sub_idx': idx,
                                'total_subs': len(sub_sentences)
                            })
                else:
                    # 没有精确定位到子句，标记为直接合并
                    item['direct_merge'] = True
            else:
                # 没有affected_subsentences信息，直接合并
                item['direct_merge'] = True
        
        # 3. 批量重写受影响的子句
        if rewrite_tasks:
            queries = [t['query'] for t in rewrite_tasks]
            vocabs = [t['vocab_str'] for t in rewrite_tasks]
            
            rewrite_result = self._rewrite_batch_with_iterations(
                queries, vocabs, max_iterations
            )
            
            # 将结果分配回原数据
            for i, (rewritten, coverage) in enumerate(zip(rewrite_result['rewrites'], rewrite_result['coverages'])):
                mapping = task_to_item_map[i]
                item_idx = mapping['item_idx']
                sub_idx = mapping['sub_idx']
                
                # 找到对应的item
                item = next(d for d in data_with_vocab if d['id'] == item_idx)

                # 更新重写结果
                item['rewritten_sentences'][sub_idx] = rewritten
                item['coverages'][sub_idx] = coverage
    
        # 4. 直接合并的句子（未定位到子句的情况）默认直接进入合并环节
    
        # 5. 合并句子
        final_results = self.merge_sentences(data_with_vocab, max_merge_iterations)
        
        # 6. 统一字段顺序
        key_order = [
            "id", "sentence", "final_rewrite", "merge_coverage",
            "sub_sentences", "rewritten_sentences", "coverages",
            "segment", "dependency_relations",
            "entities", "vocab_str",
        ]
        final_results = [{k: r[k] for k in key_order if k in r} for r in final_results]
        
        return final_results

    def _rewrite_batch_with_iterations(
        self,
        queries: List[str],
        vocabs: List[str],
        max_iterations: int
    ) -> Dict:
        """批量迭代重写（内部方法）"""
        from .zai_services import zai_rewrite_with_vocab_cot, zai_rewrite_with_feedback_cot
        
        # 第一轮重写
        result = zai_rewrite_with_vocab_cot(
            queries=queries,
            vocabs_strs=vocabs,
            model=self.model,
            temperature=self.temperature,
        )
        
        current_results = result['content']
        current_conversations = result['conversations']
        
        # 计算覆盖率
        current_coverages = []
        for i in range(len(queries)):
            vocab_set = set(vocabs[i].split('、'))
            current_coverages.append(
                self._eval_vocab_coverage(current_results[i], list(vocab_set))
            )
        
        # 迭代修正
        for _ in range(2, max_iterations + 1):
            needs_correction = []
            correction_conversations = []
            correction_wrong_words = []
            
            for i in range(len(queries)):
                if current_results[i] != "无法重写" and current_coverages[i] < 1.0:
                    vocab_set = set(vocabs[i].split('、'))
                    tokens = current_results[i].split('、')
                    wrong_words = [t for t in tokens if t not in vocab_set]
                    
                    if wrong_words:
                        needs_correction.append(i)
                        correction_conversations.append(current_conversations[i])
                        correction_wrong_words.append('、'.join(wrong_words))
            
            if not needs_correction:
                break
            
            feedback_result = zai_rewrite_with_feedback_cot(
                previous_conversations=correction_conversations,
                wrong_words_list=correction_wrong_words,
                model=self.model,
                temperature=self.temperature
            )
            
            for idx, original_idx in enumerate(needs_correction):
                current_results[original_idx] = feedback_result['content'][idx]
                current_conversations[original_idx] = feedback_result['conversations'][idx]
                
                vocab_set = set(vocabs[original_idx].split('、'))
                current_coverages[original_idx] = self._eval_vocab_coverage(
                    current_results[original_idx],
                    list(vocab_set)
                )
    
        # 标记失败的重写
        for i in range(len(queries)):
            if current_results[i] != "无法重写" and current_coverages[i] < 1.0:
                current_results[i] = "无法重写"
                current_coverages[i] = 0.0
        
        return {
            'rewrites': current_results,
            'coverages': current_coverages
        }
