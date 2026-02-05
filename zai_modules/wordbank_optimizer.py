from collections import defaultdict
import os
import logging
import random

from src.utils import (
    load_jsonl_file,
    save_jsonl_file,
    load_lines_file,
    save_lines_file,
)

def rank_weighted_choice(candidates: list, top_k: int = 5):
    """基于排名的加权随机选择函数"""

    if not candidates:
        return None
    
    pool = candidates[:top_k]
    n = len(pool)

    weights = [1.0 / (i + 1) for i in range(n)]
    # weights = [(n - i) for i in range(n)]

    return random.choices(pool, weights=weights, k=1)[0]


class WordBankOptimizer:
    """词表优化器：统计、决策、应用词表替换"""
    
    def __init__(
        self, 
        context2dict_path: str,
        selected_words_path: str,
        rewrite_eval_path: str
    ):
        """
        初始化词表优化器
        
        Args:
            context2dict_path: 上下文到字典映射文件路径
            selected_words_path: 选中词汇文件路径（按排名顺序，越靠前越好）
            rewrite_eval_path: 重写评估结果文件路径
        """
        self.context2dict_path = context2dict_path
        self.selected_words_path = selected_words_path
        self.rewrite_eval_path = rewrite_eval_path
        
        # 当前选中词汇（迭代修改，保持排名顺序）
        self.current_selected_words = load_lines_file(self.selected_words_path)
        
        # 当前重写评估结果（迭代修改）
        self.current_rewrite_eval_data = load_jsonl_file(self.rewrite_eval_path)

        # 缓存的映射数据
        self._mapping_data = None
        self._replaceable_to_context = None
        self._replaceable_weights = None
    
    def _load_mapping_data(self):
        """加载并缓存映射数据"""
        if self._mapping_data is None:
            self._mapping_data = load_jsonl_file(self.context2dict_path)
            
            # 构建replaceable_words的反向映射
            self._replaceable_to_context = {}
            for mapping in self._mapping_data:
                word = mapping['word']
                for replaceable in mapping.get('replaceable_words', []):
                    if replaceable not in self._replaceable_to_context:
                        self._replaceable_to_context[replaceable] = set()
                    self._replaceable_to_context[replaceable].add(word)
            
            self._replaceable_weights = {
                word: len(contexts) 
                for word, contexts in self._replaceable_to_context.items()
            }
        
        return self._mapping_data

    def update_selected_words(self, replacement_decision):
        """应用词汇替换决策,更新内部状态和文件，新词添加到开头"""
        
        word_to_remove = replacement_decision['word_to_remove']
        word_to_add = replacement_decision['word_to_add']
        
        # 移除旧词，将新词添加到开头
        new_list = [w for w in self.current_selected_words if w != word_to_remove]
        new_list.insert(0, word_to_add)

        # 更新内部状态和文件
        self.current_selected_words = new_list

        return self.current_selected_words
    
    def _update_rewrite_eval_file(self):
        """将当前评估数据同步到文件"""
        save_jsonl_file(self.current_rewrite_eval_data, self.rewrite_eval_path)
    
    def _update_selected_words_file(self):
        """将当前选中词汇同步到文件"""
        save_lines_file(self.current_selected_words, self.selected_words_path)

    def update_rewrite_eval_data(self, updated_records):
        """更新重写评估数据"""
        # 构建ID到记录的映射
        eval_map = {e["id"]: e for e in self.current_rewrite_eval_data}
        eval_map.update({e["id"]: e for e in updated_records})
        
        # 更新内部状态
        self.current_rewrite_eval_data = list(eval_map.values())
        
        return self.current_rewrite_eval_data
    
    def get_average_score(self):
        """获取当前评估数据的平均分数"""
        if not self.current_rewrite_eval_data:
            return 0.0
        return sum(e.get("pred_score", 0) for e in self.current_rewrite_eval_data) / len(self.current_rewrite_eval_data)
   
    
    def compute_statistics(self):
        """统计词语使用情况和未匹配词信息"""
 
        selected_words = self.current_selected_words
        selected_words_set = set(selected_words)
        rewrite_eval_data = self.current_rewrite_eval_data
        mapping_data = self._load_mapping_data()
        
        # 1. 初始化 word_usage
        word_usage = {
            word: {'valid_contribution': 0, 'count': 0, 'sentence_ids': []} 
            for word in selected_words
        }
        
        # 2. 初始化未匹配实例列表
        unmatched_instances = []
        
        # 3. 构建索引：context_id -> word -> mapping_entry
        mapping_lookup = {}
        for entry in mapping_data:
            ctx_id = entry['context_id']
            if ctx_id not in mapping_lookup:
                mapping_lookup[ctx_id] = {}
            mapping_lookup[ctx_id][entry['word']] = entry
    
        # 4. 遍历评测结果
        for item in rewrite_eval_data:
            idx = item['id']
            final_rewrite = item['final_rewrite']
            
            # --- A. 基础词频统计 (Count) ---
            rewrite_words_list = final_rewrite.split()
            rewrite_words_set = set(rewrite_words_list)
            active_selected_words = selected_words_set.intersection(rewrite_words_set)
            
            for w in active_selected_words:
                word_usage[w]['sentence_ids'].append(idx)
                word_usage[w]['count'] += 1

            # --- B. 依存匹配统计 (归因 + 失败检测) ---
            dependency_relations = item.get('dependency_relations', [])
            matched_relations = item.get('matched_relations', [])
            
            current_sentence_mappings = mapping_lookup.get(idx, {})
            
            word_dep_status = {}
            
            for dep_str, match_res in zip(dependency_relations, matched_relations):
                parts = dep_str.split()
                if len(parts) < 3: continue
                w1, relation_type, w2 = parts[0], parts[1], parts[2]
                
                is_matched = 1 if (match_res == 1 or (isinstance(match_res, dict) and match_res.get('matched') == 1)) else 0
                involved_context_words = [w1, w2]
                
                if is_matched:
                    for ctx_word in involved_context_words:
                        mapping_entry = current_sentence_mappings.get(ctx_word)
                        if not mapping_entry: continue

                        candidates = set(mapping_entry.get('replaceable_words', []))
                        contributors = []
                        for cand in candidates:
                            if cand in selected_words_set and cand in rewrite_words_set:
                                contributors.append(cand)
                        
                        if ctx_word in selected_words_set and ctx_word in rewrite_words_set:
                            if ctx_word not in contributors:
                                contributors.append(ctx_word)
                        
                        for contributor in contributors:
                            word_usage[contributor]['valid_contribution'] += 1
                
                for w in involved_context_words:
                    if w not in word_dep_status:
                        word_dep_status[w] = {'total': 0, 'matched_count': 0}
                    
                    word_dep_status[w]['total'] += 1
                    if is_matched:
                        word_dep_status[w]['matched_count'] += 1

            # --- C. 筛选未匹配实例 ---
            for ctx_word, status in word_dep_status.items():
                original_entry = current_sentence_mappings.get(ctx_word)
                if not original_entry:
                    continue
                
                # 核心判定：只有当所有依存关系都失败 (matched_count == 0) 时，才算未匹配
                if status['matched_count'] == 0 and status['total'] > 0:
                    unmatched_instances.append(original_entry)

        return word_usage, unmatched_instances

    def decide_word_replacement(self, word_usage, unmatched_instances):
        """决定剔除旧词和增添新词，并精确定位受影响的子句"""
        remove_top_k = 50
        add_top_k = 50
        
        current_words_set = set(word_usage.keys())
        word_rank = {word: idx for idx, word in enumerate(self.current_selected_words)}
        
        # 1. 删除决策
        sorted_removal_candidates = sorted(
            word_usage.items(),
            key=lambda x: (
                x[1]['valid_contribution'], 
                x[1]['count'], 
                -word_rank.get(x[0], 0)
            )
        )

        word_to_remove, word_to_remove_info = rank_weighted_choice(sorted_removal_candidates, top_k=remove_top_k)
        
        # 2. 新增决策
        if not unmatched_instances:
            return None

        candidate_scores = defaultdict(lambda: {'score': 0, 'affected_sentences': set()})
        
        for instance in unmatched_instances:
            sentence_id = instance['context_id']
            candidates = instance.get('replaceable_words', [])
            
            for cand in candidates:
                if cand in current_words_set:
                    continue
                
                entry = candidate_scores[cand]
                entry['score'] += 1
                entry['affected_sentences'].add(sentence_id)

        if not candidate_scores:
            return None

        sorted_add_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        word_to_add, add_info = rank_weighted_choice(sorted_add_candidates, top_k=add_top_k)
        
        # 3. **新增：精确定位受影响的子句**
        affected_subsentences = self._locate_affected_subsentences(
            word_to_remove, 
            word_to_add,
            word_to_remove_info['sentence_ids'],
            add_info['affected_sentences']
        )
        
        total_affected_ids = (
            set(word_to_remove_info['sentence_ids']) |
            add_info['affected_sentences']
        )
        
        return {
            'word_to_remove': word_to_remove,
            'word_to_add': word_to_add,
            
            'word_to_remove_stats': {
                'rank': word_rank.get(word_to_remove, -1),
                'usage': word_to_remove_info['count'],
                'contribution': word_to_remove_info['valid_contribution']
            },
            'word_to_add_stats': {
                'expected_gain': add_info['score'],
                'covered_sentences_count': len(add_info['affected_sentences'])
            },
            
            'sentences_affected_count': len(total_affected_ids),
            'add_affected_ids': list(add_info['affected_sentences']),
            'remove_affected_ids': word_to_remove_info['sentence_ids'],
            'sentences_affected_ids': list(total_affected_ids),
            'affected_subsentences': affected_subsentences
        }

    def _locate_affected_subsentences(
        self, 
        word_to_remove: str, 
        word_to_add: str,
        remove_affected_ids: list,
        add_affected_ids: set
    ):
        """
        精确定位受影响的子句
    
        通过判断上下文词是否出现在子句中来定位
    
        返回格式:
        {
            sentence_id: {
                'affected_indices': [子句索引列表],
                'reason': 'remove' | 'add' | 'both'
            }
        }
        """
        affected_subsentences = {}
        mapping_data = self._load_mapping_data()

        # 构建 context_id -> preprocessed_item 的映射
        preprocessed_map = {item['id']: item for item in self.current_rewrite_eval_data}
        
        # 构建 context_id -> word -> mapping_entry 的映射
        mapping_lookup = {}
        for entry in mapping_data:
            ctx_id = entry['context_id']
            word = entry['word']
            if ctx_id not in mapping_lookup:
                mapping_lookup[ctx_id] = {}
            mapping_lookup[ctx_id][word] = entry
    
        # 处理删除词影响的句子
        for sent_id in remove_affected_ids:
            if sent_id not in preprocessed_map:
                continue
            
            if sent_id not in affected_subsentences:
                affected_subsentences[sent_id] = {
                    'affected_indices': set(),
                    'reason': 'remove'
                }
            
            preprocessed_item = preprocessed_map[sent_id]
            rewritten_sentences = preprocessed_item.get('rewritten_sentences', [])
            # 通过删除词定位受影响的子句
            for sub_idx, sub_rewritten_sentence in enumerate(rewritten_sentences):
                if word_to_remove in sub_rewritten_sentence:
                    affected_subsentences[sent_id]['affected_indices'].add(sub_idx)
    
        # 处理新增词影响的句子
        for sent_id in add_affected_ids:
            if sent_id not in preprocessed_map:
                continue
            
            if sent_id not in affected_subsentences:
                affected_subsentences[sent_id] = {
                    'affected_indices': set(),
                    'reason': 'add'
                }
            else:
                affected_subsentences[sent_id]['reason'] = 'both'
            
            preprocessed_item = preprocessed_map[sent_id]
            sub_sentences = preprocessed_item.get('sub_sentences', [])
            sentence_mappings = mapping_lookup.get(sent_id, {})
            
            # 遍历所有上下文词，检查是否受 word_to_add 影响
            for context_word, mapping_entry in sentence_mappings.items():
                replaceable_words = mapping_entry.get('replaceable_words', [])
                
                # 如果该上下文词可以被 word_to_add 替换
                if word_to_add in replaceable_words:
                    # 查找该词在哪些子句中出现
                    for sub_idx, sub_sentence in enumerate(sub_sentences):
                        if context_word in sub_sentence:
                            affected_subsentences[sent_id]['affected_indices'].add(sub_idx)
    
        # 转换为列表格式并排序
        for sent_id in affected_subsentences:
            affected_subsentences[sent_id]['affected_indices'] = sorted(
                list(affected_subsentences[sent_id]['affected_indices'])
            )
        
        return affected_subsentences