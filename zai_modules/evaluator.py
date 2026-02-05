from typing import List, Dict

from .zai_services import (
    zai_dependency_parsing,
    zai_match_dependency_relations,
    zai_segment_sentences
)


class SemanticEvaluator:
    """
    语义评判器：基于依存句法分析评估句子间的语义相似度
    使用预处理的依存分析结果进行评估
    """
    
    def __init__(
        self,
        model: str = 'glm-4.5',
        temperature: float = 0.2,
        thinking: bool = False
    ):
        """
        初始化语义评判器
        
        Args:
            model: 使用的模型名称
            temperature: 温度参数
            thinking: 是否启用思考模式
        """
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
    
    def _preprocess_dependency_relations(
        self,
        sentences: List[str]
    ) -> List[List[str]]:
        """调用zai进行依存句法分析，获取依存关系列表"""

        segments = zai_segment_sentences(
            queries=sentences,
            model=self.model,
            temperature=self.temperature,
            thinking=self.thinking
        )['content']

        dep_parsing_result = zai_dependency_parsing(
            queries=segments,
            model=self.model,
            temperature=self.temperature,
            thinking=self.thinking
        )

        dependency_relations = dep_parsing_result['content']
        return dependency_relations
    
    def compute_similarity(
        self,
        sentences1: List[str],
        sentences2: List[str],
        dependency_relations: List[List[str]] = None
    ) -> Dict:
        """
        计算两组句子的语义相似度

        Args:
            sentences1: 第一组句子（参考句）
            sentences2: 第二组句子（测试句）
            dependency_relations: 预处理的依存关系列表
            
        Returns:
            包含相似度分数和详细信息的字典
        """
        if dependency_relations:
            dep_rels = dependency_relations
        else:
            dep_rels = [None] * len(sentences1)
        
        # 检查sentences2是否为空，如果为空则直接生成全0的判定结果
        matched_relations = []
        reasoning_contents = [None] * len(sentences1)
        for i in range(len(sentences1)):
            if not sentences2[i] or sentences2[i].strip() == "":
                # 对于空句子，生成与依存关系数量相同的全0列表
                matched_relations.append([0.0] * len(dep_rels[i]))
            else:
                matched_relations.append(None)  # 占位，稍后填充
        
        # 收集需要调用API的索引
        valid_indices = [i for i in range(len(sentences1)) if matched_relations[i] is None]
        
        if valid_indices:
            # 只对非空的测试句调用API
            valid_sentences1 = [sentences1[i] for i in valid_indices]
            valid_sentences2 = [sentences2[i] for i in valid_indices]
            valid_dep_rels = [dep_rels[i] for i in valid_indices]

            if dependency_relations is None:
                valid_dep_rels = self._preprocess_dependency_relations(valid_sentences1)
            
            match_result = zai_match_dependency_relations(
                gold_sentences=valid_sentences1,
                test_sentences=valid_sentences2,
                dependency_relations_list=valid_dep_rels,
                model=self.model,
                temperature=self.temperature,
                thinking=self.thinking
            )
            
            # 将API结果填入对应位置
            for idx, valid_idx in enumerate(valid_indices):
                matched_relations[valid_idx] = match_result['content'][idx]
                reasoning_contents[valid_idx] = match_result['reasoning_content'][idx]
        
        # 4. 计算相似度分数：匹配的依存关系占比
        similarity_scores = []
        for i in range(len(sentences1)):
            total_rels = len(dep_rels[i])
            if total_rels == 0:
                similarity_scores.append(0.0)
                continue
            matched_rels = sum(matched_relations[i])
            similarity = matched_rels / total_rels
            similarity_scores.append(similarity)
        
        return {
            'content': similarity_scores,
            'reasoning_content': reasoning_contents,
            'dependency_relations': dep_rels,
            'matched_relations': matched_relations
        }
    
    def evaluate_batch(
        self,
        batch: List[Dict],
        gold_scores: List[float] = None
    ) -> Dict:
        """
        批量评估句子对的相似度
        
        Args:
            batch: 包含句子对及其依存关系的字典列表
            gold_scores: 参考分数列表（可选）
            
        Returns:
            评估结果字典
        """
        sentence_pairs = [(item['sentence'], item['final_rewrite']) for item in batch]
        dependency_relations = [item['dependency_relations'] for item in batch]

        sentences1 = [pair[0] for pair in sentence_pairs]
        sentences2 = [pair[1] for pair in sentence_pairs]
        
        print(f"Evaluating {len(sentence_pairs)} sentence pairs...")
        similarity_results = self.compute_similarity(
            sentences1, 
            sentences2,
            dependency_relations=dependency_relations
        )
        
        pred_scores = similarity_results['content']
        
        results = {
            'pred_scores': pred_scores,
            'dependency_relations': similarity_results['dependency_relations'],
            'matched_relations': similarity_results['matched_relations'],
            'reasoning_content': similarity_results['reasoning_content']
        }
        
        # 如果提供了参考分数，计算相关系数
        if gold_scores is not None:
            from scipy.stats import pearsonr
            correlation, _ = pearsonr(gold_scores, pred_scores)
            results['pearson_correlation'] = correlation
            results['gold_scores'] = gold_scores
        
        return results
