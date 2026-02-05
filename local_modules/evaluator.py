from typing import List, Dict

from .local_services import (
    llm_dependency_parsing,
    llm_match_dependency_relations,
    llm_segment_sentences
)


class SemanticEvaluator:
    """
    语义评判器：基于依存句法分析评估句子间的语义相似度
    使用本地模型进行评估
    """
    
    def __init__(
        self,
        generator = None,
        temperature: float = 0.2,
        max_tokens: int = 16384
    ):
        """
        初始化语义评判器
        
        Args:
            generator: BaseGenerator 实例，用于本地模型推理
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.generator = generator
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _preprocess_dependency_relations(
        self,
        sentences: List[str],
        lang: str = "zh"
    ) -> List[List[str]]:
        """调用本地模型进行依存句法分析，获取依存关系列表"""
        
        # 分词
        segments = llm_segment_sentences(
            queries=sentences,
            generator=self.generator,
            temperature=self.temperature
        )
        
        # 依存句法分析
        dependency_relations = llm_dependency_parsing(
            queries=segments,
            generator=self.generator,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            lang=lang
        )
        
        return dependency_relations
    
    def compute_similarity(
        self,
        sentences1: List[str],
        sentences2: List[str],
        dependency_relations: List[List[str]] = None,
        lang: str = "zh"
    ) -> Dict:
        """
        计算两组句子的语义相似度
        
        Args:
            sentences1: 第一组句子（参考句）
            sentences2: 第二组句子（测试句）
            dependency_relations: 预处理的依存关系列表
            lang: 语言类型 ("zh" 或 "en")
            
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
        
        # 收集需要调用模型的索引
        valid_indices = [i for i in range(len(sentences1)) if matched_relations[i] is None]
        
        if valid_indices:
            # 只对非空的测试句调用模型
            valid_sentences1 = [sentences1[i] for i in valid_indices]
            valid_sentences2 = [sentences2[i] for i in valid_indices]
            valid_dep_rels = [dep_rels[i] for i in valid_indices]
            
            if dependency_relations is None:
                valid_dep_rels = self._preprocess_dependency_relations(
                    valid_sentences1,
                    lang=lang
                )
            
            match_result = llm_match_dependency_relations(
                gold_sentences=valid_sentences1,
                test_sentences=valid_sentences2,
                dependency_relations_list=valid_dep_rels,
                generator=self.generator,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                lang=lang
            )
            
            # 将模型结果填入对应位置
            for idx, valid_idx in enumerate(valid_indices):
                matched_relations[valid_idx] = match_result['content'][idx]
                reasoning_contents[valid_idx] = match_result['reasoning_content'][idx]
        
        # 计算相似度分数：匹配的依存关系占比
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
        gold_scores: List[float] = None,
        lang: str = "zh"
    ) -> Dict:
        """
        批量评估句子对的相似度
        
        Args:
            batch: 包含句子对及其依存关系的字典列表
            gold_scores: 参考分数列表（可选）
            lang: 语言类型 ("zh" 或 "en")
            
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
            dependency_relations=dependency_relations,
            lang=lang
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
