import os
import numpy as np
from typing import List, Dict, Tuple, Optional


class PreferenceStatistics:
    """偏好对统计信息收集器"""
    
    # 评估指标字段列表 (所有评分范围: 0-1)
    EVALUATION_FIELDS = ["fluency", "similarity", "simplicity", "coverage", "rhetoric"]
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 动态初始化字段
        self.chosen_avg = {"score": 0.0}
        self.rejected_avg = {"score": 0.0}
        
        # 为每个评估字段初始化
        for field in self.EVALUATION_FIELDS:
            self.chosen_avg[field] = 0.0
            self.rejected_avg[field] = 0.0
            
        self.avg_score_diff = 0.0
        self.count = 0
    
    def add_pair(self, chosen_scores: Tuple[float, ...], rejected_scores: Tuple[float, ...]):
        """
        添加偏好对评分
        """
        assert len(chosen_scores) == len(self.EVALUATION_FIELDS), f"期望 {len(self.EVALUATION_FIELDS)} 个评分，实际得到 {len(chosen_scores)} 个"
        assert len(rejected_scores) == len(self.EVALUATION_FIELDS), f"期望 {len(self.EVALUATION_FIELDS)} 个评分，实际得到 {len(rejected_scores)} 个"
        
        from src.evaluation.evaluate import calculate_final_score
        # 总分 = 流畅度 + 相似度 + 简单性 + 覆盖率 - 修辞
        chosen_score = calculate_final_score(*chosen_scores)
        rejected_score = calculate_final_score(*rejected_scores)
        
        self.count += 1
        n = self.count
        
        # 增量更新平均值 - 总分
        self.chosen_avg["score"] += (chosen_score - self.chosen_avg["score"]) / n
        self.rejected_avg["score"] += (rejected_score - self.rejected_avg["score"]) / n
        
        # 增量更新平均值 - 各项评分
        for i, field in enumerate(self.EVALUATION_FIELDS):
            self.chosen_avg[field] += (chosen_scores[i] - self.chosen_avg[field]) / n
            self.rejected_avg[field] += (rejected_scores[i] - self.rejected_avg[field]) / n
        
        self.avg_score_diff += (chosen_score - rejected_score - self.avg_score_diff) / n
    
    def get_statistics(self, total_samples: int) -> Dict:
        return {
            "total_samples": total_samples,
            "valid_pairs": self.count,
            **{f"chosen_average_{k}": v for k, v in self.chosen_avg.items()},
            **{f"rejected_average_{k}": v for k, v in self.rejected_avg.items()},
            "average_score_difference": self.avg_score_diff,
        }


def generate_preference(
        data: List[Dict],
        wordbank_path: str = "corpus/wordbank/Words-List.txt",
        api_key: Optional[str] = None,
        rhetoric_model_path: Optional[str] = None,   
        fluency_model_path: Optional[str] = None,   
    ) -> Tuple[List[Dict], Dict, List]:
    """
    生成偏好对数据
    
    评分说明:
    - fluency: 流畅度评分，0-1 (从原始1-5归一化)
    - coverage: 词汇覆盖率，0-1
    - rhetoric: 修辞使用，0或1
    
    Returns:
        Tuple[List[Dict], Dict, List]: (偏好对数据, 统计信息, 流畅度原始输出)
    """

    # 1. 准备评估数据
    print("准备评估数据...")
    
    # 分组candidates和references用于evaluate_fluency
    rewrites_grouped = []
    references = []
    all_tokens_flat = []
    all_retrieve_words_flat = []
    rewrites_counts = []
    valid_indices = []  # 记录 data 中有效样本的下标

    for idx, sample in enumerate(data):
        df = sample['data_frame']
        rewrites = df.get("rewrites", [])   # 原始候选文本列表
        rewrites_seg = df.get("rewrites_seg", [])   # 以空格分隔的字符串
        retrieve_words = df.get("retrieve_words", '')  # 检索词列表, 以空格分隔的字符串
        ner_texts_val = df.get("ner_texts", "")  # NER实体列表, 以空格分隔的字符串
        reference = df.get("text", "")
        
        rewrites = [rewrite.strip() for rewrite in rewrites if rewrite.strip()]  # 候选文本列表
        rewrites_tokens_list = [seg.strip().split(' ') for seg in rewrites_seg if seg.strip()]  # 分词列表, List[List[str]]
        retrieve_words_list = [word for word in retrieve_words.split(' ') if word.strip()]  # 检索词列表, List[str]
        retrieve_words_list += [ner for ner in ner_texts_val.split(' ') if ner.strip()]  # 将NER实体加入检索词列表
        
        if not rewrites or not reference:
            continue
        # 收集
        rewrites_counts.append(len(rewrites))
        valid_indices.append(idx)
        rewrites_grouped.append(rewrites)
        references.append(reference)
        all_tokens_flat.extend(rewrites_tokens_list)
        all_retrieve_words_flat.extend([set(retrieve_words_list)] * len(rewrites_tokens_list)) # 每个候选对应相同的检索词集合

    # 2. 进行批量评估
    from src.evaluation.evaluate import batch_evaluate_rewrites
    eval_res = batch_evaluate_rewrites(
        # data
        rewrites_grouped=rewrites_grouped,
        references=references,
        tokens_flat=all_tokens_flat,
        retrieve_words_flat=all_retrieve_words_flat,
        # config
        wordbank_path=wordbank_path,
        api_key=api_key,
        fluency_model_path=fluency_model_path,   
        rhetoric_model_path=rhetoric_model_path,      
    )
    fluency_raw_output = eval_res['raw_fluency_outputs']
    scores_flat = eval_res['scores_flat']
    group_lengths = eval_res['group_lengths']

    # 3. 生成偏好对并构建DPO数据
    print("生成偏好对并构建DPO数据...")
    
    stats = PreferenceStatistics()
    dpo_collection = []
    score_idx = 0
    
    for j, data_idx in enumerate(valid_indices):
        sample = data[data_idx]
        df: Dict = sample['data_frame']
        rewrites = df.get('rewrites', [])
        rewrites_count = group_lengths[j] if j < len(group_lengths) else 0
        
        # 跳过只有一个候选的样本
        if rewrites_count < 2:
            score_idx += rewrites_count
            continue
        
        sample_scores = scores_flat[score_idx: score_idx + rewrites_count]
        score_idx += rewrites_count

        # 选择最佳和最差的候选
        totals = [s.get("total") for s in sample_scores]
        best_idx = int(np.argmax(totals))
        worst_idx = int(np.argmin(totals))

        # 更新统计（五元组）
        def as_tuple(s: Dict[str, float]) -> Tuple[float, float, float, float, float]:
            return (
                float(s.get("fluency", 0.0)),
                float(s.get("similarity", 0.0)),
                float(s.get("simplicity", 0.0)),
                float(s.get("coverage", 0.0)),
                float(s.get("rhetoric", 0.0)),
            )
        stats.add_pair(as_tuple(sample_scores[best_idx]), as_tuple(sample_scores[worst_idx]))

        # 构建DPO数据项
        sample.update({
            'history': [],
            'output': "",
            'chosen': rewrites[best_idx],
            'rejected': rewrites[worst_idx],
        })

        # 记录中间结果
        def join_metric(name: str) -> str:
            return " ".join([f"{float(s.get(name, 0.0)):.4f}" for s in sample_scores])
        df.update({
            **{f"{field}_scores": join_metric(field) for field in ['fluency', 'similarity', 'simplicity', 'coverage', 'rhetoric']}
        })

        dpo_collection.append(sample)

    # 4. 获取统计信息
    status = stats.get_statistics(len(data))
    return dpo_collection, status, fluency_raw_output


if __name__ == "__main__":
    pass