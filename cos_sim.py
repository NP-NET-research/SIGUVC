import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSimilarityScorer:
    def __init__(self, model_name='/home/zhipu_fund/text2vec-large-chinese'):
        """
        初始化Sentence-BERT模型
        :param model_name: 预训练模型名称
        """

        self.model = SentenceTransformer(
            model_name,
            use_auth_token=True,
        )
        print(f"已加载模型: {model_name}")
    
    def calculate_similarity(self, batch, responses):
        """
        使用NumPy计算余弦相似度
        
        :return: 相似度分数列表（0.0-1.0）
        """
        similarity_scores = []
        for ori, para in zip(batch['batch_original_sentence'], responses):

            ref_embedding = self.model.encode(ori, convert_to_tensor=False)
            
            cand_embeddings = self.model.encode(para, convert_to_tensor=False)
        
            dot_product = np.dot(ref_embedding, cand_embeddings)
            
            norm_ref = np.linalg.norm(ref_embedding)
            norm_cand = np.linalg.norm(cand_embeddings)
            
            if norm_ref == 0 or norm_cand == 0:
                similarity_scores.append(0.0)
                continue
                
            cos_sim = dot_product / (norm_ref * norm_cand)
            
            normalized_score = (cos_sim + 1) / 2
            similarity_scores.append(normalized_score)
                
        return similarity_scores
