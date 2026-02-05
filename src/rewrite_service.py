# rewrite_service.py

from typing import Dict, List
from src.inference.prompt import PromptRegistry
from src.data.wordbank import WordBank
from src.inference.openai_utils import generate_with_api_parallel
import re
from openai import OpenAI
import os
import pkuseg
from src.inference.generator import BaseGenerator

from src.utils import load_lines_file

# os.environ.pop("HTTP_PROXY", None)
# os.environ.pop("HTTPS_PROXY", None)
# os.environ.pop("http_proxy", None)
# os.environ.pop("https_proxy", None)


# def build_input(text, wordbank_str):
#         # extra_words = load_lines_file('corpus/wordbank/extra_words.txt')
#         # extra_words = [w.strip() for w in extra_words]
#         # extra_words = [w for w in extra_words if w and w not in wordbank_str.split('、')]
#         # wordbank_str += '、' + '、'.join(extra_words)
#         return f'【原始句子】\n{text}\n\n【词表】\n{wordbank_str}\n'


class SentenceRewriteService:

    def __init__(
        self,
        generator = None,
        max_rounds: int = 5,
        temperature: float = 0.2,
    ):
        self.max_rounds = max_rounds

        # 模型推理参数
        self.temperature = temperature
        
        self.generator = generator if generator is not None else BaseGenerator()

        self.rewrite_prompt_template = PromptRegistry.get("rewrite")

        self.MIN_TOKEN_COUNT = 2  # 最小分词数量
        self.MIN_SIMILARITY = 0.5  # 最低语义相似度
        self.TARGET_VOCAB_MATCH_RATE = 1.0  # 目标词表匹配率（100%，严格约束）

    # ======================== 主流程 ========================
    
    def muti_round_rewrite(
        self,
        texts: List[str],
        retrieve_words_strs: List[str],
    ) -> List[str]:
        """
        批量改写多个句子
        参数:
            texts: 原始句子列表
            retrieve_words_strs_list: 每个句子对应的检索词表字符串
        返回: List[str] - 每个句子的改写结果
        """
        batch_size = len(texts)
        
        # 构建minivocab
        mini_vocabs = []
        for words_str in retrieve_words_strs:
            words = [w.strip() for w in words_str.split('、') if w.strip()]
            mini_vocabs.append(set(words))
        
        # 初始化每个句子的对话历史
        conversations = []
        for i in range(batch_size):
            conv = [
                {"role": "system", "content": self.rewrite_prompt_template.system},
                {"role": "user", "content": self.rewrite_prompt_template.instruction.format(
                    sentence=texts[i],
                    wordbank=retrieve_words_strs[i]
                )},
            ]
            conversations.append(conv)
        
        # 初始化结果
        results = [{"original": texts[i], "retrieve_words_strs": retrieve_words_strs[i], "final": "", "logs": []} 
                   for i in range(batch_size)]
        rewritten_texts = [""] * batch_size
        completed = [False] * batch_size
        
        # 迭代改写
        for round_i in range(self.max_rounds):
            # 收集未完成的索引
            active_indices = [i for i in range(batch_size) if not completed[i]]
            if not active_indices:
                break
            
            # 批量生成改写结果
            active_conversations = [conversations[i] for i in active_indices]
            assert isinstance(self.generator, OpenAI), "当前仅支持OpenAI兼容接口的并行推理"
            active_outputs = generate_with_api_parallel(
                self.generator,
                active_conversations,
                temperature=self.temperature
            )
            
            # 更新改写结果
            for idx, i in enumerate(active_indices):
                rewritten_texts[i] = active_outputs[idx].strip()
            
            # 批量评测
            active_originals = [texts[i] for i in active_indices]
            active_rewrittens = [rewritten_texts[i] for i in active_indices]
            active_vocabs = [mini_vocabs[i] for i in active_indices]
            
            feedbacks = self.build_feedback(
                active_originals,
                active_rewrittens,
                active_vocabs
            )
            
            # 更新日志和对话历史
            for idx, i in enumerate(active_indices):
                feedback = feedbacks[idx]
                
                results[i]["logs"].append({
                    "round": round_i + 1,
                    "candidate": rewritten_texts[i],
                    "feedback_type": feedback["type"],
                    "feedback": feedback["content"],
                    "metrics": feedback["metrics"],
                    "illegal_words": feedback["illegal_words"]
                })
                
                if feedback["type"] == "done":
                    completed[i] = True
                    results[i]["final"] = rewritten_texts[i]
                else:
                    # 添加助手回复和用户反馈到对话历史
                    feedback_content = (
                        f"根据反馈内容，严格按照约束进行改写：\n"
                        f"【反馈】{feedback['content']}\n"
                        f"【核心约束】1. 仅使用词表中的词汇（违规词已标注，需全部替换）；2. 语义与原句「{texts[i]}」高度一致（相似度≥{self.MIN_SIMILARITY}）；3. 用空格分词，无修辞，语法通顺。\n"
                        f"【输出要求】仅空格分词结果，无需额外解释。"
                    )
                    history = conversations[i]
                    initial_history = history[:2]
                    interaction_history = history[2:]
                    recent_interactions = interaction_history[-4:] if len(interaction_history) >= 4 else interaction_history

                    conversations[i] = initial_history + recent_interactions + [
                        {"role": "assistant", "content": rewritten_texts[i]},  # 当前轮助手输出
                        {"role": "user", "content": feedback_content}  # 当前轮用户反馈指令
                    ]
        
        # 处理未完成的句子
        for i in range(batch_size):
            if not completed[i]:
                results[i]["final"] = rewritten_texts[i]
        
        return results
    
    def single_round_rewrite(
        self,
        texts: List[str],
        retrieve_words_strs: List[str],
        model_name: str = "glm-4.5-Flash"
    ) -> List[List[str]]:
        """
        批量改写多个句子（单轮）
        参数:
            texts: 原始句子列表
            retrieve_words_strs_list: 每个句子对应的检索词表字符串
        返回: List[List[str]] - 每个句子的改写结果列表（每条生成多个候选）
        """
        batch_size = len(texts)
        
        # 构建minivocab
        mini_vocabs = []
        for words_str in retrieve_words_strs:
            words = [w.strip() for w in words_str.split('、') if w.strip()]
            mini_vocabs.append(set(words))
        
        # 构建对话历史
        conversations = []
        for i in range(batch_size):
            conv = [
                {"role": "system", "content": self.rewrite_prompt_template.system},
                {"role": "user", "content": self.rewrite_prompt_template.instruction + \
                    """【句子】\n{sentence}\n【词表】\n{vocab}\n\n输出：\n""".format(
                    sentence=texts[i],
                    vocab=retrieve_words_strs[i]
                )},
            ]
            conversations.append(conv)
        
        # 批量生成改写结果, 每条生成num_sequences个候选
        if isinstance(self.generator, BaseGenerator):
            rewritten_texts = self.generator.generate(
                queries=conversations,
            )
        else:
            assert isinstance(self.generator, OpenAI)
            rewritten_texts = generate_with_api_parallel(
                client=self.generator,
                conversations=conversations,
                temperature=self.temperature,
                model_name=model_name
            )

        return rewritten_texts
    
    # ======================== 评测 ========================

    def build_feedback(
        self, 
        originals: List[str], 
        rewrittens: List[str], 
        mini_vocabs: List[set]
    ) -> List[Dict]:
        """
        批量评测改写结果并构建反馈信息
        参数:
            originals: 原始句子列表
            rewrittens: 改写后的句子列表
            mini_vocabs: 每个句子对应的单义词词库
        返回: List[Dict] - 结构化反馈字典列表，包含：
            {
                "type": "done/format_error/rhetoric_error/vocab_error/similarity_error",
                "content": 反馈文本,
                "metrics": 量化指标（匹配率、相似度等）,
                "illegal_words": 违规词列表（词表外词汇）
            }
        """
        batch_size = len(originals)
        feedbacks = []
        all_tokens = []

        # 初始化结构化反馈（默认未完成）
        for i in range(batch_size):
            feedbacks.append({
                "type": "uncheck",
                "content": "",
                "metrics": {"vocab_match_rate": 0.0, "similarity": 0.0},
                "illegal_words": [],
                "rhetoric_info": ""
            })
        
        # 第一步：分词和基础检查
        for i in range(batch_size):
            rewritten = rewrittens[i].strip()
            tokens = rewritten.split()
            all_tokens.append(tokens)
            fb = feedbacks[i]
            
            if len(tokens) < self.MIN_TOKEN_COUNT:
                fb["type"] = "format_error"
                fb["content"] = "分词错误！确保使用空格进行分词。"
                continue
            fb["type"] = "to_check"
        
        indices_to_check = [i for i in range(batch_size) if feedbacks[i]["type"] == "to_check"]
        if not indices_to_check:
            return feedbacks
        
        # 第二步：批量修辞检测
        texts_to_check = ["".join(all_tokens[i]) for i in indices_to_check]
        rhetoric_results = self._judge_rhetoric_with_llm(texts_to_check)
        
        for idx, i in enumerate(indices_to_check):
            has_rhetoric, rhetoric_text = rhetoric_results[idx]
            fb = feedbacks[i]
            if has_rhetoric:
                fb["type"] = "rhetoric_error"
                fb["rhetoric_info"] = rhetoric_text
                fb["content"] = "修辞使用违规！检测到{}，请使用直白、客观的表达方式重写，不添加比喻、拟人、夸张等修辞，确保语义与原句一致。".format(
                    rhetoric_text
                )
        
        # 更新需要继续检查的索引
        indices_to_check = [i for i in indices_to_check if feedbacks[i]["type"] == "to_check"]
        if not indices_to_check:
            return feedbacks
        
        # 第三步：词库覆盖率评测
        for i in indices_to_check:
            tokens = all_tokens[i]
            mini_vocab = mini_vocabs[i]
            fb = feedbacks[i]

            # 计算词表匹配率和违规词（优化评测逻辑，避免依赖外部函数的列表嵌套）
            total = len(tokens)
            legal_tokens = [t for t in tokens if t in mini_vocab]
            illegal_tokens = list(set([t for t in tokens if t not in mini_vocab]))  # 去重，避免重复提示
            vocab_match_rate = len(legal_tokens) / total if total > 0 else 0.0

            # 更新量化指标
            fb["metrics"]["vocab_match_rate"] = vocab_match_rate
            fb["illegal_words"] = illegal_tokens

            # 未达标：提示具体违规词+替换建议
            if vocab_match_rate < self.TARGET_VOCAB_MATCH_RATE:
                fb["type"] = "vocab_error"
                # 生成词表候选建议（帮助模型快速替换，减少猜测成本）
                fb["content"] = (
                    "违规使用词表外词汇：「{}」\n"
                    "优化建议：1. 将违规词替换为词表中语义相近的词汇；2. 若无完全匹配词，用词表内词汇组合表达原含义；3. 保持分词格式（空格分隔），不新增词表外词汇。"
                ).format("、".join(illegal_tokens))
                continue
        

        # 更新待检查索引（词表通过的样本）
        indices_to_check = [i for i in indices_to_check if feedbacks[i]["type"] == "to_check"]
        if not indices_to_check:
            return feedbacks
        
        # 第四步：语义相似度评测
        sentence_pairs = [(originals[i], " ".join(all_tokens[i])) for i in indices_to_check]
        similarities = self._score_similarity_with_llm(sentence_pairs)
        
        for idx, i in enumerate(indices_to_check):
            similarity = similarities[idx]
            fb = feedbacks[i]
            fb["metrics"]["similarity"] = similarity

            if similarity < self.MIN_SIMILARITY:
                fb["type"] = "similarity_error"
                fb["content"] = (
                    "语义相似度偏低（{:.2f}，目标≥{:.2f}）！\n"
                    "优化建议：1. 保留原句核心实体；2. 调整词汇组合，确保表达的含义与原句一致；3. 仍需严格使用词表内词汇，不新增违规词。"
                ).format(
                    similarity, self.MIN_SIMILARITY,
                    # originals[i], " ".join(all_tokens[i]),
                )
            else:
                # 所有检查通过
                fb["type"] = "done"

        return feedbacks

    def score(
        self,
        originals: List[str],
        rewrittens: List[str],
        mini_vocabs: List[set]
    ) -> List[Dict]:
        """
        批量评分改写结果，从词匹配率、流畅度、语义相似度三个维度打分
        参数:
            originals: 原始句子列表
            rewrittens: 改写后的句子列表
            mini_vocabs: 每个句子对应的单义词词库
        返回: List[Dict] - 评分结果列表，包含：
            {
                "vocab_match": int (0或1),      # 词表匹配：1表示100%匹配，0表示有违规词
                "fluency": int (0或1),          # 流畅度：1表示流畅，0表示不流畅
                "similarity": float (0-1),      # 语义相似度
                "overall": float (0-1)          # 综合得分（加权平均）
            }
        """
        batch_size = len(originals)
        scores = []
        all_tokens = []

        # 第一步：分词和词表匹配率评测（二元分数）
        for i in range(batch_size):
            rewritten = rewrittens[i].strip()
            tokens = rewritten.split()
            all_tokens.append(tokens)
            
            mini_vocab = mini_vocabs[i]
            total = len(tokens)
            
            if total > 0:
                legal_tokens = [t for t in tokens if t in mini_vocab]
                # 100%匹配得1分，否则得0分
                vocab_match = 1 if len(legal_tokens) == total else 0
            else:
                vocab_match = 0
            
            scores.append({
                "vocab_match": vocab_match,
                "fluency": 0,
                "similarity": 0.0,
                "overall": 0.0
            })
        
        # 第二步：流畅度评测（二元分数）
        texts_for_fluency = [" ".join(tokens) for tokens in all_tokens]
        fluency_scores = self._score_fluency_with_llm(texts_for_fluency)
        
        for i in range(batch_size):
            scores[i]["fluency"] = fluency_scores[i]
        
        # 第三步：语义相似度评测
        # sentence_pairs = [(originals[i], " ".join(all_tokens[i])) for i in range(batch_size)]
        # similarity_scores = self._score_similarity_with_llm(sentence_pairs)
        
        # for i in range(batch_size):
        #     scores[i]["similarity"] = similarity_scores[i]
        
        # 第四步：计算综合得分，直接相加
        for i in range(batch_size):
            scores[i]["overall"] = scores[i]["vocab_match"] + scores[i]["fluency"] + scores[i]["similarity"]
        
        return scores

    # ======================== LLM 评测 ========================

    def _judge_rhetoric_with_llm(self, texts: List[str]) -> List[tuple]:
        """
        使用专门的修辞检测模型进行评估
        输入：List[str] - 需要检测的文本列表
        返回: List[tuple(bool, Optional[str])] - [(是否含有修辞, 修辞说明), ...]
        """
        from src.evaluation.scorers import llm_evaluate_rhetoric
        results = llm_evaluate_rhetoric(
            texts=texts,
            client=OpenAI(api_key="EMPTY", base_url="http://10.108.17.160:8080/v1")
        )
        
        return results

    def _score_fluency_with_llm(self, texts: List[str]) -> List[int]:
        """
        批量评估文本流畅度
        参数: texts - List[str] - 需要评估的文本列表
        返回: List[int] - 流畅度分数列表，1表示流畅，0表示不流畅
        """
        from src.evaluation.scorers import llm_evaluate_fluency
        results = llm_evaluate_fluency(
            texts=texts,
            client=self.client
        )
        return results

    def _score_similarity_with_llm(self, sentence_pairs: List[tuple]) -> List[float]:
        """
        批量评估句子对的语义相似度
        参数: sentence_pairs - List[(原句, 改写句), ...]
        返回: List[float] - 相似度分数列表 (0-1)
        """
        from src.evaluation.scorers import llm_evaluate_similarity
        results = llm_evaluate_similarity(
            sentence_pairs=sentence_pairs,
            client=self.client
        )
        
        return results

