# reward_server_vllm_openai_batch.py
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from src.inference.prompt import PromptRegistry
import logging
import os
from logging.handlers import RotatingFileHandler
import json 
import statistics
from openai import OpenAI


def repetition_penalty(tokens: List[List[str]], n: int = 3) -> List[float]:
    """
    计算 n-gram 重复惩罚分数。
    返回值范围在 [0, 1]，1 表示无重复，0 表示严重重复。
    """
    penalties = []
    for tok_list in tokens:
        if len(tok_list) < n:
            penalties.append(1.0)
            continue

        # 提取 n-grams
        ngrams = [tuple(tok_list[i:i+n]) for i in range(len(tok_list) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))

        # 重复率
        repeat_ratio = 1 - unique / total if total > 0 else 0.0

        # 惩罚函数：重复越多，惩罚越重
        penalty = 1.0 - repeat_ratio  # 线性惩罚
        penalties.append(max(0.0, min(1.0, penalty)))  # 限制在 [0,1]

    return penalties



class RewardRequest(BaseModel):
    model: str
    messages: List[str]  # query + response 拼接文本列表

class RewardServer:
    def __init__(
            self, 
            host="127.0.0.1", 
            port=38294, 
            vllm_api_base: str ="http://127.0.0.1:8080/v1", 
            rhetoric_api_base: str = "http://10.108.17.160:8080/v1",
            work_dir: str = 'saves', 
            max_concurrency: int = 4
        ):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Reward API Server")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(base_url=vllm_api_base, api_key="EMPTY")
        self.rhetoric_client = OpenAI(api_key="EMPTY", base_url=rhetoric_api_base)
        self.rhetoric_use_template = PromptRegistry.get("rhetoric_detection")
        
        self.logger = self._setup_logger(os.path.join(work_dir, "reward_server.log"))
        self.sample_log_path = os.path.join(work_dir, "rewrite_samples.json")  # 保持 JSON 文件

        self.warmup_batches = 100  # 预热批次数量
        self.alpha = 0.2   # 相似度 vs 词表覆盖率权重
        self.beta = 0.6    # 重复惩罚强度（越大惩罚越重）
        self.similarity_threshold = 0.4  # 相似度阈值
        self.vocab_coverage_threshold = 0.5  # 词表覆盖率阈值
        
        self.similarity_gamma = 2.0
        self.vocab_coverage_gamma = 2.0

        self.global_batch_id = 0
        self.max_concurrency = max(1, int(max_concurrency))
        self._register_routes()

    def _setup_logger(self, log_file: str = None) -> logging.Logger:
        """
        初始化文件与控制台日志，文件按 10MB 轮转，保留 5 份。
        """
        logger = logging.getLogger("RewardServer")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)

        if log_file is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            log_dir = os.path.join(base_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "reward_server.log")

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        logger.info(f"日志初始化完成，输出文件：{log_file}")
        return logger

    def _log_responses(self, parsed: List[Dict[str, object]]):
        """
        将解析后的回答写入 JSON 文件（数组）
        """
        try:
            records = []
            if os.path.exists(self.sample_log_path):
                try:
                    with open(self.sample_log_path, "r", encoding="utf-8") as f:
                        data = f.read().strip()
                        if data:
                            records = json.loads(data)
                            if not isinstance(records, list):
                                records = []
                except Exception:
                    records = []  # 文件损坏或非数组结构时重建

            for p in parsed:
                records.append(p)

            with open(self.sample_log_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"写入样本日志失败: {e}", exc_info=True)
    
    @staticmethod
    def extract_model_output(texts: List[str]) -> List[Dict[str, object]]:
    
        from src.utils import parse_glm4_chat_template
        messages = [parse_glm4_chat_template(text) for text in texts]

        """conv = [
                {"role": "system", "content": self.rewrite_prompt_template.system},
                {"role": "user", "content": self.rewrite_prompt_template.instruction + \
                    f"【句子】\n{sentence}\n【词表】\n{vocab}\n\n输出：".format(
                    sentence=texts[i],
                    vocab=retrieve_words_strs[i]
                )},
        ]"""
        
        parsed_results = []
        for msg in messages:    
            
            all_user_msgs = [m for m in msg if m['role'] == 'user']
            all_assistant_msgs = [m for m in msg if m['role'] == 'assistant']

            user_msg = all_user_msgs[0]            
            parsed = {"original": "", "vocab": [], "response": ""}
            
            user_content = user_msg.get('content', '')
            original_match = re.search(r'【句子】\s*(.*?)\s*【词表】', user_content, re.DOTALL)
            vocab_match = re.search(r'【词表】\s*(.*?)\s*输出：', user_content, re.DOTALL)
            parsed['original'] = original_match.group(1).strip()
            vocab_str = vocab_match.group(1).strip()
            parsed['vocab'] = [w.strip() for w in vocab_str.split('、') if w.strip()]
            
            assistant_msg = all_assistant_msgs[-1]
            parsed['response'] = assistant_msg.get('content', '').strip()
            
            parsed_results.append(parsed)
        
        return parsed_results

    
    def _compute_overall_reward(
        self,
        similarity_rewards: List[float],
        vocab_coverage_rewards: List[float],
        repetition_penalties: List[float],
    ) -> List[float]:
        """
        计算综合得分（软阈值折扣）：
        - 先计算 base = alpha * similarity + (1 - alpha) * vocab
        - 对 base 乘以两个折扣项：
            sim_discount = 1.0                 (if sim >= thr)    else (sim / thr) ** similarity_gamma
            vocab_discount = 1.0               (if vocab >= thr)  else (vocab / thr) ** vocab_coverage_gamma
        - 再叠加重复惩罚：reward = base * sim_discount * vocab_discount * (1 - beta * (1 - repetition_penalty))
        """
        def soft_discount(x: float, thr: float, gamma: float) -> float:
            # thr<=0 时不打折；x>=thr 不打折；否则按幂次折扣
            if thr is None or thr <= 0:
                return 1.0
            if x >= thr:
                return 1.0
            ratio = max(0.0, x) / thr
            return pow(ratio, max(1.0, float(gamma)))

        rewards = []
        for sim, vocab, rep_pen in zip(similarity_rewards, vocab_coverage_rewards, repetition_penalties):
            base = self.alpha * sim + (1 - self.alpha) * vocab

            # 相似度与覆盖率折扣（替代原硬截断）
            sim_disc = soft_discount(sim, self.similarity_threshold, self.similarity_gamma)
            vocab_disc = soft_discount(vocab, self.vocab_coverage_threshold, self.vocab_coverage_gamma)

            # 重复惩罚
            rep_factor = (1 - self.beta * (1 - rep_pen))

            reward = base * sim_disc * vocab_disc * rep_factor

            rewards.append(reward)

        return rewards

    def calculate_reward(self, messages: List[str]) -> List[float]:
        """
        使用 extract_model_output 逐条解析；词表覆盖率直接用tokens，语义相似度时拼接tokens。
        并将每条样本的输入、改写、词表与得分写入日志文件（JSONL）。
        """
        parsed = self.extract_model_output(messages)
        assert len(parsed) == len(messages), "解析结果数量与输入数量不匹配！"

        original_sents = [p.get("original", "") for p in parsed]
        responses_sents = [p.get("response", []) for p in parsed]
        responses_tokens = [s.split(' ') for s in responses_sents]
        vocabs = [p.get("vocab", []) for p in parsed]

        # ========== 相似度得分 ==========
        # responses_sents = ["".join(tokens) for tokens in responses_tokens]
        # similarity_rewards = self._caculate_similarity_rewards(original_sents, responses_sents)

        # ========== 是否流畅评分 ==========
        fluency_rewards = self._caculate_fluency_rewards(responses_sents)

        # ========== 词汇覆盖率得分 ==========
        vocab_coverage_rewards = self._caculate_vocab_coverage_rewards(responses_tokens, vocabs)

        # ========== 有无修辞得分 ==========
        # rhetoric_rewards = self._caculate_rhetoric_rewards(responses_sents)

        # ========== 写入日志 ==========
        self._log_responses([
            p.update({
                'vocab': "、".join(list(vocabs[i])),
                "vocab_coverage_reward": vocab_coverage_rewards[i],
                "fluency_reward": fluency_rewards[i],
                # "similarity_reward": similarity_rewards[i],
                # "rhetoric_reward": rhetoric_rewards[i],
            }) or p
            for i, p in enumerate(parsed)
        ]
        )

        # ========== 综合得分 ==========
        rewards = [
            0.5*f + 0.5*v
            for f, v in zip(fluency_rewards, vocab_coverage_rewards)
        ]

        return rewards

    # ======================== 评测 ========================
    def _caculate_similarity_rewards(self, original_sents: List[str], responses_sents: List[str]) -> List[float]:
        """
        计算相似度得分，返回列表。
        """
        sentence_pairs = [
            (orig, res)
            for orig, res in zip(original_sents, responses_sents)
        ]
        from src.evaluation.scorers import llm_evaluate_similarity
        similarity_rewards = llm_evaluate_similarity(
            sentence_pairs=sentence_pairs,
            client=self.client
        )
        return similarity_rewards

    def _caculate_vocab_coverage_rewards(self, responses_tokens: List[List[str]], vocabs: List[List[str]]) -> List[float]:
        """
        计算在词库中的词汇覆盖率得分，返回列表。
        """
        vocab_coverages = []
        for tokens, vocab in zip(responses_tokens, vocabs):
            legal_tokens = [t for t in tokens if t in vocab]
            coverage = len(legal_tokens) / len(tokens) if tokens else 0.0
            vocab_coverages.append(coverage)
        
        # 没有达到覆盖阈值的词汇，奖励为0, # 预热阶段不过滤
        if self.global_batch_id < self.warmup_batches:
            vocab_coverage_rewards = vocab_coverages
        else:
            vocab_coverage_rewards = [
                vc if vc >= self.vocab_coverage_threshold else 0.0
                for vc in vocab_coverages
            ]    
        return vocab_coverage_rewards

    def _caculate_fluency_rewards(self, responses_sents: List[str]) -> List[float]:
        """
        计算流畅性得分，返回列表。
        """
        from src.evaluation.scorers import llm_evaluate_fluency
        fluency_rewards = llm_evaluate_fluency(
            sentences=responses_sents,
            client=self.client
        )
        return fluency_rewards

    def _caculate_rhetoric_rewards(self, responses_sents: List[str], ) -> List[float]:
        """
        计算修辞手法得分，返回列表。
        """
        from src.evaluation.scorers import llm_evaluate_rhetoric
        rhetoric_rewards = llm_evaluate_rhetoric(
            sentences=responses_sents,
            client=self.rhetoric_client
        )
        return rhetoric_rewards


    # ======================== API 路由 ========================
    def _register_routes(self):
        @self.app.post("/reward")
        async def reward_endpoint(request: RewardRequest):
            rewards = self.calculate_reward(request.messages)

            batch_size = len(request.messages)
            avg_reward = (sum(rewards) / len(rewards)) if rewards else 0.0

            reward_mean = avg_reward
            reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

            self.logger.info(f'golbal batch id: {self.global_batch_id}, batch_size: {batch_size}')
            self.logger.info(f"Reward mean={reward_mean:.4f}, Reward std={reward_std:.4f})")
            self.global_batch_id += 1
            return {"scores": rewards}

    def run(self):
        self.logger.info(f"🚀 Reward server running at http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

if __name__ == "__main__":
    server = RewardServer(host="127.0.0.1", port=38294,
                          vllm_url="http://127.0.0.1:8000/v1/chat/completions",
                          work_dir="saves")
    server.run()
