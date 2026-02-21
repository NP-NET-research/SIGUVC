# reward_server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import json
import re
import logging
import os
from logging.handlers import RotatingFileHandler
import statistics
from src.utils import load_json_file

"""训练侧通信代码
def get_rewards_from_server(server_url: str, messages: list[str]) -> list["torch.Tensor"]:
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)
"""

app = FastAPI()


class RewardRequest(BaseModel):
    model: str
    messages: List[str]

class RhetoricRewardServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 38294, work_dir: str = "saves"):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Rhetoric Reward API Server")

        os.makedirs(work_dir, exist_ok=True)
        self.logger = self._setup_logger(os.path.join(work_dir, "rhetoric_reward_server.log"))
        self.sample_log_path = os.path.join(work_dir, "rhetoric_rewrite_samples.jsonl")

        # 仅在类内部构建并持有标签映射
        self.input_to_label = self._build_label_mapping()
        self.global_batch_id = 0

        self._register_routes()

    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("RhetoricRewardServer")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        logger.info(f"日志初始化完成，输出文件：{log_file}")
        return logger

    def _build_label_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        读取训练数据与标注数据，构建从句子到标注的映射：
        { sentence: { 'rhetoric_type': str, 'explanation': str } }
        """
        mapping: Dict[str, Dict[str, str]] = {}
        train_path = "/home/ljl/zhipu/data/train_ppo.json"
        label_path = "/home/ljl/zhipu/data/rhetoric_train_data.json"
        try:
            training_data = load_json_file(train_path) or []
            labeled_data = load_json_file(label_path) or []
        except Exception as e:
            self.logger.error(f"加载数据集失败: {e}", exc_info=True)
            return mapping

        cnt_total = 0
        cnt_mismatch = 0
        for train_example, label_example in zip(training_data, labeled_data):
            cnt_total += 1
            sentence = (label_example.get('sentence') or "").strip()
            m = re.search(r"句子：(.*?)\n判断：", str(train_example.get("input", "")))
            if not m:
                cnt_mismatch += 1
                self.logger.warning(f"训练样本缺少句子段或格式不匹配：{train_example.get('input', '')[:80]}...")
                continue
            train_sentence = m.group(1).strip()
            if sentence != train_sentence:
                cnt_mismatch += 1
                self.logger.warning(f"Sentence mismatch: {sentence} != {train_sentence}")

            mapping[sentence] = {
                "rhetoric_type": label_example.get("type", "").strip(),
                "explanation": label_example.get("explanation", "").strip()
            }

        self.logger.info(f"标签映射构建完成，共 {len(mapping)} 条，检测到格式/对齐问题 {cnt_mismatch}/{cnt_total}")
        return mapping

    @staticmethod
    def extract_model_output(msg: str) -> Dict[str, object]:
        """
        提取：
        - original_sentence: 输入句子
        - model_tokens: 模型回答tokens（在“判断：<|assistant|>”之后）
        """
        result = {
            "original_sentence": "",
            "model_tokens": []
        }
        # 提取句子
        sent_match = re.search(r"句子：(.*?)\n判断：", msg, re.DOTALL)
        if sent_match:
            result["original_sentence"] = sent_match.group(1).strip()

        # 提取回答
        # 与训练侧保持一致：在 '判断：<|assistant|>' 之后为回答
        resp_text = ""
        if "判断：<|assistant|>" in msg:
            resp_text = msg.split("判断：<|assistant|>", 1)[-1].strip()
        else:
            # 回退：若未包含分隔符，尝试直接在“判断：”后取文本
            fallback_match = re.search(r"判断：\s*(.*)$", msg, re.DOTALL)
            if fallback_match:
                resp_text = fallback_match.group(1).strip()

        if resp_text:
            result["model_tokens"] = resp_text.split()

        return result

    def _log_responses(self, records: List[Dict[str, object]]):
        """
        将解析后的样本记录写入 JSONL。
        字段：
        - text: 原始句子
        - prediction: 模型回答文本（由 tokens 拼接）
        - label: 标注是否有修辞（是/否）
        - reward: 本样本奖励
        - type/explanation: 标注元信息（若有）
        """
        try:
            with open(self.sample_log_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"写入样本日志失败: {e}", exc_info=True)

    def calculate_reward(self, messages: List[str]) -> List[float]:
        """
        - 解析句子与回答tokens
        - 使用标注映射判断正误：正确+1，错误-1，无标注0
        - 记录样本日志
        """
        parsed = [self.extract_model_output(m) for m in messages]
        rewards: List[float] = []
        log_records: List[Dict[str, object]] = []

        for p in parsed:
            sentence = p.get("original_sentence", "")
            resp_tokens = p.get("model_tokens", [])
            resp_text = " ".join(resp_tokens) if resp_tokens else ""

            reward = 0.0
            label_info = self.input_to_label.get(sentence)
            if label_info:
                label_affirm = bool(label_info.get("rhetoric_type", "").strip())
                pred_affirm = resp_text.startswith("是")
                reward = 1.0 if pred_affirm == label_affirm else -1.0
            else:
                reward = 0.0  # 无标注不评分

            rewards.append(reward)
            log_records.append({
                "text": sentence,
                "prediction": resp_text,
                "label": "是" if (label_info and bool(label_info.get("rhetoric_type", "").strip())) else ("未知" if not label_info else "否"),
                "type": (label_info or {}).get("rhetoric_type", ""),
                "explanation": (label_info or {}).get("explanation", ""),
                "reward": reward
            })

        # 写样本日志
        self._log_responses(log_records)
        return rewards

    def _register_routes(self):
        @self.app.post("/reward")
        async def reward_endpoint(request: RewardRequest):
            rewards = self.calculate_reward(request.messages)
            batch_size = len(request.messages)
            reward_mean = (sum(rewards) / len(rewards)) if rewards else 0.0
            reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

            self.logger.info(f"global batch id: {self.global_batch_id}, batch_size: {batch_size}")
            self.logger.info(f"Reward mean={reward_mean:.4f}, Reward std={reward_std:.4f}")
            self.global_batch_id += 1
            return {"scores": rewards}

    def run(self):
        self.logger.info(f"🚀 Rhetoric reward server running at http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

if __name__ == "__main__":
    server = RhetoricRewardServer(host="127.0.0.1", port=38294, work_dir="saves")
    server.run()
