from dataclasses import dataclass, field
from typing import Optional
import os
import json
import pickle
import jieba
import gc
import logging
from tqdm import tqdm
import copy
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from dataset_dataloader import MyDataset, Mycollater
from GLM4_API.GLM4API import GLM4_API
from checkpoint import CheckpointManager
from Wordbank.wordbank import WordEmbeddingMatcher, evaluate_wordbank
from cos_sim import SemanticSimilarityScorer
from Tensorboard.tensorboard_logger import TensorBoardLogger

logging.basicConfig(
    filename='Res/upload_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name_or_path: str = "../glm-4-9b-chat-hf"
    upcast_layernorm: bool = False

@dataclass
class GenerationConfig:
    do_sample: bool = True
    max_new_tokens: int = 512
    min_length: int = -1
    temperature: float = 0.7
    top_k: float = 0.0
    top_p: float = 1.0

@dataclass
class TrainingConfig:
    epochs: int = 4
    global_batch_size: int = 2
    mini_batch_size: int = 1
    learning_rate: float = 3e-6
    num_epochs: int = 4
    logging_dir: str = "logs"
    empty_cache_steps: int = 50
    logging_steps: int = 1
    save_steps: int = 50
    output_dir: str = "output"
    model_output_dir: str = "output/model"

if __name__ == "__main__":
    model_cfg = ModelConfig()
    gen_cfg = GenerationConfig()
    train_cfg = TrainingConfig(
        epochs=3
    )
    checkpoint_manager = CheckpointManager(output_dir=train_cfg.output_dir, model_name_or_path=model_cfg.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=True,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="fp4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base_model, tokenizer, step_count = checkpoint_manager.load_checkpoint(base_model, tokenizer)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(base_model, lora_config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        peft_model,
        trust_remote_code=True,
        load_in_4bit=True, 
    )

    model.is_peft_model = True
    model.config.use_cache = False
    model.train()

    ref_model = copy.deepcopy(base_model).eval().requires_grad_(False)

    ppo_trainer_cfg = PPOConfig(
        model_name=model_cfg.model_name_or_path,
        ppo_epochs=train_cfg.epochs,
        batch_size=train_cfg.global_batch_size,
        mini_batch_size=train_cfg.mini_batch_size,
        learning_rate=train_cfg.learning_rate,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        target_kl=10,
        cliprange_value=0.25,
        vf_coef=0.3,
        max_grad_norm=1.0,
        optimize_cuda_cache=True,
        remove_unused_columns=False,
        kl_penalty="abs",
        init_kl_coef=0.05,

    )

    ppo_trainer = PPOTrainer(
        config=ppo_trainer_cfg,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
    )
    
    encoded_dataset = MyDataset(
        prompt_path="./Res/paraphrased_system_prompt.txt",
        data_path="./Res/data_1000_modern.json",
        tokenizer_path=model_cfg.model_name_or_path,
        device='cuda:0'
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=encoded_dataset,
        batch_size=train_cfg.global_batch_size,
        shuffle=True,
        collate_fn=Mycollater,
        drop_last=True,
    )

    semantic_prompt = """
    你是一个语义相似度判断助手。你的任务是判断以上两个句子在语义上的相似度，并给出一个范围为0到1的分数，分数随相似程度递减，从1.0表示句子语义完全相似，到0表示语义完全不相似。
    以下是一些评分示例，请参考：
    1. 句子1: 张三是无事不登三宝殿。句子2: 如果张三能来，一定有原因。输出：1.0。
    2. 句子1: 西藏的秏牛，内蒙的骆驼，早像春风一般巡视过高原和沙漠。句子2: 西藏的牛和内蒙的骆驼，早已走遍了高原和沙漠。输出：0.8。
    3. 句子1: 不敢说，要是咱们里面有人露出去，我可怕挨黑枪。句子2: 我敢说，要是咱们里面有人泄露出去，我才不害怕被人暗中算计。输出：0。
    4. 句子1: 我宁愿挨生活的巴掌，不愿意委屈自已的心。句子2: 我不愿意遭受生活的打击，也不愿意违背自己的内心。输出：0。
    5. 句子1: 我失去了地位，保住了人格。句子2: 我失去了地位，却保住了人格。输出：0.4。
    请只回复分数，不要输出多余内容。
    """
    expression_prompt = """
    请判断以上句子是否包含非字面表达（修辞手法/引申义），按以下规则评分：
    0.0 = 存在明显的比喻/拟人/夸张/双关/反语/借代等修辞
    1.0 = 存在不太明显的或是无法避免的，不用它就无法很好表达句子原意的修辞
    1.0 = 仅字面含义的直述
    示例：
    1. "钢产量第7，煤第3..." 输出：1.0（平铺直叙）
    2. "芝麻大的官" 输出：0.0（夸张+比喻）
    3. "他走得很快" 输出：1.0（字面含义直述）
    4. "我的梦想插上了翅膀" 输出：0.0（"翅膀"代指机会）
    5. "这些眼睛们似乎连成一气，已经在那里咬他的灵魂。" 输出：0.0（超现实描述）
    6. "西藏的耗牛，内蒙的骆驼早已如同春风样巡游过高原和沙漠。" 输出：1.0（无法避免）
    7. "我冲入这黑绵绵的昏夜，为要寻一颗明星；为要寻一颗明星，我冲入这黑绵绵的昏夜。" 输出：0.0（排比/列举分承）
    8. "在人类生活历史的深处，有一些事物会形成闪耀夺目的宝石。" 输出：1.0（不太明显）
    请只回复分数，不要输出多余内容。
    """
    vocabulary_prompt = """
    你是一个文本处理助手，请严格按以下规则处理输入的分词文本：
    1. 识别并移除所有专有名称、命名性词汇（人名、地名、机构名等），注意，你我他之类的这种是要保留的
    2. 识别并移除所有标点符号（包括中文的标点符号，逗号、句号、问号、引号、感叹号、分号等）
    3. 保留原始的空格分隔格式
    4. 仅输出处理后的分词结果，不要添加任何解释或额外内容

    示例：
    1. 输入："今天 天气 很好 ， 李四 和 王五 去 公园 玩 。" 输出："今天 天气 很好 和 去 玩"
    2. 输入："他 叫 张三 ， 在 北京理工大学 学习 。" 输出："他 叫 在 学习"
    3. 输入："苹果 公司 位于 美国 加州 。" 输出："苹果 公司 位于"
    """

    api_key = 'e1460eea1141212d4d200e18bf5422c1.GJeATw66ztnAyh9M'
    glm4_api = GLM4_API(
        api_key=api_key,
        batch_size=train_cfg.global_batch_size,
        semantic_prompt=semantic_prompt,
        expression_prompt=expression_prompt,
        vocabulary_prompt=vocabulary_prompt
    )

    generation_kwargs = {
        "do_sample": gen_cfg.do_sample,
        "max_new_tokens": gen_cfg.max_new_tokens,
        "min_length": gen_cfg.min_length,
        "top_k": gen_cfg.top_k,
        "top_p": gen_cfg.top_p,
        "temperature": gen_cfg.temperature,
        "pad_token_id": tokenizer.pad_token_id,
    }

    vocab_file = "/home/zhipu_fund/Debug/Wordbank/dict_meaning_full.json"
    serialized_file = "/home/zhipu_fund/Debug/Wordbank/vocab.pkl"
    matcher = WordEmbeddingMatcher(vocab_file, serialized_file)

    os.makedirs(train_cfg.logging_dir, exist_ok=True)
    log_file = os.path.join(train_cfg.logging_dir, "training.log")
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    step_count = 0
    tensorboard_logger = TensorBoardLogger(log_dir=train_cfg.logging_dir)
    logger = tensorboard_logger.setup_logger(log_file)  # 复用您的日志设置

    
    for epoch in tqdm(range(train_cfg.epochs), desc="Epoch",position=0):
        flag_step = 0
        for batch in tqdm(dataloader, desc="Batch",position=1):
            torch.cuda.empty_cache()
            step_count += 1
            device = next(model.parameters()).device

            response_tensors = ppo_trainer.generate(
                batch['batch_input_ids'],
                **generation_kwargs,
                output_hidden_states=False,
                output_attentions=False,
                return_prompt=False,
                
            )
            responses = [tokenizer.decode(r.cpu(), skip_special_tokens=True) for r in response_tensors]

            rewritten_sentences = []
            tokenized_results = []

            for resp in responses:
                rewrite_pattern = r"1\.\s*改写[：:\s]*(.*?)(?:\s*2\.|\s*分词|$)"
                tokenize_pattern = r"2\.\s*分词[：:\s]*(.*)"
                
                rewrite_match = re.search(rewrite_pattern, resp, re.DOTALL)
                rewrite_text = rewrite_match.group(1).strip() if rewrite_match else ""
                tokenize_match = re.search(tokenize_pattern, resp, re.DOTALL)
                tokenize_text = tokenize_match.group(1).strip() if tokenize_match else ""
                
                if not tokenize_text:
                    lines = resp.strip().split('\n')
                    if lines:
                        tokenize_text = lines[-1].strip()
                if not rewrite_text:
                    rewrite_text = resp.strip()
                    if tokenize_text and tokenize_text in rewrite_text:
                        rewrite_text = rewrite_text.replace(tokenize_text, "").strip()
                rewritten_sentences.append(rewrite_text)
                tokenized_results.append(tokenize_text)

            tokenized_results2 = glm4_api.vocabulary_filtering(tokenized_results)
            


            print(f"step = {step_count}")
            print(f"inputs = {batch['batch_original_sentence']}")
            print(f"改写 = {rewritten_sentences}")
            print(f"分词 = {tokenized_results2}")
            print(f"tokenized = {tokenized_results2}")
            print("=" * 32)

            wordbank_results = evaluate_wordbank(tokenized_results2, matcher)
            expression_results = glm4_api.evaluate_expression_style(rewritten_sentences)

            avg_wordbank = sum(wordbank_results) / len(wordbank_results)
            avg_expression = sum(expression_results) / len(expression_results)

            rewards = []
            for wb, expr in zip(wordbank_results, expression_results):
                wb = round(wb, 3)
                expr = round(expr, 3)
                reward = round((wb * 0.6) + (expr * 0.4), 3)
                print(f"wb = {wb:.3f}, expr = {expr:.3f}, reward = {reward:.3f}")
                rewards.append(torch.tensor(reward, device=device))
            avg_reward = torch.stack(rewards).mean().item()

            stats = ppo_trainer.step(batch['batch_input_ids'], response_tensors, rewards)
            stats["custom/reward"] = avg_reward
            stats["custom/wordbank"] = avg_wordbank
            stats["custom/expression"] = avg_expression
            kl_div = stats.get("objective/kl", 0)
            print(f"当前KL散度: {kl_div:.3f}, β={ppo_trainer.config.init_kl_coef:.3f}")

            if step_count % train_cfg.logging_steps == 0:
                avg_loss = stats.get("ppo/loss/total", 0)
                logger.info(f"Epoch {epoch} Step {step_count}: Loss={avg_loss}, Reward={avg_reward}")
                logger.info(f"日志已保存到 {log_file}")
                tensorboard_logger.log_stats(step=step_count, stats=stats)

            if step_count % train_cfg.save_steps == 0:
                checkpoint_manager.save_checkpoint(model, tokenizer, step_count)
            
            if step_count % train_cfg.empty_cache_steps == 0:
                torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.memory_reserved() > 0.9 * torch.cuda.max_memory_reserved():
                    torch.cuda.memory._release_cached_allocator()
        gc.collect()
        torch.cuda.empty_cache() 

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    model.save_pretrained(train_cfg.output_dir)  
    tokenizer.save_pretrained(train_cfg.output_dir)  
    print(f"模型与分词器已保存到 {train_cfg.output_dir}，本次训练结束。")
    tensorboard_logger.close()
