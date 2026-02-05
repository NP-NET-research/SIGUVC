import gc
import os
from typing import List

from openai import OpenAI
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
from llamafy import merge, train_dpo
from src.utils import load_json_file, load_jsonl_file, load_lines_file, save_json_file, setup_save_directory, save_jsonl_file
from tqdm import tqdm
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def dpo_pipeline():
    from src.inference.generator import BaseGenerator, GenerationConfig
    from src.rewrite_service import SentenceRewriteService

    def preprocess_data(data, work_dir):

        # 读取词表并去除vocab长度大于500的样本
        max_vocab_size = 500
        
        data = [d for d in data if len(d['vocab']) <= max_vocab_size]

        queries = [d['query'] for d in data]
        vocab = [d['vocab'] for d in data]
        
        # 提取命名实体
        from src.nlp_utils import vllm_extract_entities
        all_entities = vllm_extract_entities(queries)
        for i, d in enumerate(data):
            d['entities'] = list(all_entities[i])
        
        # 保存预处理结果
        preprocessed_path = os.path.join(work_dir, "preprocessed_data.json")
        save_json_file(data, preprocessed_path)
        
        return data

    def sample_data(generator, queries, vocabs_strs):
    
        rewrite_service = SentenceRewriteService(generator)

        sample_results = rewrite_service.single_round_rewrite(
            texts=queries,
            retrieve_words_strs=vocabs_strs,
        )
        # 保存采样结果
        data_sample = []
        for query, vocabs_str, sample in zip(queries, vocabs_strs, sample_results):
            data_sample.append({
                'query': query,
                'vocab_str': vocabs_str,
                'samples': sample,
            })
        sample_output_path = os.path.join(work_dir, "sampled_data.json")
        save_json_file(data_sample, sample_output_path)
        
        return data_sample
    
    def score_samples(sampled_data):

        # 准备评测数据, 展平采样结果, 便于批量评分
        queries_flat = []
        samples_flat = []
        vocab_flat = []
        for item in sampled_data:
            query = item['query']
            vocabs_str = item['vocab_str']
            vocab = vocabs_str.split('、')
            samples = item['samples']
            for sample in samples:
                queries_flat.append(query)
                samples_flat.append(sample)
                vocab_flat.append(vocab)
        
        print(f"准备评分的样本数量: {len(samples_flat)}")
        
        # 评分：词匹配度、流畅度
        vocab_scores = []
        fluency_scores = []

        # 第一步：分词和词表匹配率评测
        for i in range(len(samples_flat)):
            rewritten = samples_flat[i]
            mini_vocab = vocab_flat[i]
            tokens = rewritten.split(' ')
        
            total = len(tokens)
            if total > 0:
                legal_tokens = [t for t in tokens if t in mini_vocab]
                # 100%匹配得1分，否则得0分
                vocab_match = 1 if len(legal_tokens) == total else 0
            else:
                vocab_match = 0
            vocab_scores.append(vocab_match)
        
        # 第二步：流畅度评分（调用评分模型API）
        from src.evaluation.scorers import vllm_evaluate_fluency
        fluency_scores = vllm_evaluate_fluency(samples_flat)

        # 聚合每个query的样本及其评分
        query_to_samples = {}
        for i in range(len(samples_flat)):
            query = queries_flat[i]
            vocab = vocab_flat[i]
            sample = samples_flat[i]
            vocab_score = vocab_scores[i]
            fluency_score = fluency_scores[i]
            if query not in query_to_samples:
                query_to_samples[query] = []
            query_to_samples[query].append({
                'vocab_str': '、'.join(vocab),
                'sample': sample,
                'vocab_score': vocab_score,
                'fluency_score': fluency_score,
            })
        
        # vocab_score 和 fluency_score 均为1分的样本数量
        successful_count = 0
        for sample_infos in query_to_samples.values():
            for sample_info in sample_infos:
                if sample_info['vocab_score'] == 1 and sample_info['fluency_score'] == 1:
                    successful_count += 1
        
        print(f"满分样本数: {successful_count}")

        save_json_file(query_to_samples, os.path.join(work_dir, "scored_samples.json"))
        return dict(query_to_samples)
    
    def generate_dpo_data(query_to_samples):

        dpo_collection = []
        for query, sample_infos in query_to_samples.items():
            
            vocabs_str = sample_infos[0]['vocab_str']

            # 按评分之和降序排序
            sample_infos.sort(key=lambda x: (x['vocab_score'], x['fluency_score']), reverse=True)
            if len(sample_infos) < 2:
                continue
                
            # 确保chosen样本满足词表覆盖和流畅度均为1分
            chosen_sample = None
            for sample_info in sample_infos:
                if sample_info['vocab_score'] == 1 and sample_info['fluency_score'] == 1:
                    chosen_sample = sample_info['sample']
                    break
            
            # 如果没有找到满足条件的chosen样本，跳过这个query
            if chosen_sample is None:
                continue
                
            rejected_sample = sample_infos[-1]['sample']
            
            # 确保chosen和rejected不同
            if chosen_sample == rejected_sample:
                continue
            
            from src.inference.prompt import PromptRegistry
            rewrite_prompt_template = PromptRegistry.get("rewrite")
            dpo_collection.append({
                'system': rewrite_prompt_template.system,
                'instruction': rewrite_prompt_template.instruction,
                'input': """【句子】\n{sentence}\n【词表】\n{vocab}\n\n输出：""".format(sentence=query, vocab=vocabs_str),
                'history': [],
                'chosen': chosen_sample,
                'rejected': rejected_sample,
            })
        
        save_json_file(dpo_collection, os.path.join(work_dir, "train_dpo.json"))
        print(f"生成偏好对数量: {len(dpo_collection)}")

        return
    
    # =========================== 配置参数 ============================
    # 模型
    model_name_or_path = "saves/dpo_pipeline/glm4-9b-dpo-merged_1"
    # model_name_or_path = "/home/ljl/Data/LLM/GLM-4-9B-0414"
    score_model_path = '/home/ljl/Data/LLM/GLM-4-32B-0414'
    score_model_api = 'http://localhost:8080/v1'  # 本地部署的评分模型API地址
    
    # 数据路径
    train_path = "corpus/train_with_vocab.jsonl"
    test_path = "corpus/test_with_vocab.jsonl"

    # 推理参数
    num_sequences = 1  # 采样序列数
    config = GenerationConfig(
            model_name_or_path=model_name_or_path,
            use_vllm=True,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=False,
            num_sequences=num_sequences,
            vllm_config={
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            }
        )
    
    work_dir = 'saves/dpo_pipeline'
    os.makedirs(work_dir, exist_ok=True)
    
    # 读取文本数据
    data = load_jsonl_file('corpus/classified_results.jsonl')
    
    # ============================ 数据预处理 ============================
    
    if not os.path.exists(os.path.join(work_dir, "preprocessed_data.json")):
        data = preprocess_data(data, work_dir)
    else:
        data = load_json_file(os.path.join(work_dir, "preprocessed_data.json"))

    print(f"数据集大小: {len(data)}")
    # 100条作为测试集
    test_data = data[:100]
    train_data = data[100:]

    data = test_data
    queries = [d['query'] for d in data]
    vocabs_strs = ['、'.join(d['entities']+d['vocab']) for d in data]

    # ============================ 采样数据 ============================
    if not os.path.exists(os.path.join(work_dir, "sampled_data.json")):
        generator = BaseGenerator(config)
        sampled_data = sample_data(generator, queries, vocabs_strs)
        del generator
        gc.collect()
    else:
        sampled_data = load_json_file(os.path.join(work_dir, "sampled_data.json"))

    # ============================ 打分 ============================

    if not os.path.exists(os.path.join(work_dir, "scored_samples.json")):
        query_to_samples = score_samples(sampled_data)
    else:
        query_to_samples = load_json_file(os.path.join(work_dir, "scored_samples.json"))
    
    # ============================ 生成偏好对 ============================

    # if not os.path.exists(os.path.join(work_dir, "train_dpo.json")):
    #     generate_dpo_data(query_to_samples)
    
    # train_preference_pairs_path = os.path.join(work_dir, "train_dpo.json")

    # ============================ DPO训练 ============================

    # dpo_adapter_path = train_dpo(
    #     model_name_or_path=model_name_or_path, 
    #     train_dataset_path=train_preference_pairs_path, 
    #     base_dir=work_dir,
    #     config_path="configs/train_dpo.yaml"
    # )
    
    # print(f"DPO训练完成，adapter保存在: {dpo_adapter_path}")

    
    # ============================= 模型合并 ============================

    # merged_model_name = "glm4-9b-dpo-merged"
    
    # merged_model_path = merge(
    #     model_name_or_path=model_name_or_path,
    #     adapter_name_or_path=dpo_adapter_path,
    #     save_model_name=merged_model_name,
    #     base_dir=work_dir,
    #     config_path="configs/merge_adapter.yaml"
    # )
    # print(f"模型合并完成，保存在: {merged_model_path}")


    # ============================ 模型评测 ============================
    
    # test_data = load_lines_file(test_path)
    # test_queries = [line.strip() for line in test_data if line.strip()]
    # test_queries = test_queries

    # print("DPO Pipeline 完成!")
    # print(f"工作目录: {work_dir}")
    return


def ppo_pipeline():

    model_name_or_path = "/home/ljl/Data/LLM/GLM-4-9B-0414"
    # model_name_or_path = 'saves/dpo_pipeline/glm4-9b-dpo-merged_0'
    
    # 数据路径
    # train_path = "corpus/train.json"
    train_path = "corpus/train_with_vocab.json"
    test_path = "corpus/test_with_vocab.json"

    # work_dir = setup_save_directory(base_dir="./saves", prefix="ppo_pipeline", input_file=train_path)
    work_dir = "saves/ppo_pipeline"
    os.makedirs(work_dir, exist_ok=True)
    
    # ============================ 准备PPO数据 ============================

    from src.inference.prompt import PromptRegistry
    rewrite_prompt_template = PromptRegistry.get("rewrite")
    
    train_data = load_json_file(train_path)
    ppo_data = []
    for df in train_data:

        input_text = """【句子】\n{sentence}\n【词表】\n{vocab}\n\n输出：""".format(
            sentence=df['query'],
            vocab='、'.join(df['entities'] + df['vocab'])
        )
        
        ppo_data.append({
            'system': rewrite_prompt_template.system,
            'instruction': rewrite_prompt_template.instruction,
            'input': input_text,
            'history': [],
            'output': ''
        })

    train_ppo_data_path = os.path.join(work_dir, "train_ppo.json")
    save_json_file(ppo_data, train_ppo_data_path)

    # ============================ 等待reward服务器 ============================
    print("watting for reward server on port 38294...")
    # 查询本地38294端口，等待2分钟
    def _wait_for_port(port, timeout=120):
        import socket
        import time
        start_time = time.time()
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    return
            if time.time() - start_time > timeout:
                raise TimeoutError(f"等待端口 {port} 超时。")
            time.sleep(5)
    
    _wait_for_port(38294, timeout=300)

    # ============================ PPO训练 ============================

    from llamafy import train_ppo
    ppo_adapter_path = train_ppo(
        train_dataset_path=train_ppo_data_path,
        train_config_path="configs/train_ppo.yaml",
        base_dir=work_dir,
    )

    print(f"PPO训练完成，adapter保存在: {ppo_adapter_path}")

    # ============================= 模型合并 ============================
    merged_model_name = f"glm4-9b-ppo-merged" 
    merged_model_path = merge(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=ppo_adapter_path,
        save_model_name=merged_model_name,
        base_dir=work_dir,
        config_path="configs/merge_adapter.yaml"
    )
    print(f"模型合并完成，保存在: {merged_model_path}")

    return merged_model_path


def evaluate_model(merged_model_path):
    
    from src.inference.generator import BaseGenerator, GenerationConfig
    from src.evaluation.scorers import vllm_evaluate_fluency
    from src.rewrite_service import SentenceRewriteService

    # 读取测试数据
    test_data = load_json_file("corpus/test_with_vocab.json")
    test_queries = [d['query'] for d in test_data]
    test_vocabs_strs = ['、'.join(d['entities'] + d['vocab']) for d in test_data]
    print(f"测试集大小: {len(test_queries)}")

    # 构建生成器
    config = GenerationConfig(
            model_name_or_path=merged_model_path,
            use_vllm=True,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=False,
            num_sequences=1,
            vllm_config={
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            }
        )
    generator = BaseGenerator(config)
    sentence_rewriter = SentenceRewriteService(generator)

    # 批量改写
    print("Batch rewriting sentences for evaluation...")
    rewritten_texts = sentence_rewriter.single_round_rewrite(
        texts=test_queries,
        retrieve_words_strs=test_vocabs_strs,
    )
    # 评测流畅度
    fluency_scores = vllm_evaluate_fluency(rewritten_texts)
    
    return


def muti_round_rewrite(train_data):
    work_dir = setup_save_directory(base_dir="./saves", prefix="debug_rewrite", input_file="corpus/train_retrieved_results.json")

    from src.rewrite_service import SentenceRewriteService
    from src.sample import interleave_from_ordered

    per_word_topk = 5
    retrieve_k = 300
    max_rounds = 10

    train_data = train_data[:100]
    
    queries = []
    entities = []
    retrieve_words_strs = []
    mini_vocab = []
    for item in tqdm(train_data, desc="构建检索词表"):
        df: dict= item

        # 使用与训练一致的方式从 ordered_retrieved 聚合词表
        ordered_retrieved = df['ordered_retrieved']  # Dict[str, str]
        ordered_lists = [words_str.split('、') for _, words_str in ordered_retrieved.items()]
        retrieve_words_str = interleave_from_ordered(
            ordered_lists=ordered_lists,
            sep='、',
            dedup=True,
            per_list_limit=per_word_topk,
            total_limit=retrieve_k,
        )
        
        retrieve_words_str = retrieve_words_str

        queries.append(df['query'])
        entities.append(list(set(df['entities'].split(' '))))
        retrieve_words_strs.append(retrieve_words_str)
        mini_vocab.append(set(retrieve_words_str.split('、')))


    sentence_rewriter = SentenceRewriteService(max_rounds=max_rounds,)
    
    # 批量提取命名实体
    print("Extracting named entities...")
    all_entities = sentence_rewriter.extract_named_entities(queries)
    
    # 批量改写
    print("Batch rewriting sentences...")
    results = []
    batch_size = 128
    for i in tqdm(range(0, len(queries), batch_size), desc="Rewriting batches"):
        batch_queries = queries[i:i+batch_size]
        batch_entities = all_entities[i:i+batch_size]
        batch_retrieve_words_strs = retrieve_words_strs[i:i+batch_size]
        batch_mini_vocab = mini_vocab[i:i+batch_size]

        batch_results = sentence_rewriter.batch_rewrite(
            texts=batch_queries,
            ner_tokens_list=batch_entities,
            retrieve_words_strs=batch_retrieve_words_strs,
            mini_vocabs=batch_mini_vocab
        )
        results.extend(batch_results)

    save_json_file(results, os.path.join(work_dir, "debug_rewrite_results.json"))

if __name__ == "__main__":
    # dpo_pipeline()
    ppo_pipeline()


