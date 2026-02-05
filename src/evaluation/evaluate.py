import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.sample import retrieve_generate_sample
from src.utils import load_json_file, load_jsonl_file, save_json_file, setup_save_directory
from src.evaluation.scorers import (
        evaluate_semantic_api as remote_sem_eval,
        evaluate_semantic_local,
        evaluate_rhetoric_usage,
        evaluate_vocabulary_coverage,
        evaluate_minivocabulary_coverage,
    )

def calculate_final_score(fluency: float, similarity: float, simplicity: float, coverage: float, rhetoric: float) -> float:
    """计算综合得分"""
    final = 0.0
    # final = (fluency + similarity + simplicity)/3 + coverage - rhetoric
    alpha = 0.5
    final = alpha * (fluency + similarity + simplicity)/3 + (1 - alpha) * (coverage - rhetoric)
    
    return round(final, 4)


def batch_evaluate_rewrites(
    rewrites_grouped: List[List[str]],
    references: List[str],
    retrieve_words_flat: List[set[str]],
    tokens_flat: List[List[str]],
    wordbank_path: str = "corpus/wordbank/Words-List.txt",
    api_key: Optional[str] = None,
    fluency_model_path: Optional[str] = '/home/ljl/Data/LLM/GLM-4-32B-0414',
    rhetoric_model_path: str = 'checkpoints/glm4-9b-0414-rhetoric-use',
) -> Dict[str, Any]:
    """
    对候选集执行综合评估（先本地修辞/词汇覆盖，再语义三分）
    返回：
      - scores_flat: List[Dict]，每个候选的五维分数与总分
      - group_lengths: List[int]，与 rewrites_grouped 对齐的分组长度
    """
    assert len(rewrites_grouped) == len(references), "候选与参考数量不匹配"

    # 展开为扁平输入
    group_lengths = [len(g) for g in rewrites_grouped]
    all_rewrites_flat = [c for g in rewrites_grouped for c in g]
    references_flat = [ref for ref, g in zip(references, rewrites_grouped) for _ in range(len(g))]

    print(f"开始评估 {len(all_rewrites_flat)} 条候选...")
    
    # 词汇覆盖
    vocab_flat, _, _ = evaluate_minivocabulary_coverage(tokens=tokens_flat, mini_vocab=retrieve_words_flat)
    print("✓ 词汇覆盖率评估完成")

    # 修辞
    rhet_flat = evaluate_rhetoric_usage(all_rewrites_flat, model_path=rhetoric_model_path)
    # rhet_flat = [0.0] * len(all_rewrites_flat)  # 暂时不计算修辞
    print("✓ 修辞评估完成")

    # 语义（API 或 本地），直接返回扁平分数
    if api_key:
        try:
            sem_flat, raw_flat = remote_sem_eval(all_rewrites_flat, references_flat, api_key)
        except Exception as e:
            sem_flat, raw_flat = [{"fluency": 0.0, "similarity": 0.0, "simplicity": 0.0}] * len(all_rewrites_flat), []
    else:
        sem_flat, raw_flat = evaluate_semantic_local(all_rewrites_flat, references_flat, fluency_model_path)

    # raw 执行按 reference 分组
    raw_flu: List[List[Any]] = []
    cursor = 0
    for size in group_lengths:
        raw_flu.append(raw_flat[cursor: cursor + size])
        cursor += size

    # 合并为单对象
    scores_flat = []
    for i, sem in enumerate(sem_flat):
        f = float(sem.get("fluency", 0.0))
        s = float(sem.get("similarity", 0.0))
        sp = float(sem.get("simplicity", 0.0))
        v = float(vocab_flat[i] if i < len(vocab_flat) else 0.0)
        r = float(rhet_flat[i] if i < len(rhet_flat) else 0.0)
        scores_flat.append({
            "fluency": f,
            "similarity": s,
            "simplicity": sp,
            "coverage": v,
            "rhetoric": r,
            "total": calculate_final_score(f, s, sp, v, r),
        })
    print("✓ 语义评估完成")

    return {
        'scores_flat': scores_flat,
        'group_lengths': group_lengths,
        'raw_fluency_outputs': raw_flu,
    }


def evaluate_model_comparison(
    test_queries: List[str],
    base_model_path: str,
    dpo_model_path: str,
    vocab_file: str = 'corpus/wordbank/Words-List.txt',
    vectorizer_model_path: str = "checkpoints/bge-large-zh-v1.5",
    retrieval_k: int = 100,
    api_key: Optional[str] = None,
    work_dir: Optional[str] = None,
    rhetoric_model_path: Optional[str] = None,
    fluency_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对比评估基础模型和DPO模型效果
    """

    # ==================== 初始化路径 ====================
    work_dir = setup_save_directory("saves", "evaluation", "model_comparison") if work_dir is None else work_dir
    os.makedirs(work_dir, exist_ok=True)

    print(f"加载 {len(test_queries)} 条测试数据")

    # ===========================================================
    # Step 1. 模型推理
    # ===========================================================
    def run_inference(model_path: str, queries: List[str], tag: str):
        """对单个模型推理"""
        print(f"\n{'='*60}\n开始推理：{tag}\n路径: {model_path}\n{'='*60}")
        
        results = retrieve_generate_sample(
            texts=queries,
            model_name_or_path=model_path,
            vocab_file=vocab_file,
            vectorizer_model_path=vectorizer_model_path,
            retrieval_k=retrieval_k,
            num_sequences=1,
            do_sample=False,
            use_vllm=True,
        )

        save_json_file(results, os.path.join(work_dir, f"{tag}_samples.json"))
        print(f"✓ {tag} 推理完成，共 {len(results)} 条样本")
        return results

    base_samples = run_inference(base_model_path, test_queries, "base_model")
    dpo_samples = run_inference(dpo_model_path, test_queries, "dpo_model")

    # ===========================================================
    # Step 2. 构建 batch_evaluate_candidates 输入
    # ===========================================================
    def build_inputs(samples: List[Dict[str, Any]]) -> Tuple[List[List[str]], List[str], List[List[str]], List[set[str]]]:
        rewrites_grouped, references, tokens_flat, retrieve_words_flat = [], [], [], []
        for s in samples:
            df = s['data_frame']
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
            rewrites_grouped.append(rewrites)
            references.append(reference)
            tokens_flat.extend(rewrites_tokens_list)
            retrieve_words_flat.extend([set(retrieve_words_list)] * len(rewrites_tokens_list)) # 每个候选对应相同的检索词集合
        return rewrites_grouped, references, tokens_flat, retrieve_words_flat

    base_cg, base_refs, base_tokens_flat, base_retrieve_words_flat = build_inputs(base_samples)
    dpo_cg, dpo_refs, dpo_tokens_flat, dpo_retrieve_words_flat = build_inputs(dpo_samples)

    # ===========================================================
    # Step 3. 调用 batch_evaluate_candidates 获取打分，（合并 base 与 dpo，一次启动评测模型）
    # ===========================================================
    from src.evaluation.evaluate import batch_evaluate_rewrites as _batch_eval
    combined_cg = base_cg + dpo_cg
    combined_refs = base_refs + dpo_refs
    combined_tokens = base_tokens_flat + dpo_tokens_flat
    combined_retrieve_words = base_retrieve_words_flat + dpo_retrieve_words_flat
    combined_res = _batch_eval(
        rewrites_grouped=combined_cg,
        references=combined_refs,
        tokens_flat=combined_tokens,
        retrieve_words_flat=combined_retrieve_words,
        wordbank_path=vocab_file,
        api_key=api_key,
        fluency_model_path=fluency_model_path,
        rhetoric_model_path=rhetoric_model_path,
    )

    # 使用 scores_flat 简化切分
    base_flat_len = sum(len(g) for g in base_cg)
    dpo_flat_len = sum(len(g) for g in dpo_cg)
    scores_flat = combined_res['scores_flat']
    base_scores = scores_flat[:base_flat_len]
    dpo_scores = scores_flat[base_flat_len: base_flat_len + dpo_flat_len]

    # raw_fluency_outputs 按 reference 数切分
    base_ref_len = len(base_refs)
    base_raw_flu = combined_res.get('raw_fluency_outputs', [])[:base_ref_len]
    dpo_raw_flu = combined_res.get('raw_fluency_outputs', [])[base_ref_len: base_ref_len + len(dpo_refs)]

    def _to_stats(tag_key: str, scores: List[Dict[str, float]], raw_flu, total_flat_len: int):
        if not scores:
            stats = {
                "total_samples": 0,
                "avg_fluency": 0.0, "avg_similarity": 0.0, "avg_simplicity": 0.0,
                "avg_coverage": 0.0, "avg_rhetoric": 0.0, "avg_total_score": 0.0,
                "std_fluency": 0.0, "std_similarity": 0.0, "std_simplicity": 0.0,
                "std_coverage": 0.0, "std_rhetoric": 0.0, "std_total_score": 0.0,
            }
        else:
            getv = lambda k: np.array([s.get(k, 0.0) for s in scores], dtype=float)
            flu = getv("fluency"); sim = getv("similarity"); simp = getv("simplicity")
            cov = getv("coverage"); rhet = getv("rhetoric")
            tot = np.array([s.get("total") for s in scores], dtype=float)
            stats = {
                "total_samples": int(total_flat_len),
                "avg_fluency": float(flu.mean()), "avg_similarity": float(sim.mean()), "avg_simplicity": float(simp.mean()),
                "avg_coverage": float(cov.mean()), "avg_rhetoric": float(rhet.mean()), "avg_total_score": float(tot.mean()),
                "std_fluency": float(flu.std()), "std_similarity": float(sim.std()), "std_simplicity": float(simp.std()),
                "std_coverage": float(cov.std()), "std_rhetoric": float(rhet.std()), "std_total_score": float(tot.std()),
            }
        eval_dir = os.path.join(work_dir, f"{tag_key}_evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        save_json_file(stats, os.path.join(eval_dir, "statistics.json"))
        save_json_file(raw_flu, os.path.join(eval_dir, "raw_fluency_output.json"))
        print(f"✓ {tag_key} 评估完成, 平均得分: {stats['avg_total_score']:.4f}")
        return stats

    base_stats = _to_stats("base_model", base_scores, base_raw_flu, base_flat_len)
    dpo_stats = _to_stats("dpo_model", dpo_scores, dpo_raw_flu, dpo_flat_len)

    # ===========================================================
    # Step 4. 汇总结果与改进对比
    # ===========================================================
    results = {"base_model": base_stats, "dpo_model": dpo_stats}
    
    print("\n" + "="*60)
    print("最终评估结果")
    print("="*60)

    for tag, stats in results.items():
        print(f"\n[{tag}]")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "="*60)
    print("DPO 模型改进情况")
    print("="*60)
    improvements = {
        m: dpo_stats.get(f"avg_{m}", 0.0) - base_stats.get(f"avg_{m}", 0.0)
        for m in ["fluency", "similarity", "simplicity", "coverage", "rhetoric", "total_score"]
    }
    for metric, imp in improvements.items():
        base_val = base_stats.get(f"avg_{metric}", 0.0)
        percent = (imp / base_val * 100) if base_val else 0
        print(f"  {metric}: {imp:+.4f} ({percent:+.2f}%) {'↑' if imp > 0 else '↓'}")

    summary = {"results": results, "improvements": improvements}
    summary_path = os.path.join(work_dir, "evaluation_summary.json")
    save_json_file(summary, summary_path)
    print(f"\n评估汇总已保存到: {summary_path}")

    return results




def evaluate_rhetoric_detection(data_path, save_dir=None):
    
    def calculate_f1_score(y_true, y_pred):
        
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

        # 过滤掉None值
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        y_true_filtered = [y_true[i] for i in valid_indices]
        y_pred_filtered = [y_pred[i] for i in valid_indices]
        
        if not y_true_filtered:
            return {"error": "No valid predictions found"}
        
        # 计算各种指标
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_filtered, y_pred_filtered, average='binary'
        )
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered).ravel()
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "total_samples": len(y_true_filtered),
            "invalid_predictions": len(y_true) - len(y_true_filtered)
        }
    
    def extract_prediction_label(rhetoric_detection_text):
        if not rhetoric_detection_text:
            return None
        
        text = rhetoric_detection_text.strip()
        if text.startswith("是"):
            return 1
        elif text.startswith("否"):
            return 0
        else:
            return None
    
    data = load_json_file(data_path)
    
    y_true_binary = []  # 二分类：有修辞/无修辞
    y_pred_binary = []
    
    for item in data:
        record = item.get("raw_data")  # 兼容不同数据格式
        true_type = record.get("type", "").strip()
        true_type = true_type if true_type != "" else "无修辞"
        true_binary = 1 if true_type != "无修辞" else 0
        y_true_binary.append(true_binary)
        rhetoric_detection = record.get("llm_pred", "")

        pred_binary = extract_prediction_label(rhetoric_detection)
        y_pred_binary.append(pred_binary)
    
    binary_results = calculate_f1_score(y_true_binary, y_pred_binary)
    results = {"binary_classification": binary_results}

    if save_dir:
        evaluation_result_path = os.path.join(save_dir, "evaluation_results.json")
        save_json_file(data=results, path=evaluation_result_path)
        print(f"评估结果已保存到: {evaluation_result_path}")

    return results

if __name__ == "__main__":
    canadidates = [
        ["我喜欢吃苹果。", "我爱吃苹果。", "我喜欢吃苹果！"],
        ["今天天气很好。", "今天的天气真好。", "今天阳光明媚。"]
    ]
    references = ["我喜欢吃苹果。", "今天天气很好。"]
