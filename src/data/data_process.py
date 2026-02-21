import json
import os
import random
import re
from typing import Dict, List, Optional, Any

import numpy as np
from src.utils import load_json_file, load_lines_file, save_json_file, save_lines_file
from src.inference.prompt import RHETORIC_INSTRUCTION, RHETORIC_SYSTEM


def split_sentences(text: str) -> list[str]:
    """
    根据标点拆分句子，句号、问号、感叹号、分号都可以拆分
    """
    # 将标点作为拆分点，但保留标点
    split_pattern = re.compile(r"([;])")
    pieces = split_pattern.split(text)
    
    # 合并标点和前面的文字
    sentences = []
    for i in range(0, len(pieces)-1, 2):
        sentence = pieces[i].strip() + pieces[i+1].strip()
        if sentence:
            sentences.append(sentence)
    
    # 处理最后一段可能没有标点的情况
    if len(pieces) % 2 != 0 and pieces[-1].strip():
        sentences.append(pieces[-1].strip())
    
    return sentences

def process_giga(min_len=5, max_len=200, valid_char_ratio=0.8, truncate_len=None):
    # 处理giga数据集
    input_path = "corpus/giga/2006.txt"
    output_path = "corpus/giga/2006_processed.txt"

    data = load_lines_file(input_path)
    print(f"原始数据集共{len(data)}行")
    
    split_data = []
    for line in data:
        for sent in split_sentences(line):
            split_data.append(sent)
    data = split_data
    print(f"拆分后数据集共{len(data)}行")
    # 去除空行
    data = [line.strip() for line in data if line.strip()]

    # 去除重复
    data = list(dict.fromkeys(data))
    print(f"去重后数据行数: {len(data)}")

    # 过滤异常字符比例过高的句子
    def is_valid_sentence(s):
        clean_s = re.sub(r"[^\u4e00-\u9fff0-9，。！？,.!?]", "", s)
        return len(clean_s) / max(1, len(s)) >= valid_char_ratio
    
    data = [s for s in data if is_valid_sentence(s)]
    # 过滤长度异常句子
    data = [s for s in data if min_len <= len(s) <= max_len]

    # 可选截断
    if truncate_len:
        data = [s[:truncate_len] for s in data]

    # 长度统计
    lengths = [len(s) for s in data]
    if lengths:
        print(f"\n长度统计信息:")
        print(f"  最小长度: {min(lengths)}")
        print(f"  最大长度: {max(lengths)}")
        print(f"  平均长度: {np.mean(lengths):.2f}")
        print(f"  中位数长度: {np.median(lengths)}")

        # 长度分布区间
        ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, float('inf'))]
        print(f"\n长度分布:")
        for start, end in ranges:
            count = sum(1 for l in lengths if start <= l < end)
            percentage = (count / len(lengths)) * 100
            range_str = f"{start}-{end}" if end != float('inf') else f"{start}+"
            print(f"  {range_str}: {count} ({percentage:.2f}%)")

    # 保存清洗后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(data))

    print(f"\n处理后的数据已保存到: {output_path}")
    print(f"有效数据行数: {len(data)}")

import random
from collections import defaultdict

def split_train_test_by_length(input_file: str,
                               train_file: str,
                               test_file: str,
                               total_samples: int = 40000,
                               train_ratio: float = 0.9,
                               min_len: int = 5,
                               max_len: int = 200,
                               seed: int = 42):
    """
    按长度分布比例抽样并划分训练集和测试集。
    
    参数:
        input_file: 处理后的语料路径
        train_file: 输出训练集路径
        test_file: 输出测试集路径
        total_samples: 总抽样数
        train_ratio: 训练集比例
        min_len, max_len: 保留句子的长度区间
        seed: 随机种子
    """
    random.seed(seed)

    # 读取语料
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 按长度过滤
    filtered_lines = [line for line in lines if min_len <= len(line) <= max_len]
    print(f"长度在 {min_len}-{max_len} 的句子数: {len(filtered_lines)}")

    # 按长度区间分组
    length_bins = [(0,50), (50,100), (100,200)]
    bin_to_lines = defaultdict(list)
    for line in filtered_lines:
        for start, end in length_bins:
            if start <= len(line) < end:
                bin_to_lines[(start,end)].append(line)
                break

    # 计算每个区间抽样数
    bin_counts = {k: len(v) for k,v in bin_to_lines.items()}
    total_available = sum(bin_counts.values())
    bin_samples = {k: int(total_samples * (v / total_available)) for k,v in bin_counts.items()}

    # 抽样
    sampled_lines = []
    for k, n in bin_samples.items():
        lines_in_bin = bin_to_lines[k]
        n = min(n, len(lines_in_bin))  # 避免区间数量不足
        sampled_lines.extend(random.sample(lines_in_bin, n))

    # 随机打乱
    random.shuffle(sampled_lines)

    # 划分训练集和测试集
    train_count = int(len(sampled_lines) * train_ratio)
    train_lines = sampled_lines[:train_count]
    test_lines = sampled_lines[train_count:]

    # 保存
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))

    print(f"训练集: {len(train_lines)} 条 -> {train_file}")
    print(f"测试集: {len(test_lines)} 条 -> {test_file}")
    print("各长度区间抽样情况:")
    for k, v in bin_samples.items():
        print(f"  {k[0]}-{k[1]}: {v} 条")


def wordbank_process(input_file: str, output_file: str):
    
    lines = load_lines_file(input_file)
    processed_words = set()
    for line in lines:
        if line:
            processed_words.add(line)
    print(f"原始单词数: {len(lines)}, 处理后单词数: {len(processed_words)}")
    processed_words = sorted(processed_words)
    save_lines_file(list(processed_words), output_file)


if __name__ == "__main__":
    
    # 使用示例
    # split_train_test_by_length(
    #     input_file="corpus/giga/2006_processed.txt",
    #     train_file="corpus/giga/train_36k.txt",
    #     test_file="corpus/giga/test_4k.txt",
    #     total_samples=40000,
    #     train_ratio=0.9
    # )
    wordbank_process(
        input_file="corpus/wordbank/Words-List-temp.txt",
        output_file="corpus/wordbank/Words-List.txt"
    )

