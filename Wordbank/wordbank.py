import os
import json
import pickle
import jieba
import time
import re

class WordEmbeddingMatcher:
    def __init__(self, vocab_file, serialized_vocab_file="Res/vocab.pkl"):
        self.vocab_file = vocab_file
        self.serialized_vocab_file = serialized_vocab_file
        if os.path.exists(self.serialized_vocab_file) and self._is_vocab_up_to_date():
            self.vocab = self.load_serialized_vocab()
            print(f"加载序列化词库文件: {self.serialized_vocab_file}")
        else:
            self.vocab = self.load_vocab(vocab_file)
            self.save_serialized_vocab()
            print(f"词库已解析并保存为序列化文件: {self.serialized_vocab_file}")
        self.vocab_set = set(self.vocab.keys())

    def _is_vocab_up_to_date(self):
        vocab_mtime = os.path.getmtime(self.vocab_file)
        serialized_mtime = os.path.getmtime(self.serialized_vocab_file)
        return vocab_mtime <= serialized_mtime

    def load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line.strip())
                word = entry['word']
                meanings = entry.get('meaning', [])
                processed_meanings = [meaning.replace('～', word) for meaning in meanings]
                vocab[word] = {"meaning": processed_meanings}
        return vocab

    def save_serialized_vocab(self):
        with open(self.serialized_vocab_file, 'wb') as f:
            pickle.dump(self.vocab, f)

    def load_serialized_vocab(self):
        with open(self.serialized_vocab_file, 'rb') as f:
            return pickle.load(f)

    def match_words(self, sentence):
        tokens = set(jieba.cut(sentence))
        matched_words = list(tokens & self.vocab_set)
        return matched_words


def extract_vocab_to_set(processed_text):
    """
    从处理后的文本中提取词汇到集合，跳过标点符号
    
    参数:
        processed_text: 经过API处理后的文本（带空格分隔的词汇）
        
    返回:
        set: 唯一词汇集合（不含标点）
    """
    punctuation_pattern = r'[，。！？、：；"“”‘’（）《》【】〃\.,!\?;:\'"\(\)\[\]\{\}<>/]'
    no_punctuation_text = re.sub(punctuation_pattern, ' ', processed_text)
    tokens = re.split(r'\s+', no_punctuation_text.strip())
    vocab_set = {token for token in tokens if token}
    return vocab_set

def evaluate_wordbank(responses, wordbank):
    wordbank_results = []
    t_start = time.time()
    for sentence in responses:
        if not sentence.strip():
            wordbank_results.append(0.0)
            continue
        tokens = extract_vocab_to_set(sentence)
        print(f"tokens = {tokens}")
        
        if not tokens:
            wordbank_results.append(0.0)
            continue

        matched_tokens = tokens & wordbank.vocab_set
        ratio = len(matched_tokens) / len(tokens)
        wordbank_results.append(ratio)
    return wordbank_results