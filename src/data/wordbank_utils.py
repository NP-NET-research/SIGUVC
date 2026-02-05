import os
import numpy as np
import faiss
from typing import Dict, List, Set

from src.utils import load_pickle_file, save_pickle_file


class SubWordBank:

    def __init__(self, cache_file: str, source_file: str = None, 
                 read_only: bool = False, force_rebuild: bool = False):
        self.cache_file = cache_file
        self.source_file = source_file
        self.read_only = read_only
        self.force_rebuild = force_rebuild

        self.data_dict: Dict[str, Set[str]] = {}  # 分级路径 -> 词集合
        self._data_set: frozenset = frozenset()   # 扁平集合

        self._initialize_data()

    def _initialize_data(self) -> None:
        if self.source_file and (self.force_rebuild or not self._is_cache_valid()):
            self._load_from_source_file()
            self._save_to_cache()
        else:
            self._load_from_cache()
    
    def _load_from_source_file(self) -> None:
        print(f"Loading wordbank from source file: {self.source_file}")
        if not self.source_file or not os.path.exists(self.source_file):
            return

        self.data_dict = {}
        try:
            current_path = None
            has_category = False
            lines = []
            with open(self.source_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
                    if line.startswith("#"):
                        has_category = True

            if has_category:
                current_path = None
                for line in lines:
                    if line.startswith("#"):
                        current_path = line.rstrip("：")
                        self.data_dict.setdefault(current_path, set())
                    elif current_path:
                        words = line.split()
                        self.data_dict[current_path].update(words)
            else:
                self.data_dict["#default"] = set(lines)

            if not self.data_dict:
                raise ValueError("Wordbank is empty after processing, please check the file.")

            self._update_data_set()
            print(f"Loaded wordbank from source file: {self.source_file}, total words: {len(self._data_set)}")
        except Exception as e:
            raise ValueError(f"Failed to read source file: {e}")
    
    def _update_data_set(self):
        """扁平集合用于快速查询"""
        all_words = set()
        for words in self.data_dict.values():
            all_words.update(words)
        self._data_set = frozenset(all_words)

    def contains_word(self, word: str) -> bool:
        if not isinstance(word, str) or not word.strip():
            return False
        return word.strip() in self._data_set

    def get_words_by_category(self, category: str) -> Set[str]:
        if not category or category not in self.data_dict:
            return set()
        return self.data_dict[category].copy()

    @property
    def words(self) -> Set[str]:
        return set(self._data_set)

    def add_words(self, path: str, words: List[str]) -> int:
        """
        path: 四级路径 '一级/二级/三级/四级'
        """
        if self.read_only or not path or not words:
            return 0
        self.data_dict.setdefault(path, set())
        initial_count = len(self.data_dict[path])
        self.data_dict[path].update(word.strip() for word in words if word.strip())
        added_count = len(self.data_dict[path]) - initial_count
        if added_count > 0:
            self._update_data_set()
            self._save_to_cache()
        return added_count

    def remove_words(self, words: List[str]) -> int:
        if self.read_only or not words:
            return 0
        removed_count = 0
        for word in words:
            word = word.strip()
            if not word:
                continue
            for word_set in self.data_dict.values():
                if word in word_set:
                    word_set.remove(word)
                    removed_count += 1
        if removed_count > 0:
            self._update_data_set()
            self._save_to_cache()
        return removed_count

    def export_to_txt(self, output_file: str) -> None:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for word in sorted(self._data_set):
                    f.write(f"{word}\n")
        except Exception as e:
            print(f"导出词库失败: {e}")

    # ----------------- 缓存相关 -----------------
    def _is_cache_valid(self) -> bool:
        if not self.source_file:
            return True
        try:
            if not os.path.exists(self.cache_file):
                return False
            return os.path.getmtime(self.source_file) <= os.path.getmtime(self.cache_file)
        except Exception:
            return False

    def _ensure_cache_dir(self) -> None:
        if self.cache_file:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _save_to_cache(self) -> bool:
        try:
            self._ensure_cache_dir()
            tmp_path = f"{self.cache_file}.tmp"
            save_pickle_file(self.data_dict, tmp_path)
            os.replace(tmp_path, self.cache_file)
            return True
        except Exception:
            return False

    def _load_from_cache(self) -> bool:
        try:
            if not os.path.exists(self.cache_file):
                return False
            data = load_pickle_file(self.cache_file)
            if isinstance(data, dict):
                self.data_dict = data
            else:
                self.data_dict = {}
            self._update_data_set()
            return True
        except Exception:
            self.data_dict = {}
            return False
        

class VocabVectorizer:
    """词库向量化组件（基于 BGE/BEG 模型 + FAISS CPU 索引，仅保存索引）"""

    def __init__(
            self, 
            encoder_model,
            faiss_index_file: str,
            force_rebuild: bool = False
        ):
        self.encoder_model = encoder_model
        self.faiss_index_file = faiss_index_file
        self.force_rebuild = force_rebuild

        self.words: List[str] = []
        self.word_vectors: np.ndarray = None
        self.index: faiss.IndexFlatIP = None
    
    # -------------------- 向量化 + 构建索引 --------------------
    def vectorize_and_build_index(self, vocab: Set[str], batch_size: int = 512):
        """对词库向量化并构建 FAISS 索引"""
        print(f"向量化 {len(vocab)} 个词...")
        self.words = list(vocab)
        self.word_vectors = []

        # 分批向量化
        for i in range(0, len(self.words), batch_size):
            batch_words = self.words[i:i+batch_size]
            batch_vectors = self.encoder_model.encode(batch_words, batch_size=len(batch_words), convert_to_numpy=True)
            batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            self.word_vectors.append(batch_vectors)

        self.word_vectors = np.vstack(self.word_vectors).astype(np.float32)

        # 构建 FAISS CPU 内积索引
        dim = self.word_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.word_vectors)
        print(f"FAISS 索引构建完成，包含 {self.index.ntotal} 条向量")
      
    # -------------------- 保存 / 加载 --------------------
    def save_faiss_index(self):
        if self.index is None:
            raise ValueError("索引未构建")
        faiss.write_index(self.index, self.faiss_index_file)
        # 保存词表
        with open(self.faiss_index_file + ".words", "w", encoding="utf-8") as f:
            f.write("\n".join(self.words))
        print(f"FAISS 索引和词表已保存: {self.faiss_index_file}")

    def load_faiss_index(self):
        if not os.path.exists(self.faiss_index_file):
            raise FileNotFoundError(f"FAISS 索引文件不存在: {self.faiss_index_file}")
        self.index = faiss.read_index(self.faiss_index_file)
        # 加载词表
        with open(self.faiss_index_file + ".words", "r", encoding="utf-8") as f:
            self.words = [line.strip() for line in f]
        print(f"FAISS 索引和词表已加载，共 {len(self.words)} 条词向量")

