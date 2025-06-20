from wordbank import WordEmbeddingMatcher, evaluate_wordbank
import pickle

vocab_file = "./dict_meaning_full.json"
serialized_file = "./vocab.pkl"
matcher = WordEmbeddingMatcher(vocab_file, serialized_file)

response = [
    "山上有许多红色的桃花，蜀江的青色水流拍打着山边流淌",
    "是以清庙茅屋，大路越席，大羹不致，粢食不凿，昭其俭也",
    "生存或者毁灭，这是一个问题",
    "他毕业于机电系"
]

result = evaluate_wordbank(response, matcher)
print(result)

with open(serialized_file, 'rb') as f:
    vocab_set = pickle.load(f)

test_words = ['许多', '桃花', '山上', '流淌', '红色', '有', '的', '水流', '蜀江', '拍打着', '山边', '青色']
for word in test_words:
    exists = word in vocab_set
    print(f"词语 '{word}': {'在' if exists else '不在'}词库中")