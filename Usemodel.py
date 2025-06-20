import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

MODEL_PATH1 = "../glm-4-9b-chat-hf"
MODEL_PATH2 = "./model_modern_big"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载模型1: {MODEL_PATH1}")
tokenizer1 = AutoTokenizer.from_pretrained(
    MODEL_PATH1,
    trust_remote_code=True
)
model1 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH1,
    trust_remote_code=True,
    device_map="auto"
)
model1.eval()

print(f"正在加载模型2: {MODEL_PATH2}")
tokenizer2 = AutoTokenizer.from_pretrained(
    MODEL_PATH2,
    trust_remote_code=True
)
model2 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH2,
    trust_remote_code=True,
    device_map="auto"
)
model2.eval()

print("正在加载嵌入模型...")
embedding_model = SentenceTransformer('../paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE)

def paraphrase_sentence(sentence, model, tokenizer):
    system_prompt = """
        请严格按步骤执行：
        1. 改写句子：去除修辞手法，使用直白表达
        2. 分词处理：将改写结果分词，词间用空格分隔

        << 绝对规则 >>
        1. 只输出两个结果：改写后的句子和分词结果
        2. 禁止添加任何解释、说明、警告或额外文字
        3. 严格遵循以下输出格式：
            1. 改写：[改写后的句子]
            2. 分词：[空格分隔的分词结果]

        << 改写规则 >>
        - 改写后的句子绝对不能含有任何修辞手法、成语、俗语、歇后语表达
        - 保留专有名称、命名性的词汇，特别是指代人名的词语
        - 使用标准现代汉语的常用词，直观易懂地表达
        - 引号引用的部分也需要全部进行改写
        - 保持语义不变

        << 分词规则 >>
        - 按词语自然边界切分，不要将词语拆分为单个汉字（除非是独立使用的单字词）
        - 保持专有名称、命名性的词语完整，特别是指代人名的词语（如人名"黎大品"、"秦大哥"等不要分开）
        - 保持常用词语完整（如"一个"、"不是"、"科研工作"等作为一个词）
        - 保留标点符号（视为一个独立的词语）
        - 输出完整的分词结果

        << 示例 >>
        输入："芝麻大的官"
        1. 改写："非常小的官职"
        2. 分词："非常 小 的 官职"

        输入："他走得很快"
        1. 改写："他走路速度很快"
        2. 分词："他 走路 速度 很快"

        输入："我的梦想插上了翅膀"
        1. 改写："我的梦想获得了发展机会"
        2. 分词："我的 梦想 获得 了 发展 机会"

        << 当前任务 >>
        输入：{sentence}
        输出：
        """
    input_text = f"""
<|system|>
{system_prompt}
<|user|>
用户输入：{sentence}
<|assistant|>
输出：
"""
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return full_response.split("</s>")[0].strip()

def calculate_similarity(sentence1, sentence2):
    embeddings = embedding_model.encode([sentence1, sentence2], convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

if __name__ == "__main__":
    test_cases = [
        "他公开向黎大品发牢骚，说黎大品只会死读书，专捡没影的科研项目干。弄好了，会被他人摘了桃子；弄不好，就只能灰溜溜滚蛋。", 
        "上山下乡的潮流把他们冲到北大荒，而回城的风潮又将他们带进这座滨海小城，当起了半熟不熟的大学生。",
        "程咬金直立着不拜道：“秦大哥，不是这等讲。古人云，自行作事自身当。这事是我做的，怎么牵累您？”",
        "有时候连买化肥都挤不出现金来，他们还硬要打肿脸充胖子。",
    ]

    for i, sentence in enumerate(test_cases):
        print(f"测试用例 {i+1}: {sentence}")
    
        result1 = paraphrase_sentence(sentence, model1, tokenizer1)
        print(f"\n模型1 ({MODEL_PATH1}) 改写结果:")
        print(result1)
        
        result2 = paraphrase_sentence(sentence, model2, tokenizer2)
        print(f"\n模型2 ({MODEL_PATH2}) 改写结果:")
        print(result2)

        print(f"\n{'='*40}")
