from GLM4API import GLM4_API
import json
import time

api_key = 'e1460eea1141212d4d200e18bf5422c1.GJeATw66ztnAyh9M'
batch_size = 4
data_path = './data_batch=4.json'
semantic_prompt = """
    你是一个语义相似度判断助手。你的任务是判断以上两个句子在语义上的相似度，并给出分数：-1.0、-0.8、-0.6、-0.4、-0.2、0.2、0.4、0.6、0.8或1.0，分数随相似程度递减，从1.0表示句子语义完全相似，到-1.0表示语义完全不相似。
    以下是一些评分示例，请参考：
    1. 句子1: 张三是无事不登三宝殿。句子2: 如果张三能来，一定有原因。输出：1.0。
    2. 句子1: 西藏的秏牛，内蒙的骆驼，早像春风一般巡视过高原和沙漠。句子2: 西藏的牛和内蒙的骆驼，早已走遍了高原和沙漠。输出：0.8。
    3. 句子1: 不敢说，要是咱们里面有人露出去，我可怕挨黑枪。句子2: 我敢说，要是咱们里面有人泄露出去，我才不害怕被人暗中算计。输出：-1.0。
    4. 句子1: 我宁愿挨生活的巴掌，不愿意委屈自已的心。句子2: 我不愿意遭受生活的打击，也不愿意违背自己的内心。输出：-0.6。
    5. 句子1: 我失去了地位，保住了人格。句子2: 我失去了地位，却保住了人格。输出：0.4。
    请只回复分数，不要输出多余内容。
    """
expression_prompt = """
    请判断以上句子是否包含非字面表达（修辞手法/引申义），按以下规则评分：
    1.0 = 存在比喻/拟人/夸张/双关/反语/借代等修辞，或词语活用/非常规组合
    0.0 = 仅字面含义的直述
    示例：
    1. "钢、煤、发电量跃居世界第7、3、14位" → 1.0（列举分承）
    2. "钢产量第7，煤第3..." → 0.0（平铺直叙）
    3. "芝麻大的官" → 1.0（夸张+比喻）
    4. "他走得很快" → 0.0
    5. "我的梦想插上了翅膀" → 1.0（"翅膀"代指机会）
    6. "这些眼睛们似乎连成一气，已经在那里咬他的灵魂。" → 1.0（超现实描述）
    7. "起初，她只认为李永恩有“恩”于她，碍于面子，不得不任其“方便”。" → 1.0（语义双关/谐音）
    8. "我冲入这黑绵绵的昏夜，为要寻一颗明星；为要寻一颗明星，我冲入这黑绵绵的昏夜。" → 1.0（排比/列举分承）
    请只回复分数，不要输出多余内容。
    """


if __name__ == '__main__':
    glm4_api = GLM4_API(
        api_key=api_key,
        batch_size=batch_size,
        semantic_prompt=semantic_prompt,
        expression_prompt=expression_prompt
    )
    with open(data_path, 'r', encoding='utf-8') as f:
        batches = json.load(f)["batches"]
        scores = []
        for batch in batches:
            batch_start = time.time()
            # debug
            score = glm4_api.evaluate_semantic_similarity(
                batch=batch,
                responses=batch["responses"]
            )
            # score = glm4_api.evaluate_expression_style(
            #     responses=batch["responses"]
            # )
            scores.append(score)
            # 打印进度
            batch_time = time.time() - batch_start
            print(f"本批次耗时: {batch_time:.4f}s")
            print(f"score: {score}")
        
        print(f"评估得分: {scores}")