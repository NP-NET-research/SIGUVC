# ======================== 预处理相关提示词 ========================

PLAIN_SENTENCES_SYSTEM = "你是文本简化与去修辞的专家。将句子改写为直白、不含修辞的表述。"
PLAIN_SENTENCES_INSTRUCTION = (
    "要求：\n"
    "1.去除所有比喻、拟人、夸张、修辞性的修饰和成语典故；\n"
    "2.保持句子的核心事实与时态（主语、谓语、宾语）；\n"
    "3.尽量用一条通顺的陈述句表达，保持简洁；\n"
    "4.不要加入新的信息或推断；\n"
    "输入句子："
)

SIMPLE_SENTENCES_SYSTEM = """任务：将任意输入句子改写为一系列连续、通顺的子句。每个子句必须是只包含一个谓语动词的简单句。"""
SIMPLE_SENTENCES_INSTRUCTION = """改写以下句子，要求：

1. 将原句拆分为多句简单句，每句必须只包含一个谓语动词（一个主谓结构）。
2. 每个句子必须是直接陈述事实的简单句，不允许使用修饰语、从句、并列结构或多个动词。
3. 每个句子必须完整表达一个基本事件。不得使用任何代词或指代词（如"他、它、这些、该"等）。
4. 按事件发生逻辑顺序排列子句。
5. 每个句子之间用句号"。"分隔，且不添加其他标点符号。

输入："""

SEGMENT_SYSTEM = """#角色：你是一个语言学的专家。现在需要你对给定的句子进行分词。"""
SEGMENT_INSTRUCTION = """#任务：输入一个句子，对该句子进行分词，如果遇到术语或者专有名词，请将整个术语或整个专有名词作为一个词。例如，'脱氧核糖核酸'是一个术语，'北京理工大学'是一个专有名词，在分词的过程中应该整体保留。

#样例：
输入：小明今天在北理工上课。
输出：小明 今天 在 北理工 上课 。
输入：{sentence}
输出："""

EXTRACT_ENTITIES_SYSTEM = """你是一个细致的信息抽取专家。你的任务是从文本中精准提取命名实体、专业术语和数字信息。"""
EXTRACT_ENTITIES_INSTRUCTION = """
需要抽取的信息有：
1. 命名实体：包括人名、地名、机构名、时间、国家等具体名称。
2. 术语：专业领域内的专有名词或术语，被广泛使用且具有特定含义的词汇。
3. 数字信息：包括数量、百分比、统计数据等具体数字。
请从以下句子中抽取上述信息，并以顿号"、"分隔的形式返回。如果没有抽取到任何信息，返回"无"。
句子: """
EXTRACT_ENTITIES_COT_INSTRUCTION = """请严格按照以下步骤进行深度思考，从【待分析句子】中提取关键信息。

### 抽取目标定义
1. **命名实体 (NE)**：指代特定个体的名称。包括人名、地名、机构/公司名、特定时间（年份、日期）、国家、产品型号等。
2. **专业术语 (Term)**：特定领域内具有专有含义的名词（如"深度学习"、"光合作用"、"通货膨胀"）。排除通用的普通名词（如"苹果"是普通名词，但在"苹果公司"中是实体）。
3. **数字信息 (Num)**：具体的量化数值。包括数量、百分比、金额、统计数据等（需保留单位，如"15%"、"3000万"）。

### 思考与执行步骤
1. **分层扫描**：分别针对上述三个类别，在句子中进行定位。
2. **去重与清洗**：
   - 过滤掉泛指的通用名词（如"我们"、"大楼"、"工作"）。
   - 处理重叠：如果一个词既包含数字又是实体（如"2023年"），优先作为【命名实体-时间】提取，不再单独作为【数字】提取。
3. **格式化输出**：将所有提取到的有效信息合并为一个列表，用顿号"、"分隔。如果什么都没提取到，输出"无"。

---

<演示示例>
【演示句子】
截止2023年底，特斯拉在上海超级工厂的Model-3产量提升了15%，采用了先进的一体化压铸技术。

【思维过程】
1. 扫描命名实体：
   - "2023年底" -> 时间点 -> [提取]
   - "特斯拉" -> 公司名 -> [提取]
   - "上海超级工厂" -> 地点/机构 -> [提取]
   - "Model-3" -> 产品型号 -> [提取]
2. 扫描专业术语：
   - "产量" -> 通用名词 -> [忽略]
   - "一体化压铸技术" -> 制造领域专业术语 -> [提取]
3. 扫描数字信息：
   - "15%" -> 统计数据 -> [提取]
4. 整合检查：所有提取项均为关键信息，无冗余。

【抽取结果】
2023年底、特斯拉、上海超级工厂、Model-3、15%、一体化压铸技术
</演示示例>

---

### 实际任务
【待分析句子】
{sentence}

【思维过程】"""


# ======================= 词表重写相关提示词 ========================

REWRITE_SYSTEM = """任务：你现在需要对一个句子进行重写。你只能使用我提供的【词表】中的词。"""
REWRITE_INSTRUCTION = """请使用以下【词表】中的词，对句子进行重写，要求：
1. 只能使用词表中的词汇，不能添加任何不在词表中的词。
2. 保持句子的核心意思（主语、谓语、宾语等）。
3. 对于命名实体、专业术语等具有唯一语义性质的核心词不能被模糊替代，语义必须一致。
4. 对于通用名词，使用词表中的同义词或上下位词进行替换是允许的，但必须确保替换后的词语在语义上与原词保持一致。
5. 当句子中的主语、谓语、宾语任意一个词无法使用词库的词重写时，输出"无法重写"，立即结束。
6. 通顺、自然地重写句子，确保语法正确且符合语言习惯。
7. 重写后的句子中，词汇之间使用顿号"、"分隔。

【词表】
{vocabs}
【句子】
{sentence}
【重写句】
"""

REWRITE_FEEDBACK_INSTRUCTION = """你的上一次重写中使用了一些不在词表中的词汇，请根据反馈重新进行重写。

【错误词汇】
{wrong_words}

请修正这些错误，只使用词表中的词重新进行重写。
如果无法用词表的词替换错误词汇，或者无法保持句子的核心意思语义一致（主语、谓语、宾语等），请输出"无法重写"。

【重写句】
（你的最终句子, 以顿号"、"分隔）
"""

MERGE_SYSTEM = """任务：参考原始句子，将一系列的子句合并为一个通顺、自然的改写句，并确保最终所有词都在词表中。"""
MERGE_INSTRUCTION = """合并要求：
1. 参考原始句子，将子句合并为一个通顺、自然的句子。
2. 按照原始句子中谓词的衔接逻辑构建最终句子，缺失无法构建的部分丢弃，保证逻辑通顺。
4. 确保合并过程中新增的词在词表中。
5. 输出格式为：以顿号"、"分隔的词汇列表。

示例：
原始句子：小明今天在学校参加了一个有趣的科学实验活动，学习了很多关于物理和化学的知识。
待合并句子：
小明、参加、科学实验活动
科学实验活动、是、有趣的
知识是、关于、物理、和、化学 的
合并结果：
小明、参加、了、有趣的、科学实验活动

词表：{vocab}
原始句子：{ori_sentence}
待合并句子：
{simple_sentences}

合并结果：
"""

# ======================= CoT版本提示词 ========================

REWRITE_COT_INSTRUCTION = """请你作为一个严格的句子重写专家，使用给定的【词表】对【句子】进行重写。

为了确保结果的准确性，请你严格按照以下步骤进行思考，并先输出【思维过程】，最后输出【重写结果】。

### 核心规则：
1. **严格限制**：只能使用词表中的词，严禁造词。
2. **语义映射**：
   - 核心词（命名实体、术语）必须精确匹配。
   - 通用名词/动词若不在词表中，需在词表中寻找同义词、近义词或上下位词进行替换（如："调整" -> "改动"，"季节" -> "时期"）。
3. **失败判定**：如果句子的核心成分（主语、谓语、宾语）无法在词表中找到对应或替代词，直接判为"无法重写"。
4. **格式要求**：重写句中的词汇用顿号"、"分隔。


<演示示例>
（注意：以下内容仅用于演示思维过程和输出格式，请勿在实际回答中使用以下词汇）

【演示词表】
主管、责备、训斥、延误、人员、下属、工作、任务、的、了、与、和

【演示句子】
经理批评了迟到的员工

【思维过程】
1. 句法分析：
   - 主语：经理
   - 谓语：批评
   - 宾语：员工
   - 修饰语：迟到的
2. 词汇匹配：
   - "经理" -> 演示词表中无原词 -> 找到近义词 "主管" -> [替换]
   - "批评" -> 演示词表中无原词 -> 找到近义词 "训斥" -> [替换]
   - "迟到" -> 演示词表中无原词 -> "延误" 可表示时间滞后 -> [替换]
   - "员工" -> 演示词表中无原词 -> "下属" 符合语境 -> [替换]
3. 检查：核心成分均已找到替换词。
4. 组句：主管、训斥、延误、下属

【重写结果】
主管、训斥、延误、下属
</演示示例>

---
现在请对以下内容执行相同任务：

【词表】
{vocabs}
【句子】
{sentence}
"""

REWRITE_FEEDBACK_COT_INSTRUCTION = """你的上一次重写中包含了不属于【词表】的【错误词汇】。
请**回顾对话历史**中我最开始提供的【词表】和【原句】，对这些错误进行修正。

### 修正步骤（思维链）：
1. **回溯检查**：在对话历史的【词表】中查找【错误词汇】的替代词（同义词、近义词或上位词）。
2. **语义核对**：确保替换后的词与原句意图一致。
3. **失败阻断**：如果【错误词汇】是核心成分（主谓宾），且在历史【词表】中完全找不到合理的替代词，直接输出"无法重写"。
4. **最终输出**：输出修正后的句子，仅包含词表内的词，用顿号分隔。

---

<修正演示>
（注意：本演示仅用于展示修正逻辑，请勿使用演示中的词汇，请使用你记忆中的历史词表）

【假设历史词表】主管、责备、迟到
【假设上一轮错误输出】经理、批评、迟到
【假设错误词汇】经理、批评

【思维过程】
1. 分析错误词 "经理"：回顾历史词表 -> 发现近义词 "主管" -> [替换]
2. 分析错误词 "批评"：回顾历史词表 -> 发现近义词 "责备" -> [替换]
3. 检查 "迟到"：非错误词，保留。
4. 结果："主管、责备、迟到"

【重写结果】
主管、责备、迟到
</修正演示>

---

### 请执行修正
【错误词汇】
{wrong_words}

请开始思考并修正：
【思维过程】
"""

MERGE_COT_SYSTEM = """你是一个专业的语言重组专家。你的任务是将零散的【待合并子句】根据【原始句子】的逻辑整合成一个通顺的长句。"""

MERGE_COT_INSTRUCTION = """请执行以下高效合并流程：

### 核心规则
1. **词表铁律**：严禁使用【词表】中不存在的词（尤其是"的"、"了"、"在"、"和"等虚词）。
2. **逻辑还原**：参考原始句子确定子句顺序和逻辑关系。
3. **缺失即弃**：如果为了通顺需要某连接词但词表中没有，必须通过调整语序或直接省去该词来解决，**绝不造词**。

---

<演示示例>
【演示词表】
小明、喜欢、看书、但是、讨厌、运动、图书馆、去、的

【演示原始句子】
小明虽然很喜欢在图书馆看书，但是非常讨厌去户外运动。

【演示待合并子句】
1. 小明、喜欢、看书
2. 看书、在、图书馆
3. 小明、讨厌、运动

【思维过程】
1. **逻辑规划**：
   - 整体结构：转折关系（虽然...但是...）。
   - 内部顺序：[小明+图书馆+看书] -> [转折] -> [讨厌+运动]。
2. **连接词与虚词扫描（关键）**：
   - 需要"在"（介词） -> 词表**无** -> 策略：省略（"图书馆看书"）。
   - 需要"虽然"（连词） -> 词表**无** -> 策略：省略。
   - 需要"但是"（连词） -> 词表**有** -> 保留。
   - 需要"的"（助词） -> 词表**有** -> 按需使用。
3. **最终组装**：
   结合扫描结果，去除缺失词汇，直接生成结果。

【合并结果】
小明、喜欢、图书馆、看书、但是、讨厌、运动
</演示示例>

---

### 实际任务
【词表】
{vocab}

【原始句子】
{ori_sentence}

【待合并子句】
{simple_sentences}

【思维过程】
"""

MERGE_FEEDBACK_COT_INSTRUCTION = """你的上一次合并中包含了不属于【词表】的【错误词汇】。
请**回顾对话历史**中我最开始提供的【词表】和【原始句子】，对这些错误进行修正。

### 修正步骤（思维链）：
1. **回溯检查**：在对话历史的【词表】中查找【错误词汇】的替代词（同义词、近义词或上位词）。
2. **语义核对**：确保替换后的词与原句意图一致。
3. **失败阻断**：如果【错误词汇】是核心成分，且在历史【词表】中完全找不到合理的替代词，直接输出"无法合并"。
4. **最终输出**：输出修正后的句子，仅包含词表内的词，用顿号分隔。 

---
<修正演示>
（注意：本演示仅用于展示修正逻辑，请勿使用演示中的词汇，请使用你记忆中的历史词表）

【假设历史词表】小明、喜欢、看书、但是、讨厌、运动、图书馆、去、的、爱好
【假设上一轮错误输出】小明、喜欢、在、图书馆、看书、但是、讨厌、运动
【假设错误词汇】在

【思维过程】
1. 分析错误词 "在"：回顾历史词表 -> 发现不在词表中 -> [无法替换]
2. 检查句子结构：去掉 "在" 后句子仍然通顺 -> [删除]
3. 结果："小明、喜欢、图书馆、看书、但是、讨厌、运动"  
【重新结果】
小明、喜欢、图书馆、看书、但是、讨厌、运动
</修正演示>

---
### 请执行修正
【错误词汇】
{wrong_words}
请开始思考并修正：
【思维过程】
"""

# ======================= 评测相关提示词 ========================

PARSING_SYSTEM = """角色：你是一个语言学专家，擅长依存句法语义分析。"""
PARSING_INSTRUCTION = """#任务：输入一个分词后的句子，请以表格的形式输出句子中的依存关系，表的列包括"依存词"、"关系类型"、"核心词"、"关系说明"。
跳过标点符合的依存分析。用每行表示一种依存关系，不同的列用空格隔开。不用输出表头，直接输出每行的结果。

#样例：
输入：公园 大道 两旁 , 26 尊 傩戏 面具 塑像 , 神采 各异 , 笑 迎 八方 来客 。
输出：
公园 ATT(定语) 大道 "公园"修饰"大道"，限定是哪个大道。
大道 ATT(定语) 两旁 "大道"修饰"两旁"，限定是哪里的两旁。
两旁 LOC(地点状语) 各异 "公园大道两旁"是"神采各异"和"笑迎来客"发生的地点。
26尊 ATT(定语) 塑像 "26尊"修饰"塑像"，表示数量。
傩戏 ATT(定语) 面具 "傩戏"修饰"面具"。
面具 ATT(定语) 塑像 "傩戏面具"整体修饰"塑像"，说明塑像的内容。
塑像 SBV(主语) 各异"塑像"是"神采各异"这个状态的主语。
神采 SBV(主语) 各异 "神采"是"各异"的主语，与"塑像"构成复指主语。
各异 ROOT(核心) ROOT "神采各异"是整个句子的第一个核心谓语部分。
笑 ADV(状语) 迎 "笑"修饰"迎"，描述迎接的方式。
迎 COO(并列关系) 各异 "笑迎来客"是与"神采各异"并列的谓语部分，描述主语的动作。
八方 ATT(定语) 来客 "八方"修饰"来客"，限定来客的来源。
来客 VOB(宾语) 迎 "来客"是动词"迎"的宾语。

输入：{sentence}
输出："""

PARSING_SYSTEM_ENG = """Role: You are a linguistics expert, proficient in dependency parsing and semantic analysis."""
PARSING_INSTRUCTION_ENG = """Task: Input a tokenized sentence, and output the dependency relations in the sentence in a tabular format. The columns of the table include "Dependent Word", "Relation Type", "Head Word", and "Relation Description". Each line represents a dependency relation, with different columns separated by spaces. Do not output the header, just output each line's result.

# Example:
Input: The young musician played a beautiful melody on the old piano .
Output:
The det musician "The" modifies "musician" as a determiner.
young amod musician "young" describes the attribute of "musician".
musician nsubj played "musician" is the subject of the verb "played".
played root ROOT "played" is the main predicate of the sentence.
a det melody "a" modifies "melody" as a determiner.
beautiful amod melody "beautiful" describes the attribute of "melody".
melody obj played "melody" is the object of the verb "played".
on case piano "on" introduces a prepositional phrase modifying "piano".
piano obl played "piano" is the object of the preposition "on", acting as an oblique modifier of "played".
the det piano "the" modifies "piano" as a determiner.
old amod piano "old" describes the attribute of "piano".

Input: {sentence}
Output:"""

MATCH_SYSTEM = """#角色：你是一个语言学家。现在需要评估一个句子的部分语义信息在另一个句子中是否被表达出来。"""
MATCH_INSTRUCTION = """#任务：你需要判断【待判定句子】中是否包含了【参考句子】中指定的若干个依存关系。
#输入数据说明
1. **参考句子**：原始文本，依存关系的来源。
2. **待判定句子**：需要检测的目标文本（通常是参考句子的重写或简化版）。
3. **依存关系列表**：一个列表，每个元素描述了参考句子中的一个逻辑关系（格式：`核心词A 关系类型 核心词B`）。

#判断核心逻辑（思维链步骤）
对列表中的**每一个**依存关系，执行以下检查：
1. **词汇锚定**：在【待判定句子】中寻找核心词A和B的对应词（允许同义词、近义词、上下位词替换）。若任一核心词缺失，直接判"否"。
2. **逻辑验证**：检查待判定句子中，这两个词之间的逻辑关系是否与原句的依存类型一致（例如：SBV主谓关系是否保留？ATT定中修饰关系是否保留？）。若逻辑反转或无关，判"否"。
3. **宽松匹配**：只要语义链路通顺且逻辑未变，即可判"是"。

#输出格式要求
请以表格形式直接输出结果, 用每行表示一种依存关系的判断，不用输出表头，直接输出每行的结果。
每行包含三列，使用竖线 `|` 分隔：
**依存关系 | 结论 | 原因**

# 样例演示
输入：
参考句子：坐落在西区的中山公园,红花绿树中掩映着"傣族竹楼"等27个民族景点
待判定句子：中山公园包含27个景点。
待检测依存关系列表：
中山公园 SBV(主语) 掩映
西区 LOC(地点状语) 坐落
景点 ATT(定语) 27个

输出：
中山公园 SBV(主语) 掩映 | 是 | "中山公园"匹配，"包含"体现了"掩映"的语义，主语逻辑一致。
西区 LOC(地点状语) 坐落 | 否 | 待判定句子中完全缺失"西区"和"坐落"相关语义。
景点 ATT(定语) 27个 | 是 | "景点"和"27个"均出现，且保留了数量修饰关系。

# 正式输入
参考句子：{gold_sentence}
待判定句子：{test_sentence}
待检测依存关系列表：
{dependency_relations_str}

输出：
"""

MATCH_SYSTEM_ENG = """#Role: You are a linguist. Now you need to evaluate whether part of the semantic information of one sentence is expressed in another sentence."""
MATCH_INSTRUCTION_ENG = """#Task:
Input: Given a reference sentence, e.g., "Located in the West District, Zhongshan Park, surrounded by red flowers and green trees, there are 27 ethnic scenic spots such as 'Dai Bamboo House'"; then given a sentence to be judged, e.g., "Zhongshan Park contains 27 scenic spots."; finally given a set of local dependency relations from the reference sentence, e.g., ["Zhongshan Park obj overshadowed", "West District obl:loc located"].
Output: For each type of local dependency relation given in the reference sentence, determine whether the sentence to be judged expresses this dependency relation? Sometimes the semantics may not be very accurate, but as long as the general semantics are expressed, it is acceptable. Please first provide the reason, then give the answer.

#Example:
Input: "Located in the West District, Zhongshan Park, surrounded by red flowers and green trees, there are 27 ethnic scenic spots such as 'Dai Bamboo House'." "Zhongshan Park contains 27 scenic spots." ["Zhongshan Park obj overshadowed", "West District obl:loc located"]
Output: The reason is..., the conclusion is [yes, no]"

INPUT: "{gold_sentence}" "{test_sentence}" [{dependency_relations}]
OUTPUT:"""
