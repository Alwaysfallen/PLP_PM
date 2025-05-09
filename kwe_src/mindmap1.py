"""在本地 8 GB GPU 环境下，不依赖远程 LLM，将已提取的关键词与 BART 摘要结果转化为结构化思维导图图形的完整代码示意。"""

"""思维导图绘制逻辑
目标： 从PDF文档中提取信息，并自动生成一个结构化的思维导图，以清晰、直观地呈现文档的主要内容。

流程：
1、文本提取与预处理：
使用 PyPDF2 库从 PDF 文档中提取文本内容。
使用正则表达式和自定义函数对文本进行清洗，去除页码、版权信息、特殊字符等干扰信息，提高后续分析的准确性。
2、摘要与关键词提取：
利用预训练的 BART 模型对清洗后的文本进行摘要，生成简洁的文本概括。
使用 KeyBERT 算法从文本中提取关键词，捕捉文档的核心概念。
3、思维导图构建：
将文件名作为思维导图的中心主题（根节点）。
将摘要按句号分割成多个子主题，作为思维导图的二级节点。
建立中心主题与子主题之间的连接关系，体现层次结构。
将提取的关键词与相应的子主题进行关联，丰富子主题的信息。
4、可视化：
使用 networkx 库构建思维导图的网络结构。
使用 matplotlib 库将思维导图可视化，并导出为图片格式。"""

# 2. 导入库
import os  # 导入os模块用于文件操作
from transformers import pipeline
from keybert import KeyBERT
import networkx as nx
import matplotlib.pyplot as plt
import PyPDF2
import re
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab') # Download punkt_tab data

# 3. 加载预训练模型
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # 使用公开的预训练模型
kw_model = KeyBERT()

# 4. 输入文本和关键词提取
# --- PDF 文本提取和清洗 ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_courseware_text(text):
    # 去除页码、版权、学校名等信息
    text = re.sub(r'\bPage \d+ of \d+\b', '', text)
    text = re.sub(r'\b(Copyright|©|All Rights Reserved).*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(University|Institute|School).*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(COPYRIGHT|CONFIDENTIAL|DRAFT|VERSION \d+)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Figure\s*\d+:.*', '', text, flags=re.IGNORECASE)

    # 替换公式为标记
    text = re.sub(r'(\$.*?\$|\\\[.*?\\\])', '[FORMULA]', text)

    # 去除奇怪的 Unicode 字符（保留英文、标点）
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 去除典型“表格标题 + 数值行”结构
    text = re.sub(r'(\d{1,3}(?:,\d{3})*\.?\d*\s+[A-Za-z ]+\s+(?:-?\d{1,3}(?:,\d{3})*\.?\d*\s*){2,})', '', text)

    # 删除数字密度过高的段落（数字比例 > 50%）
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        num_tokens = sum(1 for tok in tokens if re.match(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$', tok))
        if num_tokens / len(tokens) < 0.5:
            filtered_lines.append(line.strip())

    # 去重 + 去空行
    unique_lines = list(dict.fromkeys([l for l in filtered_lines if l]))
    text = '\n'.join(unique_lines)

    return text.strip()

pdf_path = "D:/NUS_ISS/EBA5004-project/Notes/EBA5004/NUS_CUI_Day1_v3.pdf"  # 替换为你的 PDF 文件路径
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_courseware_text(raw_text)


# 获取 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# 检查文本长度
input_ids = tokenizer(cleaned_text, truncation=True, max_length=None)['input_ids']
if len(input_ids) > 1024:  # BART 的最大输入长度限制
    # 分块处理
    from nltk import sent_tokenize
    sentences = sent_tokenize(cleaned_text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(tokenizer.tokenize(current_chunk + sentence)) <= 1024:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    summaries = []
    keywords_list = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        keywords = kw_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 2), top_n=10)
        summaries.append(summary)
        keywords_list.append(keywords)

    summary = " ".join(summaries)  # 合并摘要
    keywords = [kw for sublist in keywords_list for kw in sublist]  # 合并关键词

else:
    # 直接处理
    summary = summarizer(cleaned_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    keywords = kw_model.extract_keywords(cleaned_text, keyphrase_ngram_range=(1, 2), top_n=10)


# 5. 构建思维导图数据结构
G = nx.DiGraph()  # 创建有向图

# 中心主题识别 (使用关键词权重)
from collections import Counter
keyword_weights = Counter([kw for kw, score in keywords])  # 计算关键词权重
central_topic = keyword_weights.most_common(1)[0][0]  # 选择权重最高的关键词
G.add_node("Root", label=central_topic)

# 子主题识别和关联
topics = nltk.sent_tokenize(summary)  # 将摘要分割成句子
topic_nodes = [] # 用于存储子主题节点的标签
for topic in topics:
    topic_node = topic  # 使用句子作为节点标签
    G.add_node(topic_node, label=topic_node)
    topic_nodes.append(topic_node)
    # 关键词匹配和语义相似度计算，连接中心主题和子主题
    if central_topic in topic:  # 简单关键词匹配，可替换为更复杂的语义相似度计算
        G.add_edge("Root", topic_node)

# 关键词关联
for keyword, score in keywords:
    # 检查关键词是否与中心主题或子主题重叠
    if keyword != central_topic and keyword not in topic_nodes:
        # 找到最相关的子主题 (简单关键词匹配，可替换为更复杂的语义相似度计算)
        most_relevant_topic = None
        max_relevance = 0
        for topic in topics:
            if keyword in topic and len(topic) > max_relevance:
                most_relevant_topic = topic
                max_relevance = len(topic)  # 使用长度作为简单 relevance 指标

        # 将关键词添加到最相关的子主题下
        if most_relevant_topic:
            G.add_node(keyword, label=keyword)
            G.add_edge(most_relevant_topic, keyword)


# 6. 应用图布局算法
# # pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')  # 按顺序排列
# pos = nx.planar_layout(G) # 按顺序排列

pos = nx.spring_layout(G, k=0.5, iterations=50)  # 使用 spring_layout，并调整参数

# pos = nx.kamada_kawai_layout(G, scale=2)  # 调整 kamada_kawai_layout 参数

# pos = nx.fruchterman_reingold_layout(G, k=0.1)  # 使用 fruchterman_reingold_layout 布局算法


# 7. 可视化并导出
plt.figure(figsize=(25,20))# 增加图形尺寸

# 分别绘制不同类型的节点，并设置不同的颜色
# 手动设置中心主题节点位置 (如果需要)
pos[central_topic] = (0, 0)

# 使用 try-except 语句处理异常
nodelist = [central_topic] + topics + [kw for kw, _ in keywords]
for node in nodelist:
    try:
        # 分别绘制不同类型的节点，并设置不同的颜色
        if node == central_topic:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='plum', node_size=2000)  # 中心主题
        elif node in topics:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='lightblue', node_size=1500)  # 子主题
        # else:
        #     nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='lightgreen', node_size=1000)  # 关键词
    except KeyError:
        print(f"Node '{node}' has no position, skipping.")  # 打印警告信息

nx.draw_networkx_edges(G, pos)  # 绘制边
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")  # 绘制标签


# 保存思维导图
output_folder = "D:/NUS_ISS/EBA5004-project/mindmaps"  # 设置保存文件夹路径
filename = pdf_path.split('/')[-1].split('.')[0]  # 从pdf_path中提取文件名作为保存文件名
output_path = os.path.join(output_folder, filename + ".png")  # 拼接完整保存路径

plt.title("Mind Map1 - " + filename)  # 设置标题，包含文件名
plt.savefig(output_path)  # 保存到指定路径
plt.show()