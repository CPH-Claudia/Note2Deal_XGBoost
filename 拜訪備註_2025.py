# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 17:27:25 2025

@author: Z01788
"""

# %% 文字前處理 
# pip install numpy==1.23 --upgrade
import pandas as pd
# import numpy as np

visit = pd.read_excel("D:/備註文字探勘/拜訪0409.xlsx", sheet_name=2) 

df = visit.copy()

def is_valid_note(note):
    if pd.isna(note):
        return 0
    lines = str(note).splitlines()  # 分割成多行
    # 檢查是否有「不是以 # 或 ＃ 開頭」的有效文字行
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not (line.startswith('#') or line.startswith('＃')):
            return 1
    return 0

df['是否有備註(排除#)'] = df['拜訪備註'].apply(is_valid_note)

def has_sharp(note):
    if pd.isna(note):
        return 0
    if '#' in str(note) or '＃' in str(note):
        return 1
    return 0

# 新增欄位
df['含有#'] = df['拜訪備註'].apply(has_sharp)

def extract_non_sharp_text(note):
    if pd.isna(note):
        return ''
    
    # 將 _x000D_ 替換為換行符號（或清除 \r）
    cleaned_note = str(note).replace('_x000D_', '\n').replace('\r', '')

    lines = cleaned_note.splitlines()
    clean_lines = [
        line.strip()
        for line in lines
        if line.strip() and not (line.strip().startswith('#') or line.strip().startswith('＃'))
    ]
    return '\n'.join(clean_lines)

df['拜訪備註_文字'] = df['拜訪備註'].apply(extract_non_sharp_text)


# %% CKIP 斷詞
# change python version to <3.12.x to run tensorflow
# pip install spyder-kernels==3.0.3
# https://www.python.org/downloads/release/python-390/
# py -3.9 --version
# pip install tensorflow==2.5.0
# pip install ckiptagger
# change python interpreter


# 建立 CKIP 用詞典（視你使用的 CKIP 斷詞方法支援而定）
# 若你使用的是 CKIPTagger 可加至自訂詞典：
import tensorflow as tf
from ckiptagger import data_utils
# pip install gdown
# data_utils.download_data_gdown("./")  # 約 250MB

from ckiptagger import WS, POS, NER

ws = WS("./data")
pos = POS("./data")
ner = NER("./data")

# 對文字進行斷詞
text_list = df['拜訪備註_文字'].dropna().tolist()  # 避免空值

# word_segments = ws(text_list) # 斷詞
# pos_tags = pos(word_segments) # 詞性標註
# ner_tags = ner(word_segments, pos_tags) # 命名實體辨識

# # 如果 CKIPTagger 本身不支援加詞（目前的限制），
# # 建議你在 CKIP 斷詞後 **人工合併詞**，如下：
# def merge_custom_terms(ws_result, custom_terms):
#     merged_result = []
#     term_set = set(custom_terms)  # 加速查找
#     max_len = max(len(term) for term in custom_terms)  # 最長詞彙長度

#     for sentence in ws_result:
#         merged_sentence = []
#         i = 0
#         while i < len(sentence):
#             match = None
#             # 嘗試從長度最長往短比對
#             for l in range(min(max_len, len(sentence) - i), 0, -1):
#                 phrase = ''.join(sentence[i:i+l])
#                 if phrase in term_set:
#                     match = phrase
#                     i += l
#                     break
#             if match:
#                 merged_sentence.append(match)
#             else:
#                 merged_sentence.append(sentence[i])
#                 i += 1
#         merged_result.append(merged_sentence)
#     return merged_result

# # 原始斷詞 + 合併保險詞彙
# ws_segments = ws(text_list)

# # 斷詞前載入保險術語字典
# insurance_terms = [
#     "保單健診", "華南產", "癌症險", "旅平險", "新安東京", "三商美邦", "還本型保單", "富邦", "安達", "續保", "意外險", 
#     "公共意外險", "富邦產", "定期壽險", "重大傷病", "實支實付", "住院醫療", "理賠", "分紅躉繳", "轉介紹"
# ]

# word_segments = merge_custom_terms(ws_segments, insurance_terms)


# 載入你的保險術語字典
insurance_terms = set([
    "保單健診", "華南產", "癌症險", "旅平險", "新安東京", "還本型保單", "富邦", "安達", "續保", 
    "富邦產", "定期壽險", "重大傷病", "實支實付", "住院醫療", "理賠", "分紅躉繳", "轉介紹", 
    "保單","保險","行銷活動","防疫險","保險經紀人","健診","籃子理論","錠嵂","意外險","資產規劃", "車險需求",
    "儲蓄險","中壽","中國人壽","旅平","旅平險","三商美邦","簽約","成交","重大傷病","失能險","保經","見面三講","開門三講","退休規劃", 
    "車險","醫療險","火險","壽險","新光","遠雄","富邦","Toyota","機車險","寵物險","自動化工程師","六大保障","建議書", 
    "台灣人壽","失智症","app","OPP","保險存摺","國泰","遠雄","照會","三照","遞送","簽約","市調表","解約","美元保單","美元儲蓄", 
    "送保單" ,"照會單" ,"台壽","保誠" ,"癌症" ,"不動產","問卷","理賠","健診","轉介","簽收","建立關係","強制險","永達", 
    "觀念溝通","需求分析","六大保障","保單健診","終身","萬事利達","續保","友邦","寒暄","關心","保險存摺","年金","PHB","宏泰", 
    "南山","長照","XHB","HNRC","新生兒","約訪","年繳","美金","phb","探班","要保人",'企管副會長','意外險需求','double鑫','下週'
])
max_term_len = max(len(term) for term in insurance_terms)

# 合併術語詞的函數（套用在 CKIP 結果上）
def merge_custom_terms(ws_result, custom_terms):
    term_set = set(custom_terms)
    max_len = max(len(term) for term in term_set)
    merged_result = []

    for sentence in ws_result:
        merged_sentence = []
        i = 0
        while i < len(sentence):
            match = None
            for l in range(min(max_len, len(sentence) - i), 0, -1):
                phrase = ''.join(sentence[i:i+l])
                if phrase in term_set:
                    match = phrase
                    i += l
                    break
            if match:
                merged_sentence.append(match)
            else:
                merged_sentence.append(sentence[i])
                i += 1
        merged_result.append(merged_sentence)
    return merged_result

# --- 使用方式 ---
text_list = df['拜訪備註_文字'].dropna().tolist()

# 第一步：CKIP 斷詞
ws_segments = ws(text_list)

# 第二步：保險術語合併（基於 CKIP 結果後處理）
final_segments = merge_custom_terms(ws_segments, insurance_terms)



# 停用詞處理
import requests
from opencc import OpenCC

# 四個停用詞庫 URL（來自 https://github.com/goto456/stopwords）
stopword_urls = [
    "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/scu_stopwords.txt"
]

# 初始化簡體轉繁體轉換器
cc = OpenCC('s2t')  # Simplified to Traditional

# 儲存繁體停用詞
stopwords_trad = set()

for url in stopword_urls:
    print(f"載入中: {url}")
    response = requests.get(url)
    response.encoding = 'utf-8'
    if response.status_code == 200:
        lines = response.text.strip().splitlines()
        # 逐行轉換為繁體，去除空行
        for line in lines:
            word = cc.convert(line.strip())
            if word:
                stopwords_trad.add(word)

print(f"共獲得停用詞數量（繁體）: {len(stopwords_trad)}")

filtered_words = [
    [word for word in sentence if word not in stopwords_trad and len(word) > 1]
    for sentence in final_segments
]


# # 人工輸入停用詞
# stopwords = set(['，', '。', '的', '有', '是', '在', '了', '及', '和', '也'])

# # 去除停用詞與單字
# filtered_words = [
#     [word for word in sentence if word not in stopwords and len(word) > 1]
#     for sentence in word_segments
# ]

# 匯出斷詞後結果
import pickle

with open('D:/備註文字探勘/filtered_words_2025.pkl', 'wb') as f:
    pickle.dump(filtered_words, f)
    

# %% 變更環境到 PYTHON311
import pickle

with open('D:/備註文字探勘/filtered_words_2025.pkl', 'rb') as f:
    filtered_words = pickle.load(f)

# 詞頻統計
from collections import Counter

# 合併大小寫，排除純數字、排除類似日期格式的詞
import re

def clean_word(word):
    word = word.lower().strip()

    # 移除全是數字（含千分位逗號），如 3000000 或 3,000,000
    if re.fullmatch(r'[\d,]+', word):
        return None

    # 移除明確的日期格式，如 3/4、2025-04-11、3-31
    if re.fullmatch(r'\d{1,4}[/-]\d{1,2}([/-]\d{1,2})?', word):
        return None

    # 移除明確的年月日格式，例如 2025年4月10日（進階可選）
    if re.fullmatch(r'\d+年\d+月\d+日', word):
        return None

    # 移除太短的詞（1 個字母或空白）
    if len(word) <= 1:
        return None

    return word


# 淨化詞語後重新統計詞頻
flat_words_cleaned = [clean_word(word) for sentence in filtered_words for word in sentence]
flat_words_cleaned = [word for word in flat_words_cleaned if word]  # 去除 None

word_freq_clean = Counter(flat_words_cleaned)



# 建立詞頻表 DataFrame
freq_df = pd.DataFrame(word_freq_clean.most_common(), columns=['詞語', '出現次數'])
# freq_df.head(20)  


# Matplotlib requires numpy>=1.23
# C:\Users\z01788\AppData\Local\Programs\Python\Python313\python.exe -m pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 產生文字雲
wc = WordCloud(
    font_path='C:/Windows/Fonts/msjh.ttc',  # 微軟正黑體
    background_color='white',
    width=800,
    height=600
)

# 將換行符號從詞語中移除
word_freq_clean_fixed = {k.replace('\n', '').replace('\r', ''): v for k, v in word_freq_clean.items()}

wc.generate_from_frequencies(word_freq_clean_fixed)

# 顯示詞雲
plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

ceo = pd.read_excel("D:/備註文字探勘/拜訪0409.xlsx", sheet_name=4)
# 確保兩邊業代欄位都是字串型別（保留前導0）
visit['業代'] = visit['業代'].astype(str).str.zfill(6)
ceo['業代'] = ceo['業代'].astype(str).str.zfill(6)
ceo['CEO課程起日'] = pd.to_datetime(ceo['CEO課程起日'], errors='coerce')

# 新增CEO欄位（業代有在 ceo 表中）
ceo_after_date = ceo[ceo['CEO課程起日'] >= pd.Timestamp('2024/10/01')]
df['CEO'] = df['業代'].isin(ceo_after_date['業代']).astype(int)
print(df['CEO'].value_counts())
print(df.groupby('CEO')['業代'].nunique())

tags = pd.read_excel("D:/備註文字探勘/拜訪0409.xlsx", sheet_name=5)
# 只保留需要的欄位，避免重複欄位影響合併
tags_subset = tags[['拜訪紀錄UUID', '標籤名稱']]

# 合併資料，若一個 UUID 對應多個標籤則會產生多列
df = df.merge(tags_subset, on='拜訪紀錄UUID', how='left')


# %% 納入成交資料
# 再執行一次"% 文字前處理"
text_list = df['拜訪備註_文字'].dropna().tolist()  # 避免空值

policy = pd.read_excel("D:/備註文字探勘/拜訪0409.xlsx", sheet_name=3) 

# # 合併標籤：成交為 1，沒對到為 0
# df['成交'] = df['客戶UUID'].isin(policy['經紀人1-被保人CRM UUID']).astype(int)

# 接著取得與 filtered_words 的對應
# 注意：filtered_words 是經過 dropna 的 text_list 對應到的 → 需建立 index 對應
df_valid_notes = df[df['拜訪備註_文字'].notna()].reset_index(drop=True)


# # 先針對每個客戶 UUID，找出最近一筆（最大投保日）的保單資訊
# policy_sorted = policy.sort_values('投保日')
# latest_policy = policy_sorted.groupby('經紀人1-被保人CRM UUID').tail(1)

# # 篩出需要的欄位
# latest_policy_info = latest_policy[['經紀人1-被保人CRM UUID', '被保人性別', '被保人投保年齡']].drop_duplicates()

# # 合併進 df_valid_notes
# df_valid_notes = df_valid_notes.merge(
#     latest_policy_info,
#     left_on='客戶UUID',
#     right_on='經紀人1-被保人CRM UUID',
#     how='left'
# )

# # 可選：刪除重複欄位名稱
# df_valid_notes = df_valid_notes.drop(columns=['經紀人1-被保人CRM UUID'])



from collections import defaultdict
import numpy as np

# 確保時間格式
df_valid_notes['拜訪時間'] = pd.to_datetime(df_valid_notes['拜訪時間'])
policy['投保日'] = pd.to_datetime(policy['投保日'])

# 建立字典：UUID → 該客戶所有投保日清單（已排序）
policy_dict = defaultdict(list)
for _, row in policy.iterrows():
    uuid = row['經紀人1-被保人CRM UUID']
    policy_dict[uuid].append(row['投保日'])

# 定義：找出與拜訪時間最接近的投保日，計算「拜訪時間 - 投保日」的天數差
def get_closest_policy_diff(uuid, visit_time):
    投保日_list = policy_dict.get(uuid, [])
    if not 投保日_list:
        return pd.Series([np.nan, pd.NaT])
    
    # 計算與每個投保日的時間差
    diffs = [(visit_time - d).days for d in 投保日_list]
    closest_idx = np.argmin(np.abs(diffs))  # 找絕對值最小（最接近）
    return pd.Series([diffs[closest_idx], 投保日_list[closest_idx]])

# 新增兩個欄位：「拜訪與最近投保日天數差」、「最近投保日」
df_valid_notes[['拜訪與投保日天數差', '最近投保日']] = df_valid_notes.apply(
    lambda row: get_closest_policy_diff(row['客戶UUID'], row['拜訪時間']),
    axis=1
)

# 針對每位客戶，按照拜訪時間排序，給拜訪順序編號
df_valid_notes['拜訪次數'] = df_valid_notes.sort_values(['客戶UUID', '拜訪時間']) \
    .groupby('客戶UUID').cumcount() + 1

# 篩選成交前 21 天以內
subset = df_valid_notes[df_valid_notes['拜訪與投保日天數差'].between(-21, 0)]

# groupby 分析距離成交的天數 vs 標籤（動作）
summary = subset.groupby(['拜訪與投保日天數差', '標籤名稱']).size().unstack().fillna(0)

# 統計成交前 N 次拜訪中做了哪些動作
action_by_visit_order = df_valid_notes.groupby(['拜訪次數', '標籤名稱']).size().unstack().fillna(0)

# 計算比例
columns_to_analyze = ['建議書', '成交', '約訪', '需求確認', '面談']
summary['總和'] = summary[columns_to_analyze].sum(axis=1)
for col in columns_to_analyze:
    summary[f'{col}_比例'] = summary[col] / summary['總和']



df_成交 = df_valid_notes[df_valid_notes['是否成交'] == 1]
df_未成交 = df_valid_notes[df_valid_notes['是否成交'] == 0]

df_未成交_last = df_未成交.groupby('客戶UUID').tail(1) # 每位未成交者的「最後一次拜訪」

# 統計未成交者的最後一次拜訪次序分布
未成交_次序分布 = df_未成交_last['拜訪次數'].value_counts().sort_index()

# 拜訪目的統計
未成交_目的分布 = df_未成交_last['標籤名稱'].value_counts()

df_成交['距離成交日天數'] = (pd.to_datetime(df_成交['拜訪時間']) - pd.to_datetime(df_成交['最近投保日'])).dt.days
df_成交當次 = df_成交[df_成交['距離成交日天數'] == 0]

成交_次序分布 = df_成交當次['拜訪次數'].value_counts().sort_index()
成交_目的分布 = df_成交當次['標籤名稱'].value_counts()

df_未成交_last['卡關點'] = df_未成交_last['拜訪次數'].astype(str) + '次_' + df_未成交_last['標籤名稱']
卡關統計 = df_未成交_last['卡關點'].value_counts()

# plt.figure(figsize=(14, 6))
# bottom = [0] * len(summary)
# for col in ['建議書', '成交', '約訪', '需求確認', '面談']:
#     plt.bar(summary.index, summary[col], bottom=bottom, label=col)
#     bottom = [i + j for i, j in zip(bottom, summary[col])]

# plt.title('距成交日前不同天數的拜訪目的分布（堆疊長條圖）')
# plt.xlabel('與投保日的天數')
# plt.ylabel('拜訪紀錄數')
# plt.legend(title='拜訪目的')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# from collections import defaultdict

# 1. 建立字典：客戶UUID 對應所有投保日（datetime）
policy_dict = defaultdict(list)
for _, row in policy.iterrows():
    uuid = row['經紀人1-被保人CRM UUID']
    policy_dict[uuid].append(row['投保日'])

# 2. 新邏輯：拜訪日在任一投保日前 21 天內 → 判為 1
def check_within_21_days(uuid, visit_time):
    投保日_list = policy_dict.get(uuid, [])
    return int(any((0 <= (投保日 - visit_time).days) for 投保日 in 投保日_list))

# 3. 套用邏輯
df_valid_notes['是否成交'] = df_valid_notes.apply(
    lambda row: check_within_21_days(row['客戶UUID'], row['拜訪時間']),
    axis=1
)



# df_valid_notes = df_valid_notes.merge(
#     policy[['經紀人1-被保人CRM UUID', '商品險種次類別', '投保日', '受理件數', '保額-匯率', '繳款保費new']],
#     left_on='客戶UUID',
#     right_on='經紀人1-被保人CRM UUID',
#     how='left'
# )

# %% 基本統計(詞頻、字數)
# 計算每筆備註的字數（不含空白）
df_valid_notes['備註字數'] = df_valid_notes['拜訪備註_文字'].apply(lambda x: len(str(x).replace(" ", "").replace("\n", "")))

# 計算每筆備註的句數（以句點、問號、驚嘆號、頓號、換行等作為句子的分隔符）
def count_sentences(text):
    if pd.isna(text):
        return 0
    # 使用正則表達式依標點拆句
    sentences = re.split(r'[。！？\n\r]+', str(text))
    # 移除空句
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

df_valid_notes['備註行數'] = df_valid_notes['拜訪備註_文字'].apply(count_sentences)

# 每個業務員的平均字數
agent_avg_length = df_valid_notes.groupby('業代')['備註字數'].mean().reset_index()
agent_avg_length = agent_avg_length.rename(columns={'備註字數': '平均備註字數'})

# 計算百分位
# 僅保留 備註字數 > 0 的紀錄來做平均
df_valid_notes_nonzero = df_valid_notes[df_valid_notes['備註字數'] > 0]

# 計算每個業代的平均備註字數（不含 0 字數的紀錄）
agent_avg_length = df_valid_notes_nonzero.groupby('業代')['備註字數'].mean().reset_index()
agent_avg_length = agent_avg_length.rename(columns={'備註字數': '平均備註字數'})

# 計算百分位
agent_avg_length['平均字數百分位'] = agent_avg_length['平均備註字數'].rank(pct=True)

# 分區標籤
def categorize_percentile(p):
    if p <= 0.25:
        return '低'
    elif p <= 0.75:
        return '中'
    else:
        return '高'

agent_avg_length['平均字數區間'] = agent_avg_length['平均字數百分位'].apply(categorize_percentile)


agent_avg_length['平均字數百分位'] = agent_avg_length['平均備註字數'].rank(pct=True)


# 根據百分位數分區
def categorize_percentile(p):
    if p <= 0.25:
        return '低'
    elif p <= 0.75:
        return '中'
    else:
        return '高'

agent_avg_length['平均字數區間'] = agent_avg_length['平均字數百分位'].apply(categorize_percentile)

# 把結果合併回原資料
df_valid_notes = df_valid_notes.merge(agent_avg_length[['業代', '平均字數區間', '平均字數百分位']], on='業代', how='left')

# 把結果合併回原資料
# 1. 計算每位業代的總受理件數
policy_by_agent = policy.groupby('經紀人業代')['保單申請案號'].nunique().reset_index()
policy_by_agent = policy_by_agent.rename(columns={'經紀人業代': '業代', '保單申請案號': '總受理件數'})

# 2. 計算每位業代的計績FYC總和
fyc_by_agent = policy.groupby('經紀人業代')['計績FYC'].sum().reset_index()
fyc_by_agent = fyc_by_agent.rename(columns={'經紀人業代': '業代', '計績FYC': '總FYC'})

# 3. 合併兩個指標
policy_summary = policy_by_agent.merge(fyc_by_agent, on='業代', how='outer')

# 4. 計算件均FYC（注意除以0的情況）
policy_summary['件均FYC'] = policy_summary['總FYC'] / policy_summary['總受理件數']
policy_summary['件均FYC'] = policy_summary['件均FYC'].replace([float('inf'), -float('inf')], pd.NA)

valid_fyc = policy_summary['總FYC'].dropna()

# 產生四分位分組標籤
quartile_labels = ['低', '中低', '中高', '高']

# 建立一個新欄位「總FYC四分位」
policy_summary['總FYC四分位'] = pd.qcut(policy_summary['總FYC'], q=4, labels=quartile_labels)

# 件均FYC 四分位分組
policy_summary['件均FYC四分位'] = pd.qcut(policy_summary['件均FYC'], q=4, labels=quartile_labels)

# 5. 合併回原始 df_valid_notes
df_valid_notes = df_valid_notes.merge(policy_summary, on='業代', how='left')


# %% 檢定
target_cols = ['業代', '備註字數', '總FYC', '平均字數區間', '標籤名稱']

# 先排除空值
filtered_df = df_valid_notes[
    (df_valid_notes['總FYC'].notna()) & 
    (df_valid_notes['備註字數'] > 0)
]

# 依業代分組後，保留第一筆紀錄
first_per_agent = filtered_df.groupby('業代', as_index=False).first()

# 再選取所需欄位
first_per_agent = first_per_agent[target_cols]


from scipy.stats import spearmanr

for group in first_per_agent['平均字數區間'].unique():
    subset = first_per_agent[first_per_agent['平均字數區間'] == group]
    if len(subset) >= 2:
        r, p = spearmanr(subset['備註字數'], subset['總FYC'])
        print(f"{group}（Spearman）→ ρ: {r:.3f}, p: {p:.4f}, N={len(subset)}")



# %% 3D Plot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 計算每個業務員的三個指標
agent_stats = df_valid_notes.groupby('業代').agg(
    平均字數=('備註字數', 'mean'),
    拜訪數=('拜訪紀錄UUID', 'nunique'),
    客戶數=('客戶UUID', 'nunique')  # 確認這個是你的客戶識別欄位名稱
).reset_index()

fyc = pd.read_excel("D:/備註文字探勘/FYC.xlsx", sheet_name=0) 
agent_stats = agent_stats.merge(fyc, on='業代', how='left')

# 排除離群值
# 計算 IQR (四分位數範圍)
Q1 = agent_stats['計績FYC'].quantile(0.25)
Q3 = agent_stats['計績FYC'].quantile(0.75)
IQR = Q3 - Q1

# 定義合理範圍：Q1 - 1.5*IQR ~ Q3 + 1.5*IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 篩選留下合理範圍內的業務員
agent_stats_filtered = agent_stats[
    (agent_stats['計績FYC'] >= lower_bound) &
    (agent_stats['計績FYC'] <= upper_bound)
]

# 計算三個指標的平均
avg_x = agent_stats_filtered['平均字數'].mean()
avg_y = agent_stats_filtered['拜訪數'].mean()
avg_z = agent_stats_filtered['客戶數'].mean()

# 畫互動式 3D 散點圖
fig = px.scatter_3d(
    agent_stats_filtered,
    x='平均字數',
    y='拜訪數',
    z='客戶數',
    # text='業代',            # 懸浮顯示業代名稱
    color='計績FYC',       # 用平均字數著色（可改成其他指標）
    opacity=0.7,
    size_max=10
)

fig.update_traces(marker=dict(size=5))  # 點大小


# 平均字數（X固定）的線
fig.add_trace(go.Scatter3d(
    x=[avg_x, avg_x],
    y=[agent_stats_filtered['拜訪數'].min(), agent_stats_filtered['拜訪數'].max()],
    z=[agent_stats_filtered['客戶數'].min(), agent_stats_filtered['客戶數'].min()],
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='平均字數線'
))

# 拜訪數（Y固定）的線
fig.add_trace(go.Scatter3d(
    x=[agent_stats_filtered['平均字數'].min(), agent_stats_filtered['平均字數'].max()],
    y=[avg_y, avg_y],
    z=[agent_stats_filtered['客戶數'].min(), agent_stats_filtered['客戶數'].min()],
    mode='lines',
    line=dict(color='blue', dash='dash'),
    name='平均拜訪數線'
))

# 客戶數（Z固定）的線
fig.add_trace(go.Scatter3d(
    x=[agent_stats_filtered['平均字數'].min(), agent_stats_filtered['平均字數'].min()],
    y=[agent_stats_filtered['拜訪數'].min(), agent_stats_filtered['拜訪數'].min()],
    z=[avg_z, avg_z],
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='平均客戶數線'
))

# 圖面微調
fig.update_layout(
    title='業務員3D分布圖（含平均線）',
    scene=dict(
        xaxis_title='平均備註字數',
        yaxis_title='拜訪數',
        zaxis_title='客戶數'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show(renderer="browser")

# %% 情感分數
from snownlp import SnowNLP
import seaborn as sns

sentiments = []

for i, comment in enumerate(text_list, start=1):
    if isinstance(comment, str) and comment.strip():  # 非空字串
        try:
            s = SnowNLP(comment)
            score = s.sentiments  # 介於 0~1
            print(f"Comment {i}: Sentiment Score: {score:.4f}")
            sentiments.append(score)
        except Exception as e:
            print(f"Comment {i}: Error - {e}")
            sentiments.append(None)
    else:
        print(f"Comment {i}: Skipped (not valid text)")
        sentiments.append(None)

sns.kdeplot(sentiments, fill=True, color='skyblue')
plt.title('Kernel Density Estimate of Sentiment Distribution')
plt.xlabel('Polarity')
plt.ylabel('Density')
plt.show()

df_valid_notes['情感分數'] = sentiments

# %% LDA 主題建模
# ① 計算最佳主題數（coherence_count）
# ② 訓練 LDA 模型（使用最佳主題數）
# ③ print_topics()
# ④ pyLDAvis 可視化儲存
# ⑤ 對 LDA 主題結果加上語意標籤

# https://reurl.cc/eMdVWQ 
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')

filtered_words_cleaned = [
    [clean_word(word) for word in sentence]
    for sentence in filtered_words
]

# 移除 None 或空值的詞語
filtered_words_cleaned = [
    [word for word in sentence if word]  # 避免 None 或 '' 留下來
    for sentence in filtered_words_cleaned
]

df_valid_notes['拜訪備註_詞語'] = filtered_words_cleaned
df_valid_notes['詞數'] = df_valid_notes['拜訪備註_詞語'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# 依照客戶UUID與拜訪時間排序後加上累計次數
df_valid_notes = df_valid_notes.sort_values(by=['客戶UUID', '拜訪時間'])
df_valid_notes['拜訪次數'] = df_valid_notes.groupby('客戶UUID').cumcount() + 1



# 建立字典與語料
dictionary = corpora.Dictionary(filtered_words_cleaned) # 使用corpora放入filtered_words生成辭典
corpus = [dictionary.doc2bow(text) for text in filtered_words_cleaned] # 將辭典中單詞轉換為向量

# # 列出各主題內容及分數
# def coherence(num_topics):
#     ldamodel = LdaModel(corpus, num_topics = num_topics, id2word = dictionary, passes = 30, random_state = 42)
#     print(ldamodel.print_topics(num_topics = num_topics, num_words = 15))
#     ldacm = CoherenceModel(model = ldamodel, texts = filtered_words, dictionary = dictionary, coherence="c_v")
#     print(ldacm.get_coherence())
#     return ldacm.get_coherence()

def coherence(num_topics):
    ldamodel = LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, 
        passes=30, iterations=100, random_state=42
    )
    ldacm = CoherenceModel(model=ldamodel, texts=filtered_words_cleaned, dictionary=dictionary, coherence="c_v")
    coherence_score = ldacm.get_coherence()
    print(f"主題數: {num_topics}, coherence: {coherence_score:.4f}")
    return coherence_score


def coherence_count(num_topics):
    x = range(1,num_topics+1)
    y = [coherence(i) for i in x]
    return x, y
    
x, y = coherence_count(20)

# 找出最高的 coherence 對應的主題數
best_idx = y.index(max(y))
best_x = x[best_idx]
best_y = y[best_idx]

# 繪圖
plt.plot(x, y, marker='o', zorder=1)
plt.scatter(best_x, best_y, color='red', zorder=2, label=f'最佳主題數: {best_x}')
plt.xlabel("主題數目")
plt.ylabel("coherence大小")
plt.title("主題-coherence變化情形")
plt.legend()
plt.show()
# coherence 越高代表分類效果越好
# run time: 1hr


best_num_topics = best_x  
lda = LdaModel(corpus, num_topics = best_num_topics, id2word = dictionary, passes = 30, random_state = 42) 
# passes: 每篇文章會被反覆使用多少次來更新主題分佈
topics_lst = lda.print_topics()
print(topics_lst)

import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()
data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(data, f"D:/備註文字探勘/{best_num_topics}_topic_model.html")
# λ越小會呈現出在這個主題中獨特性越高的單詞


# 顯示前50筆
# for i, (k, v) in enumerate(word_freq_clean_fixed.items()):
#     if i >= 50:
#         break
#     print(k, v)


# 預設主題提示詞字典
# theme_keywords = {
#     "家庭與人生": ["二寶", "預產期", "懷孕", "媽媽", "爸爸", "家庭", "責任", "人生", "恭喜"],
#     "缺口規劃": ["保障", "足額", "缺口", "補強", "保單健診", "補強", "重大傷病", "癌症", "險理"],
#     "商品與契約": ["保單", "簽約", "分紅躉繳", "躉繳", "台幣", "50萬", "保誠", "台壽", "凱基", "台銀", "進度", "車險", "國泰產", "產保單", "華南"],
#     "經濟狀況": ["收入", "貸款", "做黑", "博弈", "事業", "工作", "2.1萬", "7.5萬"]}

theme_keywords = {
    "家庭與人生": ["媽媽", "懷孕", "寶寶", "家庭", "二寶", "小孩", "小朋友", "生小孩", "人生", "生育", "父母"],
    "缺口與健診": ["健診", "保障", "保單健診", "缺口", "補強", "足額", "險種", "調整保額", "保額不足", "保單整理", "補保"],
    "商品與契約": ["簽約", "契約", "躉繳", "續期", "台幣", "美元", "年繳", "月繳", "繳別", "投保", "受益人", "投保公司"],
    "保險品牌與公司": ["國泰", "新光", "保誠", "富邦", "台壽", "凱基", "南山", "台銀", "友邦", "華南", "三商"],
    "健康與疾病": ["癌症", "重大傷病", "住院", "手術", "失能", "醫療險", "長照", "疾病", "醫療保障", "殘扶"],
    "經濟與財務": ["收入", "貸款", "工作", "博弈", "事業", "負債", "房貸", "現金流", "薪水"],
    "產物與車險": ["產險", "車險", "車禍", "強制險", "國泰產", "第三人責任險", "車輛", "華南產", "事故"],
    "時間與進度": ["預約", "進度", "時間", "排程", "重新安排", "等保單", "處理中", "未回覆", "等待"],
}

# 對 LDA 主題結果加上語意標籤
def label_topic_by_keywords(topic_words, theme_dict):
    for theme, keywords in theme_dict.items():
        if any(word in topic_words for word in keywords):
            return theme
    return "未標註"

# 主題語意對照表
topic_list = lda.show_topics(num_topics=best_num_topics, num_words=30, formatted=False)
topic_labels = []
for idx, topic in topic_list:
    words = [w for w, _ in topic]
    label = label_topic_by_keywords(words, theme_keywords)
    topic_labels.append((idx, label, words))

# 輸出結果
for idx, label, words in topic_labels:
    print(f"主題 {idx}｜{label}｜關鍵詞：{', '.join(words)}")
    
# %%%% 將主題分佈結果整合到 df_valid_notes
# 取得每篇文章的主主題
doc_topics = lda.get_document_topics(corpus)
dominant_topic = [max(doc, key=lambda x: x[1])[0] for doc in doc_topics]  # 取最大機率主題

# 建立主題對應語意標籤的對照字典
topic_id_to_label = {idx: label for idx, label, _ in topic_labels}

# 把標籤套用到文章主題
topic_labels_for_docs = [topic_id_to_label.get(tid, -1) for tid in dominant_topic]

# 加入 df_valid_notes
df_valid_notes['主題編號'] = dominant_topic
df_valid_notes['主題標籤'] = topic_labels_for_docs

doc_topic_distributions = []

for doc_id, doc_topics_dist in enumerate(doc_topics):
    for topic_id, prob in doc_topics_dist:
        doc_topic_distributions.append({
            "文檔ID": doc_id,
            "主題編號": topic_id,
            "主題標籤": topic_id_to_label.get(topic_id, "未標註"),
            "機率": prob
        })

df_doc_topics = pd.DataFrame(doc_topic_distributions)

# 建立每篇評論在每個主題下的權重表（文檔ID x 主題）
topic_weight_wide = df_doc_topics.pivot_table(
    index="文檔ID", columns="主題編號", values="機率", fill_value=0
)

# 重新命名欄位（加上前綴）
topic_weight_wide.columns = [f"主題_{i}" for i in topic_weight_wide.columns]

# 合併回 df_notes_with_comment
df_valid_notes = pd.concat([df_valid_notes, topic_weight_wide], axis=1)

# 建立一個新的 DataFrame
df_notes_with_comment = df_valid_notes[df_valid_notes["是否有備註(排除#)"] == 1].reset_index(drop=True)
df_deal = df_notes_with_comment[df_notes_with_comment["拜訪後成交"] == 1].reset_index(drop=True)

df_exploded = df_valid_notes.explode('拜訪備註_詞語')  # 將 list 拆成一列一個詞語


# 可選：重新命名欄位以符合 Tableau 習慣
df_exploded = df_exploded.rename(columns={'拜訪備註_詞語': '詞語'})

# 計算每個業代的拜訪紀錄數量
visit_count = df.groupby('業代')['拜訪紀錄UUID'].count()

# 計算每個業代「是否有備註」為1的數量
remark_count = df[df['是否有備註(排除#)'] == 1].groupby('業代')['是否有備註(排除#)'].count()

# 合併結果
result = pd.DataFrame({
    '拜訪紀錄數量': visit_count,
    '有備註數量': remark_count
}).fillna(0).astype(int)

# 輸出
with pd.ExcelWriter("D:/備註文字探勘/拜訪備註分析_2025.xlsx", engine='openpyxl') as writer:
    df_valid_notes.to_excel(writer, sheet_name='words', index=False)
    
    # df_word_freq = pd.DataFrame(word_freq_clean_fixed.items(), columns=['詞語', '出現次數'])
    # df_word_freq.to_excel(writer, sheet_name='Frequency', index=False)
    df_exploded.to_excel(writer, sheet_name='exploded', index=False)
df_exploded.to_csv("D:/備註文字探勘/exploded_2025.csv", index=False, encoding='utf-8-sig')

# %% 成交機率預測模型
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

# 特徵欄位
features = ['主題_0', '主題_1', '主題_2', '情感分數', '拜訪次數']
df_notes_with_comment['拜訪與投保日天數差'] = df_notes_with_comment['拜訪與投保日天數差'].fillna(999)
df_model = df_notes_with_comment[df_notes_with_comment['拜訪與投保日天數差'] != 999]

X = df_model[features]
y = df_model['拜訪與投保日天數差']

# 拆分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 模型建立與訓練
# model = RandomForestClassifier(class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)



from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=9, 
    # learning_rate=0.05,
    # max_depth=4,
    # min_child_weight=3,
    # n_estimators=200, 
    use_label_encoder=False, 
    eval_metric='logloss'
    )
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from xgboost import XGBRegressor

model = XGBRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
df_model.loc[X_test.index, '預測天數差'] = y_pred



# # Find best parameters
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'min_child_weight': [3, 5],
#     'learning_rate': [0.05, 0.1],
#     'n_estimators': [100, 200],
# }

# grid = GridSearchCV(
#     estimator=XGBClassifier(scale_pos_weight=5, eval_metric='logloss'),
#     param_grid=param_grid,
#     scoring='f1',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )

# grid.fit(X_train, y_train)
# print(grid.best_params_)



# ====== 5. ROC 曲線與 AUC ======
y_prob = model.predict_proba(X_test)[:, 1]  # 機率預測
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# 加上 precision/recall
print("AUC:", auc_score)
print("AUC_PR", average_precision_score(y_test, y_prob))

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 預測是否成交（門檻 = 0.5，可改）
y_pred = (y_prob >= 0.5).astype(int)

# 整合成 DataFrame
df_pred_result = X_test.copy()
df_pred_result['預測成交機率'] = y_prob
df_pred_result['預測是否成交'] = y_pred
df_pred_result['實際是否成交'] = y_test.values

cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

# 顯示數值
print("混淆矩陣：")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["成交", "未成交"])
disp.plot(cmap=plt.cm.Blues)
plt.title("混淆矩陣（預測 vs 實際）")
plt.show()


# 檢測特徵重要姓
importances = model.feature_importances_
for name, score in zip(X.columns, importances):
    print(f"{name}: {score:.4f}")

# %% 成交文字特徵統計
from collections import Counter

# 每則備註的詞語為 list 格式
df_notes_with_comment['詞語_列表'] = df_notes_with_comment['拜訪備註_詞語'].apply(eval)  # 如果是字串的 list，要轉換

word_stats = {}

for _, row in df_notes_with_comment.iterrows():
    for word in row['詞語_列表']:
        if word not in word_stats:
            word_stats[word] = {'count': 0, 'success': 0}
        word_stats[word]['count'] += 1
        word_stats[word]['success'] += row['拜訪後_成交']

# 轉為 DataFrame
import pandas as pd
word_df = pd.DataFrame([
    {'詞語': k, '出現次數': v['count'], '成交次數': v['success'], '成交率': v['success']/v['count']}
    for k, v in word_stats.items() if v['count'] >= 5  # 可設定最低門檻
])

