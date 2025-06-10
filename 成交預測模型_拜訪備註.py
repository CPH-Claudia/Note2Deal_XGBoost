# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:07:20 2025

@author: Z01788
"""

import pandas as pd
import numpy as np
import random
import os

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, classification_report, 
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False 

# %% 文字前處理 
# pip install numpy==1.23 --upgrade
visit = pd.read_excel("D:/備註文字探勘/拜訪_2024冬夏賽.xlsx", sheet_name="VISIT") 
visit['拜訪時間'] = pd.to_datetime(visit['拜訪時間'], errors='coerce')

# 篩選 2024/2/1 ~ 2024/6/30 和 2024/8/1 ~ 2024/12/31 的筆數
visit_filtered = visit[
    ((visit['拜訪時間'] >= pd.Timestamp('2024-02-01')) & (visit['拜訪時間'] <= pd.Timestamp('2024-06-30'))) |
    ((visit['拜訪時間'] >= pd.Timestamp('2024-08-01')) & (visit['拜訪時間'] <= pd.Timestamp('2024-12-31')))
]

df = visit_filtered.copy()

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

# 計算每位業代對每個客戶的拜訪次數
df['拜訪次數'] = (
    df.groupby(['業代', '客戶UUID'])['拜訪紀錄UUID']
    .transform('count')
)

# 計算每位業代對所有客戶的平均拜訪次數
df['平均每客戶拜訪次數'] = (
    df.groupby('業代')['拜訪次數']
    .transform('mean')
)
# df['平均每客戶拜訪次數'].describe().T

le = LabelEncoder()
df['營業單位_編碼'] = le.fit_transform(df['營業單位'])

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

with open('D:/備註文字探勘/filtered_words_2024.pkl', 'wb') as f:
    pickle.dump(filtered_words, f)
    
# %% 變更環境到 PYTHON311
import pickle

with open('D:/備註文字探勘/filtered_words_2024.pkl', 'rb') as f:
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


# %% 合併資料
# %%%% 納入成交資料
# 再執行一次"% 文字前處理"
tags = pd.read_excel("D:/備註文字探勘/拜訪_2024冬夏賽.xlsx", sheet_name="TAGS")
# 只保留需要的欄位，避免重複欄位影響合併
tags_subset = tags[['拜訪紀錄UUID', '標籤名稱']]

# 合併資料，若一個 UUID 對應多個標籤則會產生多列
df = df.merge(tags_subset, on='拜訪紀錄UUID', how='left')

# ceo = pd.read_excel("D:/備註文字探勘/拜訪_冬賽.xlsx", sheet_name="CEO", dtype={'業代': str})

# # 新增CEO欄位（業代有在 ceo 表中）
# df['CEO'] = df['業代'].isin(ceo['業代']).astype(int)
# # print(df.groupby('CEO')['業代'].nunique())

# %%%% 納入業務資料
agent = pd.read_excel("D:/備註文字探勘/拜訪_2024冬夏賽.xlsx", sheet_name="AGENT", dtype={'業代': str})

agent_stage = {'CB': 0, 'JB': 1, 'PB': 2, 'SB': 3}
agent['職級_代碼'] = agent['月結檔 | 職級'].map(agent_stage)

# 1 取每位業代的最後一筆基本資料
agent_sorted = agent.sort_values(['業代', '計績年月'])  # 確保時間排序

# 加上排序與筆次
agent_sorted['筆次'] = agent_sorted.groupby('業代').cumcount()
agent_sorted['上一期職級'] = agent_sorted.groupby('業代')['職級_代碼'].shift(1)
# 判斷是否是升遷，排除第一筆
agent_sorted['是否晉升'] = (agent_sorted['筆次'] > 0) & (agent_sorted['職級_代碼'] > agent_sorted['上一期職級'])
# 找出每位業代的第一次晉升時間
agent_sorted['晉升日'] = agent_sorted.where(agent_sorted['是否晉升']).groupby('業代')['計績年月'].transform('min')

last_info = agent_sorted.groupby('業代').last().reset_index()[['業代', '職級_代碼', '目前年齡', '性別', '年資']]

# 2 夏賽（202403~202406）活動參與率 & FYC
last = agent[(agent['計績年月'] >= 202303) & (agent['計績年月'] <= 202306) | 
            (agent['計績年月'] >= 202309) & (agent['計績年月'] <= 202312)]
last_summary = last.groupby('業代').agg({
    '活動參與率': 'mean',
    '新繳款FYC': 'mean'
}).reset_index().rename(columns={
    '活動參與率': '上年度活動參與率',
    '新繳款FYC': '上年度FYC'
})

# 3 冬賽（202409~202412）活動參與率 & FYC
now = agent[(agent['計績年月'] >= 202403) & (agent['計績年月'] <= 202406) | 
            (agent['計績年月'] >= 202409) & (agent['計績年月'] <= 202412)]
now_summary = now.groupby('業代').agg({
    '活動參與率': 'mean',
    '新繳款FYC': 'mean'
}).reset_index().rename(columns={
    '活動參與率': '今年度活動參與率',
    '新繳款FYC': '今年度FYC'
})

# 4 得獎數（不限月份）
award_count = agent[agent['得獎時間(年度)'].notna()] \
    .groupby('業代')['人事狀況碼-內容'].count().reset_index() \
    .rename(columns={'人事狀況碼-內容': '得獎數'})

# 5 合併所有資料
agent_summary = last_info \
    .merge(last_summary, on='業代', how='left') \
    .merge(now_summary, on='業代', how='left') \
    .merge(award_count, on='業代', how='left')

# 6 補值處理
agent_summary[['上年度活動參與率', '上年度FYC', '今年度活動參與率', '今年度FYC']] = agent_summary[
    ['上年度活動參與率', '上年度FYC', '今年度活動參與率', '今年度FYC']
].fillna(0)

agent_summary['得獎數'] = agent_summary['得獎數'].fillna(0).astype(int)


# 只留下有晉升日的業代（非空）
has_promotion = agent_sorted[agent_sorted['晉升日'].notna()][['業代', '晉升日']].drop_duplicates('業代')

# 合併晉升日資料
agent_summary = agent_summary.merge(has_promotion, on='業代', how='left')

# 無晉升者 → 填 0（也可依需求填 '無' 或 NaT）
agent_summary['晉升日'] = agent_summary['晉升日'].fillna(0)


# 7 命名欄位與職級編碼
agent_summary = agent_summary.rename(columns={
    '職級_代碼': '最新職級',
    '目前年齡': '業務目前年齡',
    '性別': '業務性別',
    '年資': '目前年資'
})

sex_stage = {'男': 0, '女': 1}
agent_summary['業務性別'] = agent_summary['業務性別'].map(sex_stage)

# 8 合併進主 df
df = df.merge(agent_summary, on='業代', how='left')

# 計算晉升
# 1 保留晉升日轉換
def convert_yyyymm_to_datetime(x):
    if x == 0 or pd.isna(x):
        return pd.NaT
    else:
        x = int(x)
        year = x // 100
        month = x % 100
        return pd.to_datetime(f"{year}-{month:02d}-01")

df['晉升日期_dt'] = df['晉升日'].apply(convert_yyyymm_to_datetime)

# 2 是否有晉升（只要有晉升日，不看拜訪先後）
df['是否晉升'] = df['晉升日期_dt'].apply(lambda x: 1 if pd.notna(x) else 0)

# 3 計算距離晉升天數（可為負數）
df['距離晉升天數'] = (df['晉升日期_dt'] - df['拜訪時間']).dt.days

df['是否晉升'] = df.apply(
    lambda row: 1 if pd.notna(row['晉升日期_dt']) and 0 <= (row['晉升日期_dt'] - row['拜訪時間']).days  else 0,
    axis=1
)

# %%%% 納入增員資料
member = pd.read_excel("D:/備註文字探勘/拜訪_2024冬夏賽.xlsx", sheet_name="MEMBER", dtype={'業代': str})

# 篩選前一個賽季 + 當年度 202309~202412
previous = member[(member['計績年月'] >= 202309) & (member['計績年月'] <= 202312) | 
                  (member['計績年月'] >= 202403) & (member['計績年月'] <= 202406) | 
                  (member['計績年月'] >= 202409) & (member['計績年月'] <= 202412)]
previous_count = previous.groupby('引薦主管業代')['業代'].nunique().reset_index()
previous_count.columns = ['業代', '加前一賽季增員數']

# 篩選當年度 202403~20246 & 202409~202412
current = member[(member['計績年月'] >= 202409) & (member['計績年月'] <= 202412)]
current_count = current.groupby('引薦主管業代')['業代'].nunique().reset_index()
current_count.columns = ['業代', '當年度賽季增員數']

# 合併兩期統計資料
referral_summary = pd.merge(previous_count, current_count, on='業代', how='outer').fillna(0)

# 確保數字為整數
referral_summary[['加前一賽季增員數', '當年度賽季增員數']] = referral_summary[['加前一賽季增員數', '當年度賽季增員數']].astype(int)

# 合併進主表 df
df = df.merge(referral_summary, on='業代', how='left')

# 若某些業代沒引薦過人 → 補 0
df[['加前一賽季增員數', '當年度賽季增員數']] = df[['加前一賽季增員數', '當年度賽季增員數']].fillna(0).astype(int)


# %%%% 斷詞詞語合併
filtered_words_cleaned = [
    [clean_word(word) for word in sentence]
    for sentence in filtered_words
]

# 移除 None 或空值的詞語
filtered_words_cleaned = [
    [word for word in sentence if word]  # 避免 None 或 '' 留下來
    for sentence in filtered_words_cleaned
]

df['拜訪備註_詞語'] = filtered_words_cleaned
df['詞數'] = df['拜訪備註_詞語'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# %%%% 納入 customer 資料
customer = pd.read_excel("D:/備註文字探勘/拜訪_2024冬夏賽.xlsx", sheet_name="CUSTOMER", dtype={'業代': str})
customer['建立時間'] = pd.to_datetime(customer['建立時間'])
# 篩選 2024/1/1 ~ 2024/7/31 的筆數
customer_filtered = customer[(customer['建立時間'] >= '2023-07-01') & (customer['建立時間'] <= '2023-12-31')]

# 建立分類欄位：「準客戶」與「新增保戶」
def classify_customer_type(ctype):
    if pd.isna(ctype):
        return '未知'
    elif '準客戶' in ctype:
        return '準客戶'
    elif '錠嵂保戶' in ctype:
        return '新增保戶'
    else:
        return '其他'

customer_filtered['分類'] = customer_filtered['客戶類型'].apply(classify_customer_type)

# 計算每個業代在上半年新增的準客戶數與新增保戶數（以客戶UUID唯一計算）
customer_stats = (
    customer_filtered
    .groupby(['業代', '分類'])['客戶UUID']
    .nunique()
    .unstack(fill_value=0)
    .reset_index()
)

# 確保兩欄都有，即使其中之一為 0 也存在
if '準客戶' not in customer_stats.columns:
    customer_stats['準客戶'] = 0
if '新增保戶' not in customer_stats.columns:
    customer_stats['新增保戶'] = 0

# 欄位重新命名
customer_stats = customer_stats.rename(columns={
    '準客戶': '上半年準客戶數',
    '新增保戶': '上半年新增保戶數'
})

# 合併到原始 df
df = df.merge(customer_stats, on='業代', how='left')
df[['上半年準客戶數', '上半年新增保戶數']] = df[['上半年準客戶數', '上半年新增保戶數']].fillna(0).astype(int)

# %%%% 納入客戶資料
info = pd.read_excel("D:/備註文字探勘/拜訪_2024冬夏賽.xlsx", sheet_name="INFO")
customer_summary = info.groupby('經紀人1-被保人CRM UUID').agg({
    '被保人性別': 'last', 
    '被保人目前年齡': 'last', 
    '要保人目前年齡': 'last', 
    '保單申請案號': pd.Series.nunique, 
    '繳款保費new': 'sum' 
}).reset_index()

# 重新命名欄位
customer_summary = customer_summary.rename(columns={
    '保單申請案號': '件數',
    '繳款保費new': '總保費'
})

sex_stage = {'男': 0, '女': 1, '法人': 2, '校正': 3}
customer_summary['被保人性別'] = customer_summary['被保人性別'].map(sex_stage)

# 只保留性別為 0 或 1 的資料（排除法人與校正）
customer_summary = customer_summary[customer_summary['被保人性別'].isin([0, 1])]
df = df.merge(customer_summary, 
              how='left', 
              left_on='客戶UUID', 
              right_on='經紀人1-被保人CRM UUID')
df = df.drop(columns=['經紀人1-被保人CRM UUID'])

df_drop = df.dropna(subset=['被保人性別', '被保人目前年齡', '要保人目前年齡', '件數', '總保費'])

# 取得與 filtered_words 的對應
# 注意：filtered_words 是經過 dropna 的 text_list 對應到的 → 需建立 index 對應
df_valid_notes = df_drop[df_drop['拜訪備註_文字'].notna()].reset_index(drop=True)


policy = pd.read_excel("D:/備註文字探勘/拜訪_2024冬夏賽.xlsx", sheet_name="POLICY") 
policy['是否為網路投保'] = np.where(policy['進件別'] == '網路投保', 1, 0)

# 確保時間格式正確
df_valid_notes['拜訪時間'] = pd.to_datetime(df_valid_notes['拜訪時間'])
policy['投保日'] = pd.to_datetime(policy['投保日'])

# 建立字典：UUID → 投保日清單（已排序）
policy_dict = defaultdict(list)
for _, row in policy.iterrows():
    uuid = row['經紀人1-被保人CRM UUID']
    policy_dict[uuid].append(row)

# 定義：回傳與拜訪時間最近的投保紀錄（含是否為網路投保）
def get_nearest_policy_info(uuid, visit_time):
    records = policy_dict.get(uuid, [])
    if not records:
        return pd.Series([np.nan, pd.NaT, np.nan])

    # 分成拜訪後與拜訪前的
    after = [r for r in records if r['投保日'] > visit_time]
    before = [r for r in records if r['投保日'] <= visit_time]

    if after:
        r = sorted(after, key=lambda x: x['投保日'])[0]
    elif before:
        r = sorted(before, key=lambda x: x['投保日'], reverse=True)[0]
    else:
        return pd.Series([np.nan, pd.NaT, np.nan])

    # 回傳：天數差、最近投保日、是否為網路投保
    return pd.Series([(visit_time - r['投保日']).days, r['投保日'], r['是否為網路投保']])

df_valid_notes[['拜訪與投保日天數差', '最近投保日', '最近是否為網路投保']] = df_valid_notes.apply(
    lambda row: get_nearest_policy_info(row['客戶UUID'], row['拜訪時間']),
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

# %% 文字特徵 (詞頻、字數)
# 備註字數（不含空白）
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

# 平均字數區間 - 根據百分位數分區
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

# 5. 合併回原始 df_valid_notes
df_valid_notes = df_valid_notes.merge(policy_by_agent, on='業代', how='left')

def compute_gender_diff_v2(row):
    if row['業務性別'] == row['被保人性別']:
        return row['業務性別']  # 0 or 1，性別相同
    elif row['業務性別'] == 0 and row['被保人性別'] == 1:
        return 2  # 業務男 → 客戶女
    elif row['業務性別'] == 1 and row['被保人性別'] == 0:
        return 3  # 業務女 → 客戶男
    else:
        return pd.NA  # 保險防呆，若資料異常不丟錯

df_valid_notes['業務客戶性別組合'] = df_valid_notes.apply(compute_gender_diff_v2, axis=1)
# 業務客戶年齡差距
df_valid_notes['業務客戶年齡差距'] = df_valid_notes['要保人目前年齡'] - df_valid_notes['業務目前年齡']

# %% 切分增員/銷售資料集 + 特徵整理
# 先將標籤名稱轉為字串，NaN 變成空字串
df_valid_notes['標籤名稱'] = df_valid_notes['標籤名稱'].fillna('').astype(str)
# 包含 "(增)" 的標籤資料集
df_member = df_valid_notes[df_valid_notes['標籤名稱'].str.contains(r'\(增\)', regex=True)]
# 不包含 "(增)" 的標籤資料集
df_insurence = df_valid_notes[~df_valid_notes['標籤名稱'].str.contains(r'\(增\)', regex=True)]

# 映射原始標籤為進度分數（1~5）
visit_stage = {'約訪': 0, '面談': 1, '需求確認': 2, '建議書': 3, '成交': 4}
df_insurence['拜訪目的'] = df_insurence['標籤名稱'].map(visit_stage)

# 按業代 + 客戶 組合計算
visit_freq = df_insurence.groupby(['業代', '客戶UUID'])['拜訪時間'].agg(['min', 'max', 'count']).reset_index()
visit_freq['拜訪天數差'] = (visit_freq['max'] - visit_freq['min']).dt.days + 1
visit_freq['每位客戶平均拜訪次數'] = visit_freq.apply(
    lambda row: row['count'] / row['拜訪天數差'] if row['count'] > 1 else 0,
    axis=1
)

# 計算每位業代對每位客戶的平均拜訪間隔
def avg_visit_interval(x):
    if len(x) < 2:
        return 0
    x_sorted = x.sort_values()
    diffs = x_sorted.diff().dropna().dt.days
    return diffs.mean()

avg_interval = df_insurence.groupby(['業代', '客戶UUID'])['拜訪時間'].apply(avg_visit_interval).reset_index(name='平均拜訪間隔天數')
visit_freq = visit_freq.merge(avg_interval, on=['業代', '客戶UUID'], how='left')

# 計算業代的總拜訪週期長度
df_insurence['週'] = df_insurence['拜訪時間'].dt.to_period('W').astype(str)

weekly_customer_count = df_insurence.groupby(['業代', '週'])['客戶UUID'].nunique().reset_index()
weekly_customer_count_summary = weekly_customer_count.groupby('業代')['客戶UUID'].mean().reset_index()
weekly_customer_count_summary.columns = ['業代', '每週平均拜訪客戶數']


df_insurence = df_insurence.merge(visit_freq[['業代', '客戶UUID', '拜訪天數差', '平均拜訪間隔天數']], on=['業代', '客戶UUID'], how='left')
df_insurence = df_insurence.merge(
    weekly_customer_count_summary,
    on='業代',
    how='left'
)


# 將備註斷詞結果組成文字
df_insurence['備註文字_處理'] = df_insurence['拜訪備註_詞語'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# 移除缺資料者
df_insurence_1 = df_insurence[df_insurence['拜訪目的'].notna()]
df_insurence_1 = df_insurence_1[df_insurence_1['備註文字_處理'].str.strip() != '']

unit_stats = df_insurence_1.groupby('營業單位').agg(
    業務員數=('業代', 'nunique'),
    客戶數=('客戶UUID', 'nunique'),
    成交前拜訪數=('拜訪紀錄UUID', 'nunique')
).reset_index()
unit_stats['平均每客戶拜訪次數'] = unit_stats['成交前拜訪數'] / unit_stats['客戶數']


# %% 計算有意義詞數
from gensim.models import Word2Vec

# 假設你已經將所有斷詞存在 list 格式（每一筆是一個詞語 list）
token_lists = df_insurence_1['備註文字_處理'].dropna().apply(lambda x: x.split()).tolist()

# 訓練 Word2Vec（也可以載入外部保險語料）
model_w2v = Word2Vec(sentences=token_lists, vector_size=150, window=5, min_count=2, workers=4)

# 你關心的保險詞
seed_words = ['保障', '保單', '理賠', '投保', '保費', '變更', '簽名', 
              "保單健診", "華南產", "癌症險", "旅平險", "新安東京", "還本型保單", "富邦", "安達", "續保", 
            "富邦產", "定期壽險", "重大傷病", "實支實付", "住院醫療", "理賠", "分紅躉繳", "轉介紹", 
            "保單","保險","行銷活動","防疫險","保險經紀人","健診","籃子理論","錠嵂","意外險","資產規劃", "車險需求",
            "儲蓄險","中壽","中國人壽","旅平","旅平險","三商美邦","簽約","成交","重大傷病","失能險","保經","見面三講","開門三講","退休規劃", 
            "車險","醫療險","火險","壽險","新光","遠雄","富邦","Toyota","機車險","寵物險","自動化工程師","六大保障","建議書", 
            "台灣人壽","失智症","app","OPP","保險存摺","國泰","遠雄","照會","三照","遞送","簽約","市調表","解約","美元保單","美元儲蓄", 
            "送保單" ,"照會單" ,"台壽","保誠" ,"癌症" ,"不動產","問卷","理賠","健診","轉介","簽收","建立關係","強制險","永達", 
            "觀念溝通","需求分析","六大保障","保單健診","終身","萬事利達","續保","友邦","寒暄","關心","保險存摺","年金","PHB","宏泰", 
            "南山","長照","XHB","HNRC","新生兒","約訪","年繳","美金","phb","探班","要保人",'企管副會長','意外險需求','double鑫','下週']  
meaningful_words_set = set(seed_words)

# 加入語意相近的詞（例如距離前 10 名，距離需 < 0.6）
for word in seed_words:
    if word in model_w2v.wv:
        similar_words = model_w2v.wv.most_similar(word, topn=20)
        for sim_word, score in similar_words:
            if score > 0.6:  # 可自行調整門檻
                meaningful_words_set.add(sim_word)


def count_word2vec_meaningful(text):
    if pd.isna(text): return 0
    tokens = text.split()
    return sum(1 for word in tokens if word in meaningful_words_set)

df_insurence_1['有意義詞數'] = df_insurence_1['備註文字_處理'].apply(count_word2vec_meaningful)

df_insurence_1['meaningful_ratio'] = df_insurence_1.apply(
    lambda row: row['有意義詞數'] / row['詞數'] if row['詞數'] > 0 else 0, axis=1)

# %% 拜訪進度預測
# 篩選資料：只使用成交者，且「拜訪與投保日天數差」為負值
df_model = df_insurence_1[df_insurence_1['拜訪與投保日天數差'].notna()].copy()
df_model = df_model[df_model['拜訪與投保日天數差'] <= 0].copy()

# 基本統計：各營業單位的業代數與客戶數
unit_stats_1 = df_model.groupby('營業單位').agg(
    業務員數=('業代', 'nunique'),
    客戶數=('客戶UUID', 'nunique'),
    成交前拜訪數=('拜訪紀錄UUID', 'nunique')
).reset_index()
unit_stats_1['平均每客戶拜訪次數'] = unit_stats_1['成交前拜訪數'] / unit_stats_1['客戶數']

df_model.groupby('標籤名稱')['拜訪與投保日天數差'].describe()
df_model.groupby('標籤名稱')['拜訪與投保日天數差'].mean().sort_values()

# ===== [1] 成交距離分段分類標籤 =====

# 假設你看完平均值後決定如下
def classify_days(diff):
    if diff <= -60:
        return '距離成交遙遠'
    elif diff <= -30:
        return '中距離'
    elif diff <= -7:
        return '接近成交'
    else:
        return '即將成交'

df_model['成交距離分段'] = df_model['拜訪與投保日天數差'].apply(classify_days)

# ===== [2] 特徵工程 =====

# TF-IDF 向量化（備註斷詞結果欄為 list）
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(df_model['備註文字_處理'])

# 數值特徵標準化
numerical_features = df_model[['備註字數', '備註行數', '拜訪次數', '拜訪目的', '業代客戶拜訪頻率', 'CEO']].fillna(0)
X_num = StandardScaler().fit_transform(numerical_features)
# 合併所有特徵
X_all = csr_matrix(hstack([X_text, X_num]))


# y = df_model['成交距離分段']
# le_y = LabelEncoder()
# y_encoded = le_y.fit_transform(df_model['成交距離分段'])  # 類別轉為整數 0~3
# 自訂分類順序 → 類別文字對應到整數
category_order = ['即將成交', '接近成交', '中距離', '距離成交遙遠']
category_map = {k: i for i, k in enumerate(category_order)}
reverse_map = {v: k for k, v in category_map.items()}

# 編碼 y
df_model['成交距離分段編碼'] = df_model['成交距離分段'].map(category_map)
y = df_model['成交距離分段編碼']

# 分層 K 折交叉驗證
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化收集器
all_y_true = []
all_y_pred = []
avg_cm = np.zeros((len(category_order), len(category_order)), dtype=int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y), 1):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # 累計混淆矩陣
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(category_order))))
    avg_cm += cm

    print(f"\n📊 Fold {fold} 分類報告：")
    print(classification_report(y_test, y_pred, target_names=category_order))

# 平均混淆矩陣視覺化
avg_cm = avg_cm // skf.get_n_splits()
disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm, display_labels=category_order)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("平均混淆矩陣：成交進度預測（交叉驗證）")
plt.grid(False)
plt.show()

# 總體分類報告
print("\n📈 全部資料整體分類報告：")
print(classification_report(all_y_true, all_y_pred, target_names=category_order))

# 先重建原始 index 對應的資料（與 y 對齊）
df_result = df_model.reset_index(drop=True).copy()

# 製作對照欄位
df_result['實際分類值'] = all_y_true
df_result['預測分類值'] = all_y_pred

# 整數 → 進度文字（使用你自訂的 reverse_map）
df_result['實際成交距離'] = df_result['實際分類值'].map(reverse_map)
df_result['預測成交距離'] = df_result['預測分類值'].map(reverse_map)

# 判斷錯誤與距離落差
df_result['是否預測錯誤'] = df_result['預測分類值'] != df_result['實際分類值']
df_result['分類落差'] = abs(df_result['預測分類值'] - df_result['實際分類值'])

# # 可匯出 CSV
# df_result.to_excel("D:/備註文字探勘/df_results_0123.xlsx", index=False)


# %%%% 拜訪距成交天數預測
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 把 y 換成連續目標值：拜訪與投保日天數差
y = df_model['拜訪與投保日天數差']

# 先確認 y 沒有缺值（若有則過濾）
mask = y.notna()
X_all = X_all[mask]
y = y[mask]

# 使用回歸模型
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_y_true, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_all), 1):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# 轉換為 NumPy 陣列
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# 評估指標
mse = mean_squared_error(all_y_true, all_y_pred)
mae = mean_absolute_error(all_y_true, all_y_pred)
r2 = r2_score(all_y_true, all_y_pred)

print(f"📉 MSE: {mse:.2f}")
print(f"📉 MAE: {mae:.2f}")
print(f"📈 R² : {r2:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(all_y_true, all_y_pred, alpha=0.3)
plt.plot([all_y_true.min(), all_y_true.max()],
         [all_y_true.min(), all_y_true.max()], 'r--')
plt.xlabel("True Days")
plt.ylabel("Predicted Days")
plt.title("預測 vs 實際（拜訪距投保日天數差）")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%%% 整體混淆矩陣
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 假設你已經有 df_result
# 這裡直接使用預測文字欄位 & 實際文字欄位

cm_table = pd.crosstab(df_result['實際成交距離'], df_result['預測成交距離'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_table, annot=True, cmap='Blues', fmt='d')
plt.title("預測 vs 實際成交距離分段（熱力圖）")
plt.ylabel("實際進度")
plt.xlabel("預測進度")
plt.show()

# %%%% 數值型特徵重要性評估
import shap
import xgboost as xgb

# 只使用數值型特徵（X_num）
X = X_num  # 或 X_num_scaled
y = df_model['成交距離分段編碼']

# 重新訓練一個只用數值欄位的 XGBoost 模型
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model_simple = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_simple.fit(X_train, y_train)

# SHAP 針對此模型建立解釋器
explainer = shap.Explainer(model_simple)
shap_values = explainer(X)

# 畫圖：bar chart 呈現每個變數平均重要性
shap.summary_plot(shap_values, 
                  features=X, 
                  feature_names=['備註字數', '備註行數', '拜訪次數', '拜訪目的', '業代客戶拜訪頻率', 'CEO'], 
                  plot_type='bar')

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# summary plot for 4 classes 
for i, class_name in enumerate(category_order):
    print(f"📊 正在產生 Class {i}（{class_name}）的 SHAP summary plot...")
    
    shap.summary_plot(
        shap_values[..., i],         # 多分類 SHAP 值：每個類別一組
        features=X,
        feature_names=['備註字數', '備註行數', '拜訪次數', '拜訪目的', '業代客戶拜訪頻率', 'CEO'],
        show=False
    )
    
    plt.title(f"SHAP Summary Plot - Class {i}：{class_name}")
    plt.tight_layout()
    plt.show()

# %% 成交機率預測
random.seed(42)
np.random.seed(42)
from gensim.models import Word2Vec

# ===== Step 1: 建立成交標籤 =====
df_model_1 = df_insurence_1[df_insurence_1['平均每客戶拜訪次數'] > 4]
df_model_1['是否成交'] = df_model_1['拜訪與投保日天數差'].apply(lambda x: 1 if pd.notna(x) and x <= 30 else 0)
y = df_model_1['是否成交']

# ===== Step 2: 數值特徵處理 =====
numerical_cols = [
    '業務客戶性別組合', '最新職級', '拜訪目的', 
    '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', # '拜訪紀錄密度', 
    '備註字數', '有意義詞數', 
    '目前年資', '營業單位_編碼', # '當年度賽季增員數', '加前一賽季增員數', '最新職級', 
    '上半年準客戶數', '今年度活動參與率', '上年度FYC', '距離晉升天數', 
    '件數', '總保費' # '業務客戶性別組合', 
]
# '拜訪次數', '拜訪天數差', '備註行數', '業務性別', '被保人性別', '最近是否為網路投保', 'CEO', '夏賽活動參與率', '夏賽FYC', 
# '是否晉升', '得獎數', '夏賽增員數', '冬賽增員數', '業務目前年齡', '被保人目前年齡', '今年度活動參與率', '今年度FYC', 

# 描述統計表
summary_var = df_insurence_1[numerical_cols].describe().T

X_num = df_model_1[numerical_cols].fillna(0)
X_num_scaled = StandardScaler().fit_transform(X_num)

# ===== Step 3: Word2Vec 建模與 TF-IDF 加權 =====
sentences = df_model_1['備註文字_處理']
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2)

# TF-IDF 權重
# tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
def identity(x):
    return x

tfidf_vectorizer = TfidfVectorizer(
    tokenizer=identity,
    preprocessor=identity,
    token_pattern=None
)
tfidf_vectorizer.fit(sentences)
tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

# 取加權向量
def vectorize_sentence_weighted(sentence):
    vecs, weights = [], []
    for word in sentence:
        if word in w2v_model.wv and word in tfidf_dict:
            vecs.append(w2v_model.wv[word] * tfidf_dict[word])
            weights.append(tfidf_dict[word])
    return np.sum(vecs, axis=0) / np.sum(weights) if vecs else np.zeros(w2v_model.vector_size)

X_w2v_weighted = np.array([vectorize_sentence_weighted(s) for s in sentences])
valid_idx = df_model_1['備註文字_處理'].notna()
X_w2v_full = np.zeros((len(df_model_1), w2v_model.vector_size))
X_w2v_full[valid_idx] = X_w2v_weighted

# ===== Step 4: 初步模型取得 Word2Vec Top 10 特徵 =====
X_all = np.hstack([X_w2v_full, X_num_scaled])
model_init = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_init.fit(X_all, y)

w2v_importances = model_init.feature_importances_[:X_w2v_full.shape[1]]
top_k = 10
w2v_top_indices = np.argsort(w2v_importances)[::-1][:top_k]
X_w2v_top = X_w2v_full[:, w2v_top_indices]
w2v_top_feature_names = [f'w2v_{i}' for i in w2v_top_indices]

# ===== Step 5: 合併 Word2Vec Top10 + 所有數值特徵 =====
X_combined = np.hstack([X_w2v_top, X_num_scaled])
final_feature_names = w2v_top_feature_names + numerical_cols

# ===== Step 6: Hold-out 測試切分 + 特徵篩選 =====
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_combined, y, test_size=0.2, stratify=y, random_state=42)

# 轉為 DataFrame
X_trainval_df = pd.DataFrame(X_trainval, columns=final_feature_names)
X_trainval_df.to_csv("D:/備註文字探勘/models/train_reference.csv", index=False)

model_init = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_init.fit(X_trainval, y_trainval)

# importances = model_init.feature_importances_
# top_indices = np.argsort(importances)[::-1][:10]
# top_feature_names = [final_feature_names[i] for i in top_indices]
# X_trainval_top = X_trainval[:, top_indices]
# X_test_top = X_test[:, top_indices]

# ===== Step 7: 交叉驗證（在 trainval 上）=====
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_scores, pr_scores = [], []
all_y_true, all_y_pred, all_y_proba = [], [], []

for train_idx, val_idx in skf.split(X_trainval, y_trainval):
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.6).astype(int)

    roc_scores.append(roc_auc_score(y_val, y_proba))
    pr_scores.append(average_precision_score(y_val, y_proba))
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

print("\n📊 Cross-Validation (Train Set)")
print(classification_report(all_y_true, all_y_pred))
print(f"Average ROC AUC: {np.mean(roc_scores):.4f}")
print(f"Average PR AUC : {np.mean(pr_scores):.4f}")

# ===== Step 8: Hold-out 測試集評估 =====
final_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
final_model.fit(X_trainval, y_trainval)

y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.6).astype(int)

print("\n🧪 Final Evaluation on Hold-out Test Set")
print(classification_report(y_test, y_test_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"PR AUC : {average_precision_score(y_test, y_test_proba):.4f}")



# === 交叉驗證結果 ===
proba_df = pd.DataFrame({
    '預測機率': all_y_proba,
    '實際標籤': all_y_true,
    '預測標籤': all_y_pred
})

# 對應交叉驗證樣本 index（這是 X_trainval 裡的 row index）
# 注意：all_y_proba 的長度 = y_trainval 的長度
trainval_index = y_trainval.index[:len(all_y_proba)]
meta_cols = df_model_1.loc[trainval_index, ['客戶UUID', '業代', '拜訪紀錄UUID', '拜訪時間', '拜訪備註', '備註文字_處理']].reset_index(drop=True)
model_vars = df_model_1.loc[trainval_index, numerical_cols].reset_index(drop=True)
proba_df = pd.concat([meta_cols, model_vars, proba_df], axis=1)

# === 測試集結果 ===
test_result_df = pd.DataFrame({
    '預測機率': y_test_proba,
    '實際標籤': y_test,
    '預測標籤': y_test_pred
}).reset_index(drop=True)

# 使用 y_test 的 index 去抓 df_model_1 對應的原始欄位
meta_test = df_model_1.loc[y_test.index, ['客戶UUID', '業代', '拜訪紀錄UUID', '拜訪時間', '拜訪備註', '備註文字_處理']].reset_index(drop=True)
model_vars_test = df_model_1.loc[y_test.index, numerical_cols].reset_index(drop=True)
test_result_df = pd.concat([meta_test, model_vars_test, test_result_df], axis=1)

combined_df = pd.concat([proba_df, test_result_df], ignore_index=True)

def classify_probability(p):
    if p >= 0.90:
        return "極高潛力"
    elif p >= 0.75:
        return "高潛力"
    elif p >= 0.50:
        return "中潛力"
    elif p >= 0.25:
        return "低潛力"
    else:
        return "極低潛力"

combined_df['預測潛力分群'] = combined_df['預測機率'].apply(classify_probability)

# %% 寫入 Excel，不同工作表
# with pd.ExcelWriter("D:/備註文字探勘/成交預測_結果.xlsx", engine='xlsxwriter') as writer:
#     proba_df.to_excel(writer, sheet_name='交叉驗證結果', index=False)
#     test_result_df.to_excel(writer, sheet_name='測試集結果', index=False)
#     combined_df.to_excel(writer, sheet_name='合併結果', index=False)

# 儲存模型
with open('D:/備註文字探勘/models/xgb_model_final.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('D:/備註文字探勘/models/word2vec_model.pkl', 'wb') as f:
    pickle.dump(w2v_model, f)
import dill
with open('D:/備註文字探勘/models/tfidf_vectorizer.pkl', 'wb') as f:
    dill.dump(tfidf_vectorizer, f)

with open('D:/備註文字探勘/models/scaler.pkl', 'wb') as f:
    pickle.dump(StandardScaler().fit(X_num), f)


# # Odds Ratio
# from sklearn.linear_model import LogisticRegression

# def compute_odds_ratio(X_scaled, y, feature_names):
#     model_lr = LogisticRegression(max_iter=1000)
#     model_lr.fit(X_scaled, y)

#     coef = model_lr.coef_[0]
#     or_values = np.exp(coef)

#     odds_ratio_df = pd.DataFrame({
#         '變數名稱': feature_names,
#         '迴歸係數': coef,
#         'odds ratio': or_values
#     })

#     return odds_ratio_df.sort_values(by='odds ratio', ascending=False)

# trainval_index = y_trainval.index[:len(all_y_proba)]
# X_trainval_num = X_num_scaled[trainval_index, :]
# or_df = compute_odds_ratio(X_num_scaled, y, numerical_cols)
# print(or_df)

# %% 模型評估
# 模型訓練（一次性，不用 cross-validation）
model.fit(X_all, y)

# 建立 feature 名稱（假設 Word2Vec 是前 100 維）
w2v_feature_names = [f'w2v_{i}' for i in range(X_w2v_full.shape[1])]
num_feature_names = list(X_num.columns)
feature_names = w2v_feature_names + num_feature_names

# 特徵重要性
importances = model.feature_importances_
feat_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# 顯示前20名特徵
plt.figure(figsize=(10, 6))
plt.barh(feat_importance_df['feature'][:20][::-1], feat_importance_df['importance'][:20][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Important Features (XGBoost)")
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定義每次加入的特徵群（你可以依需求調整分群）
feature_groups = {
    'M1': ['拜訪天數差', '每週平均拜訪客戶數', '目前年資', '拜訪目的', '目前年齡', '職級_代碼', '拜訪次數', 'CEO', '得獎數',  
           '有意義詞數', '備註字數', '詞數', 'meaningful_ratio', '備註行數'],
    'M2': ['拜訪天數差', '每週平均拜訪客戶數', '目前年資', '拜訪目的', '目前年齡', '職級_代碼', '拜訪次數', 'CEO', '得獎數',  
           '有意義詞數', '備註字數', '詞數', '備註行數'],
    'M3': ['拜訪天數差', '每週平均拜訪客戶數', '目前年資', '拜訪目的', '目前年齡', '職級_代碼', '拜訪次數', 'CEO', '得獎數',  
           '備註字數', 'meaningful_ratio', '備註行數'],
    '備註文字': 'w2v',  # 特別標註，非數值欄位 # 改名清楚一些
    '數值型特徵': numerical_cols,
    '全部特徵': 'all'
}

results = []

for name, features in tqdm(feature_groups.items()):
    if features == 'w2v':
        X_subset = csr_matrix(X_w2v_full)  # 轉 sparse 以相容後續流程
    elif features == 'all':
        X_subset = csr_matrix(np.hstack([X_w2v_full, X_num_scaled]))
    else:
        X_num_subset = StandardScaler().fit_transform(df_model_1[features].fillna(0))
        X_subset = csr_matrix(X_num_subset)

    all_y_true, all_y_proba = [], []

    for train_idx, test_idx in skf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)

    # 評估指標
    auc_roc = roc_auc_score(all_y_true, all_y_proba)
    ap = average_precision_score(all_y_true, all_y_proba)
    results.append({'Feature': name, 'ROC AUC': auc_roc, 'PR AUC': ap})

# 輸出結果
df_result = pd.DataFrame(results)
df_result = df_result.sort_values('ROC AUC', ascending=False)

# 可視化
plt.figure(figsize=(10, 5))
plt.plot(df_result['Feature'], df_result['ROC AUC'], marker='o', label='ROC AUC')
plt.plot(df_result['Feature'], df_result['PR AUC'], marker='x', label='PR AUC')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Score')
plt.title('每組變數對模型效能影響')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 單一變數
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt

results = []

# 加入 文字 先評估
X_w2v_sparse = csr_matrix(X_w2v_full)  # 加入 Word2Vec（文字向量）比較
all_y_true, all_y_proba = [], []
for train_idx, test_idx in skf.split(X_w2v_full, y):
    X_train, X_test = X_w2v_full[train_idx], X_w2v_full[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    all_y_true.extend(y_test)
    all_y_proba.extend(y_proba)

results.append({
    'Feature': 'Word2Vec',
    'ROC AUC': roc_auc_score(all_y_true, all_y_proba),
    'PR AUC': average_precision_score(all_y_true, all_y_proba)
})

# 數值型變數：逐個變數進行測試
for col in tqdm(numerical_cols):
    X_single = StandardScaler().fit_transform(df_model_1[[col]].fillna(0))
    X_single = csr_matrix(X_single)

    all_y_true, all_y_proba = [], []

    for train_idx, test_idx in skf.split(X_single, y):
        X_train, X_test = X_single[train_idx], X_single[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)

    results.append({
        'Feature': col,
        'ROC AUC': roc_auc_score(all_y_true, all_y_proba),
        'PR AUC': average_precision_score(all_y_true, all_y_proba)
    })

# 彙整結果
df_result = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)


# 可視化
plt.figure(figsize=(10, 5))
plt.plot(df_result['Feature'], df_result['ROC AUC'], marker='o', label='ROC AUC')
plt.plot(df_result['Feature'], df_result['PR AUC'], marker='x', label='PR AUC')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Score')
plt.title('單一變數對模型效能的影響')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% 建立分群欄位：根據分位數切成三段
df_model_1['拜訪頻率分群'] = pd.qcut(df_model_1['每位客戶平均拜訪次數'], q=3, labels=['低頻', '中頻', '高頻'])

# SHAP
for group in ['低頻', '中頻', '高頻']:
    group_df = df_model_1[df_model_1['拜訪頻率分群'] == group]
    X_group = group_df[numerical_cols]
    y_group = group_df['是否成交']

    X_scaled = StandardScaler().fit_transform(X_group)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_scaled, y_group)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    shap.summary_plot(
        shap_values,
        features=X_scaled,
        feature_names=numerical_cols,
        show=False,
        plot_type="dot"
    )
    plt.title(f"拜訪頻率分群：{group}")
    plt.tight_layout()
    plt.show()
    
# pd.qcut(df_model_1['業代客戶拜訪頻率'], q=3).unique() 

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)

groups = df_model_1['拜訪頻率分群'].dropna().unique()
group_metrics = {}  # 用來儲存所有分群結果

for group in groups:
    print(f"\n===== 分析拜訪頻率分群: {group} =====")
    
    df_group = df_model_1[df_model_1['拜訪頻率分群'] == group].reset_index(drop=True)

    tfidf = TfidfVectorizer(max_features=1000)
    tfidf.fit(df_model_1['備註文字_處理'].fillna(''))
    X_text = tfidf.transform(df_group['備註文字_處理'].fillna(''))
    X_num = df_group[numerical_cols].fillna(0)
    X_num_scaled = StandardScaler().fit_transform(X_num)
    X_all = csr_matrix(hstack([X_text, X_num_scaled]))
    y = df_group['是否成交']

    if y.value_counts().min() == 0:
        print("⚠️ 該群組正負樣本不平衡（其中一類為0），略過此群。")
        continue

    scale_ratio = y.value_counts()[0] / y.value_counts()[1]

    model = XGBClassifier(
        scale_pos_weight=scale_ratio,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    all_y_true, all_y_proba, all_y_pred = [], [], []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"[{group}] ROC Curve")
    plt.subplot(1, 2, 2)
    plt.title(f"[{group}] Precision-Recall Curve")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y), 1):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.6).astype(int)

        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)
        all_y_pred.extend(y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"Fold {fold}")

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f"Fold {fold}")

    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ====== 效能指標 ======
    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    all_y_pred = np.array(all_y_pred)

    metrics = {
        'ROC AUC': roc_auc_score(all_y_true, all_y_proba),
        'PR AUC': average_precision_score(all_y_true, all_y_proba),
        'F1': f1_score(all_y_true, all_y_pred),
        'Accuracy': accuracy_score(all_y_true, all_y_pred),
        'Precision': precision_score(all_y_true, all_y_pred),
        'Recall': recall_score(all_y_true, all_y_pred),
        'Total Samples': len(all_y_true),
        'Positive Rate': np.mean(all_y_true)
    }

    group_metrics[group] = metrics

    print("效能指標：")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


# %% 整體混淆矩陣
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 假設你已經有 df_result
# 這裡直接使用預測文字欄位 & 實際文字欄位

cm_table = pd.crosstab(df_result['是否成交'], df_result['預測是否成交'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_table, annot=True, cmap='Blues', fmt='d')
plt.title("預測 vs 實際成交（熱力圖）")
plt.ylabel("實際成交")
plt.xlabel("預測成交")
plt.show()

# %% 整體變數解釋-SHAP
import shap
import xgboost as xgb

np.random.seed(42)

# 只使用數值型特徵（X_num）
X = X_num_scaled  # 或 X_num
y = df_model_1['是否成交']  

# 重新訓練一個只用數值欄位的 XGBoost 模型
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model_simple = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_simple.fit(X_train, y_train)

# SHAP 針對此模型建立解釋器
explainer = shap.Explainer(model_simple)
shap_values = explainer(X)

# 畫圖：bar chart 呈現每個變數平均重要性
shap.summary_plot(shap_values, features=X, feature_names=numerical_cols, plot_type='bar')

shap.summary_plot(
    shap_values,
    features=X,
    feature_names=numerical_cols
)

# 個別變數解釋
explainer = shap.Explainer(final_model, X_trainval, feature_names=final_feature_names)
shap_values = explainer(X_trainval)

# 畫出前 10 個最重要特徵
shap.plots.bar(shap_values, max_display=20)

# 以「距離晉升天數」為例
# shap.plots.scatter(shap_values[:, final_feature_names.index("距離晉升天數")], 
#                    color=shap_values, 
#                    show=True)

# 繪製 SHAP interaction plot
shap.plots.scatter(shap_values[:, final_feature_names.index('距離晉升天數')],
                   color=shap_values[:, final_feature_names.index('備註字數')],
                   show=True)

# 使用 SHAP interaction values: 兩個變數交互下對成交機率的影響
import shap
import xgboost as xgb

# 訓練 XGBoost 模型（假設你已有特徵 X 和標籤 y）
model = xgb.XGBClassifier()
model.fit(X_num, y)

# 建立 SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_num)

# SHAP 互動值 (兩變數之間的交互影響)
shap_interaction_values = explainer.shap_interaction_values(X_num)

# 繪圖：互動 SHAP 值圖（以 '拜訪次數' 和 '拜訪天數差' 為例）
shap.dependence_plot(
    ('拜訪次數', '拜訪天數差'),  # 主變數與交互變數
    shap_interaction_values, 
    X_num, 
    display_features=X_num, 
    interaction_index='拜訪天數差'
)





import pandas as pd
import numpy as np

# 假設 shap_values 是 shape=(n_samples, n_features)
# 且 feature_names 與 X_trainval 對應

feature_index = final_feature_names.index("距離晉升天數")
x = X_trainval[:, feature_index]
shap_val = shap_values[:, feature_index].values

# 建立 dataframe
df_shap = pd.DataFrame({
    "值": x,
    "SHAP": shap_val
})

# 按照值排序
df_shap = df_shap.sort_values("值")

# 平滑處理（可選擇 rolling mean）
df_shap["SHAP_smooth"] = df_shap["SHAP"].rolling(window=20, min_periods=1).mean()

# 找出第一個 SHAP 轉正的值
threshold = df_shap[df_shap["SHAP_smooth"] > 0]["值"].min()
print(f"『距離晉升天數』對成交有正面影響的起始值：約為 {threshold:.0f} 天")
# 約為 -3 天 -> 晉升後3天的拜訪

# 個變數SHAP趨勢 (Find Critical Value)
plt.figure(figsize=(8, 4))
plt.plot(df_shap["值"], df_shap["SHAP_smooth"])
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(threshold, color='red', linestyle='--', label=f'轉正臨界點 = {threshold:.0f}')
plt.xlabel("距離晉升天數")
plt.ylabel("SHAP 值")
plt.title("距離晉升天數 對成交的 SHAP 趨勢")
plt.legend()
plt.grid(True)
plt.show()


# %% shap 各變數解釋
num_vars = [
    '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', # '拜訪紀錄密度', 
    '備註字數', '有意義詞數', 
    '目前年資', # '當年度賽季增員數', 
    '上半年準客戶數', '今年度活動參與率', '上年度FYC', '距離晉升天數', 
    '件數', '總保費'
]

cat_vars = ['業務客戶性別組合', '最新職級', '拜訪目的', '營業單位_編碼']


import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False

# 輸出資料夾
output_dir = "D:/備註文字探勘/shap_2024"
os.makedirs(output_dir, exist_ok=True)

# 個別變數解釋
explainer = shap.Explainer(final_model, 
                           X_trainval, 
                           feature_names=final_feature_names, 
                           model_output='raw', 
                           feature_perturbation="interventional") # 減少隨機性

shap_values = explainer(X_trainval)

def plot_shap_bin_auto_with_summary_dual_x(
    X_data, shap_values, feature_names, variables,
    mean_dict, scale_dict,
    window=20, output_dir=None,
    min_range_width=0.1, merge_gap=0.05
):
    summary_list = []

    for var in variables:
        try:
            i = feature_names.index(var)
            x = X_data[:, i]
            shap_val = shap_values[:, i].values

            df = pd.DataFrame({"值": x, "SHAP": shap_val}).sort_values("值").reset_index(drop=True)
            df["SHAP_smooth"] = df["SHAP"].rolling(window=window, min_periods=1).mean()

            df["is_neg"] = (df["SHAP_smooth"] < 0).astype(int)
            df["neg_group"] = (df["is_neg"].diff(1) != 0).cumsum()
            neg_segments = df[df["is_neg"] == 1].groupby("neg_group")
            neg_ranges = [(seg["值"].min(), seg["值"].max()) for _, seg in neg_segments]

            value_min, value_max = df["值"].min(), df["值"].max()
            positive_ranges = []
            current_start = value_min
            for neg_start, neg_end in sorted(neg_ranges):
                if current_start < neg_start:
                    positive_ranges.append((current_start, neg_start))
                current_start = max(current_start, neg_end)
            if current_start < value_max:
                positive_ranges.append((current_start, value_max))

            positive_ranges = [(lo, hi) for lo, hi in positive_ranges if (hi - lo) >= min_range_width]

            # 合併破碎段
            merged_ranges = []
            for lo, hi in sorted(positive_ranges):
                if not merged_ranges:
                    merged_ranges.append([lo, hi])
                else:
                    prev_lo, prev_hi = merged_ranges[-1]
                    if lo - prev_hi <= merge_gap:
                        merged_ranges[-1][1] = hi
                    else:
                        merged_ranges.append([lo, hi])

            # 還原原始數值
            mean, scale = mean_dict[var], scale_dict[var]
            restored_ranges = [(lo * scale + mean, hi * scale + mean) for lo, hi in merged_ranges]

            # 繪圖開始
            fig, ax1 = plt.subplots(figsize=(8, 4))

            peak_idx = df["SHAP_smooth"].idxmax()
            peak_x = df.loc[peak_idx, "值"]
            peak_raw = peak_x * scale + mean

            ax1.scatter(df["值"], df["SHAP"], alpha=0.3, label="原始 SHAP")
            ax1.plot(df["值"], df["SHAP_smooth"], color='blue', label="平滑趨勢")
            ax1.axhline(0, color='gray', linestyle='--')
            ax1.axvline(peak_x, color='red', linestyle='--', label=f'最大貢獻點 = {peak_raw:.2f}')

            for idx, ((lo, hi), (lo_raw, hi_raw)) in enumerate(zip(merged_ranges, restored_ranges)):
                ax1.axvspan(lo, hi, color='lightgreen', alpha=0.3)
                summary_list.append({
                    '變數': var,
                    '原始值區間_起': round(lo_raw, 2),
                    '原始值區間_迄': round(hi_raw, 2),
                    '標準化值_起': round(lo, 2),
                    '標準化值_迄': round(hi, 2)
                })

            # 設定主 X 軸（標準化值）標籤在上方
            ax1.xaxis.set_label_position('top')
            ax1.xaxis.tick_top()
            ax1.set_xlabel(f"標準化", labelpad=10)
            ax1.set_ylabel("SHAP 值")
            ax1.set_title(f"{var} 對成交的 SHAP 趨勢")
            
            # 雙 X 軸：下方顯示原始數值，對齊標準化 X 軸
            def to_raw(x): return x * scale + mean
            def to_std(x): return (x - mean) / scale
            
            # 替代 ax2 = ax1.twiny()
            secax = ax1.secondary_xaxis('bottom', functions=(to_raw, to_std))
            secax.set_xlabel(f"原始數值")
            
            # 平均與標準差說明
            mean_text = f"原始平均值: {mean:.2f}\n原始標準差: {scale:.2f}"
            ax1.text(0.98, 0.95, mean_text, transform=ax1.transAxes,
                     ha='right', va='top', fontsize=8, color='dimgray',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))

            ax1.legend(loc='upper left')
            ax1.grid(True)

            # if output_dir:
            #     filename = os.path.join(output_dir, f"{var}_shap_trend.png")
            #     plt.savefig(filename, dpi=300, bbox_inches='tight')
            #     plt.close()
            #     print(f"✅ 已儲存：{filename}")
            # else:
            #     plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ {var} 失敗：{e}")

    return pd.DataFrame(summary_list)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)  # <== 這是你原本做的

mean_dict = dict(zip(numerical_cols, scaler.mean_))
scale_dict = dict(zip(numerical_cols, scaler.scale_))

df_summary = plot_shap_bin_auto_with_summary_dual_x(
    X_data=X_trainval,
    shap_values=shap_values,
    feature_names=final_feature_names,
    variables=num_vars,
    mean_dict=mean_dict,
    scale_dict=scale_dict,
    window=20,
    output_dir="D:/備註文字探勘/shap_2024",
    min_range_width=0.1,
    merge_gap=0.05
)

# # 匯出為 CSV
# df_summary.to_csv("D:/備註文字探勘/shap_正向貢獻區間總表.csv", index=False)

# 類別型變數 SHAP 平均值圖
def plot_shap_by_category(X_data, shap_values, feature_names, var_name, output_dir): # 
    try:
        idx = feature_names.index(var_name)
        x = proba_df[var_name].values
        shap_val = shap_values[:, idx].values

        df = pd.DataFrame({
            '類別值': x,
            'SHAP': shap_val
        })

        shap_mean = df.groupby('類別值')['SHAP'].mean().reset_index()

        plt.figure(figsize=(6, 4))
        bars = plt.bar(shap_mean['類別值'].astype(str), shap_mean['SHAP'], color='skyblue')
        
        # 加上文字標籤
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height - 0.015 if height >= 0 else height + 0.015,
                f"{height:.2f}",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=8,
                color='black'
            )
            
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel(var_name)
        plt.ylabel("平均 SHAP 值")
        plt.title(f"{var_name} 各類別對成交的平均 SHAP 貢獻")
        # plt.grid(True, axis='y')
        plt.tight_layout()

        # 儲存圖片
        filename = os.path.join(output_dir, f"{var_name}_shap_bar.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已儲存：{filename}")

    except Exception as e:
        print(f"❌ {var_name} 失敗：{e}")

for v in cat_vars:
    plot_shap_by_category(X_trainval, shap_values, final_feature_names, v, output_dir) # 
    
    
# %% 決策樹自動分點建議

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def analyze_feature_with_prediction_and_actual(df, feature, pred_col='預測機率', target='實際標籤', bins=5, min_samples_leaf=100):
    df_valid = df[[feature, pred_col, target]].dropna()

    # 分箱（以分位數為主，若失敗則退回等寬）
    try:
        df_valid['區間'] = pd.qcut(df_valid[feature], q=bins, duplicates='drop')
    except ValueError:
        df_valid['區間'] = pd.cut(df_valid[feature], bins=bins)

    # 分箱統計：實際成交率、預測機率平均
    bin_stats = df_valid.groupby('區間').agg(
        樣本數=('實際標籤', 'count'),
        成交率=('實際標籤', 'mean'),
        平均成交機率=(pred_col, 'mean')
    )

    print(f"\n📊 【{feature}】分箱分析")
    
    # 雙軸圖繪製
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax2 = ax1.twinx()
    bin_labels = bin_stats.index.astype(str)

    # 成交率柱狀圖（左軸）
    ax1.bar(bin_labels, bin_stats['成交率'], color='steelblue', alpha=0.7, label='實際成交率')
    ax1.set_ylabel('成交率', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 1)
    
    # 為成交率加上數值標籤
    for i, val in enumerate(bin_stats['成交率']):
        ax1.text(i, val + 0.04, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color='steelblue')

    # 預測機率折線圖（右軸）
    ax2.plot(bin_labels, bin_stats['平均成交機率'], color='darkorange', marker='o', label='預測成交機率')
    ax2.set_ylabel('平均預測成交機率', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1)

    # 為預測機率加上數值標籤
    for i, val in enumerate(bin_stats['平均成交機率']):
        ax2.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color='darkorange')

    plt.title(f'{feature} 分箱成交率 vs 預測成交機率')
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

    # 決策樹切點
    X = df_valid[[feature]].values
    y = df_valid[target].values
    clf = DecisionTreeClassifier(max_depth=1, min_samples_leaf=min_samples_leaf)
    clf.fit(X, y)

    threshold = clf.tree_.threshold[0]
    gain = clf.tree_.impurity[0] - sum([
        clf.tree_.impurity[1] * clf.tree_.n_node_samples[1] / clf.tree_.n_node_samples[0],
        clf.tree_.impurity[2] * clf.tree_.n_node_samples[2] / clf.tree_.n_node_samples[0]
    ])

    print(f"\n🌳 自動切點建議：{feature} ≈ {threshold:.2f}")
    print(f"📈 推薦行動建議：當【{feature} ≥ {threshold:.2f}】時，成交率明顯提高（資訊增益 ≈ {gain:.4f}）")

    return bin_stats, threshold

def analyze_categorical_feature(df, cat_var, pred_col='預測機率', target='是否成交'):
    df_valid = df[[cat_var, pred_col, target]].dropna()

    cat_stats = df_valid.groupby(cat_var).agg(
        樣本數=(target, 'count'),
        成交率=(target, 'mean'),
        平均預測機率=(pred_col, 'mean')
    ).sort_values('成交率', ascending=False)

    print(f"\n📊【{cat_var}】類別成交率 vs 預測機率")
    
    # 雙軸圖
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    labels = cat_stats.index.astype(str)

    ax1.bar(labels, cat_stats['成交率'], color='steelblue', alpha=0.7)
    ax1.set_ylabel('成交率', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 1)
    
    # 為成交率加上標籤
    for i, val in enumerate(cat_stats['成交率']):
       ax1.text(i, val + 0.04, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color='steelblue')

    ax2.plot(labels, cat_stats['平均預測機率'], color='darkorange', marker='o')
    ax2.set_ylabel('平均預測成交機率', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1)
    
    # 為預測機率加上標籤
    for i, val in enumerate(cat_stats['平均預測機率']):
        ax2.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color='darkorange')

    plt.title(f'{cat_var} 類別成交率 vs 預測機率')
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

    return cat_stats


results_summary_all = []

# 數值型分析（含分箱、切點、雙軸圖）
for feature in num_vars:
    print(f"\n📊 數值變數分析：{feature}")
    try:
        bin_stats, threshold = analyze_feature_with_prediction_and_actual(
            df=combined_df,
            feature=feature,
            pred_col='預測機率',
            target='實際標籤',
            bins=5,
            min_samples_leaf=100
        )

        results_summary_all.append({
            '變數': feature,
            '型別': '數值',
            # '子類別': '',  # 數值型不需要子類別
            '樣本數': bin_stats['樣本數'].sum(),
            '最高區間成交率': round(bin_stats['成交率'].max(), 3),
            '最低區間成交率': round(bin_stats['成交率'].min(), 3),
            '最高區間預測機率': round(bin_stats['平均成交機率'].max(), 3),
            '最低區間預測機率': round(bin_stats['平均成交機率'].min(), 3),
            '樣本數最大區間': str(bin_stats['樣本數'].idxmax()),
            '建議門檻': round(threshold, 2)
        })

    except Exception as e:
        print(f"⚠️ {feature} 分析失敗：{e}")

# 類別型分析（不需門檻，只看各類別表現）
for feature in cat_vars:
    print(f"\n📊 類別變數分析：{feature}")
    try:
        cat_stats = analyze_categorical_feature(
            df=combined_df,
            cat_var=feature,
            pred_col='預測機率',
            target='實際標籤'
        )

        for idx, row in cat_stats.iterrows():
            results_summary_all.append({
                '變數': feature,
                '型別': '類別',
                '子類別': idx,
                '樣本數': row['樣本數'],
                '最高區間成交率': round(row['成交率'], 3),
                '最低區間成交率': round(row['成交率'], 3),
                '最高區間預測機率': round(row['平均預測機率'], 3),
                '最低區間預測機率': round(row['平均預測機率'], 3),
                '樣本數最大區間': '',
                '建議門檻': ''
            })

    except Exception as e:
        print(f"⚠️ {feature} 分析失敗：{e}")


df_summary_all = pd.DataFrame(results_summary_all)
# df_summary_all.to_csv('變數分析總表.csv', index=False)



import seaborn as sns
sns.boxplot(x=df_model_1['是否成交'], y=df_model_1['每位客戶平均拜訪次數'])
plt.title("每位客戶平均拜訪次數 vs 是否成交")

# %% 檢查共線性
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt

# ====== 步驟 1：選擇你要檢查的欄位 ======
# 你可以依自己的欄位進行調整
numerical_columns = ['拜訪次數', '拜訪目的', '拜訪天數差', '每週平均拜訪客戶數', 
                  '備註字數', '有意義詞數', '備註行數', 
                  '目前年資', '業務目前年齡', '夏賽增員數', '得獎數', '距離晉升天數', '冬賽增員數',
                  '上半年新客數', '冬賽活動參與率', '冬賽FYC', '夏賽活動參與率', '夏賽FYC', 
                  '被保人目前年齡', '件數', '總保費']
#  '業務性別', '被保人性別', , '最近是否為網路投保''職級_代碼', 
# '是否晉升', 'CEO',  

# 選出要檢查的特徵變數
X_vif = df_model_1[numerical_columns].copy()

# # 加入常數項
# X_const = add_constant(X)

# ====== 步驟 2：計算 VIF ======
vif_df = pd.DataFrame()
vif_df["變數名稱"] = X_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# ====== 步驟 3：依照 VIF 排序，顯示高共線性變數 ======
vif_df = vif_df.sort_values("VIF", ascending=False)
print("📌 所有變數的 VIF 值：")
print(vif_df)

# ====== 步驟 4：標示可能共線性的欄位 ======
high_vif = vif_df[vif_df["VIF"] > 5]
to_drop = vif_df[vif_df["VIF"] > 10]["變數名稱"].tolist()

print("\n⚠️ 建議檢查以下有中度或高度共線性的欄位 (VIF > 5)：")
print(high_vif)

if to_drop:
    print("\n❌ 以下變數 VIF > 10，建議刪除：")
    print(to_drop)
else:
    print("\n✅ 無高度共線性問題（VIF > 10）")

# ====== 步驟 5（可選）：繪製相關係數熱力圖 ======
plt.figure(figsize=(12, 10))
sns.heatmap(X_vif.corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("數值變數相關性熱力圖")
plt.show()


# %% 真實成交資料視覺化
# 實際成交 vs 實際未成交 密度分布圖
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution_by_label(df, feature_list, target_col='是否成交', bins=50):
    for col in feature_list:
        plt.figure(figsize=(8, 4))
        if df[col].nunique() > 10:
            # 連續變數用 KDE 分布
            sns.kdeplot(data=df, x=col, hue=target_col, hue_order=[0, 1], common_norm=False, fill=True, alpha=0.3)
        else:
            # 類別/離散變數用條形圖
            sns.histplot(data=df, x=col, hue=target_col, hue_order=[0, 1], multiple='dodge', bins=bins)

        plt.title(f'{col} - 成交 vs 未成交 分布')
        plt.xlabel(col)
        plt.ylabel('密度' if df[col].nunique() > 10 else '人數')
        # plt.legend(title='是否成交', labels=['成交 (1)', '未成交 (0)'])
        plt.tight_layout()
        plt.show()
        
plot_distribution_by_label(df_model_1, numerical_cols)

# 針對實際成交自動找切點 (長條圖)
def analyze_feature_vs_conversion(df, feature, target='是否成交', bins=5, min_samples_leaf=100):
    df_valid = df[[feature, target]].dropna()

    # 分箱（分位數或等寬）
    try:
        df_valid['區間'] = pd.qcut(df_valid[feature], q=bins, duplicates='drop')
    except ValueError:
        df_valid['區間'] = pd.cut(df_valid[feature], bins=bins)

    # 每區間成交率
    bin_stats = df_valid.groupby('區間')[target].agg(['count', 'sum'])
    bin_stats['成交率'] = bin_stats['sum'] / bin_stats['count']
    bin_stats.columns = ['樣本數', '成交數', '成交率']

    print(f"\n🔍 【{feature}】分箱成交率分析")
    # display(bin_stats)

    # 繪圖
    plt.figure(figsize=(8, 4))
    plt.bar(bin_stats.index.astype(str), bin_stats['成交率'], color='steelblue')
    plt.xticks(rotation=45)
    plt.ylabel('成交率')
    plt.title(f'{feature} 各區間成交率')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # 決策樹找最佳切點
    X = df_valid[[feature]].values
    y = df_valid[target].values
    clf = DecisionTreeClassifier(max_depth=1, min_samples_leaf=min_samples_leaf)
    clf.fit(X, y)

    threshold = clf.tree_.threshold[0]
    gain = clf.tree_.impurity[0] - sum([
        clf.tree_.impurity[1] * clf.tree_.n_node_samples[1] / clf.tree_.n_node_samples[0],
        clf.tree_.impurity[2] * clf.tree_.n_node_samples[2] / clf.tree_.n_node_samples[0]
    ])
    
    print(f"\n🌳 自動切點建議：{feature} ≈ {threshold:.2f}")
    print(f"📈 推薦行動建議：當【{feature} ≥ {threshold:.2f}】時，成交率明顯提高（資訊增益 ≈ {gain:.4f}）")

    return bin_stats, threshold

analyze_feature_vs_conversion(df_model_1, '每週平均拜訪客戶數')
results_summary = []

for feature in numerical_cols:
    print(f"\n📊 分析變數：{feature}")
    try:
        bin_stats, threshold = analyze_feature_vs_conversion(df_model_1, feature)
        results_summary.append({
            '變數': feature,
            '建議門檻': round(threshold, 2),
            '最高區間成交率': bin_stats['成交率'].max(),
            '最低區間成交率': bin_stats['成交率'].min(),
            '樣本數最大區間': bin_stats['樣本數'].idxmax()
        })
    except Exception as e:
        print(f"⚠️ {feature} 分析失敗：{e}")


# %% 模型比較
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix

# ===== 1. 資料與參數設定 =====
text_col = '備註文字_處理'
numerical_cols = ['備註字數', '備註行數', '拜訪次數', '拜訪目的', '業代客戶拜訪頻率', 'CEO']
target_col = '是否成交'
threshold = 0.6  # 預測門檻

# ===== 2. 資料分割（Train / Test）=====
df_train, df_test = train_test_split(df_model_1, test_size=0.2, stratify=df_model_1[target_col], random_state=42)

# ===== 3. 分群邏輯設定（用訓練資料的分位數）=====
q1, q2 = df_train['業代客戶拜訪頻率'].quantile([0.33, 0.67])

def assign_group(x):
    if x <= q1:
        return '低頻'
    elif x <= q2:
        return '中頻'
    else:
        return '高頻'

df_train['分群'] = df_train['業代客戶拜訪頻率'].apply(assign_group)
df_test['分群'] = df_test['業代客戶拜訪頻率'].apply(assign_group)

# ===== 4. 建立 TF-IDF 向量器（全體訓練資料 fit）=====
tfidf = TfidfVectorizer(max_features=1000)
tfidf.fit(df_train[text_col].fillna(''))

# ===== 5. 統一模型：整體資料建模 =====
X_text_all = tfidf.transform(df_train[text_col].fillna(''))
X_num_all = df_train[numerical_cols].fillna(0)
scaler_all = StandardScaler().fit(X_num_all)
X_num_scaled_all = scaler_all.transform(X_num_all)
X_all = csr_matrix(hstack([X_text_all, X_num_scaled_all]))
y_all = df_train[target_col]

model_all = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_all.fit(X_all, y_all)

# 測試資料轉換
X_test_text = tfidf.transform(df_test[text_col].fillna(''))
X_test_num = scaler_all.transform(df_test[numerical_cols].fillna(0))
X_test_all = csr_matrix(hstack([X_test_text, X_test_num]))
y_test = df_test[target_col]

# 預測
y_proba_all = model_all.predict_proba(X_test_all)[:, 1]
y_pred_all = (y_proba_all >= threshold).astype(int)

# 評估
print("====== 統一模型 ======")
print("F1-score:", f1_score(y_test, y_pred_all))
print("PR AUC:", average_precision_score(y_test, y_proba_all))
print("ROC AUC:", roc_auc_score(y_test, y_proba_all))


# ===== 6. 分群模型：針對每個群組訓練模型 =====
group_models = {}
group_scalers = {}

preds_group = []
probas_group = []
truth_group = []

for group in ['低頻', '中頻', '高頻']:
    df_sub_train = df_train[df_train['分群'] == group]
    df_sub_test = df_test[df_test['分群'] == group]

    if len(df_sub_train[target_col].unique()) < 2:
        print(f"⚠️ {group} 群體樣本不平衡，跳過建模")
        continue

    # 特徵處理
    X_text = tfidf.transform(df_sub_train[text_col].fillna(''))
    X_num = df_sub_train[numerical_cols].fillna(0)
    scaler = StandardScaler().fit(X_num)
    X_num_scaled = scaler.transform(X_num)
    X_train = csr_matrix(hstack([X_text, X_num_scaled]))
    y_train = df_sub_train[target_col]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    group_models[group] = model
    group_scalers[group] = scaler

    # 測試組
    X_text_test = tfidf.transform(df_sub_test[text_col].fillna(''))
    X_num_test = df_sub_test[numerical_cols].fillna(0)
    X_num_test_scaled = scaler.transform(X_num_test)
    X_test_group = csr_matrix(hstack([X_text_test, X_num_test_scaled]))

    y_proba = model.predict_proba(X_test_group)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    preds_group.extend(y_pred)
    probas_group.extend(y_proba)
    truth_group.extend(df_sub_test[target_col])

# ===== 7. 分群總結效能 =====
print("\n====== 分群模型整體表現 ======")
print("F1-score:", f1_score(truth_group, preds_group))
print("PR AUC:", average_precision_score(truth_group, probas_group))
print("ROC AUC:", roc_auc_score(truth_group, probas_group))


# # 切分資料
# X_train, X_test, y_train, y_test = train_test_split(X_all, y, stratify=y, test_size=0.2, random_state=42)

# # 建立模型
# # model = RandomForestClassifier(class_weight='balanced', random_state=42)
# # model.fit(X_train, y_train)

# from xgboost import XGBClassifier
# # 計算類別不平衡比例（未成交 : 成交）
# scale_ratio = y.value_counts()[0] / y.value_counts()[1]

# # 資料切分
# X_train, X_test, y_train, y_test = train_test_split(X_all, y, stratify=y, test_size=0.2, random_state=42)

# # 建立 XGBoost 模型
# model = XGBClassifier(
#     scale_pos_weight=scale_ratio,  # 調整不平衡
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42
# )

# model.fit(X_train, y_train)

# # 預測與機率
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]  # 預測成交機率

# # 評估
# print(classification_report(y_test, y_pred))
# print("AUC Score:", roc_auc_score(y_test, y_proba))

# threshold = 0.7
# y_pred_adj = (y_proba >= threshold).astype(int)

# print(f"📊 門檻調整為 {threshold}")
# print(classification_report(y_test, y_pred_adj))

# # 若要預測整份資料：
# df_model_1['成交機率'] = model.predict_proba(X_all)[:, 1].round(3)
# df_model_1['預測是否成交'] = model.predict(X_all)

# # 匯出給 Tableau
# df_model_1[['業代', '客戶UUID', '拜訪紀錄UUID', '標籤名稱', '是否成交', '成交機率', '預測是否成交']].to_csv("成交預測模型結果.csv", index=False)


# %% 不要用
# # %% ===== [5] 特徵重要性（只限結構特徵和目的）=====
# importances = model.feature_importances_
# tfidf_n = X_text.shape[1]
# num_n = X_num.shape[1]

# # TF-IDF 特徵名稱 + 數值特徵名稱（順序需與 X_num 一致）
# labels = list(tfidf.get_feature_names_out()) + ['備註字數', '備註行數', '拜訪次數', '拜訪目的', '業代客戶拜訪頻率']  

# # 選出 top 20 特徵
# top_idx = np.argsort(importances)[-20:]

# plt.figure(figsize=(10, 8))
# plt.barh(range(len(top_idx)), importances[top_idx])
# plt.yticks(range(len(top_idx)), [labels[i] for i in top_idx])
# plt.title("特徵重要性（Top 20）")
# plt.xlabel("Importance")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # 單獨分析結構特徵重要性 (重建特徵 + 評估重要性)
# # 建立這 5 欄的特徵矩陣
# selected_features = ['備註字數', '備註行數', '拜訪次數', '拜訪目的', '業代客戶拜訪頻率', 'CEO']
# X_struct = df_model[selected_features].fillna(0)

# # 建模
# X_scaled = StandardScaler().fit_transform(X_struct)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# model.fit(X_train, y_train)

# # 抽取重要性
# importances = model.feature_importances_
# plt.figure(figsize=(8, 6))
# plt.barh(range(len(selected_features)), importances)
# plt.yticks(range(len(selected_features)), selected_features)
# plt.xlabel("Important Score")
# plt.title("六個結構特徵的重要性（XGBoost）")
# plt.grid(True)
# plt.show()

# # 排序結果（可印出）
# feature_importance_df = pd.DataFrame({
#     '特徵': selected_features,
#     '重要性分數': importances
# }).sort_values(by='重要性分數', ascending=False)
# print(feature_importance_df)


# # %% 拜訪目的 (標籤名稱) 預測
# le = LabelEncoder()
# y_cat = le.fit_transform(df_model['標籤名稱'])

# # 3. TF-IDF 向量化
# tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
# X = tfidf.fit_transform(df_model['備註文字_處理'])

# # 4. 訓練/測試集切分
# X_train, X_test, y_train, y_test = train_test_split(X, y_cat, stratify=y_cat, test_size=0.2, random_state=42)

# # 5. 建立分類模型（Logistic Regression）
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # 6. 預測與評估
# y_pred = model.predict(X_test)
# print("📊 模型評估報告：")
# print(classification_report(y_test, y_pred, target_names=le.classes_))

# df_model['預測拜訪目的'] = le.inverse_transform(model.predict(X))  # 或 model.predict(X_text)

# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
# disp.plot(cmap='Blues', xticks_rotation=45)

# # # 以 TF-IDF 向量做語意分群
# # kmeans = KMeans(n_clusters=6, random_state=42)
# # clusters = kmeans.fit_predict(X)

# # df_model['自動分群標籤'] = clusters

# # # 顯示每群數量
# # print(df_model['自動分群標籤'].value_counts())


# # %% 導出 Tableau 使用欄位 
# # df_model.columns
# tableau_df = df_model[['營業單位', '業代', '客戶UUID', '拜訪紀錄UUID', '拜訪時間', '是否有拜訪備註',
#        '是否有備註(排除#)', '含有#', '拜訪備註_文字', '標籤名稱', '拜訪備註_詞語', '詞數', '拜訪與投保日天數差', 
#        '最近投保日', '備註字數', '備註行數', '備註文字_處理', '拜訪目的', '預測拜訪目的', '預測拜訪進度', '預測拜訪進度 (%)']]
# tableau_df.to_excel("D:/備註文字探勘/df_model.xlsx", index=False)
