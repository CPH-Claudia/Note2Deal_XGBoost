# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:48:44 2025

@author: Z01788
"""
# %% combine 
import pandas as pd
import numpy as np

# visit
visit = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="visit") 
visit['拜訪時間 年/月/日'] = pd.to_datetime(visit['拜訪時間 年/月/日'], errors='coerce')

# 計算每位業代對每個客戶的拜訪次數
visit['拜訪次數'] = (
    visit.groupby(['業代', '客戶UUID'])['拜訪紀錄UUID']
    .transform('count')
)

# 計算每位業代對所有客戶的平均拜訪次數
visit['平均每客戶拜訪次數'] = (
    visit.groupby('業代')['拜訪次數']
    .transform('mean')
)

# 複製一個表作為彙整總表
df0 = visit.copy()

# tags
tags = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="tags") 
tags['拜訪時間 年/月/日'] = pd.to_datetime(tags['拜訪時間 年/月/日'], errors='coerce')

# 1. 篩選出有「(增)拜訪目的」的 UUID
target_uuid = tags.loc[tags['標籤子分類'] == '(增)拜訪目的', '拜訪紀錄UUID'].unique()

# 2. 從 tags 中保留這些 UUID 對應的所有資料（包含方式與目的）
filtered_tags = tags[tags['拜訪紀錄UUID'].isin(target_uuid)]

# 3. 將資料 reshape 成一筆拜訪紀錄一列，每個標籤子分類一欄
pivot_tags = filtered_tags.pivot_table(
    index='拜訪紀錄UUID',
    columns='標籤子分類',
    values='標籤名稱',
    aggfunc='first'  # 如果一個欄位有多個值只取第一個（理論上應該只有一筆）
).reset_index()

# 4. 合併回 df 資料
df1 = df0.merge(pivot_tags, on='拜訪紀錄UUID', how='inner')

# import matplotlib.pyplot as plt
# plt.rc('font', family = 'Microsoft JhengHei')
# plt.rcParams['axes.unicode_minus'] = False 
# from matplotlib_venn import venn2

# # 取出 UUID 集合
# uuid_visit = set(visit['拜訪紀錄UUID'])
# uuid_tags = set(tags['拜訪紀錄UUID'])

# # 繪製 Venn Diagram
# plt.figure(figsize=(6, 6))
# venn2([uuid_visit, uuid_tags], set_labels=('visit', 'tags'))
# plt.title('交集 Venn Diagram: 拜訪紀錄UUID')
# plt.show()

# agent
agent = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="agent") 

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

last_info = agent_sorted.groupby('業代').last().reset_index()[[
    '業代', '業務員', '簽約日 年/月/日', '合終日期 年/月/日', '職級_代碼', '目前年齡', '性別', '生日', '年資']]

# 只留下有晉升日的業代（非空）
has_promotion = agent_sorted[agent_sorted['晉升日'].notna()][['業代', '晉升日']].drop_duplicates('業代')

# 合併晉升日資料
agent_summary = last_info.merge(has_promotion, on='業代', how='left')

# 無晉升者 → 填 0（也可依需求填 '無' 或 NaT）
agent_summary['晉升日'] = agent_summary['晉升日'].fillna(0)


# 命名欄位與職級編碼
agent_summary = agent_summary.rename(columns={
    '職級_代碼': '最新職級',
    '目前年齡': '業務目前年齡',
    '性別': '業務性別',
    '生日': '業務生日', 
    '年資': '目前年資'
})

# 合併進主 df
df2 = df1.merge(agent_summary, on='業代', how='left')

# 匯入各營業單位地址
address = pd.read_excel("D:/地區名稱.xlsx", sheet_name="營業單位地址") 
df2 = df2.merge(address, on='營業單位', how='left')

# 晉升日轉換
def convert_yyyymm_to_datetime(x):
    if x == 0 or pd.isna(x):
        return pd.NaT
    else:
        x = int(x)
        year = x // 100
        month = x % 100
        return pd.to_datetime(f"{year}-{month:02d}-01")

df2['晉升日期_dt'] = df2['晉升日'].apply(convert_yyyymm_to_datetime)

# 是否有晉升（只要有晉升日，不看拜訪先後）
df2['是否晉升'] = df2['晉升日期_dt'].apply(lambda x: 1 if pd.notna(x) else 0)

# 計算距離晉升天數（可為負數）
df2['距離晉升天數'] = (df2['晉升日期_dt'] - df2['拜訪時間 年/月/日']).dt.days

df2['業務生日'] = pd.to_datetime(df2['業務生日'], format='%Y%m%d', errors='coerce')  

# 計算簽約當下年齡（以年為單位，向下取整）
df2['簽約時年齡'] = ((df2['簽約日 年/月/日'] - df2['業務生日']).dt.days // 365).astype(int)

# customer
customer = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="customer", dtype={"職業類別代碼": str}) 

# 只保留個人的資料（排除法人與校正）
customer_filter = customer[customer['性別'].isin(['男', '女'])]
# customer_filter['生日 年/月/日'] = pd.to_datetime(customer_filter['生日 年/月/日'])
# customer_filter['建立時間 年/月/日'] = pd.to_datetime(customer_filter['建立時間 年/月/日'])

# 從 customer_filter 中取出需要的欄位
customer_basic = customer_filter[['客戶UUID', '客戶姓名', '性別', '生日 年/月/日', '客戶目前年齡', '通訊地郵遞區號']]

# 建立分類欄位：「準客戶」與「新增保戶」
def classify_customer_type(ctype):
    if pd.isna(ctype):
        return '未知'
    elif '準增' in ctype:
        return '準增'
    elif '錠嵂業務' in ctype:
        return '新增業務'
    else:
        return '其他'

customer_filter['業務分類'] = customer_filter['客戶類型'].apply(classify_customer_type)

# 計算每個業代在上半年新增的準客戶數與新增保戶數（以客戶UUID唯一計算）
customer_stats = (
    customer_filter
    .groupby(['業代', '業務分類'])['客戶UUID']
    .nunique()
    .unstack(fill_value=0)
    .reset_index()
)

# 確保兩欄都有，即使其中之一為 0 也存在
if '準增' not in customer_stats.columns:
    customer_stats['準增'] = 0
if '新增業務' not in customer_stats.columns:
    customer_stats['新增業務'] = 0

# 欄位重新命名
customer_stats = customer_stats.rename(columns={
    '準增': '歷年準增數',
    '新增業務': '歷年新增業務數'
})

customer_type = customer_filter[['客戶UUID', '客戶類型']].drop_duplicates()

# 合併到原始 df
df3 = df2.merge(customer_stats, on='業代', how='left') \
         .merge(customer_basic, on='客戶UUID', how='left') \
         .merge(customer_type, on='客戶UUID', how='left')
         
df3[['歷年準增數', '歷年新增業務數']] = df3[['歷年準增數', '歷年新增業務數']].fillna(0).astype(int)

# policy
policy = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="policy") 

# 確保日期欄位正確格式
policy['投保日 年/月/日'] = pd.to_datetime(policy['投保日 年/月/日'], errors='coerce')
policy['簽約日 年/月/日'] = pd.to_datetime(policy['簽約日 年/月/日'], errors='coerce')

# 對每位經紀人1-被保人做加總
policy_summary = (
    policy
    .groupby('經紀人1-被保人CRM UUID', as_index=False)[['受理件數', '繳款保費new']]
    .sum()
)

# 合併到原本的 df 中（假設 df 有 "經紀人1-被保人CRM UUID" 欄位）
# 將 policy_summary 合併進 df3，對應欄位不同，但不更動原始欄位名稱
df4 = df3.merge(
    policy_summary,
    left_on='客戶UUID',
    right_on='經紀人1-被保人CRM UUID',
    how='left'
)

# 只保留 df3 原始欄位名稱
df4 = df4.drop(columns=['經紀人1-被保人CRM UUID'])


# # 建立是否業務員標記
# policy['是否為業務'] = policy['被保人業代'].notna().astype(int)

# # 資料清理：保證保費與受理件數為數值
# policy['繳款保費new'] = pd.to_numeric(policy['繳款保費new'], errors='coerce')
# policy['受理件數'] = pd.to_numeric(policy['受理件數'], errors='coerce')

# # 分別篩選是業務員與非業務員
# policy_agent_before = policy[(policy['是否為業務'] == 1) & (policy['投保日 年/月/日'] < policy['簽約日 年/月/日'])]
# policy_non_agent = policy[policy['是否為業務'] == 0]

# # 計算加總
# agent_summary = (
#     policy_agent_before.groupby(['被保人', '被保人業代', '經紀人1-被保人CRM UUID'])
#     .agg(
#         總保費=('繳款保費new', 'sum'),
#         總受理件數=('受理件數', 'sum')
#     )
#     .reset_index()
# )
# agent_summary['群體'] = '業務員'

# # 4. 以被保人計算非業務員群體：總保費與件數
# non_agent_summary = (
#     policy_non_agent.groupby(['被保人', '經紀人1-被保人CRM UUID'])
#     .agg(
#         總保費=('繳款保費new', 'sum'),
#         總受理件數=('受理件數', 'sum')
#     )
#     .reset_index()
# )
# non_agent_summary['群體'] = '非業務員'

# # 合併資料
# final = pd.concat([agent_summary, non_agent_summary], axis=0, ignore_index=True)

# # 合併到原本的 df 中（假設 df 有 "經紀人1-被保人CRM UUID" 欄位）
# # 將 policy_summary 合併進 df3，對應欄位不同，但不更動原始欄位名稱
# df4 = df3.merge(
#     final,
#     left_on='客戶UUID',
#     right_on='經紀人1-被保人CRM UUID',
#     how='left'
# )

# # 只保留 df3 原始欄位名稱
# df4 = df4.drop(columns=['經紀人1-被保人CRM UUID'])

# 1. 處理 agent 資料
df4['業務生日'] = pd.to_datetime(df4['業務生日'], format='%Y%m%d', errors='coerce')  # 將 19901020 -> datetime
df4['業務姓名生日性別key'] = df4['業務員'].astype(str) + '_' + df4['業務生日'].dt.strftime('%Y-%m-%d') + '_' + df4['業務性別']

# 2. 處理 customer 資料
df4['客戶生日'] = pd.to_datetime(df4['生日 年/月/日'], errors='coerce')
df4['客戶姓名生日性別key'] = df4['客戶姓名'].astype(str) + '_' + df4['客戶生日'].dt.strftime('%Y-%m-%d') + '_' + df4['性別']

# 建立業務姓名生日性別key集合（整體的 set，避免逐行比對）
agent_keys = set(df4['業務姓名生日性別key'].dropna())

# # 判斷客戶是否成為業務：看客戶key是否出現在業務key集合中
# df4['客戶是否為業務'] = df4['客戶姓名生日性別key'].isin(agent_keys).astype(int)


# summary_1 = (
#     df4.groupby('客戶是否為業務')
#     .agg(
#         人數=('業代', 'nunique'),
#         平均簽約年齡=('簽約時年齡', 'mean'),
#         中位數簽約年齡=('簽約時年齡', 'median'),
#         平均簽約年限=('簽約日 年/月/日', lambda s: ((pd.Timestamp('today') - s).dt.days/365).mean())
#     )
#     .reset_index()
# )

# # 建立業務 key → 簽約日對照表（如一位業務有多筆，只取最早）
# agent_sign_dates = (
#     df4[['業務姓名生日key', '簽約日 年/月/日']]
#     .dropna()
#     .drop_duplicates('業務姓名生日key')  # 一個 key 一筆
#     .set_index('業務姓名生日key')
# )

# # 將業務簽約日合併到 df4（根據客戶是否成為業務，將其 key 對應到業務 key 的簽約日）
# df4['對應業務簽約日'] = df4['客戶姓名生日key'].map(agent_sign_dates['簽約日 年/月/日'])

# # 判斷是否為有效拜訪：成為業務且拜訪日在簽約日前
# df4['是否有效拜訪'] = (
#     (df4['客戶是否為業務'] == 1) &
#     (df4['簽約日 年/月/日'] >= df4['拜訪時間 年/月/日'])
# ).astype(int)

# 5. 計算年齡差與性別差
df4['客戶業務年齡差距'] = (
    (df4['客戶生日'] - df4['業務生日']).dt.days / 365
).round(1)

# 業務客戶匹配
sex_stage = {'男': 0, '女': 1}
df4['客戶性別'] = df4['性別'].map(sex_stage)
df4['業務性別'] = df4['業務性別'].map(sex_stage)

def gender_diff(row):
    if row['業務性別'] == row['客戶性別']:
        return row['業務性別']  # 0 or 1，性別相同
    elif row['業務性別'] == 0 and row['客戶性別'] == 1:
        return 2  # 業務男 → 客戶女
    elif row['業務性別'] == 1 and row['客戶性別'] == 0:
        return 3  # 業務女 → 客戶男
    else:
        return pd.NA  # 保險防呆，若資料異常不丟錯

df4['業務客戶性別組合'] = df4.apply(gender_diff, axis=1)

# %% 計算業務客戶距離
import pandas as pd
import xml.etree.ElementTree as ET

# 載入 XML 檔案（你可以先下載到本地，或用 requests 讀網址）
tree = ET.parse("C:/Users/Z01788/Downloads/1050812_行政區經緯度(toPost).xml")  # 檔名請用你的實際路徑
root = tree.getroot()

# 建立空 list 裝資料
data = []

# 找出每一筆行政區記錄
for item in root.findall(".//_x0031_050429_行政區經緯度_x0028_toPost_x0029_"):
    name = item.findtext("行政區名")
    zipcode = item.findtext("_x0033_碼郵遞區號")
    lon = item.findtext("中心點經度")
    lat = item.findtext("中心點緯度")
    
    if zipcode and lat and lon:
        data.append({
            "郵遞區號": zipcode.strip(),
            "行政區名": name.strip(),
            "緯度": float(lat),
            "經度": float(lon)
        })

# 轉為 DataFrame
zipcode_df = pd.DataFrame(data)
zipcode_df["郵遞區號"] = zipcode_df["郵遞區號"].astype(str)

# 客戶與業務資料補上經緯度
# 客戶
df4["客戶郵遞區號"] = df4["通訊地郵遞區號"].astype(str).str[:3]
df4 = df4.merge(zipcode_df.rename(columns={
    "郵遞區號": "客戶郵遞區號",
    "緯度": "客戶緯度",
    "經度": "客戶經度"
}), on="客戶郵遞區號", how="left")

# 業務（若你已轉換營業單位為郵遞區號）
df4["營業單位郵遞區號"] = df4["郵遞區號"].astype(str).str[:3]
df4 = df4.merge(zipcode_df.rename(columns={
    "郵遞區號": "營業單位郵遞區號",
    "緯度": "業務緯度",
    "經度": "業務經度"
}), on="營業單位郵遞區號", how="left")

from geopy.distance import geodesic

def calc_distance(row):
    if pd.notna(row["業務緯度"]) and pd.notna(row["客戶緯度"]):
        a = (row["業務緯度"], row["業務經度"])
        b = (row["客戶緯度"], row["客戶經度"])
        return geodesic(a, b).km
    return None

df4["距離_km"] = df4.apply(calc_distance, axis=1)


# %% ckip
import pandas as pd
from ckiptagger import WS
import os, pickle
from opencc import OpenCC
import requests

import tensorflow as tf
# from ckiptagger import data_utils
# data_utils.download_data_gdown("./")

def extract_and_segment_notes(df, note_column='拜訪備註', ckip_model_path='./data'):
    # 初始化 CKIP 斷詞工具
    ws = WS(ckip_model_path)

    # 自定保險術語字典（可擴充）
    insurance_terms = set([
        "保單健診", "華南產", "癌症險", "旅平險", "新安東京", "還本型保單", "富邦", "安達", "續保", 
        "富邦產", "定期壽險", "重大傷病", "實支實付", "住院醫療", "理賠", "分紅躉繳", "轉介紹", "增援", "增員", 
        "保單","保險","行銷活動","防疫險","保險經紀人","健診","籃子理論","錠嵂","意外險","資產規劃", "車險需求", 
        "儲蓄險","中壽","中國人壽","旅平","旅平險","三商美邦","成交","重大傷病","失能險","保經","見面三講","開門三講","退休規劃", 
        "車險","醫療險","火險","壽險","新光","遠雄","富邦","Toyota","機車險","寵物險","自動化工程師","六大保障","建議書", 
        "台灣人壽","失智症","app","OPP","保險存摺","國泰","遠雄","照會","三照","遞送","簽約","市調表","解約","美元保單","美元儲蓄", 
        "送保單" ,"照會單" ,"台壽","保誠" ,"癌症" ,"不動產","問卷","理賠","健診","轉介","簽收","建立關係","強制險","永達", 
        "觀念溝通","需求分析","保單健診","終身","萬事利達","續保","友邦","寒暄","關心","保險存摺","年金","PHB","宏泰", 
        "南山","長照","XHB","HNRC","新生兒","約訪","年繳","美金","phb","探班","要保人",'企管副會長','意外險需求','double鑫','下週'
    ])

    # 1. 清洗備註文字
    def clean_text(note):
        if pd.isna(note): return ''
        note = str(note).replace('_x000D_', '').replace('\r', '').replace('\n', ' ')  # ✅ 核心修正
        return ' '.join([
            line.strip() for line in note.splitlines()
            if line.strip() and not line.strip().startswith(('＃', '#'))
        ])

    df = df.copy()
    df['備註_清理'] = df[note_column].apply(clean_text)
    text_list = df['備註_清理'].tolist()

    # 2. CKIP 斷詞
    ws_result = ws(text_list)

    # 3. 合併保險術語
    def merge_custom_terms(ws_result, term_set):
        max_len = max(len(term) for term in term_set)
        merged = []
        for sent in ws_result:
            i, merged_sent = 0, []
            while i < len(sent):
                match = None
                for l in range(min(max_len, len(sent) - i), 0, -1):
                    phrase = ''.join(sent[i:i+l])
                    if phrase in term_set:
                        match = phrase
                        i += l
                        break
                if match:
                    merged_sent.append(match)
                else:
                    merged_sent.append(sent[i])
                    i += 1
            merged.append(merged_sent)
        return merged

    word_list = merge_custom_terms(ws_result, insurance_terms)

    # 4. 載入繁體停用詞
    stopword_urls = [
        "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt",
        "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt",
        "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt",
        "https://raw.githubusercontent.com/goto456/stopwords/master/scu_stopwords.txt"
    ]
    cc = OpenCC('s2t')
    stopwords_trad = set()
    for url in stopword_urls:
        try:
            r = requests.get(url)
            r.encoding = 'utf-8'
            if r.status_code == 200:
                for line in r.text.strip().splitlines():
                    word = cc.convert(line.strip())
                    if word: stopwords_trad.add(word)
        except:
            print(f"⚠️ 停用詞載入失敗: {url}")

    # 5. 去除停用詞與過短詞
    filtered_words = [
        [w for w in sent if w not in stopwords_trad and len(w) > 1]
        for sent in word_list
    ]

    # 回存處理結果
    df['斷詞結果'] = filtered_words
    return df

df4 = extract_and_segment_notes(df4, note_column='拜訪備註', ckip_model_path='./data')

# %% 新進業務員是否為保戶
from sklearn.preprocessing import LabelEncoder

# 篩選簽約日在 2020/1/1 ~ 2024/12/31 之間的新進業務
agent2 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="agent2") 
# 確保兩欄都是 datetime 格式
agent2['簽約日 年/月/日'] = pd.to_datetime(agent2['簽約日 年/月/日'], errors='coerce')
agent2['生日'] = pd.to_datetime(agent2['生日'], format='%Y%m%d', errors='coerce')  
agent2['性別'] = agent2['性別'].astype(str)

# 計算簽約當下年齡（以年為單位，向下取整）
agent2['簽約時年齡'] = ((agent2['簽約日 年/月/日'] - agent2['生日']).dt.days // 365).astype(int)
agent2['姓名生日性別key'] = agent2['業務員'].astype(str) + '_' + agent2['生日'].astype(str) + '_' + agent2['性別']

policy3 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="工作表3") 
policy3['被保人生日 年/月/日'] = pd.to_datetime(policy3['被保人生日 年/月/日'], format='%Y%m%d', errors='coerce')  

policy3['姓名生日性別key'] = policy3['被保人'].astype(str) + '_' + policy3['被保人生日 年/月/日'].astype(str) + '_' + policy3['被保人性別']

# 2. 建立：客戶 -> 最早交易日的查詢表
policy_key_date = (
    policy3.sort_values('投保日 年/月/日')  # 先排序
    .drop_duplicates('姓名生日性別key', keep='first')  # 保留最早投保日
    .set_index('姓名生日性別key')['投保日 年/月/日']
    .to_dict()
)

# 3. 判斷是否為保戶
def is_existing_customer(row):
    key = row['姓名生日性別key']
    trade_date = policy_key_date.get(key)
    if pd.notnull(trade_date) and trade_date < row['簽約日 年/月/日']:
        return 1
    return 0

# 4. 套用判斷
agent2['成為業務前是否為保戶'] = agent2.apply(is_existing_customer, axis=1)

# 只保留需要的欄位
agent_key = (
    agent2[['姓名生日性別key', '簽約日 年/月/日', '成為業務前是否為保戶']]
    .drop_duplicates(subset='姓名生日性別key')
)

# 合併簽約日到保單資料
policy_agent = policy3.merge(agent_key, on='姓名生日性別key', how='inner')

# 篩選簽約日前的保單
policy_before = policy_agent[policy_agent['投保日 年/月/日'] < policy_agent['簽約日 年/月/日']].copy()


# 計算每位業務在簽約前的保費與保單種類
summary = (
    policy_before.groupby('姓名生日性別key')
    .agg(
        成為業務前是否為保戶=('成為業務前是否為保戶', 'first'),
        簽約日=('簽約日 年/月/日', 'first'), 
        累積保費=('繳款保費new', 'sum'),
        累積受理件數=('受理件數', 'sum')
    )
    .reset_index()
)


df4['生日 年/月/日'] = pd.to_datetime(df4['生日 年/月/日'], format='%Y%m%d', errors='coerce')  
df4['姓名生日性別key'] = df4['客戶姓名'].astype(str) + '_' + df4['生日 年/月/日'].astype(str) + '_' + df4['性別']

# 合併，summary 的姓名生日key 對應到 df4 的 客戶姓名生日key
df5 = df4.merge(
    summary,
    on='姓名生日性別key',
    how='left',
    suffixes=('', '_簽約前')
)

df5['客戶簽約時年齡'] = (
    ((df5['簽約日'] - df5['生日 年/月/日']).dt.days // 365)
    .where(df5['簽約日'].notna())
)

# 若簽約前的精準值有資料，則取代原本欄位
df5['累積保費'] = df5['累積保費'].combine_first(df5['繳款保費new'])
df5['累積受理件數'] = df5['累積受理件數'].combine_first(df5['受理件數'])

# 確保日期格式正確
df5['拜訪時間 年/月/日'] = pd.to_datetime(df5['拜訪時間 年/月/日'], errors='coerce')
df5['簽約日'] = pd.to_datetime(df5['簽約日'], errors='coerce')

# 計算每位客戶的第一次拜訪日
first_visit = (
    df5.groupby('客戶UUID', as_index=False)['拜訪時間 年/月/日']
    .min()
    .rename(columns={'拜訪時間 年/月/日': '第一次拜訪日'})
)

# 合併第一次拜訪日到 df5
df5 = df5.merge(first_visit, on='客戶UUID', how='left')

# 計算時長
today = pd.Timestamp('today').normalize()  # 取今天日期（不含時間）

df5['拜訪到簽約天數'] = np.where(
    df5['簽約日'].notna(),
    (df5['簽約日'] - df5['第一次拜訪日']).dt.days,
    (today - df5['第一次拜訪日']).dt.days
)

purp_stage = {'(增)增援接觸': 0, '(增)約訪': 1, '(增)面談': 2, '(增)主管面談': 3, '(增)簽約': 4}
df5['拜訪目的'] = df5['(增)拜訪目的'].map(purp_stage).fillna(-1)

# 2.2 方式 Label Encoding
le = LabelEncoder()
df5['方式_num'] = le.fit_transform(df5['方式'].astype(str))

# 刪除多餘欄位
df5.drop(columns=['繳款保費new', '受理件數'], inplace=True)




df5_filtered = df5[
    ~(
        (df5['成為業務前是否為保戶'] == 1) & 
        (df5['拜訪時間 年/月/日'] > df5['簽約日'])
    )
].copy()

df5_filtered['是否有效拜訪'] = (
    (df5_filtered['成為業務前是否為保戶'] == 1) &
    (df5_filtered['簽約日'] >= df5_filtered['拜訪時間 年/月/日'])
).astype(int)

# %% 有意義的詞數
from gensim.models import Word2Vec

def extract_recruit_features_from_tokenlist(df,
                                            tokenlist_column='斷詞結果',
                                            rawtext_column='備註_清理',
                                            min_sim_score=0.6,
                                            model_vector_size=150):
    """
    df: 包含斷詞欄位（list 格式）與原始備註欄位（str）
    tokenlist_column: 斷詞欄位，格式為 list
    rawtext_column: 原始清理後備註欄位，用來計算字數
    """

    # === Step 1: 增員種子詞 ===
    recruit_seed_words = [
        "增員", "轉職", "推薦", "創業", "副業", "面談", "面試", "事業", "邀約", "詢問",
        "邀請工作", "主動", "邀請面談", "說明工作", "對工作有興趣", "適合這份工作", 
        "尋找工作", "問工作內容", "時間彈性", "自由收入", "職涯", "事業規劃", "邀請加入", 
        "簽約", "報名", "新人", "計畫", "考照班", "考照", "證照", "尖兵", "報名"
    ]

    # === Step 2: Word2Vec 訓練 ===
    token_lists = df[tokenlist_column].dropna().tolist()
    model = Word2Vec(sentences=token_lists, vector_size=model_vector_size, window=5, min_count=2, workers=4)

    # === Step 3: 擴充語意詞集合 ===
    recruit_words_set = set(recruit_seed_words)
    for word in recruit_seed_words:
        if word in model.wv:
            for sim_word, score in model.wv.most_similar(word, topn=20):
                if score >= min_sim_score:
                    recruit_words_set.add(sim_word)

    print(f"📌 擴充後的語意詞數量：{len(recruit_words_set)}")

    # === Step 4: 計算三項特徵 ===
    def count_recruit_words(tokens):
        if not isinstance(tokens, list): return 0
        return sum(1 for word in tokens if word in recruit_words_set)

    df = df.copy()
    df['詞數'] = df[tokenlist_column].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['增員語意詞數'] = df[tokenlist_column].apply(count_recruit_words)
    df['recruit_ratio'] = df.apply(
        lambda row: row['增員語意詞數'] / row['詞數'] if row['詞數'] > 0 else 0, axis=1
    )

    # === Step 5: 原始備註字數計算（不含空白與換行）
    def count_characters(text):
        if pd.isna(text): return 0
        return len(str(text).replace('\n', '').replace(' ', '').replace('\r', ''))

    df['備註字數'] = df[rawtext_column].apply(count_characters)

    return df, recruit_words_set

df_new, recruit_words = extract_recruit_features_from_tokenlist(df5_filtered)

# %% filter 
# 缺失值處理
df_drop = df_new.dropna(subset=['客戶性別', '客戶目前年齡', '業務目前年齡'])

keep_cols = [
    '營業單位', '營業單位代碼', '業代', '客戶UUID', '拜訪紀錄UUID', '拜訪時間 年/月/日', '拜訪備註',
    '拜訪次數', '平均每客戶拜訪次數', '(增)拜訪目的', '方式', '業務員', '簽約日 年/月/日', '最新職級',
    '業務目前年齡', '業務性別', '業務生日', '目前年資', '晉升日', '距離晉升天數', '歷年新增業務數',
    '歷年準增數', '客戶姓名', '性別', '客戶生日', '客戶類型', 
    '是否有效拜訪', '客戶業務年齡差距', '業務客戶性別組合', 
    '距離_km', '成為業務前是否為保戶', '拜訪到簽約天數', '拜訪目的', '方式_num',
    '簽約日', '累積保費', '累積受理件數', '客戶簽約時年齡', 
    '備註_清理', '斷詞結果', '增員語意詞數', '備註字數'
]

df_cleaned = df_drop[keep_cols].copy()

df_cleaned[['累積保費', '累積受理件數']] = df_cleaned[[
    '累積保費', '累積受理件數']].fillna(0)
# df_drop[['對應業務簽約日', '合終日期 年/月/日', '晉升日期_dt', '拜訪備註', 
#      '客戶姓名', '客戶姓名生日key', ]] = df_drop[[
#          '對應業務簽約日', '合終日期 年/月/日', '晉升日期_dt', '拜訪備註', 
#           '客戶姓名', '客戶姓名生日key', '客戶類型']].fillna(pd.NaT)

missing_report = df_cleaned.isna().sum()
missing_report = missing_report[missing_report > 0].sort_values(ascending=False)


# # 篩選 2024/1/1 ~ 2024/12/31 的筆數
# df_filtered = df_cleaned[
#     ((df_cleaned['拜訪時間 年/月/日'] >= pd.Timestamp('2024-01-01')) & (df_cleaned['拜訪時間 年/月/日'] <= pd.Timestamp('2024-12-31')))
# ]

df_filtered_1 = df_cleaned[df_cleaned['平均每客戶拜訪次數'] > 4]

# invalid_visits = df_new[
#     (df_new['客戶是否為業務'] == 1) &
#     (df_new['簽約日'] < df_new['拜訪時間 年/月/日'])
# ]

unit_stats = df_filtered_1.groupby('營業單位').agg(
    業務員數=('業代', 'nunique'),
    客戶數=('客戶UUID', 'nunique'),
    拜訪次數=('拜訪紀錄UUID', 'nunique')
).reset_index()
# 業務數: 411 / 客戶數: 4852

# %% model-xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

# ===== 1. 篩選與建立成交標籤 =====
# df_model_1 = df_filtered_1[df_filtered_1['平均每客戶拜訪次數'] > 4].copy()
# df_model_1['是否成交'] = df_model_1['是否有效拜訪']  # 或改為你要的成交定義
describe = df_filtered_1.describe()
y = df_filtered_1['是否有效拜訪']
groups = df_filtered_1['客戶UUID']  # 分群欄位：同一位潛在業務員的拜訪紀錄歸為一群


# 1. 數值特徵
# 需要標準化的欄位
cols_to_scale = [
    '拜訪次數', '平均每客戶拜訪次數', '目前年資', '距離晉升天數',
    '備註字數', '增員語意詞數', '累積受理件數', '拜訪到簽約天數', # '客戶簽約時年齡', 
    '累積保費', '客戶業務年齡差距', '距離_km'
]

# 不標準化欄位
cols_not_scale = ['最新職級', '業務客戶性別組合']

# 分開處理
scaler = StandardScaler()
X_scaled_part = scaler.fit_transform(df_filtered_1[cols_to_scale])
X_not_scaled_part = df_filtered_1[cols_not_scale].values

# 組合完整數值特徵
X_num_final = np.hstack([X_scaled_part, X_not_scaled_part])

# 2. 類別特徵
# 2.1 拜訪目的映射
purp_stage = {'(增)增援接觸': 0, '(增)約訪': 1, '(增)面談': 2, '(增)主管面談': 3, '(增)簽約': 4}
df_filtered_1['拜訪目的_num'] = df_filtered_1['(增)拜訪目的'].map(purp_stage).fillna(-1)

# 2.2 方式 Label Encoding
le = LabelEncoder()
df_filtered_1['方式_num'] = le.fit_transform(df_filtered_1['方式'].astype(str))

X_cat = df_filtered_1[['拜訪目的_num', '方式_num']].fillna(-1).values

# 3. Word2Vec + TF-IDF 加權詞向量
sentences = df_filtered_1['斷詞結果'].dropna().tolist()

w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2)

tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
tfidf.fit(sentences)
tfidf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

def vectorize_sentence_weighted(sentence):
    if isinstance(sentence, list):
        vecs, weights = [], []
        for word in sentence:
            if word in w2v_model.wv and word in tfidf_dict:
                vecs.append(w2v_model.wv[word] * tfidf_dict[word])
                weights.append(tfidf_dict[word])
        return np.sum(vecs, axis=0) / np.sum(weights) if vecs else np.zeros(w2v_model.vector_size)
    else:
        return np.zeros(w2v_model.vector_size)

X_w2v_weighted = np.vstack(df_filtered_1['斷詞結果'].apply(vectorize_sentence_weighted))

# 4. 組合所有特徵
X_combined = np.hstack([
    X_w2v_weighted,  # 100維
    X_num_final,    # 數值欄位
    X_cat            # 類別欄位
])

print(f'最終特徵矩陣形狀：{X_combined.shape}')


# 沒切分客戶uuid
from sklearn.model_selection import StratifiedKFold

# 資料切分
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_combined,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 初始化交叉驗證器（不分群）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_scores, pr_scores = [], []
all_y_true, all_y_pred, all_y_proba = [], [], []

for train_idx, val_idx in skf.split(X_trainval, y_trainval):
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    scale_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    model = XGBClassifier(scale_pos_weight=scale_ratio, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.6).astype(int)

    roc_scores.append(roc_auc_score(y_val, y_proba))
    pr_scores.append(average_precision_score(y_val, y_proba))
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

# ===== 3. 報告交叉驗證結果 =====
print("\n📊 Cross-Validation (Train Set, Grouped by 客戶UUID)")
print(classification_report(all_y_true, all_y_pred))
print(f"Average ROC AUC: {np.mean(roc_scores):.4f}")
print(f"Average PR AUC : {np.mean(pr_scores):.4f}")


# ===== Hold-out 測試集評估 =====
final_model = XGBClassifier(scale_pos_weight=scale_ratio, use_label_encoder=False, eval_metric='logloss', random_state=42)
final_model.fit(X_trainval, y_trainval)

y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.6).astype(int)

print("\n🧪 Final Evaluation on Hold-out Test Set")
print(classification_report(y_test, y_test_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"PR AUC : {average_precision_score(y_test, y_test_proba):.4f}")






# ===================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# ===== 1. 基本欄位設定 =====
y = df_filtered_1['是否有效拜訪']
groups = df_filtered_1['客戶UUID']  # 分群欄位

numerical_cols = [
    '拜訪次數', '平均每客戶拜訪次數', '目前年資', '距離晉升天數', '最新職級', 
    '備註字數', '增員語意詞數', '累積受理件數', '客戶簽約時年齡', 
    '累積保費', '客戶業務年齡差距', '業務客戶性別組合', '距離_km', 
    '拜訪目的', '方式_num'
]

categorical_cols = []

# ===== 2. 區分標準化與否的數值欄位 =====
cols_to_scale = [
    '拜訪次數', '平均每客戶拜訪次數', '目前年資', '距離晉升天數',
    '備註字數', '增員語意詞數', '累積受理件數', '累積保費',
    '客戶業務年齡差距', '距離_km'
]

cols_no_scale = ['最新職級', '客戶簽約時年齡', '業務客戶性別組合']

# ===== 3. 切分資料 =====
feature_data = df_filtered_1[numerical_cols + categorical_cols + ['斷詞結果', '客戶UUID']].copy()

X_temp_trainval, X_temp_test, y_trainval, y_test, groups_trainval, groups_test = train_test_split(
    feature_data, 
    y, 
    groups,
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# ===== 4. Word2Vec & TF-IDF (僅使用訓練集) =====
train_sentences = X_temp_trainval['斷詞結果'].dropna().tolist()

w2v_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=2)

tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
tfidf.fit(train_sentences)
tfidf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

def vectorize_sentence_weighted(sentence):
    if not isinstance(sentence, list):
        return np.zeros(w2v_model.vector_size)
    vecs, weights = [], []
    for word in sentence:
        if word in w2v_model.wv and word in tfidf_dict:
            vecs.append(w2v_model.wv[word] * tfidf_dict[word])
            weights.append(tfidf_dict[word])
    if vecs:
        return np.sum(vecs, axis=0) / np.sum(weights)
    else:
        return np.zeros(w2v_model.vector_size)

# 計算詞向量
X_w2v_trainval = np.vstack(X_temp_trainval['斷詞結果'].apply(vectorize_sentence_weighted))
X_w2v_test = np.vstack(X_temp_test['斷詞結果'].apply(vectorize_sentence_weighted))

# ===== 5. 數值欄位處理 =====
scaler = StandardScaler()

X_num_trainval_scaled = X_temp_trainval[cols_to_scale].fillna(0)
X_num_test_scaled = X_temp_test[cols_to_scale].fillna(0)

X_num_trainval_scaled = scaler.fit_transform(X_num_trainval_scaled)
X_num_test_scaled = scaler.transform(X_num_test_scaled)

X_num_trainval_final = np.hstack([
    X_num_trainval_scaled,
    X_temp_trainval[cols_no_scale].fillna(0).values
])

X_num_test_final = np.hstack([
    X_num_test_scaled,
    X_temp_test[cols_no_scale].fillna(0).values
])

# ===== 6. 類別欄位 One-Hot 處理 =====
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_temp_trainval[categorical_cols])

df_cat_trainval = encoder.transform(X_temp_trainval[categorical_cols])
df_cat_test = encoder.transform(X_temp_test[categorical_cols])

# ===== 7. 組合最終特徵 =====
X_trainval = np.hstack([
    X_w2v_trainval,              
    X_num_trainval_final,       
    df_cat_trainval              
])

X_test = np.hstack([
    X_w2v_test,                  
    X_num_test_final,           
    df_cat_test                  
])

print(f"最終特徵維度 - 訓練集: {X_trainval.shape}, 測試集: {X_test.shape}")


# ===== 7. 交叉驗證（在訓練+驗證集上）=====
print("\n📊 開始交叉驗證...")

# 選擇交叉驗證策略
# 選項1: 不考慮客戶分群
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = skf.split(X_trainval, y_trainval)

# 選項2: 考慮客戶分群（取消註解使用）
# sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# cv_splits = sgkf.split(X_trainval, y_trainval, groups_trainval)

roc_scores, pr_scores = [], []
all_y_true, all_y_pred, all_y_proba = [], [], []

for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
    print(f"  處理第 {fold} 折...")
    
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
    
    # 計算類別權重
    scale_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # 訓練模型
    model = XGBClassifier(
        scale_pos_weight=scale_ratio, 
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 預測
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.6).astype(int)
    
    # 記錄結果
    roc_scores.append(roc_auc_score(y_val, y_proba))
    pr_scores.append(average_precision_score(y_val, y_proba))
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

# ===== 8. 報告交叉驗證結果 =====
print("\n📊 Cross-Validation Results")
print("=" * 50)
print(classification_report(all_y_true, all_y_pred))
print(f"Average ROC AUC: {np.mean(roc_scores):.4f} (±{np.std(roc_scores):.4f})")
print(f"Average PR AUC : {np.mean(pr_scores):.4f} (±{np.std(pr_scores):.4f})")

# ===== 9. 最終模型訓練與測試集評估 =====
print("\n🧪 最終模型評估...")

# 計算最終的類別權重
final_scale_ratio = y_trainval.value_counts()[0] / y_trainval.value_counts()[1]

# 訓練最終模型
final_model = XGBClassifier(
    scale_pos_weight=final_scale_ratio, 
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42
)
final_model.fit(X_trainval, y_trainval)

# 在測試集上預測
y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.6).astype(int)

# ===== 10. 最終結果報告 =====
print("\n🎯 Final Evaluation on Hold-out Test Set")
print("=" * 50)
print(classification_report(y_test, y_test_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"PR AUC : {average_precision_score(y_test, y_test_proba):.4f}")

# ===== 11. 特徵重要性分析 =====
print("\n🔍 Top 10 Feature Importances:")
# Word2Vec 特徵
w2v_feature_names = [f'w2v_{i}' for i in range(X_w2v_trainval.shape[1])]

# 數值特徵
num_feature_names = list(X_temp_trainval[numerical_cols].columns)

# 類別特徵
if df_cat_trainval.ndim == 2:
    cat_feature_names = [f'cat_{i}' for i in range(df_cat_trainval.shape[1])]
else:
    cat_feature_names = []

# 最終特徵名稱
feature_names = w2v_feature_names + num_feature_names + cat_feature_names

importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df.head(10).to_string(index=False))




# ===================================================
# Step 1: 取得每位客戶的唯一 ID
unique_customers = df_filtered_1['客戶UUID'].unique()

# Step 2: 分層切分客戶（先抽出 test 客戶）
trainval_customers, test_customers = train_test_split(
    unique_customers,
    test_size=0.2,
    random_state=42,
    stratify=df_filtered_1.drop_duplicates('客戶UUID')['是否有效拜訪']
)

# Step 3: 用客戶UUID篩選出對應資料
trainval_mask = df_filtered_1['客戶UUID'].isin(trainval_customers)
test_mask = df_filtered_1['客戶UUID'].isin(test_customers)

# ===== 4. 資料切分與模型建構 =====
X_trainval, X_test = X_combined[trainval_mask], X_combined[test_mask]
y_trainval, y_test = y[trainval_mask], y[test_mask]

# ===== 1. 初始化交叉驗證器 =====
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

roc_scores, pr_scores = [], []
all_y_true, all_y_pred, all_y_proba = [], [], []

# ===== 2. 執行交叉驗證（注意 groups 要與 X_trainval 對齊） =====
for train_idx, val_idx in sgkf.split(X_trainval, y_trainval, groups=groups[trainval_mask]):
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    scale_ratio = df_filtered_1['是否有效拜訪'].value_counts()[0] / df_filtered_1['是否有效拜訪'].value_counts()[1]
    model = XGBClassifier(scale_pos_weight=scale_ratio, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.6).astype(int)

    roc_scores.append(roc_auc_score(y_val, y_proba))
    pr_scores.append(average_precision_score(y_val, y_proba))

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)



# %% shap
# 總體解釋
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False 


# 只取數值欄位對應的 SHAP 值與資料
num_start_idx = X_combined.shape[1] - len(numerical_cols)  # 數值特徵起始位置
X_test_num = X_test[:, num_start_idx:]

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

shap_values_num = shap_values[:, num_start_idx:]

# 特徵名稱
feature_names = numerical_cols

# ===== 1️⃣ 特徵重要性（條狀圖） =====
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'MeanAbsSHAP': np.abs(shap_values_num).mean(axis=0)
}).sort_values(by='MeanAbsSHAP', ascending=True)

# 繪圖
plt.figure(figsize=(8, 6))
plt.barh(shap_importance_df['Feature'], shap_importance_df['MeanAbsSHAP'])
plt.xlabel('Mean |SHAP value|')
plt.title('Feature Importance (Numerical Only)')
plt.tight_layout()
plt.show()

# ===== 2️⃣ SHAP summary plot（顏色代表數值大小） =====
shap.summary_plot(shap_values_num, X_test_num, feature_names=feature_names)









# 各變數解釋
num_vars = [
    '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', # '拜訪紀錄密度', 
    '備註字數', '有意義詞數', 
    '目前年資', # '當年度賽季增員數', 
    '上半年準客戶數', '今年度活動參與率', '上年度FYC', '距離晉升天數', 
    '件數', '總保費'
]

cat_vars = ['業務客戶性別組合', '最新職級', '拜訪目的', '營業單位_編碼']


# # 輸出資料夾
# output_dir = "D:/備註文字探勘/shap_2024"
# os.makedirs(output_dir, exist_ok=True)

# # 個別變數解釋
# explainer = shap.Explainer(final_model, 
#                            X_trainval, 
#                            feature_names=final_feature_names, 
#                            model_output='raw', 
#                            feature_perturbation="interventional") # 減少隨機性

# shap_values = explainer(X_trainval)

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

            ax1.scatter(df["值"], df["SHAP"], alpha=0.3, label="原始 SHAP")
            ax1.plot(df["值"], df["SHAP_smooth"], color='blue', label="平滑趨勢")
            ax1.axhline(0, color='gray', linestyle='--')
            # ax1.axvline(peak_x, color='red', linestyle='--', 
            #             label=f'最大貢獻點 = {peak_raw:,.2f}')  # ⭐ 貨幣格式)

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
            ax1.set_xlabel("標準化", labelpad=10)
            ax1.set_ylabel("SHAP 值")
            ax1.set_title(f"{var} 對成交的 SHAP 趨勢")
            
            # 雙 X 軸：下方顯示原始數值，對齊標準化 X 軸
            def to_raw(x): return x * scale + mean
            def to_std(x): return (x - mean) / scale
            
            # 替代 ax2 = ax1.twiny()
            secax = ax1.secondary_xaxis('bottom', functions=(to_raw, to_std))
            secax.set_xlabel("原始數值")
            
            # 平均與標準差說明
            mean_text = f"原始平均值: {mean:,.2f}\n原始標準差: {scale:,.2f}"  # ⭐ 貨幣格式
            ax1.text(0.98, 0.95, mean_text, transform=ax1.transAxes,
                     ha='right', va='top', fontsize=8, color='dimgray',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))

            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            if output_dir:
                filename = os.path.join(output_dir, f"{var}_shap_plot2.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ 已儲存：{filename}")
            else:
                plt.tight_layout()
            
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

# %% model-lstm 
import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight

def prepare_lstm_data(df, time_steps=5, embedding_dim=100):
    from gensim.models import Word2Vec
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 1. 將斷詞結果變為字串以供 TfidfVectorizer 使用
    df = df.copy()
    df['斷詞字串'] = df['斷詞結果'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')


    # 2. 訓練 TF-IDF 向量器
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['斷詞字串'])  # ✅ 必須先 fit 才能用 idf_
    idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))  # ✅ 正確取得詞權重

    # 3. 訓練 Word2Vec
    token_lists = df['斷詞結果'].dropna().tolist()
    w2v_model = Word2Vec(sentences=token_lists, vector_size=embedding_dim, window=3, min_count=1, workers=4)

    # 4. 加權平均詞向量（僅取前 10 高權重詞）
    def sentence_to_vec(sentence):
        tokens = [word for word in sentence if word in w2v_model.wv and word in idf_dict]
        sorted_tokens = sorted(tokens, key=lambda w: idf_dict[w], reverse=True)[:10]
        weighted_vectors = []
        weights = []

        for word in sorted_tokens:
            weight = idf_dict[word]
            weighted_vectors.append(w2v_model.wv[word] * weight)
            weights.append(weight)

        if weighted_vectors:
            return np.sum(weighted_vectors, axis=0) / np.sum(weights)
        else:
            return np.zeros(embedding_dim)

    # 5. 時序轉換
    df_sorted = df.sort_values('拜訪時間 年/月/日')
    X_data, y_data, uuid_list = [], [], []

    grouped = df_sorted.groupby('客戶UUID')

    for uid, group in grouped:
        vec_seq = [sentence_to_vec(row['斷詞結果']) for _, row in group.iterrows()]
        if len(vec_seq) < time_steps:
            pad_len = time_steps - len(vec_seq)
            vec_seq = [np.zeros(embedding_dim)] * pad_len + vec_seq
        else:
            vec_seq = vec_seq[-time_steps:]

        X_data.append(vec_seq)
        y_data.append(group['是否有效拜訪'].iloc[0])
        uuid_list.append(uid)

    return np.array(X_data, dtype='float32'), np.array(y_data, dtype='int32'), uuid_list, w2v_model, tfidf


# === 載入資料 ===
X_lstm, y_lstm, uuids, w2v_model, tfidf_model = prepare_lstm_data(df_filtered_1, time_steps=5, embedding_dim=100)

# === 建立模型（固定 input_shape，避開 symbolic tensor）===
model = Sequential([
    LSTM(64, input_shape=(5, 100)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === class_weight 計算 ===
classes = np.unique(y_lstm)
weights = compute_class_weight('balanced', classes=classes, y=y_lstm)
class_weight_dict = {int(c): w for c, w in zip(classes, weights)}

# 評估分類模型預測效果
import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False 

# 訓練模型並記錄歷程
history = model.fit(
    X_lstm, y_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict
)

# 畫出 Loss 與 Accuracy 曲線
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 對驗證集（validation_split=0.2）做切分
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, stratify=y_lstm, random_state=42)

# 預測機率與類別
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

# 評估指標
print("\n📋 Classification Report:")
print(classification_report(y_val, y_pred))

# 混淆矩陣
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# %% 新進業務員是否為保戶
# 篩選簽約日在 2020/1/1 ~ 2024/12/31 之間的新進業務
agent2 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="agent2") 
# 確保兩欄都是 datetime 格式
agent2['簽約日 年/月/日'] = pd.to_datetime(agent2['簽約日 年/月/日'], errors='coerce')
agent2['生日'] = pd.to_datetime(agent2['生日'], format='%Y%m%d', errors='coerce')  
agent2['性別'] = agent2['性別'].astype(str)

# 計算簽約當下年齡（以年為單位，向下取整）
agent2['簽約時年齡'] = ((agent2['簽約日 年/月/日'] - agent2['生日']).dt.days // 365).astype(int)

# policy2 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="policy2") 
# policy2['被保人生日 年/月/日'] = pd.to_datetime(policy2['被保人生日 年/月/日'], format='%Y%m%d', errors='coerce')
# policy2['性別'] = policy2['性別'].astype(str)

# 1. 建立比對 key
agent2['姓名生日性別key'] = agent2['業務員'].astype(str) + '_' + agent2['生日'].astype(str) + '_' + agent2['性別']
# policy2['姓名生日性別key'] = policy2['被保人'].astype(str) + '_' + policy2['被保人生日 年/月/日'].astype(str) + '_' + policy2['性別']

# # 2. 建立：客戶 -> 最早交易日的查詢表
# policy_key_date = policy2.set_index('姓名生日性別key')['最小值 投保日'].to_dict()

# # 3. 判斷是否為保戶
# def is_existing_customer(row):
#     key = row['姓名生日性別key']
#     trade_date = policy_key_date.get(key)
#     if pd.notnull(trade_date) and trade_date < row['簽約日 年/月/日']:
#         return 1
#     return 0

# # 4. 套用判斷
# agent2['成為業務前是否為保戶'] = agent2.apply(is_existing_customer, axis=1)

# # 5. 統計
# summary = (
#     agent2.groupby('成為業務前是否為保戶')
#     .agg(
#         人數=('業代', 'nunique'),
#         平均簽約年齡=('簽約時年齡', 'mean'),
#         中位數簽約年齡=('簽約時年齡', 'median'),
#         平均簽約年限=('簽約日 年/月/日', lambda s: ((pd.Timestamp('today') - s).dt.days/365).mean())
#     )
#     .reset_index()
# )

# # 6. 占比
# total = agent2['業代'].nunique()
# summary['占比'] = summary['人數'] / total

# print(summary)

# # 近五年新進業務員
# mask = (agent2['簽約日 年/月/日'] >= '2020-01-01') & (agent2['簽約日 年/月/日'] <= '2024-12-31')
# agent2_filtered = agent2[mask].copy()

policy3 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="工作表3") 
policy3['被保人生日 年/月/日'] = pd.to_datetime(policy3['被保人生日 年/月/日'], format='%Y%m%d', errors='coerce')  

policy3['姓名生日性別key'] = policy3['被保人'].astype(str) + '_' + policy3['被保人生日 年/月/日'].astype(str) + '_' + policy3['被保人性別']

# 2. 建立：客戶 -> 最早交易日的查詢表
policy_key_date = (
    policy3.sort_values('投保日 年/月/日')  # 先排序
    .drop_duplicates('姓名生日性別key', keep='first')  # 保留最早投保日
    .set_index('姓名生日性別key')['投保日 年/月/日']
    .to_dict()
)

# 3. 判斷是否為保戶
def is_existing_customer(row):
    key = row['姓名生日性別key']
    trade_date = policy_key_date.get(key)
    if pd.notnull(trade_date) and trade_date < row['簽約日 年/月/日']:
        return 1
    return 0

# 4. 套用判斷
agent2['成為業務前是否為保戶'] = agent2.apply(is_existing_customer, axis=1)

# 只保留需要的欄位
agent_key = (
    agent2[['姓名生日性別key', '簽約日 年/月/日', '成為業務前是否為保戶']]
    .drop_duplicates(subset='姓名生日性別key')
)

# 合併簽約日到保單資料
policy_agent = policy3.merge(agent_key, on='姓名生日性別key', how='inner')

# 篩選簽約日前的保單
policy_before = policy_agent[policy_agent['投保日 年/月/日'] < policy_agent['簽約日 年/月/日']].copy()

# 計算每位業務在簽約前的保費與保單種類
summary = (
    policy_before.groupby('姓名生日性別key')
    .agg(
        成為業務前是否為保戶=('成為業務前是否為保戶', 'first'),
        簽約日=('簽約日 年/月/日', 'first'), 
        簽約前總保費=('繳款保費new', 'sum'),
        簽約前受理件數=('受理件數', 'sum')
    )
    .reset_index()
)


df4['生日 年/月/日'] = pd.to_datetime(df4['生日 年/月/日'], format='%Y%m%d', errors='coerce')  
df4['姓名生日性別key'] = df4['客戶姓名'].astype(str) + '_' + df4['生日 年/月/日'].astype(str) + '_' + df4['性別']

# 合併，summary 的姓名生日key 對應到 df4 的 客戶姓名生日key
df5 = df4.merge(
    summary,
    left_on='姓名生日性別key',
    right_on='姓名生日性別key',
    how='left'
)

df5['客戶簽約時年齡'] = (
    ((df5['簽約日'] - df5['生日 年/月/日']).dt.days // 365)
    .where(df5['簽約日'].notna())
)


# %% Plot
import seaborn as sns

# 設定風格
plt.style.use('default')
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False 

unique_agent = (
    agent2.drop_duplicates(subset='姓名生日性別key')
)

# 業務員背景 2020~2024
mask_agent = (unique_agent['簽約日 年/月/日'] >= '2020-01-01') & (unique_agent['簽約日 年/月/日'] <= '2024-12-31')
unique_agent_5y = unique_agent[mask_agent].copy()
unique_agent_5y['成為業務前是否為保戶'].value_counts()

# 計算簽約年資（以今日為基準）
unique_agent_5y['簽約年資'] = ((pd.Timestamp('today') - unique_agent_5y['簽約日 年/月/日']).dt.days / 365)

# 分群統計
group_summary_5y = (
    unique_agent_5y.groupby('成為業務前是否為保戶')
    .agg(
        業務員數=('業代', 'nunique'),
        平均簽約年齡=('簽約時年齡', 'mean'),
        中位數簽約年齡=('簽約時年齡', 'median'),
        最小簽約年齡=('簽約時年齡', 'min'),
        最大簽約年齡=('簽約時年齡', 'max'),
        平均簽約年資=('簽約年資', 'mean'),
        中位數簽約年資=('簽約年資', 'median')
    )
    .reset_index()
)

total = unique_agent_5y['業代'].nunique()
group_summary_5y['占比'] = group_summary_5y['業務員數'] / total

filter_test = policy3[
    policy3['姓名生日性別key'].str.startswith('何佳*_1966-02-26_女', na=False)
]
# ======================


# 分組繪圖
for group_value, group_data in unique_agent.groupby('成為業務前是否為保戶'):
    
    # 轉換群體標籤
    group_label = '原為保戶' if group_value == 1 else '非保戶'

    # 抓出有效年齡資料
    data = group_data['簽約時年齡'].dropna()

    plt.figure()
    sns.histplot(data, bins=20, kde=True)
    plt.title(f'{group_label} 簽約時年齡分布 (n={len(data)})')
    plt.xlabel('簽約時年齡')
    plt.ylabel('人數')
    plt.show()




# 有拜訪紀錄的
# 1. 簽約時年齡分布（排除缺漏值）
plt.figure()
sns.histplot(df5['客戶簽約時年齡'].dropna(), bins=20, kde=True)
plt.title(f'簽約時年齡分布 (n={len(data)})')
plt.xlabel('簽約時年齡')
plt.ylabel('人數')
plt.show()

# 2. 性別比例
plt.figure()
data = df5['性別'].dropna()
data.value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title(f'性別比例 (n={len(data)})')
plt.ylabel('')
plt.show()

# 3. 客戶類型分布
plt.figure()
df5['客戶類型'].value_counts().plot.bar()
plt.title('客戶類型分布')
plt.xlabel('客戶類型')
plt.ylabel('人數')
plt.show()

# 4. 總保費分布（去除極端值，上限設為保費95%分位數）
upper_limit = df5['總保費'].quantile(0.95)
data = df5[df5['總保費'] <= upper_limit]['總保費'].dropna()
plt.figure()
sns.histplot(data, bins=30, kde=True)
plt.title(f'總保費分布 (排除極端值) (n={len(data)})')
plt.xlabel('總保費')
plt.ylabel('人數')
plt.show()

# 5. 群體分布（業務員 vs 非業務員）
plt.figure()
data = df5['群體'].dropna()
data.value_counts().plot.bar()
plt.title(f'群體分布（是否為業務員） (n={len(data)})')
plt.xlabel('群體')
plt.ylabel('人數')
plt.show()



# describe_2 = agent2_filtered.describe().T

# # 重新統計僅針對 2020~2024 的新進業務
# summary = (
#     agent2_filtered.groupby('成為業務前是否為保戶')
#     .agg(
#         人數=('業代', 'nunique'),
#         平均簽約年齡=('簽約時年齡', 'mean'),
#         中位數簽約年齡=('簽約時年齡', 'median'),
#         平均簽約年限=('簽約日 年/月/日', lambda s: ((pd.Timestamp('today') - s).dt.days / 365).mean())
#     )
#     .reset_index()
# )

# # 計算占比
# total = agent2_filtered['業代'].nunique()
# summary['占比'] = summary['人數'] / total

# print(summary)



# 以業代為單位，整理業務員資訊
agent_summary = (
    agent2.groupby('業代')
    .agg(
        業務員=('業務員', 'first'),
        成為業務前是否為保戶=('成為業務前是否為保戶', 'first'),
        簽約時年齡=('簽約時年齡', 'first'),
        簽約日=('簽約日 年/月/日', 'first'), 
        性別=('性別', 'first'), 
        目前年齡=('目前年齡', 'first')
    )
    .reset_index()
)

# 計算簽約年資（以今日為基準）
agent_summary['簽約年資'] = ((pd.Timestamp('today') - agent_summary['簽約日']).dt.days / 365)


# 篩選近五年新進業務員
agent_summary_5y = agent_summary[
    (agent_summary['簽約日'] >= '2020-01-01') & (agent_summary['簽約日'] <= '2024-12-31')
]

# 分群統計
group_summary_5y = (
    agent_summary_5y.groupby('成為業務前是否為保戶')
    .agg(
        業務員數=('業代', 'nunique'),
        平均簽約年齡=('簽約時年齡', 'mean'),
        中位數簽約年齡=('簽約時年齡', 'median'),
        最小簽約年齡=('簽約時年齡', 'min'),
        最大簽約年齡=('簽約時年齡', 'max'),
        平均簽約年資=('簽約年資', 'mean'),
        中位數簽約年資=('簽約年資', 'median')
    )
    .reset_index()
)

group_summary_5y['身分說明'] = group_summary_5y['成為業務前是否為保戶'].map({1: '原為保戶', 0: '非保戶'})
group_summary_5y = group_summary_5y[['身分說明', '業務員數', '平均簽約年齡', '中位數簽約年齡', '最小簽約年齡', '最大簽約年齡', '平均簽約年資', '中位數簽約年資']]

print(group_summary_5y)

