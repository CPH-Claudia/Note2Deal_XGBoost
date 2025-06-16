# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:48:44 2025

@author: Z01788
"""

import pandas as pd

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
customer_basic = customer_filter[['客戶UUID', '客戶姓名', '性別', '生日 年/月/日', '客戶目前年齡']]

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

# 業務客戶匹配
# 1. 處理 agent 資料
df4['業務生日'] = pd.to_datetime(df4['業務生日'], format='%Y%m%d', errors='coerce')  # 將 19901020 -> datetime
df4['業務姓名生日key'] = df4['業務員'] + df4['業務生日'].dt.strftime('%Y-%m-%d')

# 2. 處理 customer 資料
df4['客戶生日'] = pd.to_datetime(df4['生日 年/月/日'], errors='coerce')
df4['客戶姓名生日key'] = df4['客戶姓名'] + df4['客戶生日'].dt.strftime('%Y-%m-%d')

# 建立業務姓名生日key集合（整體的 set，避免逐行比對）
agent_keys = set(df4['業務姓名生日key'].dropna())

# 判斷客戶是否成為業務：看客戶key是否出現在業務key集合中
df4['客戶是否為業務'] = df4['客戶姓名生日key'].isin(agent_keys).astype(int)


summary_1 = (
    df4.groupby('客戶是否為業務')
    .agg(
        人數=('業代', 'nunique'),
        平均簽約年齡=('簽約時年齡', 'mean'),
        中位數簽約年齡=('簽約時年齡', 'median'),
        平均簽約年限=('簽約日 年/月/日', lambda s: ((pd.Timestamp('today') - s).dt.days/365).mean())
    )
    .reset_index()
)

# 建立業務 key → 簽約日對照表（如一位業務有多筆，只取最早）
agent_sign_dates = (
    df4[['業務姓名生日key', '簽約日 年/月/日']]
    .dropna()
    .drop_duplicates('業務姓名生日key')  # 一個 key 一筆
    .set_index('業務姓名生日key')
)

# 將業務簽約日合併到 df4（根據客戶是否成為業務，將其 key 對應到業務 key 的簽約日）
df4['對應業務簽約日'] = df4['客戶姓名生日key'].map(agent_sign_dates['簽約日 年/月/日'])

# 判斷是否為有效拜訪：成為業務且拜訪日在簽約日前
df4['是否有效拜訪'] = (
    (df4['客戶是否為業務'] == 1) &
    (df4['對應業務簽約日'] >= df4['拜訪時間 年/月/日'])
).astype(int)

# 5. 計算年齡差與性別差
df4['客戶業務年齡差距'] = (
    (df4['客戶生日'] - df4['業務生日']).dt.days / 365
).round(1)

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



# 缺失值處理
df_drop = df4.dropna(subset=['客戶性別', '客戶目前年齡', '業務目前年齡'])

keep_cols = [
    '營業單位', '營業單位代碼', '業代', '客戶UUID', '拜訪紀錄UUID', '拜訪時間 年/月/日', '拜訪備註',
    '拜訪次數', '平均每客戶拜訪次數', '(增)拜訪目的', '方式', '業務員', '簽約日 年/月/日', '最新職級',
    '業務目前年齡', '業務性別', '業務生日', '目前年資', '晉升日', '距離晉升天數', '歷年新增業務數',
    '歷年準增數', '客戶姓名', '性別', '生日 年/月/日', '客戶類型', '受理件數', '繳款保費new',
    '客戶生日', '客戶是否為業務', '是否有效拜訪', '客戶業務年齡差距', '業務客戶性別組合'
]

df_cleaned = df_drop[keep_cols].copy()

df_cleaned[['距離晉升天數', '受理件數', '繳款保費new']] = df_cleaned[[
    '距離晉升天數', '受理件數', '繳款保費new']].fillna(0)
# df_drop[['對應業務簽約日', '合終日期 年/月/日', '晉升日期_dt', '拜訪備註', 
#      '客戶姓名', '客戶姓名生日key', ]] = df_drop[[
#          '對應業務簽約日', '合終日期 年/月/日', '晉升日期_dt', '拜訪備註', 
#           '客戶姓名', '客戶姓名生日key', '客戶類型']].fillna(pd.NaT)

missing_report = df_cleaned.isna().sum()
missing_report = missing_report[missing_report > 0].sort_values(ascending=False)


# 篩選 2024/1/1 ~ 2024/12/31 的筆數
df_filtered = df_cleaned[
    ((df_drop['拜訪時間 年/月/日'] >= pd.Timestamp('2024-01-01')) & (df_drop['拜訪時間 年/月/日'] <= pd.Timestamp('2024-12-31')))
]

df_filtered_1 = df_filtered[df_filtered['平均每客戶拜訪次數'] > 4]

invalid_visits = df4[
    (df4['客戶是否為業務'] == 1) &
    (df4['對應業務簽約日'] < df4['拜訪時間 年/月/日'])
]

# %% 斷詞
import pandas as pd
from ckiptagger import WS
import os, pickle
from opencc import OpenCC
import requests

from ckiptagger import data_utils
data_utils.download_data_gdown("./")

def extract_and_segment_notes(df, note_column='拜訪備註', ckip_model_path='./CKIP'):
    # 初始化 CKIP 斷詞工具
    ws = WS(ckip_model_path)

    # 自定保險術語字典（可擴充）
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

    # 1. 清洗備註文字
    def clean_text(note):
        if pd.isna(note): return ''
        lines = str(note).replace('_x000D_', '\n').replace('\r', '').splitlines()
        return '\n'.join([
            line.strip() for line in lines
            if line.strip() and not (line.startswith('#') or line.startswith('＃'))
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

df4 = extract_and_segment_notes(df4, note_column='拜訪備註', ckip_model_path='./CKIP')


# %% Model 



# %% 新進業務員過去是否已經是錠嵂保戶
agent2 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="agent2") 
# 確保兩欄都是 datetime 格式
agent2['簽約日 年/月/日'] = pd.to_datetime(agent2['簽約日 年/月/日'], errors='coerce')
agent2['生日'] = pd.to_datetime(agent2['生日'], format='%Y%m%d', errors='coerce')  

# 計算簽約當下年齡（以年為單位，向下取整）
agent2['簽約時年齡'] = ((agent2['簽約日 年/月/日'] - agent2['生日']).dt.days // 365).astype(int)

policy2 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="policy2") 
policy2['被保人生日 年/月/日'] = pd.to_datetime(policy2['被保人生日 年/月/日'], format='%Y%m%d', errors='coerce')

# 1. 建立比對 key
agent2['姓名生日key'] = agent2['業務員'].astype(str) + '_' + agent2['生日'].astype(str)
policy2['姓名生日key'] = policy2['被保人'].astype(str) + '_' + policy2['被保人生日 年/月/日'].astype(str)

# 2. 建立：客戶 -> 最早交易日的查詢表
policy_key_date = policy2.set_index('姓名生日key')['最小值 投保日'].to_dict()

# 3. 判斷每位業務在簽約前是否為客戶（flag 為 1 表示有，0 表示無）
def is_existing_customer(row):
    key = row['姓名生日key']
    trade_date = policy_key_date.get(key)
    if pd.notnull(trade_date) and trade_date < row['簽約日 年/月/日']:
        return 1
    return 0

agent2['成為業務前是否為保戶'] = agent2.apply(is_existing_customer, axis=1)

# 5. 統計
summary = (
    agent2.groupby('成為業務前是否為保戶')
    .agg(
        人數=('業代', 'nunique'),
        平均簽約年齡=('簽約時年齡', 'mean'),
        中位數簽約年齡=('簽約時年齡', 'median'),
        平均簽約年限=('簽約日 年/月/日', lambda s: ((pd.Timestamp('today') - s).dt.days/365).mean())
    )
    .reset_index()
)

# 6. 占比
total = agent2['業代'].nunique()
summary['占比'] = summary['人數'] / total

print(summary)
