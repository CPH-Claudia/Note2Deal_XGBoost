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


# 篩選 2024/2/1 ~ 2024/6/30 和 2024/8/1 ~ 2024/12/31 的筆數
df_filtered = df_cleaned[
    ((df_drop['拜訪時間 年/月/日'] >= pd.Timestamp('2024-02-01')) & (df_drop['拜訪時間 年/月/日'] <= pd.Timestamp('2024-06-30'))) |
    ((df_drop['拜訪時間 年/月/日'] >= pd.Timestamp('2024-08-01')) & (df_drop['拜訪時間 年/月/日'] <= pd.Timestamp('2024-12-31')))
]

df_filtered_1 = df_filtered[df_filtered['平均每客戶拜訪次數'] > 4]



