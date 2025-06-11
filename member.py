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
    '業代', '業務員', '簽約日 年/月/日', '合終日期 年/月/日', '職級_代碼', '目前年齡', '性別', '年資']]

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
customer = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="customer") 
customer['生日 年/月/日'] = pd.to_datetime(customer['生日 年/月/日'])

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

customer['分類'] = customer['客戶類型'].apply(classify_customer_type)

# 計算每個業代在上半年新增的準客戶數與新增保戶數（以客戶UUID唯一計算）
customer_stats = (
    customer
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




# 篩選 2024/2/1 ~ 2024/6/30 和 2024/8/1 ~ 2024/12/31 的筆數
visit_filtered = visit[
    ((visit['拜訪時間 年/月/日'] >= pd.Timestamp('2024-02-01')) & (visit['拜訪時間 年/月/日'] <= pd.Timestamp('2024-06-30'))) |
    ((visit['拜訪時間 年/月/日'] >= pd.Timestamp('2024-08-01')) & (visit['拜訪時間 年/月/日'] <= pd.Timestamp('2024-12-31')))
]