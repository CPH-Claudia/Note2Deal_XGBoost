# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 17:07:53 2025

@author: Z01788
"""

import pandas as pd


# 篩選簽約日在 2020/1/1 ~ 2024/12/31 之間的新進業務
agent2 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="agent2") 
# 確保兩欄都是 datetime 格式
agent2['簽約日 年/月/日'] = pd.to_datetime(agent2['簽約日 年/月/日'], errors='coerce')
agent2['生日'] = pd.to_datetime(agent2['生日'], format='%Y%m%d', errors='coerce')  
agent2['性別'] = agent2['性別'].astype(str)

# 計算簽約當下年齡（以年為單位，向下取整）
agent2['簽約時年齡'] = ((agent2['簽約日 年/月/日'] - agent2['生日']).dt.days // 365).astype(int)
agent2['姓名生日性別key'] = agent2['業務員'].astype(str) + '_' + agent2['生日'].astype(str) + '_' + agent2['性別']


policy3 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="policy_agent", dtype={"被保人業代": str}) 
policy3['被保人生日 年/月/日'] = pd.to_datetime(policy3['被保人生日 年/月/日'], format='%Y%m%d', errors='coerce')  
policy3['投保日 年/月/日'] = pd.to_datetime(policy3['投保日 年/月/日'], format='%Y年%m月%d日', errors='coerce')
policy3['姓名生日性別key'] = policy3['被保人'].astype(str) + '_' + policy3['被保人生日 年/月/日'].astype(str) + '_' + policy3['被保人性別']

# 2. 建立：客戶 -> 最早交易日的查詢表
policy_key_date = (
    policy3.sort_values('投保日 年/月/日')  # 先排序
    .drop_duplicates('被保人業代', keep='first')  # 保留最早投保日
    .set_index('被保人業代')['投保日 年/月/日']
    .to_dict()
)

# 3. 判斷是否為保戶
def is_existing_customer(row):
    key = row['業代']
    
    # 先判斷是否有這個 key
    if key not in policy_key_date:
        return 0
    
    trade_date = policy_key_date.get(key)
    
    # 確保兩邊都是 datetime 再比較
    if pd.notnull(trade_date) and pd.notnull(row['簽約日 年/月/日']):
        if trade_date < row['簽約日 年/月/日']:
            return 1
    return 0

# 4. 套用判斷
agent2['成為業務前是否為保戶'] = agent2.apply(is_existing_customer, axis=1)

# 定義職級映射
agent_stage = {'CB': 0, 'JB': 1, 'PB': 2, 'SB': 3}
# 將職級轉為數值方便比較
agent2['職級_num'] = agent2['月結檔 | 職級'].map(agent_stage)

# 確保計績年月格式正確
agent2['計績年月'] = agent2['計績年月'].astype(int)

# 針對每位業務員，找到簽約當下的資料
sign_rank = (
    agent2.sort_values(['業代', '計績年月'])
    .drop_duplicates(['業代', '簽約日 年/月/日'], keep='first')
    [['業代', '職級_num']]
    .rename(columns={'職級_num': '簽約時職級'})
)

# 針對每位業務員，找到最新資料
latest_rank = (
    agent2.sort_values(['業代', '計績年月'])
    .groupby('業代')
    .tail(1)
    [['業代', '職級_num']]
    .rename(columns={'職級_num': '最新職級'})
)

# 排序方便計算
agent2_sorted = agent2.sort_values(['業代', '計績年月'])

# 計算每位業務員職級變化
agent2_sorted['職級變化'] = agent2_sorted.groupby('業代')['職級_num'].diff()

# 升遷次數
promotion = (
    agent2_sorted.groupby('業代')['職級變化']
    .apply(lambda x: (x > 0).sum())
    .reset_index()
    .rename(columns={'職級變化': '升遷次數'})
)

# 降職次數
demotion = (
    agent2_sorted.groupby('業代')['職級變化']
    .apply(lambda x: (x < 0).sum())
    .reset_index()
    .rename(columns={'職級變化': '降職次數'})
)

# 保留每位業務員的靜態資訊
agent_static = agent2.drop_duplicates('業代')[[
    '業代', '營業單位', '營業單位代碼', '業務員', '性別', '生日', '目前年齡', 
    '簽約日 年/月/日', '合終日期 年/月/日', '年資', '姓名生日性別key', '成為業務前是否為保戶'
]]

# 合併結果
final_df = (
    agent_static
    .merge(sign_rank, on='業代', how='left')
    .merge(latest_rank, on='業代', how='left')
    .merge(promotion, on='業代', how='left')
    .merge(demotion, on='業代', how='left')
)

# policy
# 保單資料合併簽約日
policy_merged = (
    policy3.merge(
        final_df[['業代', '姓名生日性別key', '簽約日 年/月/日']],
        left_on='被保人業代',
        right_on='業代', 
        how='left'
    )
    .rename(columns={'姓名生日性別key_y': '姓名生日性別key'})
    .drop(columns=['姓名生日性別key_x'])
)


# 累積件數、累積保費、第一次投保日、最近一次投保日
total_summary = (
    policy_merged
    .groupby('被保人業代')
    .agg(
        累積件數=('保單申請案號', 'nunique'),
        累積保費=('繳款保費new', 'sum'),
        累積fyc=('計績FYC', 'sum'),
        第一次投保日=('投保日 年/月/日', 'min'), 
        最近一次投保日=('投保日 年/月/日', 'max')
    )
)


# 簽約後資料
# 1. 標記是否為簽約日後保單
policy_merged['是否簽約後保單'] = policy_merged['投保日 年/月/日'] > policy_merged['簽約日 年/月/日']

# 2. 簽約後保單：加總件數、保費、第一次與最後一次投保日
after_summary = (
    policy_merged[policy_merged['是否簽約後保單']]
    .groupby('被保人業代')
    .agg(
        簽約後累積件數=('保單申請案號', 'nunique'),
        簽約後累積保費=('繳款保費new', 'sum'),
        簽約後累積fyc=('計績FYC', 'sum'), 
        簽約後第一次投保日=('投保日 年/月/日', 'min'),
        簽約後最近一次投保日=('投保日 年/月/日', 'max')
    )
)

# 3. 簽約後保單：商品類別 + 主附約別 橫向拆開
after_product_pivot = (
    policy_merged[policy_merged['是否簽約後保單']]
    .pivot_table(index='被保人業代', columns=['商品險種主類別', '主附約別'], aggfunc='size', fill_value=0)
)
after_product_pivot.columns = [f'簽約後商品類別_{col[0]}_{col[1]}_保單數' for col in after_product_pivot.columns]
after_product_pivot.reset_index(inplace=True)

# filter_test = policy_merged[
#     policy_merged['姓名生日性別key'].str.startswith('白睿*_1998-07-17_男', na=False)
# ]

# 簽約前資料
policy_merged['是否簽約前保單'] = policy_merged['投保日 年/月/日'] < policy_merged['簽約日 年/月/日']

before_summary = (
    policy_merged[policy_merged['是否簽約前保單']]
    .groupby('被保人業代')
    .agg(
        簽約前累積件數=('保單申請案號', 'nunique'),
        簽約前累積保費=('繳款保費new', 'sum'), 
        簽約前累積fyc=('計績FYC', 'sum')
    )
)

before_product_pivot = (
    policy_merged[policy_merged['是否簽約前保單']]
    .pivot_table(index='被保人業代', columns=['商品險種主類別', '主附約別'], aggfunc='size', fill_value=0)
)
before_product_pivot.columns = [f'簽約前商品類別_{col[0]}_{col[1]}' for col in before_product_pivot.columns]



# 主類別橫向拆開
# 透過 pivot_table 拆開 主類別 & 主附約別
product_pivot = (
    policy_merged
    .pivot_table(index='被保人業代', columns=['商品險種主類別', '主附約別'], aggfunc='size', fill_value=0)
)

# 重命名欄位
product_pivot.columns = [
    f'{col[0]}_{col[1]}_保單數' for col in product_pivot.columns
]

product_pivot.reset_index(inplace=True)



final_df = (
    final_df
    .merge(total_summary, left_on='業代', right_on='被保人業代', how='left')
    .merge(after_summary, left_on='業代', right_on='被保人業代', how='left')
    # .merge(after_product_pivot, left_on='業代', right_on='被保人業代', how='left')
    .merge(before_summary, left_on='業代', right_on='被保人業代', how='left')
    # .merge(before_product_pivot, left_on='業代', right_on='被保人業代', how='left')
    # .merge(product_pivot, left_on='業代', right_on='被保人業代', how='left')
)

final_df = (final_df
    .rename(columns={'被保人業代_y': '被保人業代'})
    .drop(columns=['被保人業代_x']))

final_df.to_csv("D:/增員/業務簽約前後狀況.csv", index=False)

# una = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="工作表1") 
# una['簽約前是否為保戶_數值'] = una['簽約前是否為保戶'].map({'是': 1, '否': 0})
# # 合併兩資料，僅保留必要欄位
# temp = una.merge(
#     final_df[['業代', '成為業務前是否為保戶']], 
#     on='業代', 
#     how='left'
# )
# # 篩選出兩者結果不同的業代
# 不一致業代 = temp[temp['簽約前是否為保戶_數值'] != temp['成為業務前是否為保戶']]




# 分析報告
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False 

# === 1. 基礎分組 ===
final_df['是否為保戶分類'] = final_df['成為業務前是否為保戶'].fillna(0).astype(int)
final_df['是否為保戶分類_str'] = final_df['是否為保戶分類'].map({0: '非保戶', 1: '保戶'})

# === 2. 整體保戶占比圓餅圖 ===
plt.figure(figsize=(5, 5))
final_df['是否為保戶分類_str'].value_counts().plot.pie(
    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(final_df))}人)',
    startangle=90, counterclock=False, wedgeprops={'edgecolor': 'white'}
)
plt.title('簽約前是否為保戶分布')
plt.ylabel('')
plt.show()

# === 3. 保戶簽約前購買狀況 ===
customer = final_df[final_df['是否為保戶分類'] == 1]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(customer['簽約前累積保費'], bins=20, kde=True)
plt.title('保戶 - 簽約前累積保費分布')

plt.subplot(1, 2, 2)
sns.histplot(customer['簽約前累積件數'], bins=20, kde=True)
plt.title('保戶 - 簽約前累積件數分布')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=customer, x='簽約前累積保費', showfliers=False)
plt.title('保戶 - 簽約前累積保費箱型圖（視覺排除離群值）')
plt.xlabel('簽約前累積保費')
plt.show()


# 商品類別購買偏好
product_cols = [col for col in final_df.columns if col.startswith('簽約前商品類別_')]
# 整理資料，移除前綴
product_summary = (
    final_df.loc[final_df['是否為保戶分類'] == 1, product_cols]
    .sum()
    .rename(lambda x: x.replace('簽約前商品類別_', ''))  # 只留下類別名稱
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 4))
sns.barplot(x=product_summary.values, y=[f'{i}' for i in product_summary.index])
plt.title('保戶 - 簽約前商品類別購買數量')
plt.xlabel('購買數量')
plt.show()

# 性別、年齡、營業單位分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=customer, x='性別', ax=axes[0])
axes[0].set_title('保戶 - 性別分布')
sns.histplot(customer['目前年齡'], bins=20, kde=True, ax=axes[1])
axes[1].set_title('保戶 - 年齡分布')
sns.countplot(data=customer, y='營業單位', order=customer['營業單位'].value_counts().index, ax=axes[2])
axes[2].set_title('保戶 - 營業單位分布')
plt.tight_layout()
plt.show()

# === 4. 保戶 vs 非保戶 簽約後累積保費、件數比較 ===
col_cf = ['累積保費', '累積件數']

for col in col_cf:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=final_df, x='是否為保戶分類_str', y=col)
    plt.title(f'{col} (簽約前是否為保戶分組)')
    plt.show()

# === 5. 簽約前購買年齡 vs 簽約時年齡差異 (保戶限定) ===
customer['購買年齡'] = ((customer['簽約前第一次投保日'] - customer['生日']).dt.days // 365).astype('Int64')
customer['簽約時年齡'] = ((customer['簽約日 年/月/日'] - customer['生日']).dt.days // 365).astype('Int64')

plt.figure(figsize=(6, 4))
sns.histplot(customer['簽約時年齡'] - customer['購買年齡'], bins=20, kde=True)
plt.title('保戶 - 簽約時年齡與首次購買年齡差異分布')
plt.xlabel('年齡差 (歲)')
plt.show()








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