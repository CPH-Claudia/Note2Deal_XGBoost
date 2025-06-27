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

policy3 = pd.read_excel("D:/增員/tableau_增員.xlsx", sheet_name="policy_agent") 
policy3['被保人生日 年/月/日'] = pd.to_datetime(policy3['被保人生日 年/月/日'], format='%Y%m%d', errors='coerce')  
policy3['投保日 年/月/日'] = pd.to_datetime(policy3['投保日 年/月/日'], errors='coerce')

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