# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:21:54 2025

@author: Z01788
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import f_oneway
import os
from datetime import datetime

# 讀取客戶過去投保資料
purchase = pd.read_excel("D:/客戶名單/全部客戶過去投保資料.xlsx", dtype={"保單申請案號": str}, sheet_name=0) 
purchase = purchase[~purchase['繳別'].str.contains("躉繳", na=False)]
purchase['投保日'] = pd.to_datetime(purchase['投保日'])
purchase = purchase.sort_values(by=['經紀人1-被保人crm Uuid', '投保日'])
split_data = purchase['商品名稱'].str.split(r'險', n=1, expand=True)
purchase.loc[:, '商品名稱_主體'] = split_data[0] + '險'
purchase.loc[:, '商品名稱_括號'] = split_data[1]

# %% 篩選條件：近5年有投保過，但2024年沒投保過的客戶
# 找出 2019~2023 年投保的客戶                     395498
past_customers = purchase.loc[purchase["投保日"].dt.year.between(2019, 2023), "經紀人1-被保人crm Uuid"]
# 找出 2024 年投保的客戶                          73593
current_customers = purchase.loc[purchase["投保日"].dt.year == 2024, "經紀人1-被保人crm Uuid"]
# 找出 2024 年未投保但 2019~2023 年有投保的客戶   223295
target_customers = set(past_customers) - set(current_customers)
# 篩選符合條件的客戶資料  #429802
df_target = purchase[purchase["經紀人1-被保人crm Uuid"].isin(target_customers)]


# 讀取險種關鍵字資料
allCompanyInsItem = pd.read_excel("D:/allCompanyInsItems-商品標籤.xlsx", sheet_name=0)
allCompanyInsItem = allCompanyInsItem[allCompanyInsItem['主約'] == 1]
allCompanyInsItem = allCompanyInsItem[['險種中文名稱','險種外部代碼(顯示用代碼)','險種關鍵字','投保單位','停售']]
allCompanyInsItem['險種中文名稱'] = allCompanyInsItem['險種中文名稱'].str.replace(r'[a-z]+', lambda x: x.group().upper(), regex=True)
allCompanyInsItem[['險種中文名稱_主體', '險種中文名稱_其餘']] = allCompanyInsItem['險種中文名稱'].str.split(r'[（()\-\s－]', n=1, expand=True)
allCompanyInsItem['險種中文名稱_主體'] = allCompanyInsItem['險種中文名稱_主體'].fillna(allCompanyInsItem['險種中文名稱'])
allCompanyInsItem.rename(columns={'險種中文名稱_主體': '商品名稱_主體'}, inplace=True)

# 建立推薦商品名稱的對應關係
allCompanyInsItem1 = allCompanyInsItem.copy()
filtered_df = allCompanyInsItem.groupby('商品名稱_主體', as_index=False).last()
allCompanyInsItem['險種關鍵字'] = allCompanyInsItem['險種關鍵字'].apply(lambda x: str(x).split(';') if pd.notnull(x) else [])

combine_data = pd.merge(df_target, filtered_df[['商品名稱_主體','險種中文名稱','險種關鍵字','停售']],on='商品名稱_主體',how='left')
combine_data = combine_data.dropna(subset=['險種中文名稱'])
combine_data['停售'] = combine_data['停售'].fillna(0)
# combine_data = combine_data[combine_data['停售'] == 0]
combine_data['年期'] = combine_data['商品名稱_括號'].str.extract(r'\(([^)]+)\)')
combine_data['年期'] = combine_data['年期'].apply(lambda x: x if x.isdigit() else '0').astype(int)
combine_data = combine_data[combine_data['險種關鍵字'].notna() & (combine_data['險種關鍵字'] != '')]
combine_data['險種關鍵字'] = combine_data['險種關鍵字'].str.split(";")



# %% 定義函數檢查關鍵字匹配
def contains_all_keywords(target_keywords, product_keywords):
    if not isinstance(product_keywords, list):
        return False
    return set(target_keywords).issubset(set(product_keywords))

# 設定篩選條件的關鍵字
# 鑫旺達
target_keywords = ['分紅', '終身保障', '壽險']  # 可根據需求更改關鍵字

# 投資型
target_keywords = ['投資型商品']  # 可根據需求更改關鍵字

# 篩選符合條件的客戶
combine_data['匹配'] = combine_data['險種關鍵字'].apply(lambda x: contains_all_keywords(target_keywords, x))
filtered_customers = combine_data[combine_data['匹配']]


# phbn
# must_have_keywords = {'不分紅'} # , '終身保障'
optional_keywords = {'老年醫療', '定額醫療', '意外醫療', '實支實付醫療', '醫療保障', '醫療日額', '醫療保障/健康保險'} # , '終身醫療'
excluded_keywords = {'終身保障', '終身醫療'}  # 這些關鍵字 **不能出現**

def keyword_filter(keywords):
    if not isinstance(keywords, list):
        return False

    keywords_set = set(keywords)

    # # 必須包含 `must_have_keywords`
    # if not must_have_keywords.issubset(keywords_set):
    #     return False

    # 必須包含 `optional_keywords` 中的至少一個
    if not keywords_set & optional_keywords:
        return False

    # 不能包含 `excluded_keywords` 中的任何一個
    if keywords_set & excluded_keywords:
        return False

    return True

# 過濾符合關鍵字條件的客戶
combine_data['匹配'] = combine_data['險種關鍵字'].apply(keyword_filter)
filtered_customers = combine_data[combine_data['匹配']]



def process_data(df):
    def merge_keywords(x):
        # 先將 "商品名稱" 和 "險種關鍵字" 進行配對
        unique_products = {}  # 用來儲存商品名稱對應的險種關鍵字
        for product, keywords in zip(x['商品名稱'], x['險種關鍵字']):
            if product not in unique_products:
                unique_products[product] = keywords  # 若商品名稱第一次出現，儲存其險種關鍵字
            else:
                continue  # 若商品已存在，不重複存入

        # 合併 "商品名稱"（逗號分隔）與 "險種關鍵字"（用 ";" 分隔不同商品）
        merged_keywords = ';'.join([','.join(k) for k in unique_products.values()])
        merged_products = ','.join(unique_products.keys())

        return pd.Series([merged_products, merged_keywords])

    # 進行 groupby，合併相同經紀人1-被保人crm Uuid
    filter_id = df.groupby('經紀人1-被保人crm Uuid').apply(lambda x: pd.Series({
        '經紀人營業單位': x['經紀人營業單位'].iloc[0],
        '經紀人 (執行)處經理業代': x['經紀人 (執行)處經理業代'].iloc[0],
        '經紀人 (執行)處經理': x['經紀人 (執行)處經理'].iloc[0],
        '經紀人 A級sb業代': x['經紀人 A級sb業代'].iloc[0],
        '經紀人 A級sb': x['經紀人 A級sb'].iloc[0],
        '經紀人業代': x['經紀人業代'].iloc[0],
        '經紀人姓名': x['經紀人姓名'].iloc[0],
        '被保人': x['被保人'].iloc[0],
        '被保人目前年齡': x['被保人目前年齡'].iloc[-1],
        '被保人性別': x['被保人性別'].iloc[0],
        '被保人生日': x['被保人生日'].iloc[0],
        '要保人姓名': x['要保人姓名'].iloc[0],
        '要保人生日': x['要保人生日'].iloc[0],
        '保單申請案號': len(x),
        '保額new': x['保額new'].sum(),
        '繳款保費new': x['繳款保費new'].sum(),
        '投保日': x['投保日'].iloc[-1],
        **dict(zip(['歷年相關商品名稱', '險種關鍵字'], merge_keywords(x)))  # 呼叫 `merge_keywords()`
    })).reset_index()

    # 重命名欄位
    filter_id.rename(columns={
        '保額new': '總保額',
        '繳款保費new': '總保費',
        '保單申請案號': '總件數',
        '投保日': '最後一次投保日'
    }, inplace=True)

    column_order = list(filter_id.columns)
    column_order.insert(column_order.index('經紀人姓名'), column_order.pop(column_order.index('經紀人1-被保人crm Uuid')))
    filter_id = filter_id[column_order]
    
    # 計算最後投保日距離今天的年數和月數
    def calculate_years_months(x):
        if pd.isnull(x):
            return None
        today = datetime.today()
        years = today.year - x.year
        months = today.month - x.month
        if months < 0:
            years -= 1
            months += 12
        return f"{years}年{months}月"

    filter_id['最後一次投保距今年月'] = filter_id['最後一次投保日'].apply(calculate_years_months)
    column_order.append(column_order.pop(column_order.index('險種關鍵字')))
    filter_id = filter_id[column_order]
    return filter_id

filtered_customers = process_data(filtered_customers)

# selected_customers = {
#     "1104547", "1085329", "853942",
#     "1702980", "1708803", "884222",
#     "864526", "884488", "1205901"
# }

# # 過濾符合的客戶
# filtered_selected_customers = filtered_customers[filtered_customers['經紀人1-被保人crm Uuid'].astype(str).isin(selected_customers)]

# with pd.ExcelWriter("D:/特定客戶.xlsx", engine='xlsxwriter') as writer:
#     filtered_selected_customers.to_excel(writer, sheet_name="特定客戶", index=False)


# 計算保額加總平均數
保額平均 = filtered_customers['總保額'].mean()
# 保額中位數 = filtered_customers['保額new'].median()

# 選出低於保額加總平均的客戶
filtered_customers_1 = filtered_customers[filtered_customers['總保額'] < 保額平均]

# %% 輸出全公司
agent_customer_count_total = filtered_customers_1.groupby(['經紀人營業單位', # '經紀人 (執行)處經理業代', '經紀人 (執行)處經理',
                                           '經紀人 A級sb業代', '經紀人 A級sb']).agg(
    不重複業務數=('經紀人業代', 'nunique'), 
    客戶數=('經紀人1-被保人crm Uuid', 'count'),
    # 保額加總=('保額new', 'sum'),
    # 保額平均=('保額new', 'mean'),
    ).reset_index()

# # 依據營業單位排序
# agent_customer_count_total = agent_customer_count_total.sort_values(by=['營業單位'])

agent_customer_count_total_unit = filtered_customers.groupby(['經紀人營業單位']).agg(
    不重複業務數=('經紀人業代', 'nunique'), 
    客戶數=('經紀人1-被保人crm Uuid', 'count'),
    保額平均=('總保額', 'mean'),
    ).reset_index()


summary_row = {
    '經紀人營業單位': '總計',
    '不重複業務數': agent_customer_count_total_unit['不重複業務數'].sum(),
    '客戶數': agent_customer_count_total_unit['客戶數'].sum(),
    '保額平均': agent_customer_count_total_unit['保額平均'].mean()  # 這裡用平均值，你也可以改成 sum()
}

# 新增總計行到 DataFrame
agent_customer_count_total_unit = pd.concat([agent_customer_count_total_unit, pd.DataFrame([summary_row])], ignore_index=True)


total_output_path = 'D:/客戶名單/全公司_鑫旺達.xlsx'
with pd.ExcelWriter(total_output_path) as writer:
    filtered_customers_1.to_excel(writer, sheet_name='低於平均保額客戶名單', index=False)
    agent_customer_count_total.to_excel(writer, sheet_name='統計', index=False)
    agent_customer_count_total_unit.to_excel(writer, sheet_name='單位統計', index=False)
    

# %% 輸出特定 6 個單位
# selected_units = {'北十', '中四', '北七', '中十七', '北六', '中十三'}
selected_units = set(filtered_customers_1['經紀人營業單位'].unique())

# 分組並輸出不同的 Excel 檔案
grouped = filtered_customers_1.groupby('經紀人營業單位')

# 檔案名稱
output_dir = "D:/客戶名單/"
os.makedirs(output_dir, exist_ok=True)
total_output_path = os.path.join(output_dir, '全單位_鑫旺達.xlsx')

# 初始化要合併的 DataFrame
all_filtered_data = []
all_summary_unit = []
all_unit_summary_unit = []

for unit, data in grouped:
    if unit in selected_units:
        # 針對特定單位篩選條件
        # if unit == "中十三":
        #     data = data[(data['被保人目前年齡'] <= 45) & (data['被保人性別'] == "女")]  # 只保留 45 歲以下女性客戶
        # else:
        #     data = data[data['被保人目前年齡'] <= 65]  # 其他單位都只保留 65 歲以下客戶

        # 工作表1: 過濾掉特定欄位
        filtered_data = data.drop(columns=['經紀人 (執行)處經理業代', '經紀人 (執行)處經理']) # , '經紀人營業單位'
        all_filtered_data.append(filtered_data)
        
        # 工作表2: 彙總經紀人業代的不重覆客戶數
        summary_unit = data.groupby(['經紀人營業單位', '經紀人 A級sb業代', '經紀人 A級sb', '經紀人業代']
                                    )['經紀人1-被保人crm Uuid'].nunique().reset_index()
        summary_unit.rename(columns={'經紀人1-被保人crm Uuid': '不重覆客戶數'}, inplace=True)
        
        all_summary_unit.append(summary_unit)
        
        # 工作表3: 各單位區經理客戶統計
        unit_summary_unit = data.groupby([
            '經紀人營業單位', '經紀人 A級sb業代', '經紀人 A級sb'
        ]).agg(
            業務數=('經紀人業代', 'nunique'),
            客戶數=('經紀人1-被保人crm Uuid', 'nunique'), 
            保額平均=('總保額', 'mean')
        ).reset_index()
            
        # 計算 "平均客戶數"（客戶數 / 業務數）
        unit_summary_unit['平均客戶數'] = unit_summary_unit['客戶數'] / unit_summary_unit['業務數']
        
        # 計算全公司總計行
        total_summary = {
            '經紀人營業單位': '總計',
            '業務數': unit_summary_unit['業務數'].sum(),
            '客戶數': unit_summary_unit['客戶數'].sum(),
            '保額平均': unit_summary_unit['保額平均'].mean(),
            '平均客戶數': unit_summary_unit['客戶數'].sum() / unit_summary_unit['業務數'].sum()
        }
        all_unit_summary_unit.append(unit_summary_unit)

# 合併所有單位的資料
final_filtered_data = pd.concat(all_filtered_data, ignore_index=True)
final_summary_unit = pd.concat(all_summary_unit, ignore_index=True)
final_unit_summary_unit = pd.concat(all_unit_summary_unit, ignore_index=True)
# 修正：只加總一次，不分單位
total_summary = {
    '經紀人營業單位': '總計',
    '業務數': final_unit_summary_unit['業務數'].sum(),
    '客戶數': final_unit_summary_unit['客戶數'].sum(),
    '保額平均': final_unit_summary_unit['保額平均'].mean(),
    '平均客戶數': final_unit_summary_unit['客戶數'].sum() / final_unit_summary_unit['業務數'].sum()
}

# 移除先前的「總計」行，確保只有一個整體總計
final_unit_summary_unit = final_unit_summary_unit[final_unit_summary_unit['經紀人營業單位'] != '總計']
final_unit_summary_unit = pd.concat([final_unit_summary_unit, pd.DataFrame([total_summary])], ignore_index=True)

# 儲存到 Excel
with pd.ExcelWriter(total_output_path, engine='xlsxwriter') as writer:
    final_filtered_data.to_excel(writer, sheet_name="客戶名單", index=False)
    final_summary_unit.to_excel(writer, sheet_name="每個業務的不重覆客戶數", index=False)
    final_unit_summary_unit.to_excel(writer, sheet_name="各單位區經理客戶統計", index=False)



# %% **第二部分**：篩選特定區經理的客戶
selected_managers = {
    "012874", "H08756", "F02879", "F06325", "014515", "015552", "014007",
    "B01719", "A05307", "F05518", "E03304", "E04223", "P08915", "P10470",
    "W09605", "012494", "A07482", "013525", "018738", "015396", "C10149"
}

# 過濾符合區經理名單的客戶
filtered_managers = filtered_customers_1[filtered_customers_1['經紀人 A級sb業代'].isin(selected_managers)]
filtered_managers = filtered_managers.drop(columns=['經紀人 (執行)處經理業代', '經紀人 (執行)處經理'])

all_summary_unit = []
all_unit_summary_unit = []

# 工作表2: 彙總經紀人業代的不重覆客戶數，依營業單位合併
summary_unit = filtered_managers.groupby(['經紀人營業單位', '經紀人 A級sb業代', '經紀人 A級sb', '經紀人業代']
                                         )['經紀人1-被保人crm Uuid'].nunique().reset_index()
summary_unit.rename(columns={'經紀人1-被保人crm Uuid': '不重覆客戶數'}, inplace=True)
all_summary_unit.append(summary_unit)

# 工作表3: 各單位區經理客戶統計，依營業單位合併
unit_summary_unit = filtered_managers.groupby([
    '經紀人營業單位', '經紀人 A級sb業代', '經紀人 A級sb'
]).agg(
    業務數=('經紀人業代', 'nunique'),
    客戶數=('經紀人1-被保人crm Uuid', 'nunique'), 
    保額平均=('總保額', 'mean')
).reset_index()
all_unit_summary_unit.append(unit_summary_unit)

# **合併為單一 DataFrame**
final_summary_unit = pd.concat(all_summary_unit, ignore_index=True)
final_unit_summary_unit = pd.concat(all_unit_summary_unit, ignore_index=True)

# **輸出 Excel 檔案**
output_dir = "D:/客戶名單/"
os.makedirs(output_dir, exist_ok=True)

with pd.ExcelWriter(os.path.join(output_dir, "區經理_投資型.xlsx"), engine='xlsxwriter') as writer:
    filtered_managers.to_excel(writer, sheet_name="保額低於平均", index=False)
    final_summary_unit.to_excel(writer, sheet_name="特定區經理業務客戶數", index=False)
    final_unit_summary_unit.to_excel(writer, sheet_name="特定區經理客戶統計", index=False)
