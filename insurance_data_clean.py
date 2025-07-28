# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:31:14 2025

@author: Z01788
"""

# === 📂 7份 Excel 工作表的資料清理流程 ===

import pandas as pd
import re
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# === 1. VISIT（拜訪資料）===
# def clean_visit(file_path):
#     visit = pd.read_excel(file_path, sheet_name="VISIT")
#     visit['拜訪時間 年/月/日'] = pd.to_datetime(visit['拜訪時間 年/月/日'], errors='coerce')

#     def extract_non_sharp_text(note):
#         if pd.isna(note): return ''
#         lines = str(note).replace('_x000D_', '\n').replace('\r', '').splitlines()
#         return '\n'.join([
#             line.strip() for line in lines
#             if line.strip() and not (line.startswith('#') or line.startswith('＃'))
#         ])

#     visit['拜訪備註_文字'] = visit['拜訪備註'].apply(extract_non_sharp_text)
#     visit['備註字數'] = visit['拜訪備註_文字'].apply(lambda x: len(str(x).replace(" ", "").replace("\n", "")))
    
#     # 計算每位業代對每個客戶的拜訪次數
#     visit['拜訪次數'] = (
#         visit.groupby(['業代', '客戶UUID'])['拜訪紀錄UUID']
#         .transform('count')
#     )

#     # 計算每位業代對所有客戶的平均拜訪次數
#     visit['平均每客戶拜訪次數'] = (
#         visit.groupby('業代')['拜訪次數']
#         .transform('mean')
#     )

#     return visit



def clean_visit(file_path):
    visit = pd.read_excel(file_path, sheet_name="VISIT")
    visit['拜訪時間 年/月/日'] = pd.to_datetime(visit['拜訪時間 年/月/日'], errors='coerce')

    def clean_hashtag(raw_tag):
        """清理標籤內容：去除頭尾雜訊、轉小寫，不保留#"""
        tag = re.sub(r'^[^a-zA-Z0-9\u4e00-\u9fa5]+', '', raw_tag)
        tag = re.sub(r'[^\w\u4e00-\u9fa5]+$', '', tag)
        return tag.lower()

    def extract_hashtags(note):
        """從備註中擷取所有標準化後的標籤（不含#）"""
        if pd.isna(note):
            return []
        text = str(note).replace('＃', '#').replace('_x000D_', '')
        text = re.sub(r'#{2,}', '#', text)
        raw_tags = re.findall(r'#([^\s#]+)', text)  # 注意：只取出 #後的內容，不含#
        return [clean_hashtag(tag) for tag in raw_tags]

    def extract_non_sharp_text(note):
        """去除開頭是 # 的行，保留其他純文字"""
        if pd.isna(note): return ''
        lines = str(note).replace('_x000D_', '\n').replace('\r', '').splitlines()
        return '\n'.join([
            line.strip() for line in lines
            if line.strip() and not (line.startswith('#') or line.startswith('＃'))
        ])

    # 擷取非 # 行文字
    visit['拜訪備註_文字'] = visit['拜訪備註'].apply(extract_non_sharp_text)
    
    # 擷取標籤文字（清洗後、不含#）
    visit['拜訪備註_標籤'] = visit['拜訪備註'].apply(extract_hashtags)
    visit['拜訪備註_標籤'] = visit['拜訪備註_標籤'].apply(lambda tags: ' '.join(tags))
    
    # 字數計算
    visit['備註字數'] = visit['拜訪備註_文字'].apply(lambda x: len(str(x).replace(" ", "").replace("\n", "")))

    # 拜訪次數與平均
    visit['拜訪次數'] = visit.groupby(['業代', '客戶UUID'])['拜訪紀錄UUID'].transform('count')
    visit['平均每客戶拜訪次數'] = visit.groupby('業代')['拜訪次數'].transform('mean')

    return visit



# === 2. TAGS（拜訪標籤）===
def clean_tags(file_path):
    tags = pd.read_excel(file_path, sheet_name="TAGS")
    tags['標籤名稱'] = tags['標籤名稱'].fillna('').astype(str) # 先將標籤名稱轉為字串，NaN 變成空字串
    tags_stage = {'約訪': 0, '面談': 1, '需求確認': 2, '建議書': 3, '成交': 4}
    tags['拜訪目的'] = tags['標籤名稱'].map(tags_stage)
    return tags[['拜訪紀錄UUID', '標籤名稱', '拜訪目的']]

# === 3. AGENT（業代資訊）===
def clean_agent(file_path):
    agent = pd.read_excel(file_path, sheet_name="AGENT", dtype={'業代': str})
    agent_stage = {'CB': 0, 'JB': 1, 'PB': 2, 'SB': 3}
    agent['職級_代碼'] = agent['月結檔 | 職級'].map(agent_stage)
    sex_stage = {'男': 0, '女': 1}
    agent['業務性別'] = agent['性別'].map(sex_stage)

    # 1. 排序後計算筆次與上一期職級
    agent_sorted = agent.sort_values(['業代', '計績年月'])
    agent_sorted['筆次'] = agent_sorted.groupby('業代').cumcount()
    agent_sorted['上一期職級'] = agent_sorted.groupby('業代')['職級_代碼'].shift(1)
    
    # 2. 判斷是否晉升（排除第一筆）
    agent_sorted['是否晉升'] = (agent_sorted['筆次'] > 0) & (
        agent_sorted['職級_代碼'] > agent_sorted['上一期職級']
    )
    
    # 3. 找出每位業代的第一次晉升時間
    agent_sorted['晉升日'] = agent_sorted.where(agent_sorted['是否晉升']).groupby('業代')['計績年月'].transform('min')
    
    # 4. 最新一筆基本資料
    latest_info = agent_sorted.groupby('業代').last().reset_index().rename(columns={
    '職級_代碼': '最新職級',
    '年資': '目前年資', 
    '目前年齡': '業務目前年齡'
    })[['業代', '計績年月', '最新職級', '業務目前年齡', '業務性別', '目前年資']]
    
    # Step 1：動態抓計績年月區間（來自保單/拜訪/agent 任一表都可）
    max_month = agent['計績年月'].max()
    max_year = max_month // 100
    max_m = max_month % 100
    current_half = 1 if 3 <= max_m <= 6 else 2
    
    this_start = max_year * 100 + (3 if current_half == 1 else 9)
    this_end = max_year * 100 + (6 if current_half == 1 else 12)
    last_start = (max_year - 1 if current_half == 1 else max_year) * 100 + (9 if current_half == 1 else 3)
    last_end = (max_year - 1 if current_half == 1 else max_year) * 100 + (12 if current_half == 1 else 6)
    
    # Step 2：今年度/上年度 活動參與率與 FYC 計算
    this_year_df = agent[(agent['計績年月'] >= this_start) & (agent['計績年月'] <= this_end)]
    last_year_df = agent[(agent['計績年月'] >= last_start) & (agent['計績年月'] <= last_end)]
    
    this_summary = this_year_df.groupby('業代')[['活動參與率', '新繳款FYC']].mean().reset_index().rename(columns={
        '活動參與率': '今年度活動參與率', '新繳款FYC': '今年度FYC'
    })
    last_summary = last_year_df.groupby('業代')[['活動參與率', '新繳款FYC']].mean().reset_index().rename(columns={
        '活動參與率': '上年度活動參與率', '新繳款FYC': '上年度FYC'
    })
    
    
    # 合併
    result = latest_info \
        .merge(this_summary, on='業代', how='left') \
        .merge(last_summary, on='業代', how='left')
        
    # 9. 加入晉升日（只保留有晉升者）
    has_promotion = agent_sorted[agent_sorted['晉升日'].notna()][['業代', '晉升日']].drop_duplicates('業代')
    agent_summary = result.merge(has_promotion, on='業代', how='left')
    agent_summary['晉升日'] = agent_summary['晉升日'].fillna(0)

    # 11. 合併進 visit_df 並計算距離晉升天數
    def convert_yyyymm_to_datetime(x):
        if x == 0 or pd.isna(x): return pd.NaT
        year, month = divmod(int(x), 100)
        return pd.to_datetime(f"{year}-{month:02d}-01")
    
    agent_summary['晉升日_dt'] = agent_summary['晉升日'].apply(convert_yyyymm_to_datetime)

    # 10. 補值處理
    agent_summary[['上年度活動參與率', '上年度FYC', '今年度活動參與率', '今年度FYC']] = agent_summary[
        ['上年度活動參與率', '上年度FYC', '今年度活動參與率', '今年度FYC']
    ].fillna(0)

    return agent_summary

# === 4. MEMBER（增員資料）===
def clean_member(file_path):
    member = pd.read_excel(file_path, sheet_name="MEMBER", dtype={'業代': str})
    count_summary = member.groupby('引薦主管業代')['業代'].nunique().reset_index()
    count_summary.columns = ['業代', '當年度增員數']
    return count_summary

# # === 5. CUSTOMER（準客戶與新增保戶）===
from dateutil.relativedelta import relativedelta

def clean_customer(file_path, reference_dates):
    customer = pd.read_excel(file_path, sheet_name="CUSTOMER", dtype={'業代': str})
    customer['建立時間 年/月/日'] = pd.to_datetime(customer['建立時間 年/月/日'], errors='coerce')

    # === 動態計算要抓的半年時間區間 ===
    # 找出最早的時間點
    min_ref_date = min(reference_dates)
    
    # 計算上一個半年範圍（上半年或下半年）
    if min_ref_date.month <= 6:
        start_date = pd.Timestamp(f"{min_ref_date.year - 1}-07-01")
        end_date = pd.Timestamp(f"{min_ref_date.year - 1}-12-31")
    else:
        start_date = pd.Timestamp(f"{min_ref_date.year}-01-01")
        end_date = pd.Timestamp(f"{min_ref_date.year}-06-30")

    # 篩選該期間內的資料
    customer_filtered = customer[
        (customer['建立時間 年/月/日'] >= start_date) &
        (customer['建立時間 年/月/日'] <= end_date)
    ]

    # 分類與彙總
    def classify(ctype):
        if pd.isna(ctype): return '未知'
        if '準客戶' in ctype: return '準客戶'
        if '錠嵂保戶' in ctype: return '新增保戶'
        return '其他'

    customer_filtered['分類'] = customer_filtered['客戶類型'].apply(classify)
    stats = customer_filtered.groupby(['業代', '分類'])['客戶UUID'].nunique().unstack(fill_value=0).reset_index()
    return stats.rename(columns={'準客戶': '上半年準客戶數', '新增保戶': '上半年新增保戶數'})


# === 6. INFO（保戶基本資料）===
def clean_info(file_path):
    info = pd.read_excel(file_path, sheet_name="INFO")
    sex_map = {'男': 0, '女': 1, '法人': 2, '校正': 3}
    info['被保人性別'] = info['被保人性別'].map(sex_map)
    info = info[info['被保人性別'].isin([0, 1])]

    summary = info.groupby('經紀人1-被保人CRM UUID').agg({
        '被保人性別': 'last',
        '被保人目前年齡': 'last',
        '要保人目前年齡': 'last',
        '保單申請案號': pd.Series.nunique,
        '繳款保費new': 'sum'
    }).reset_index().rename(columns={
        '保單申請案號': '件數',
        '繳款保費new': '總保費'
    })
    return summary

# === 7. POLICY（實際成交）===
def clean_policy(file_path):
    policy = pd.read_excel(file_path, sheet_name="POLICY")
    policy['投保日 年/月/日'] = pd.to_datetime(policy['投保日 年/月/日'])
    policy['是否為網路投保'] = np.where(policy['進件別'] == '網路投保', 1, 0)
    return policy

# === 8. TAGS_LABEL（個人化標籤）===
def clean_personal_tags(file_path):
    label = pd.read_excel(file_path, sheet_name='TAGS_LAB')
    label = label.dropna(subset=['客戶UUID', '標籤子分類', '標籤名稱'])

    # 分開處理背景與銷售類型
    tag_types = {
        '個人化標籤(背景)': '個人化標籤_背景',
        '個人化標籤(銷售)': '個人化標籤_銷售'
    }

    tag_agg = []

    for tag_subtype, new_col in tag_types.items():
        df_sub = label[label['標籤子分類'] == tag_subtype]
        grouped = df_sub.groupby('客戶UUID')['標籤名稱'].apply(lambda x: ','.join(sorted(set(x)))).reset_index()
        grouped.columns = ['客戶UUID', new_col]
        tag_agg.append(grouped)

    # 合併兩類標籤資料
    df_tags_merged = tag_agg[0]
    for df in tag_agg[1:]:
        df_tags_merged = df_tags_merged.merge(df, on='客戶UUID', how='outer')

    return df_tags_merged


# === ✅ 整合資料打包函式 ===
def prepare_model_dataset(file_path):
    visit_df = clean_visit(file_path)
    visit_dates = pd.to_datetime(visit_df['拜訪時間 年/月/日'], errors='coerce').dropna().tolist()
    tags_df = clean_tags(file_path)
    agent_df = clean_agent(file_path)
    member_df = clean_member(file_path)
    customer_df = clean_customer(file_path, reference_dates=visit_dates)
    info_df = clean_info(file_path)
    policy_df = clean_policy(file_path)
    label_df = clean_personal_tags(file_path)

    # 合併標籤
    visit_df = visit_df.merge(tags_df, on='拜訪紀錄UUID', how='left')

    # 合併 AGENT
    visit_df = visit_df.merge(agent_df, on='業代', how='left')
    visit_df['拜訪時間 年/月/日'] = pd.to_datetime(visit_df['拜訪時間 年/月/日'], errors='coerce')
    visit_df['距離晉升天數'] = (visit_df['晉升日_dt'] - visit_df['拜訪時間 年/月/日']).dt.days

    # 合併 MEMBER 增員數
    visit_df = visit_df.merge(member_df, on='業代', how='left')
    visit_df['當年度增員數'] = visit_df['當年度增員數'].fillna(0).astype(int)

    # 合併 CUSTOMER 客戶統計
    visit_df = visit_df.merge(customer_df, on='業代', how='left')
    for col in ['上半年準客戶數', '上半年新增保戶數']:
        if col in visit_df.columns:
            visit_df[col] = visit_df[col].fillna(0).astype(int)

    # 合併 INFO 客戶資訊
    visit_df = visit_df.merge(info_df, how='left', left_on='客戶UUID', right_on='經紀人1-被保人CRM UUID')
    visit_df = visit_df.drop(columns=['經紀人1-被保人CRM UUID'], errors='ignore')
    
    # 合併 TAGS_LAB 個人化標籤資料
    visit_df = visit_df.merge(label_df, on='客戶UUID', how='left')
    
    # from sklearn.preprocessing import MultiLabelBinarizer
    # def encode_personal_tags(df):
    #     # 合併兩欄標籤
    #     combined_tags = (
    #         df['個人化標籤_背景'].fillna('') + ',' +
    #         df['個人化標籤_銷售'].fillna('')
    #     ).str.strip(',').str.split(',')
    
    #     mlb = MultiLabelBinarizer()
    #     tag_matrix = mlb.fit_transform(combined_tags)
    
    #     tag_df = pd.DataFrame(tag_matrix, columns=[f"標籤_{t}" for t in mlb.classes_])
    #     tag_df.index = df.index
    
    #     df = pd.concat([df, tag_df], axis=1)
    #     return df
    
    # def encode_personal_tags(df):
    #     tag_types = {
    #         '個人化標籤_背景': '背景',
    #         '個人化標籤_銷售': '銷售'
    #     }
    #     for original_col, tag_type_label in tag_types.items():
    #         if original_col in df.columns:
    #             dummies = df[original_col].str.get_dummies(sep=',')
    #             dummies.columns = [f"標籤({tag_type_label})_{c}" for c in dummies.columns]
    #             df = pd.concat([df, dummies], axis=1)
    #     return df

    # visit_df = encode_personal_tags(visit_df)

    # 合併後處理
    visit_df['備註字數'] = visit_df['拜訪備註_文字'].apply(lambda x: len(str(x).replace(" ", "").replace("\n", "")))
    
    # 斷詞（跨環境支援）
    # 匯入預先斷好的 filtered_words，並依照 index 指派回 visit_df
    try:
        pkl_path = file_path.replace(".xlsx", "").replace(".xls", "") + ".pkl"
        with open(pkl_path, "rb") as f:
            filtered_words = pickle.load(f)
        visit_df = visit_df.reset_index(drop=True)
        visit_df['拜訪備註_詞語'] = filtered_words[:len(visit_df)]
        visit_df['備註文字_處理'] = visit_df['拜訪備註_詞語'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    except Exception as e:
        print("❌ 無法載入斷詞結果 filtered_words_test.pkl：", e)
    
    # 衍生欄位計算
    def compute_gender_diff_v2(row):
        if row['業務性別'] == row['被保人性別']:
            return row['業務性別']  # 0 or 1
        elif row['業務性別'] == 0 and row['被保人性別'] == 1:
            return 2
        elif row['業務性別'] == 1 and row['被保人性別'] == 0:
            return 3
        else:
            return pd.NA

    visit_df['業務客戶性別組合'] = visit_df.apply(compute_gender_diff_v2, axis=1)
    visit_df['業務客戶年齡差距'] = visit_df['要保人目前年齡'] - visit_df['業務目前年齡']

    # 平均拜訪間隔天數
    visit_df = visit_df.sort_values(['客戶UUID', '拜訪時間 年/月/日'])
    
    def avg_visit_interval(x):
        if len(x) < 2:
            return 0
        x_sorted = x.sort_values()
        diffs = x_sorted.diff().dropna().dt.days
        return diffs.mean()

    interval_summary = visit_df.groupby(['業代', '客戶UUID'])['拜訪時間 年/月/日'].apply(avg_visit_interval).reset_index(name='平均拜訪間隔天數')
    visit_df = visit_df.merge(interval_summary, on=['業代', '客戶UUID'], how='left')


    # 每週平均拜訪客戶數
    visit_df['週'] = visit_df['拜訪時間 年/月/日'].dt.to_period('W').astype(str)
    weekly = visit_df.groupby(['業代', '週'])['客戶UUID'].nunique().reset_index()
    weekly_summary = weekly.groupby('業代')['客戶UUID'].mean().reset_index()
    weekly_summary.columns = ['業代', '每週平均拜訪客戶數']
    visit_df = visit_df.merge(weekly_summary, on='業代', how='left')

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
    
    from gensim.models import Word2Vec

    # 假設你已經將所有斷詞存在 list 格式（每一筆是一個詞語 list）
    token_lists = visit_df['備註文字_處理'].dropna().apply(lambda x: x.split()).tolist()
    
    # 訓練 Word2Vec（也可以載入外部保險語料）
    model_w2v = Word2Vec(sentences=token_lists, vector_size=150, window=5, min_count=2, workers=4)

    # 加入語意相近的詞（例如距離前 10 名，距離需 < 0.6）
    for word in seed_words:
        if word in model_w2v.wv:
            similar_words = model_w2v.wv.most_similar(word, topn=20)
            for sim_word, score in similar_words:
                if score > 0.6:  # 可自行調整門檻
                    meaningful_words_set.add(sim_word)
        
    # 有意義詞數 (需有斷詞結果與保險語彙集)
    def count_meaningful(text):
        if pd.isna(text): return 0
        tokens = text.split()
        return sum(1 for word in tokens if word in meaningful_words_set)  # 須預先定義 meaningful_words_set

    if '備註文字_處理' in visit_df.columns:
        visit_df['有意義詞數'] = visit_df['備註文字_處理'].apply(count_meaningful)

    le = LabelEncoder()
    visit_df['營業單位_編碼'] = le.fit_transform(visit_df['營業單位'])

    # 不包含 "(增)" 的標籤資料集
    visit_df = visit_df[~visit_df['標籤名稱'].str.contains(r'\(增\)', regex=True, na=False)]
    
    # 過濾必要欄位
    df_valid = visit_df.dropna(subset=[
        '被保人性別', '被保人目前年齡', '要保人目前年齡', '件數', '總保費'
    ])
    
    from collections import defaultdict

    # === 補充：建立「拜訪與最近投保日」關係欄位 ===
    # 先確保欄位名稱與時間格式一致
    df_valid['拜訪時間'] = pd.to_datetime(df_valid['拜訪時間 年/月/日'], errors='coerce')
    policy_df['投保日'] = pd.to_datetime(policy_df['投保日 年/月/日'], errors='coerce')
    
    # 是否為網路投保
    policy_df['是否為網路投保'] = np.where(policy_df['進件別'] == '網路投保', 1, 0)
    
    # 建立 UUID 對應投保記錄 dict
    policy_dict = defaultdict(list)
    for _, row in policy_df.iterrows():
        uuid = row['經紀人1-被保人CRM UUID']
        policy_dict[uuid].append(row)
    
    # 比對函數：根據拜訪時間找出最接近的投保紀錄
    def get_nearest_policy_info(uuid, visit_time):
        records = policy_dict.get(uuid, [])
        if not records or pd.isna(visit_time):
            return pd.Series([np.nan, pd.NaT, np.nan])
    
        after = [r for r in records if r['投保日'] > visit_time]
        before = [r for r in records if r['投保日'] <= visit_time]
    
        if after:
            r = sorted(after, key=lambda x: x['投保日'])[0]
        elif before:
            r = sorted(before, key=lambda x: x['投保日'], reverse=True)[0]
        else:
            return pd.Series([np.nan, pd.NaT, np.nan])
    
        return pd.Series([(visit_time - r['投保日']).days, r['投保日'], r['是否為網路投保']])
    
    # 套用到每一列
    df_valid[['拜訪與投保日天數差', '最近投保日', '最近是否為網路投保']] = df_valid.apply(
        lambda row: get_nearest_policy_info(row['客戶UUID'], row['拜訪時間']),
        axis=1
    )

    return df_valid, policy_df


# ✅ 執行
if __name__ == '__main__':
    file = "D:/備註文字探勘/repeater/新資料_0704.xlsx"
    df_ready, policy_df = prepare_model_dataset(file)
    print("✅ 資料整合與欄位齊備，筆數：", len(df_ready))
    
