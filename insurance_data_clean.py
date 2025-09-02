# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:31:14 2025

@author: Z01788
"""

# === 📂 7份 Excel 工作表的資料清理流程 ===
# file_path = "D:/備註文字探勘/repeater/新資料_0731.xlsx"

import pandas as pd
import re
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# === 1. VISIT（拜訪資料）===
def clean_visit(file_path):
    visit = pd.read_excel(file_path, sheet_name="VISIT")
    visit['拜訪時間 年/月/日'] = pd.to_datetime(visit['拜訪時間 年/月/日'], errors='coerce')
    
    # 斷詞（跨環境支援）
    # 匯入預先斷好的 filtered_words，並依照 index 指派回 visit_df
    try:
        pkl_path = file_path.replace(".xlsx", "").replace(".xls", "") + ".pkl"
        with open(pkl_path, "rb") as f:
            filtered_words = pickle.load(f)
        visit = visit.reset_index(drop=True)
        visit['拜訪備註_詞語'] = filtered_words[:len(visit)]
        visit['備註文字_處理'] = visit['拜訪備註_詞語'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    except Exception as e:
        print("❌ 無法載入斷詞結果 filtered_words_test.pkl：", e)
    
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
        # 移除 # 後的空白（包含全形空格 \u3000）
        text = re.sub(r'#\s*', '#', text)     
        text = re.sub(r'#\u3000*', '#', text)  
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
    
    # # 與你相容的清理器：保留中英數與底線，去掉頭尾雜訊並轉小寫
    # def clean_hashtag(raw_tag: str) -> str:
    #     tag = re.sub(r'^[^a-zA-Z0-9\u4e00-\u9fff_]+', '', raw_tag)
    #     tag = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff_]+$', '', tag)
    #     return tag.lower()
    
    # # 允許的標籤 token（不含空白）；\w 含字母數字底線，另外補 CJK、常見連接符
    # TAG_PATTERN = re.compile(r'(?<![\w])[#＃]([A-Za-z0-9_\u4e00-\u9fff\-\+\/]+)', flags=re.UNICODE)
    
    # def split_note_text_and_tags(note: str):
    #     """回傳 (純文字, 標籤list[清理後])；會移除行內 #標籤 token，本體以外文字保留"""
    #     if pd.isna(note):
    #         return "", []
    
    #     s = str(note).replace('_x000D_', '\n').replace('\r', '')
    #     # 全形轉半形
    #     s = s.replace('＃', '#')
    #     # 與你一致：把 '#   '（含全形空格）收斂成 '#'
    #     s = re.sub(r'#[\s\u3000]*', '#', s)
    #     # 與你一致：連續 '##' 壓成單一 '#'
    #     s = re.sub(r'#{2,}', '#', s)
    
    #     # 擷取原始標籤
    #     raw_tags = [m.group(1) for m in TAG_PATTERN.finditer(s)]
    #     # 套用同款清理器（確保與你一致）
    #     tags = [clean_hashtag(t) for t in raw_tags if clean_hashtag(t)]
    
    #     # 行內移除標籤 token（僅去 '#xxx' 本體，保留標點與其餘文字）
    #     s_no_tags = TAG_PATTERN.sub('', s)
    
    #     # 清理多餘空白（含跨行）
    #     lines = [re.sub(r'\s+', ' ', ln).strip() for ln in s_no_tags.splitlines()]
    #     text_clean = '\n'.join([ln for ln in lines if ln])
    
    #     return text_clean, tags
    
    # # 套到你的 DataFrame
    # # 會得到兩欄：拜訪備註_文字（行內已去標籤）、拜訪備註_標籤（list）
    # visit[['拜訪備註_文字', '拜訪備註_標籤']] = visit['拜訪備註'].apply(
    #     lambda x: pd.Series(split_note_text_and_tags(x))
    # )
    

    # 擷取非 # 行文字
    visit['拜訪備註_文字'] = visit['拜訪備註'].apply(extract_non_sharp_text)
    
    # 擷取標籤文字（清洗後、不含#）
    # 👉 先保留成清單，之後要做 count、multi-hot、embedding 都方便
    visit['拜訪備註_標籤_list'] = visit['拜訪備註'].apply(extract_hashtags)

    # 👉 這裡就是你要的「# 數量」：直接用清單長度最準
    visit['hashtag_count'] = visit['拜訪備註_標籤_list'].apply(len).astype('int64')
    
    # 擷取標籤文字（清洗後、不含#）
    visit['拜訪備註_標籤'] = visit['拜訪備註'].apply(extract_hashtags)
    visit['拜訪備註_標籤'] = visit['拜訪備註_標籤'].apply(lambda tags: ' '.join(tags))
    
    # 字數計算
    visit['備註字數'] = visit['拜訪備註_文字'].apply(lambda x: len(str(x).replace(" ", "").replace("\n", "")))

    # 拜訪次數與平均
    visit['拜訪次數'] = visit.groupby(['業代', '客戶UUID'])['拜訪紀錄UUID'].transform('count')
    visit['平均每客戶拜訪次數'] = visit.groupby('業代')['拜訪次數'].transform('mean')
    
    # 先按照拜訪時間排序，再依照業代+客戶分組加上序號
    visit = visit.sort_values(by='拜訪時間 年/月/日')
    visit['拜訪序號'] = visit.groupby(['業代', '客戶UUID']).cumcount() + 1
    
    # 增加賽季欄位
    def classify_season(date):
        if 2 <= date.month <= 6:
            return '夏賽'
        elif 8 <= date.month <= 12:
            return '冬賽'
        else:
            return '非賽季'
    
    visit['賽季'] = visit['拜訪時間 年/月/日'].apply(classify_season)
    season_map = {'非賽季': 0, '夏賽': 1, '冬賽': 2}
    visit['賽季'] = visit['賽季'].map(season_map)

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
    from pandas.tseries.offsets import DateOffset
    
    # 將計績年月轉換為 datetime 格式（每月設為當月 1 號）
    agent['計績年月_dt'] = pd.to_datetime(agent['計績年月'].astype(str) + '01', format='%Y%m%d', errors='coerce')
    
    # === Step 1: 抓每位業代的最後一筆計績年月，回推半年計算 活動參與率 ===
    last_month = agent.groupby('業代')['計績年月_dt'].max().reset_index()
    last_month['起'] = last_month['計績年月_dt'] - DateOffset(months=6)
    last_month['迄'] = last_month['計績年月_dt']
    
    # 合併回原資料，篩出半年內的活動參與率
    agent_rate = agent.merge(last_month[['業代', '起', '迄']], on='業代')
    rate_df = agent_rate[(agent_rate['計績年月_dt'] >= agent_rate['起']) & (agent_rate['計績年月_dt'] <= agent_rate['迄'])]
    rate_summary = rate_df.groupby('業代')['活動參與率'].mean().reset_index().rename(columns={'活動參與率': '最近半年活動參與率'})
    
    # === Step 2: 抓每位業代的第一筆計績年月，回推半年計算 新繳款FYC ===
    first_month = agent.groupby('業代')['計績年月_dt'].min().reset_index()
    first_month['起'] = first_month['計績年月_dt'] - DateOffset(months=6)
    first_month['迄'] = first_month['計績年月_dt']
    
    # 合併回原資料，篩出半年內的 FYC
    agent_fyc = agent.merge(first_month[['業代', '起', '迄']], on='業代')
    fyc_df = agent_fyc[(agent_fyc['計績年月_dt'] >= agent_fyc['起']) & (agent_fyc['計績年月_dt'] <= agent_fyc['迄'])]
    fyc_summary = fyc_df.groupby('業代')['新繳款FYC'].mean().reset_index().rename(columns={'新繳款FYC': '上一個半年度FYC'})
    
    # === Step 3: 合併兩者 ===
    activity_summary = rate_summary.merge(fyc_summary, on='業代', how='outer')
    
    
    # 合併
    result = latest_info \
        .merge(activity_summary, on='業代', how='left')
        
        
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
    agent_summary[['最近半年活動參與率', '上一個半年度FYC']] = agent_summary[
        ['最近半年活動參與率', '上一個半年度FYC']
    ].fillna(0)
    
    # ✅ 新增：回傳完整晉升歷史 promotion_df（後續 merge_asof 用）
    promotion_df = agent_sorted[agent_sorted['是否晉升']][['業代', '計績年月']].copy()
    promotion_df = promotion_df.dropna(subset=['計績年月'])  # 去除空值
    promotion_df['晉升日_dt'] = pd.to_datetime(promotion_df['計績年月'].astype(str) + '01', format='%Y%m%d', errors='coerce')
    promotion_df = promotion_df.dropna(subset=['晉升日_dt'])  # 避免 merge_asof 錯誤
    promotion_df = promotion_df[['業代', '晉升日_dt']]  # 確保欄位乾淨
    all_agents = agent['業代'].drop_duplicates()
    promotion_df = all_agents.to_frame().merge(promotion_df, on='業代', how='left')

    return agent_summary, promotion_df


# # === 4. MEMBER（增員資料）===
# def clean_member(file_path):
#     member = pd.read_excel(file_path, sheet_name="MEMBER", dtype={'業代': str})
#     count_summary = member.groupby('引薦主管業代')['業代'].nunique().reset_index()
#     count_summary.columns = ['業代', '當年度增員數']
#     return count_summary

# # === 5. CUSTOMER（準客戶與新增保戶）===
# from dateutil.relativedelta import relativedelta

# def clean_customer(file_path, reference_dates):
#     customer = pd.read_excel(file_path, sheet_name="CUSTOMER", dtype={'業代': str})
#     customer['建立時間 年/月/日'] = pd.to_datetime(customer['建立時間 年/月/日'], errors='coerce')

#     # === 動態計算要抓的半年時間區間 ===
#     # 找出最早的時間點
#     min_ref_date = min(reference_dates)
    
#     # 計算上一個半年範圍（上半年或下半年）
#     if min_ref_date.month <= 6:
#         start_date = pd.Timestamp(f"{min_ref_date.year - 1}-07-01")
#         end_date = pd.Timestamp(f"{min_ref_date.year - 1}-12-31")
#     else:
#         start_date = pd.Timestamp(f"{min_ref_date.year}-01-01")
#         end_date = pd.Timestamp(f"{min_ref_date.year}-06-30")

#     # 篩選該期間內的資料
#     customer_filtered = customer[
#         (customer['建立時間 年/月/日'] >= start_date) &
#         (customer['建立時間 年/月/日'] <= end_date)
#     ]

#     # 分類與彙總
#     def classify(ctype):
#         if pd.isna(ctype): return '未知'
#         if '準客戶' in ctype: return '準客戶'
#         if '錠嵂保戶' in ctype: return '新增保戶'
#         return '其他'

#     customer_filtered['分類'] = customer_filtered['客戶類型'].apply(classify)
#     stats = customer_filtered.groupby(['業代', '分類'])['客戶UUID'].nunique().unstack(fill_value=0).reset_index()
#     return stats.rename(columns={'準客戶': '上半年準客戶數', '新增保戶': '上半年新增保戶數'})


# === 6. INFO（保戶基本資料）===
def clean_info(file_path):
    info = pd.read_excel(file_path, sheet_name="INFO")
    sex_map = {'男': 0, '女': 1, '法人': 2, '校正': 3}
    info['被保人性別'] = info['被保人性別'].map(sex_map)
    info = info[info['被保人性別'].isin([0, 1])]

    # summary = info.groupby('經紀人1-被保人CRM UUID').agg({
    #     '被保人性別': 'last',
    #     '被保人目前年齡': 'last',
    #     '要保人目前年齡': 'last',
    #     '保單申請案號': pd.Series.nunique,
    #     '繳款保費new': 'sum'
    # }).reset_index().rename(columns={
    #     '保單申請案號': '件數',
    #     '繳款保費new': '總保費'
    # })
    return info

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
    visit_df = visit_df.sort_values(['客戶UUID', '拜訪時間 年/月/日'])
    # visit_dates = pd.to_datetime(visit_df['拜訪時間 年/月/日'], errors='coerce').dropna().tolist()
    tags_df = clean_tags(file_path)
    agent_df, promotion_df = clean_agent(file_path)
    # member_df = clean_member(file_path)
    # customer_df = clean_customer(file_path, reference_dates=visit_dates)
    info_df = clean_info(file_path)
    policy_df = clean_policy(file_path)
    label_df = clean_personal_tags(file_path)

    # 合併標籤
    visit_df = visit_df.merge(tags_df, on='拜訪紀錄UUID', how='left')
    
    # ----------------------------------------------------
    # A) 以拜訪日為基準：客戶累積件數 / 累積保費（截至拜訪日）
    # ----------------------------------------------------
    # 假設 policy_df 內有「經紀人1-被保人CRM UUID」作為「客戶UUID」的對應鍵
    info_df['投保日 年/月/日'] = pd.to_datetime(info_df['投保日 年/月/日'], errors='coerce')
    
    cust_daily = (
        info_df
        .groupby(['經紀人1-被保人CRM UUID', '投保日 年/月/日'], as_index=False)
        .agg(件數=('保單申請案號', 'nunique'),
             總保費=('繳款保費new', 'sum'))
        .rename(columns={'經紀人1-被保人CRM UUID': '客戶UUID',
                         '投保日 年/月/日': '投保日'})
        .sort_values(['客戶UUID', '投保日'])
    )
    # 每位客戶做累積（截至當日）
    cust_daily['截至當日累積件數'] = cust_daily.groupby('客戶UUID')['件數'].cumsum()
    cust_daily['截至當日累積保費'] = cust_daily.groupby('客戶UUID')['總保費'].cumsum()
    
    # === 分組 asof：對每個客戶獨立 merge_asof，再 concat 回來 ===
    def merge_asof_by_customer(visit_df, cust_daily, include_same_day=True):
        # 右表先分組以加速
        right_groups = {cid: g.sort_values('投保日')
                        for cid, g in cust_daily.groupby('客戶UUID', sort=False)}
    
        out = []
        for cid, sub in visit_df.groupby('客戶UUID', sort=False):
            sub = sub.sort_values('拜訪時間 年/月/日')
            right = right_groups.get(cid)
    
            if right is None or right.empty:
                # 該客戶沒有任何保單紀錄 → 直接補 0
                sub['截至當日累積件數'] = 0
                sub['截至當日累積保費'] = 0.0
                out.append(sub)
                continue
    
            tmp = pd.merge_asof(
                left=sub,
                right=right[['投保日','截至當日累積件數','截至當日累積保費']],
                left_on='拜訪時間 年/月/日',
                right_on='投保日',
                direction='backward',
                allow_exact_matches=include_same_day  # 含當日/不含當日開關
            )
            tmp = tmp.drop(columns=['投保日'])
            tmp['截至當日累積件數'] = tmp['截至當日累積件數'].fillna(0).astype(int)
            tmp['截至當日累積保費'] = tmp['截至當日累積保費'].fillna(0.0)
            out.append(tmp)
    
        return pd.concat(out, ignore_index=True)
    
    # 呼叫：把你 build 好的 cust_daily 與 visit_df 丟進去
    visit_df = merge_asof_by_customer(visit_df, cust_daily, include_same_day=True)
    
    # ----------------------------------------------------
    # B) 以拜訪日為基準：業代月快照（單位 / 職級 / 年資 / 年齡 / 性別）
    #     用 as-of 取拜訪日前最近月份
    # ----------------------------------------------------
    from dateutil.relativedelta import relativedelta
    
    def parse_yyyymmdd(s):
        return pd.to_datetime(s, format='%Y%m%d', errors='coerce')
    
    def parse_yyyymm(s):
        # 將 YYYYMM 視為每月1號
        return pd.to_datetime(s.astype(str) + '01', format='%Y%m%d', errors='coerce')
    
    def months_between(start, ref):
        if pd.isna(start) or pd.isna(ref):
            return None
        # 以月為單位計算，含當月：若要更嚴格可依日扣1個月
        rd = relativedelta(ref, start)
        m = rd.years * 12 + rd.months
        # 若需按「日」嚴格扣除未滿一月，可加上：
        if ref.day < start.day:
            m = max(m - 1, 0)
        return max(m, 0)
    
    def calc_age_years(birth, ref):
        if pd.isna(birth) or pd.isna(ref):
            return None
        return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))
    
    # 讀原始 AGENT
    agent_raw = pd.read_excel(file_path, sheet_name="AGENT", dtype={'業代': str})
    
    # 計績年月（每月1號）
    agent_raw['計績年月_dt'] = parse_yyyymm(agent_raw['計績年月'])
    
    # 生日 → 動態年齡
    agent_raw['生日_dt'] = parse_yyyymmdd(agent_raw['生日'])
    agent_raw['截至該月業務年齡'] = agent_raw.apply(
        lambda r: calc_age_years(r['生日_dt'], r['計績年月_dt']), axis=1
    )
    
    def calc_months_between(start_dt, ref_dt):
        if pd.isna(start_dt) or pd.isna(ref_dt):
            return None
        rd = relativedelta(ref_dt, start_dt)
        m = rd.years * 12 + rd.months
        # 若 ref 的「日」尚未達到 start 的「日」，視為未滿一整月 → 扣 1
        if ref_dt.day < start_dt.day:
            m = max(m - 1, 0)
        return max(m, 0)
    
    # 職級
    agent_stage = {'CB': 0, 'JB': 1, 'PB': 2, 'SB': 3}
    agent_raw['職級_代碼'] = agent_raw['月結檔 | 職級'].map(agent_stage)
    
    # --- 1) 解析欄位成 datetime ---
    # 計績年月（以當月1日為基準）
    agent_raw['計績年月_dt'] = pd.to_datetime(
        agent_raw['計績年月'].astype(str) + '01',
        format='%Y%m%d', errors='coerce'
    )
    
    # 簽約日
    agent_raw['簽約日_dt'] = pd.to_datetime(
        agent_raw['簽約日 年/月/日'], errors='coerce'
    )
    
    # --- 2) 以「簽約日」→「計績年月」計算年資 ---
    agent_raw['截至該月年資(月)'] = agent_raw.apply(
        lambda r: calc_months_between(r['簽約日_dt'], r['計績年月_dt']),
        axis=1
    )
    
    # 轉成年（保留1位小數；若要整數年可用 // 12）
    agent_raw['截至該月年資'] = agent_raw['截至該月年資(月)'].apply(
        lambda m: None if pd.isna(m) else round(m / 12, 1)
    )
    
    # 性別轉碼
    sex_stage = {'男': 0, '女': 1}
    if '性別' in agent_raw.columns:
        agent_raw['業務性別'] = agent_raw['性別'].map(sex_stage)
    
    # 準備快照欄位
    snap_cols = ['業代', '計績年月_dt', '營業單位', '職級_代碼',
                 '截至該月年資', '截至該月年資(月)', '截至該月業務年齡', '業務性別']
    snap_cols = [c for c in snap_cols if c in agent_raw.columns]
    agent_snap = (agent_raw[snap_cols]
                  .rename(columns={'計績年月_dt':'計績年月'})
                  .sort_values(['業代','計績年月']))
    
    # 與拜訪 as-of 對齊
    visit_df = visit_df.sort_values(['業代', '拜訪時間 年/月/日'])
    agent_snap = agent_snap.sort_values(['業代', '計績年月'])
    
    # 分組版本：對每個業代各自 merge_asof，再 concat 回來
    def merge_asof_by_agent(visit_df, agent_snap, include_same_day=True): 
        # 右表先依業代分組並各自按「計績年月」排好
        right_groups = {
            aid: g.sort_values('計績年月')
            for aid, g in agent_snap.groupby('業代', sort=False)
        }
    
        out = []
        for aid, left_sub in visit_df.groupby('業代', sort=False):
            left_sub = left_sub.sort_values('拜訪時間 年/月/日')
    
            right_sub = right_groups.get(aid)
            if right_sub is None or right_sub.empty:
                # 沒快照就補 NA 欄位
                for col in ['營業單位','職級_代碼','截至該月年資','截至該月年資(月)','截至該月業務年齡','業務性別']:
                    if col not in left_sub.columns:
                        left_sub[col] = np.nan
                out.append(left_sub)
                continue
    
            merged = pd.merge_asof(
                left=left_sub,
                right=right_sub,
                left_on='拜訪時間 年/月/日',
                right_on='計績年月',
                direction='backward',
                allow_exact_matches=include_same_day
            )
            merged = merged.drop(columns=['計績年月'])
            out.append(merged)
    
        return pd.concat(out, ignore_index=True)
    
    # 執行（含同日；若你要嚴格小於拜訪日，改成 include_same_day=False）
    visit_df = merge_asof_by_agent(visit_df, agent_snap, include_same_day=True)
    visit_df = visit_df.drop(columns=['業代_y']).rename(columns={'業代_x': '業代'})
    
    
    
    # 2) 數值欄位標準化（不存在就補 0）
    for col in ['活動參與率', '新繳款FYC']:
        if col not in agent_raw.columns:
            agent_raw[col] = np.nan
    
    # 僅保留需要欄位，清 NaT
    agent_m = agent_raw[['業代', '計績年月_dt', '活動參與率', '新繳款FYC']].dropna(subset=['業代', '計績年月_dt'])
    agent_m['業代'] = agent_m['業代'].astype(str)
    
    # 3) 依「業代 + 月份」排序，做各欄位的「累積總和」與「有效月數累積」
    agent_m = agent_m.sort_values(['業代', '計績年月_dt'])
    
    def add_cum_and_count(df, value_col, prefix, by_col='業代', time_col='計績年月_dt'):
        """
        針對每位業務(by_col)，依 time_col 排序後，計算：
          - {prefix}cnt      ：該月是否有值(非空=1, 否則0)
          - {prefix}cnt_cum ：有效月份的累積計數（每位業務各自累加）
          - {prefix}sum_cum ：value 的累積總和（缺值視為0；每位業務各自累加）
        """
        df = df.copy()
    
        # 型別與排序（很重要：必須先依 業代+月份 排序再做 cumsum）
        df[by_col] = df[by_col].astype(str)
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values([by_col, time_col], kind='mergesort')
    
        # 計有效月 (非空 = 1)
        df[f'{prefix}cnt'] = df[value_col].notna().astype(int)
    
        # 針對每位業務做 transform 累積
        g = df.groupby(by_col, sort=False)
        df[f'{prefix}cnt_cum'] = g[f'{prefix}cnt'].transform(lambda s: s.cumsum())
        df[f'{prefix}sum_cum'] = g[value_col].transform(lambda s: s.fillna(0).cumsum())
    
        return df
    
    # 你的主流程中這樣呼叫（確保 agent_m 先有 '計績年月_dt'）
    # agent_m['計績年月_dt'] = pd.to_datetime(agent_m['計績年月'].astype(str)+'01', format='%Y%m%d', errors='coerce')
    
    agent_m = add_cum_and_count(agent_m, value_col='活動參與率', prefix='rate_',
                                by_col='業代', time_col='計績年月_dt')
    agent_m = add_cum_and_count(agent_m, value_col='新繳款FYC', prefix='fyc_',
                                by_col='業代', time_col='計績年月_dt')
    
    # 只取用到的欄位做 as-of（降記憶體）
    right_core = agent_m[['業代', '計績年月_dt',
                          'rate_cnt_cum', 'rate_sum_cum',
                          'fyc_cnt_cum',  'fyc_sum_cum']]
    
    # ========= 分組 as-of 工具：避開 merge_asof + by 的排序雷 =========
    def asof_by_agent(
        left_df, right_df,
        by_col='業代',
        left_time_col='拜訪時間 年/月/日',
        right_time_col='計績年月_dt',
        value_cols=None,
        include_same_day=True
    ):
        """
        對每個業代各自做 merge_asof，再 concat；缺欄位自動補 0，避免 KeyError。
        """
        if value_cols is None:
            value_cols = ['rate_cnt_cum','rate_sum_cum','fyc_cnt_cum','fyc_sum_cum']
    
        left = left_df.copy()
        right = right_df.copy()
    
        # dtype / 時間欄位
        left[by_col] = left[by_col].astype(str)
        right[by_col] = right[by_col].astype(str)
        left[left_time_col]   = pd.to_datetime(left[left_time_col],   errors='coerce')
        right[right_time_col] = pd.to_datetime(right[right_time_col], errors='coerce')
    
        left  = left.dropna(subset=[by_col, left_time_col])
        right = right.dropna(subset=[by_col, right_time_col])
    
        # 右表欄位檢查；若缺少，就先加空欄（0），避免合併時找不到
        for c in value_cols:
            if c not in right.columns:
                right[c] = 0
    
        # 只保留會用到的業代，加速
        used_agents = left[by_col].unique()
        right = right[right[by_col].isin(used_agents)]
    
        # 分組緩存
        right_groups = {
            aid: g.sort_values(right_time_col)
            for aid, g in right.groupby(by_col, sort=False)
        }
    
        out = []
        for aid, lsub in left.groupby(by_col, sort=False):
            lsub = lsub.sort_values(left_time_col)
            rsub = right_groups.get(aid)
    
            if rsub is None or rsub.empty:
                # 該業代右表完全沒有資料 → 直接補 0
                for c in value_cols:
                    lsub[c] = 0
                out.append(lsub)
                continue
    
            merged = pd.merge_asof(
                left=lsub,
                right=rsub[[right_time_col, *value_cols]],
                left_on=left_time_col,
                right_on=right_time_col,
                direction='backward',
                allow_exact_matches=include_same_day
            ).drop(columns=[right_time_col])
    
            # 合併後再確保欄位存在且補值
            for c in value_cols:
                if c not in merged.columns:
                    merged[c] = 0
                merged[c] = merged[c].fillna(0)
    
            out.append(merged)
    
        return pd.concat(out, ignore_index=True)
    
    # ========= 1) 以「拜訪日」對齊（含當日）：as-of 累積 =========
    cols = ['rate_cnt_cum','rate_sum_cum','fyc_cnt_cum','fyc_sum_cum']

    visit_df = asof_by_agent(
        left_df=visit_df,
        right_df=right_core,
        by_col='業代',
        left_time_col='拜訪時間 年/月/日',
        right_time_col='計績年月_dt',
        value_cols=cols, 
        include_same_day=True  # 含拜訪當月
    )
    
    # 在做六個月前 as-of 之前，先把「拜訪日」的累積備份一份
    for c in cols:
        visit_df[f'{c}_v'] = visit_df[c]
        
    # 刪掉基礎欄位，避免第二次 as-of 欄名衝突
    visit_df.drop(columns=cols, inplace=True, errors='ignore')
    
    # ========= 2) 以「拜訪日往回 6 個月」對齊（不含當日） =========
    visit_df['_six_months_ago'] = visit_df['拜訪時間 年/月/日'] - pd.DateOffset(months=6)
    
    visit_df = asof_by_agent(
        left_df=visit_df,
        right_df=right_core,
        by_col='業代',
        left_time_col='_six_months_ago',
        right_time_col='計績年月_dt',
        value_cols=cols, 
        include_same_day=False  # 起點不含當月，視窗為 (t-6M, t]
    )
    
    # 兩組欄位目前同名，為了差分先做副本（或直接在下一步計算時用 _x/_y 也可以）
    # 這裡用後綴區分：v 代表「拜訪日」、v6 代表「六個月前」
    for c in ['rate_cnt_cum','rate_sum_cum','fyc_cnt_cum','fyc_sum_cum']:
        visit_df[f'{c}_v6'] = visit_df[c]
    # 再把拜訪日那次的結果重新 as-of 一次到新的欄位，避免覆蓋；或你前面保留副本也行
    # 簡化起見：我們在第二次 as-of 後立刻把第一次的欄位複製到 *_v（請在第一次 as-of 之後補下列一段）
    
    
    # =====================================================
    
    # 安全處理：0 除錯
    def safe_avg(num, den):
        return np.where(den > 0, num / den, 0.0)
    
    # 3) 差分 → 最近半年平均值
    # 活動參與率（平均）
    rate_num = visit_df['rate_sum_cum_v'] - visit_df['rate_sum_cum_v6']
    rate_den = visit_df['rate_cnt_cum_v'] - visit_df['rate_cnt_cum_v6']
    visit_df['最近半年活動參與率'] = safe_avg(rate_num, rate_den)
    
    # FYC（平均）
    fyc_num = visit_df['fyc_sum_cum_v'] - visit_df['fyc_sum_cum_v6']
    fyc_den = visit_df['fyc_cnt_cum_v'] - visit_df['fyc_cnt_cum_v6']
    visit_df['上一個半年度FYC'] = safe_avg(fyc_num, fyc_den)
    
    # 4) 清理暫存欄位
    visit_df.drop(columns=[
        '_six_months_ago',
        'rate_cnt_cum','rate_sum_cum','fyc_cnt_cum','fyc_sum_cum',
        'rate_cnt_cum_v','rate_sum_cum_v','fyc_cnt_cum_v','fyc_sum_cum_v',
        'rate_cnt_cum_v6','rate_sum_cum_v6','fyc_cnt_cum_v6','fyc_sum_cum_v6'
    ], inplace=True, errors='ignore')
    
    
    
    # ----------------------------------------------------
    # C) 以拜訪日為基準：「上半年準客戶數 / 上半年新增保戶數」的滾動半年重新計算
    #     直接覆蓋同名欄位
    # ----------------------------------------------------
    customer_raw = pd.read_excel(file_path, sheet_name="CUSTOMER", dtype={'業代': str})
    customer_raw['建立時間 年/月/日'] = pd.to_datetime(customer_raw['建立時間 年/月/日'], errors='coerce')
    
    # 分類：與你原本邏輯一致（準客戶 / 新增保戶）
    def classify(ctype):
        if pd.isna(ctype): return '未知'
        if '準客戶' in ctype: return '準客戶'
        if '錠嵂保戶' in ctype: return '新增保戶'
        return '其他'
    
    customer_raw['分類'] = customer_raw['客戶類型'].apply(classify)
    
    # 先彙總到「業代 + 建立日」的日計數，拆成兩欄（準客戶 / 新增保戶）
    cust_counts_daily = (
        customer_raw
        .groupby(['業代', '建立時間 年/月/日', '分類'], as_index=False)['客戶UUID']
        .nunique()
        .rename(columns={'客戶UUID': 'count'})
    )
    # 只保留需要的兩類
    cust_counts_daily = cust_counts_daily[cust_counts_daily['分類'].isin(['準客戶', '新增保戶'])]
    
    # 轉寬
    cust_counts_pivot = (
        cust_counts_daily
        .pivot_table(index=['業代', '建立時間 年/月/日'],
                     columns='分類', values='count', fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if '準客戶' not in cust_counts_pivot.columns:
        cust_counts_pivot['準客戶'] = 0
    if '新增保戶' not in cust_counts_pivot.columns:
        cust_counts_pivot['新增保戶'] = 0
    
    cust_counts_pivot = cust_counts_pivot.sort_values(['業代', '建立時間 年/月/日'])
    
    # 做「累積」：之後用差分得到「近半年內」的數量
    cust_counts_pivot['cum_準客戶'] = cust_counts_pivot.groupby('業代')['準客戶'].cumsum()
    cust_counts_pivot['cum_新增保戶'] = cust_counts_pivot.groupby('業代')['新增保戶'].cumsum()
    
    # as-of 取「拜訪日」的累積
    visit_df = visit_df.sort_values(['業代', '拜訪時間 年/月/日'])
    cust_counts_pivot = cust_counts_pivot.sort_values(['業代', '建立時間 年/月/日'])
    
    def asof_counts_by_agent(
        visit_df,
        counts_df,
        by_col='業代',
        left_time_col='拜訪時間 年/月/日',
        right_time_col='建立時間 年/月/日',
        value_cols=('cum_準客戶','cum_新增保戶'),
        out_prefix='asof_',
        include_same_day=True
    ):
        visit_df = visit_df.copy()
        counts_df = counts_df.copy()
    
        visit_df[by_col] = visit_df[by_col].astype(str)
        counts_df[by_col] = counts_df[by_col].astype(str)
    
        visit_df[left_time_col]   = pd.to_datetime(visit_df[left_time_col],   errors='coerce')
        counts_df[right_time_col] = pd.to_datetime(counts_df[right_time_col], errors='coerce')
    
        visit_df  = visit_df.dropna(subset=[by_col, left_time_col])
        counts_df = counts_df.dropna(subset=[by_col, right_time_col])
    
        used_agents = visit_df[by_col].unique()
        counts_df = counts_df[counts_df[by_col].isin(used_agents)]
    
        right_groups = {
            aid: g.sort_values(right_time_col)
            for aid, g in counts_df.groupby(by_col, sort=False)
        }
    
        out = []
        for aid, left_sub in visit_df.groupby(by_col, sort=False):
            left_sub = left_sub.sort_values(left_time_col)
            right_sub = right_groups.get(aid)
    
            if right_sub is None or right_sub.empty:
                for c in value_cols:
                    left_sub[f'{out_prefix}{c}'] = 0
                out.append(left_sub)
                continue
    
            merged = pd.merge_asof(
                left=left_sub,
                right=right_sub[[right_time_col, *value_cols]],
                left_on=left_time_col,
                right_on=right_time_col,
                direction='backward',
                allow_exact_matches=include_same_day  # 含當日；不含當日改 False
            ).drop(columns=[right_time_col])
    
            for c in value_cols:
                merged[f'{out_prefix}{c}'] = merged[c].fillna(0)
                if c in merged.columns:
                    merged.drop(columns=[c], inplace=True)
    
            out.append(merged)
    
        return pd.concat(out, ignore_index=True)
    
    # 1) 以「拜訪日」為基準的累積（得到 asof_cum_準客戶 / asof_cum_新增保戶）
    visit_df = asof_counts_by_agent(
        visit_df=visit_df,
        counts_df=cust_counts_pivot[['業代','建立時間 年/月/日','cum_準客戶','cum_新增保戶']],
        by_col='業代',
        left_time_col='拜訪時間 年/月/日',
        right_time_col='建立時間 年/月/日',
        value_cols=('cum_準客戶','cum_新增保戶'),
        out_prefix='asof_',               # 會產出 asof_cum_準客戶 / asof_cum_新增保戶
        include_same_day=True             # 拜訪日含當日
    )
    
    # 2) 建立「六個月前」時間，對齊 6 個月前的累積（得到 asof6m_cum_*）
    visit_df['_six_months_ago'] = visit_df['拜訪時間 年/月/日'] - pd.DateOffset(months=6)
    
    visit_df = asof_counts_by_agent(
        visit_df=visit_df,
        counts_df=cust_counts_pivot[['業代','建立時間 年/月/日','cum_準客戶','cum_新增保戶']],
        by_col='業代',
        left_time_col='_six_months_ago',  # 以六個月前為基準對齊
        right_time_col='建立時間 年/月/日',
        value_cols=('cum_準客戶','cum_新增保戶'),
        out_prefix='asof6m_',             # 會產出 asof6m_cum_準客戶 / asof6m_cum_新增保戶
        include_same_day=False            # 視窗起點要「排除」當天，建議 False
    )
    
    # 3) 差分 → 近半年新增（覆蓋原欄位名）
    visit_df['上半年準客戶數'] = (
        visit_df['asof_cum_準客戶'] - visit_df['asof6m_cum_準客戶']
    ).fillna(0).astype(int)
    
    visit_df['上半年新增保戶數'] = (
        visit_df['asof_cum_新增保戶'] - visit_df['asof6m_cum_新增保戶']
    ).fillna(0).astype(int)
    
    # 清理暫存欄位
    visit_df.drop(columns=['_six_months_ago'], inplace=True, errors='ignore')
    
    
    # ----------------------------------------------------
    # D) 原流程其餘合併（INFO / 個人化標籤 / 晉升距離等）
    # ----------------------------------------------------
    # INFO 客戶資料
    # --- 1) 從 info_df 精簡出每位客戶一筆：被保人/要保人 性別與生日 ---

    # 先確認欄位為 datetime
    for col in ['被保人生日 年/月/日', '要保人生日 年/月/日']:
        if col in info_df.columns:
            info_df[col] = pd.to_datetime(info_df[col], errors='coerce')
    
    # 性別如果是字串，轉成 0/1（男=0, 女=1）
    sex_map = {'男': 0, '女': 1, 0: 0, 1: 1}
    for col in ['被保人性別', '要保人性別']:
        if col in info_df.columns:
            info_df[col] = info_df[col].map(sex_map).astype('float').astype('Int64')
    
    # 針對同一客戶，取「最後一個非空值」（避免多筆保單重複）
    def last_valid(s):
        s = s.dropna()
        return s.iloc[-1] if len(s) else pd.NA
    
    info_core = (info_df
        .groupby('經紀人1-被保人CRM UUID', as_index=False)
        .agg({
            '被保人性別': last_valid,
            '被保人生日 年/月/日': last_valid,
            '要保人性別': last_valid,
            '要保人生日 年/月/日': last_valid,
        })
        .rename(columns={'經紀人1-被保人CRM UUID': '客戶UUID'})
    )
    
    # --- 2) 合併到 visit_df（不會放大列數） ---
    visit_df = visit_df.merge(info_core, on='客戶UUID', how='left')
    
    # # 確保拜訪日是 datetime
    # visit_df['拜訪時間 年/月/日'] = pd.to_datetime(visit_df['拜訪時間 年/月/日'], errors='coerce')
    
    # --- 3) 計算要保人在「拜訪日」的年齡，並與業務年齡/性別比較 ---
    # def calc_age_years(birth, ref):
    #     if pd.isna(birth) or pd.isna(ref):
    #         return pd.NA
    #     return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))
    
    # 要保人當下年齡（以拜訪日為基準）
    visit_df['要保人年齡_拜訪日'] = visit_df.apply(
        lambda r: calc_age_years(r['要保人生日 年/月/日'], r['拜訪時間 年/月/日']),
        axis=1
    )
    
    # 被保人年齡
    # visit_df['被保人年齡_拜訪日'] = visit_df.apply(
    #     lambda r: calc_age_years(r['被保人生日 年/月/日'], r['拜訪時間 年/月/日']),
    #     axis=1
    # )
    
    # 與業務比較（你前面已經有「截至該月業務年齡」與「業務性別」等欄位）
    # 若你是用「截至該月業務年齡」，直接拿來比較；若另有「截至拜訪日業務年齡」，改用那欄
    age_col_agent = '截至該月業務年齡' if '截至該月業務年齡' in visit_df.columns else '業務目前年齡'
    
    visit_df['業務要保人年齡差'] = visit_df['要保人年齡_拜訪日'] - visit_df[age_col_agent]
    
    # def combine_sex(agent_sex, cust_sex):
    #     # agent_sex / cust_sex 預期為 0/1（男/女）；回傳：
    #     # 0: 同為男、1: 同為女、2: 業務男-要保女、3: 業務女-要保男
    #     if pd.isna(agent_sex) or pd.isna(cust_sex):
    #         return pd.NA
    #     if agent_sex == cust_sex:
    #         return agent_sex  # 0 或 1
    #     return 2 if (agent_sex == 0 and cust_sex == 1) else 3
    
    # visit_df['業務要保人性別組合'] = visit_df.apply(
    #     lambda r: combine_sex(r.get('業務性別', pd.NA), r.get('要保人性別', pd.NA)),
    #     axis=1
    # )
    
    # === 新的性別組合 one-hot 特徵處理 ===
    def combine_sex_onehot(agent_sex, cust_sex):
        # 回傳四種情境的 key
        if pd.isna(agent_sex) or pd.isna(cust_sex):
            return pd.NA
        if agent_sex == 0 and cust_sex == 0:
            return "業務男_要保男"
        elif agent_sex == 1 and cust_sex == 1:
            return "業務女_要保女"
        elif agent_sex == 0 and cust_sex == 1:
            return "業務男_要保女"
        elif agent_sex == 1 and cust_sex == 0:
            return "業務女_要保男"
        return pd.NA
    
    # 建立組合欄位
    visit_df['業務要保人性別組合'] = visit_df.apply(
        lambda r: combine_sex_onehot(r.get('業務性別', pd.NA), r.get('要保人性別', pd.NA)),
        axis=1
    )
    
    # 轉換成 one-hot
    visit_df = pd.get_dummies(visit_df, columns=['業務要保人性別組合'], prefix='', prefix_sep='')

    
    # 被保人的性別組合 / 年齡差
    # visit_df['業務被保人年齡差'] = visit_df[age_col_agent] - visit_df['被保人年齡_拜訪日']
    # visit_df['業務被保人性別組合'] = visit_df.apply(
    #     lambda r: combine_sex(r.get('業務性別', pd.NA), r.get('被保人性別', pd.NA)),
    #     axis=1
    # )
    
    # 個人化標籤
    visit_df = visit_df.merge(label_df, on='客戶UUID', how='left')
    
    # # 備註字數
    # visit_df['備註字數'] = visit_df['拜訪備註_文字'].apply(lambda x: len(str(x).replace(" ", "").replace("\n", "")))
    
    
    # 平均拜訪間隔天數（保留）
    visit_df = visit_df.sort_values(['客戶UUID', '拜訪時間 年/月/日'])
    def avg_visit_interval(x):
        if len(x) < 2:
            return 0
        diffs = x.sort_values().diff().dropna().dt.days
        return diffs.mean()
    interval_summary = (
        visit_df.groupby(['業代', '客戶UUID'])['拜訪時間 年/月/日']
        .apply(avg_visit_interval)
        .reset_index(name='平均拜訪間隔天數')
    )
    visit_df = visit_df.merge(interval_summary, on=['業代', '客戶UUID'], how='left')
    
    # 每週平均拜訪客戶數（保留）
    visit_df['週'] = visit_df['拜訪時間 年/月/日'].dt.to_period('W').astype(str)
    weekly = visit_df.groupby(['業代', '週'])['客戶UUID'].nunique().reset_index()
    weekly_summary = weekly.groupby('業代')['客戶UUID'].mean().reset_index()
    weekly_summary.columns = ['業代', '每週平均拜訪客戶數']
    visit_df = visit_df.merge(weekly_summary, on='業代', how='left')
    
    # === 最近晉升日 / 距離最近晉升天數（沿用你現有作法） ===
    promotion_df['晉升日_dt'] = pd.to_datetime(promotion_df['晉升日_dt'])
    promotion_lookup = promotion_df.groupby('業代')['晉升日_dt'].apply(list).to_dict()
    
    def find_nearest_promotion(row):
        agt = row['業代']
        vdt = row['拜訪時間 年/月/日']
        if agt not in promotion_lookup or pd.isna(vdt):
            return pd.Series([pd.NaT, None])
        dates = [d for d in promotion_lookup[agt] if pd.notna(d)]
        if not dates:
            return pd.Series([pd.NaT, None])
        deltas = [abs((d - vdt).days) for d in dates]
        idx = int(np.argmin(deltas))
        closest = dates[idx]
        return pd.Series([closest, (closest - vdt).days])
    
    visit_df[['最近晉升日_dt', '距離最近晉升天數']] = visit_df.apply(find_nearest_promotion, axis=1)

    
    

    # # 合併 AGENT
    # visit_df = visit_df.merge(agent_df, on='業代', how='left')
    # visit_df['拜訪時間 年/月/日'] = pd.to_datetime(visit_df['拜訪時間 年/月/日'], errors='coerce')
    # # visit_df['距離晉升天數'] = (visit_df['晉升日_dt'] - visit_df['拜訪時間 年/月/日']).dt.days
    # # === 處理 promotion_df，計算「距離最近晉升天數」 ===
    # # 確保格式正確
    # promotion_df['晉升日_dt'] = pd.to_datetime(promotion_df['晉升日_dt'])
    
    # # 建立 lookup：每個業代對應的所有晉升日 list
    # promotion_lookup = promotion_df.groupby('業代')['晉升日_dt'].apply(list).to_dict()
    
    # # # 對每筆拜訪，計算距離最近的晉升日（可以為正/負）
    # # def find_nearest_promotion(row):
    # #     agent = row['業代']
    # #     visit_date = row['拜訪時間 年/月/日']
    # #     if agent not in promotion_lookup:
    # #         return pd.Series([pd.NaT, None])
    
    # #     dates = promotion_lookup[agent]
    # #     deltas = [abs((d - visit_date).days) for d in dates]
    # #     closest_idx = deltas.index(min(deltas))
    # #     closest_date = dates[closest_idx]
    # #     distance = (closest_date - visit_date).days  # 可為正負
    # #     return pd.Series([closest_date, distance])
    
    # # # 套用到 visit_df
    # # visit_df[['最近晉升日_dt', '距離最近晉升天數']] = visit_df.apply(find_nearest_promotion, axis=1)
    # # visit_df['距離最近晉升天數'] = visit_df['距離最近晉升天數'].fillna(0)
    
    # # # 合併 MEMBER 增員數
    # # visit_df = visit_df.merge(member_df, on='業代', how='left')
    # # visit_df['當年度增員數'] = visit_df['當年度增員數'].fillna(0).astype(int)

    # # 合併 CUSTOMER 客戶統計
    # visit_df = visit_df.merge(customer_df, on='業代', how='left')
    # for col in ['上半年準客戶數', '上半年新增保戶數']:
    #     if col in visit_df.columns:
    #         visit_df[col] = visit_df[col].fillna(0).astype(int)

    # # 合併 INFO 客戶資訊
    # visit_df = visit_df.merge(info_df, how='left', left_on='客戶UUID', right_on='經紀人1-被保人CRM UUID')
    # visit_df = visit_df.drop(columns=['經紀人1-被保人CRM UUID'], errors='ignore')
    
    # # 合併 TAGS_LAB 個人化標籤資料
    # visit_df = visit_df.merge(label_df, on='客戶UUID', how='left')

    # # 合併後處理
    # visit_df['備註字數'] = visit_df['拜訪備註_文字'].apply(lambda x: len(str(x).replace(" ", "").replace("\n", "")))
    
    
    
    # # 衍生欄位計算
    # def compute_gender_diff_v2(row):
    #     if row['業務性別'] == row['被保人性別']:
    #         return row['業務性別']  # 0 or 1
    #     elif row['業務性別'] == 0 and row['被保人性別'] == 1:
    #         return 2
    #     elif row['業務性別'] == 1 and row['被保人性別'] == 0:
    #         return 3
    #     else:
    #         return pd.NA

    # visit_df['業務客戶性別組合'] = visit_df.apply(compute_gender_diff_v2, axis=1)
    # visit_df['業務客戶年齡差距'] = visit_df['要保人目前年齡'] - visit_df['業務目前年齡']

    # # 平均拜訪間隔天數
    # visit_df = visit_df.sort_values(['客戶UUID', '拜訪時間 年/月/日'])
    
    # def avg_visit_interval(x):
    #     if len(x) < 2:
    #         return 0
    #     x_sorted = x.sort_values()
    #     diffs = x_sorted.diff().dropna().dt.days
    #     return diffs.mean()

    # interval_summary = visit_df.groupby(['業代', '客戶UUID'])['拜訪時間 年/月/日'].apply(avg_visit_interval).reset_index(name='平均拜訪間隔天數')
    # visit_df = visit_df.merge(interval_summary, on=['業代', '客戶UUID'], how='left')


    # # 每週平均拜訪客戶數
    # visit_df['週'] = visit_df['拜訪時間 年/月/日'].dt.to_period('W').astype(str)
    # weekly = visit_df.groupby(['業代', '週'])['客戶UUID'].nunique().reset_index()
    # weekly_summary = weekly.groupby('業代')['客戶UUID'].mean().reset_index()
    # weekly_summary.columns = ['業代', '每週平均拜訪客戶數']
    # visit_df = visit_df.merge(weekly_summary, on='業代', how='left')

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
    
    # # 過濾必要欄位
    # df_valid = visit_df.dropna(subset=[
    #     '要保人性別', '被保人目前年齡', '要保人目前年齡'
    # ])
    df_valid = visit_df.dropna(subset=['要保人性別'])
    
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
    
        return pd.Series([(r['投保日'] - visit_time).days, r['投保日'], r['是否為網路投保']])
    
    # 套用到每一列
    df_valid[['拜訪與投保日天數差', '最近投保日', '最近是否為網路投保']] = df_valid.apply(
        lambda row: get_nearest_policy_info(row['客戶UUID'], row['拜訪時間']),
        axis=1
    )

    return df_valid, policy_df


# # ✅ 執行
# if __name__ == '__main__':
#     file = "D:/備註文字探勘/repeater/新資料_0731.xlsx"
#     df_ready, policy_df = prepare_model_dataset(file)
#     print("✅ 資料整合與欄位齊備，筆數：", len(df_ready))
    
