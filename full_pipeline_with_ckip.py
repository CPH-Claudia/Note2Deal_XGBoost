# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 17:44:13 2025

@author: Z01788
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# === 📂 斷詞 + 清理 + 預測自動化流程 ===

import pandas as pd
import numpy as np
import pickle
from ckiptagger import WS
import os
from datetime import datetime

# === 初始化斷詞工具 ===
ws = WS("./data")  # 指向 CKIP model 資料夾

# === 斷詞 + 清理流程 ===
def full_pipeline_with_ckip(file_path):
    # === Step 1: 讀取 VISIT 備註欄位進行斷詞 ===
    visit_raw = pd.read_excel(file_path, sheet_name="VISIT")
    visit_raw['拜訪時間 年/月/日'] = pd.to_datetime(visit_raw['拜訪時間 年/月/日'], errors='coerce')

    def extract_non_sharp_text(note):
        if pd.isna(note): return ''
        lines = str(note).replace('_x000D_', '\n').replace('\r', '').splitlines()
        return '\n'.join([
            line.strip() for line in lines
            if line.strip() and not (line.startswith('#') or line.startswith('＃'))
        ])

    visit_raw['拜訪備註_文字'] = visit_raw['拜訪備註'].apply(extract_non_sharp_text)

    # 斷詞
    text_list = visit_raw['拜訪備註_文字'].fillna('').tolist()
    
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
    
    # 第一步：CKIP 斷詞
    ws_segments = ws(text_list)

    # 第二步：保險術語合併（基於 CKIP 結果後處理）
    word_list = merge_custom_terms(ws_segments, insurance_terms)
    
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
        for sentence in word_list
    ]


    # 存檔
    output_dir = "D:/備註文字探勘"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]  # 取得檔名不含副檔名
    pkl_path = os.path.join(output_dir, f"repeater/{base_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(filtered_words, f)

    print(f"✅ 斷詞完成，共 {len(filtered_words)} 筆，接續資料整併...")

    # === Step 2: 清理與欄位整合 ===
    from insurance_data_clean import prepare_model_dataset
    df_ready, policy_df = prepare_model_dataset(file_path)
    print("📊 資料整併完成，樣本數：", len(df_ready))

    # === Step 3: 模型預測與輸出 ===
    from insurance_data_clean import prepare_model_dataset
    from predict_model import predict_with_model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(output_dir, f"results/預測_{timestamp}.xlsx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predict_with_model(df_ready, output_path, source_file=file)

    print("✅ 模型預測完成，結果輸出：", output_path)
    return df_ready, policy_df


# ✅ 範例執行
if __name__ == '__main__':
    file = "D:/備註文字探勘/repeater/拜訪_TEST.xlsx"
    df_ready, policy_df = full_pipeline_with_ckip(file)
    print("✅ 全流程完成，可用於預測，樣本數：", len(df_ready))