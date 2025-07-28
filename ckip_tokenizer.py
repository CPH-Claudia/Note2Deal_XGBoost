# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:49:28 2025

@author: Z01788
"""

# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from ckiptagger import WS
from opencc import OpenCC
import requests

file_path = "D:/備註文字探勘/repeater/拜訪_TEST.xlsx"
ws = WS("./data")  # CKIP model path

# 讀取備註欄位
visit_df = pd.read_excel(file_path, sheet_name="VISIT")
def clean_note(note):
    if pd.isna(note): return ''
    lines = str(note).replace('_x000D_', '\n').replace('\r', '').splitlines()
    return '\n'.join([line.strip() for line in lines if line.strip() and not (line.startswith('#') or line.startswith('＃'))])
visit_df['拜訪備註_文字'] = visit_df['拜訪備註'].apply(clean_note)
text_list = visit_df['拜訪備註_文字'].fillna('').tolist()

# 斷詞
ws_result = ws(text_list)

# 合併保險術語略
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

# 載入停用詞並繁體轉換
stopword_urls = [
    "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/scu_stopwords.txt"
]
cc = OpenCC('s2t')
stopwords_trad = set()
for url in stopword_urls:
    res = requests.get(url)
    if res.status_code == 200:
        stopwords_trad.update({cc.convert(line.strip()) for line in res.text.splitlines() if line.strip()})

filtered_words = [
    [word for word in sentence if word not in stopwords_trad and len(word) > 1]
    for sentence in word_list
]

# 儲存斷詞結果為 .pkl（與 Excel 同名）
pkl_path = file_path.replace(".xlsx", ".pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(filtered_words, f)

print(f"✅ CKIP 斷詞完成，共 {len(filtered_words)} 筆，儲存至：{pkl_path}")
