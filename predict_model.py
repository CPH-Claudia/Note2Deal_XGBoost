# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 09:34:53 2025

@author: Z01788
"""

import pickle
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def load_models():
    with open("D:/備註文字探勘/models/xgb_model_final.pkl", 'rb') as f: model = pickle.load(f)
    with open("D:/備註文字探勘/models/word2vec_model.pkl", 'rb') as f: w2v = pickle.load(f)
    with open("D:/備註文字探勘/models/tfidf_vectorizer.pkl", 'rb') as f: tfidf = dill.load(f, encoding='latin1')
    with open("D:/備註文字探勘/models/scaler.pkl", 'rb') as f: scaler = pickle.load(f)
    return model, w2v, tfidf, scaler

def classify_probability(p):
    if p >= 0.9: return "極高潛力"
    elif p >= 0.75: return "高潛力"
    elif p >= 0.5: return "中潛力"
    elif p >= 0.25: return "低潛力"
    else: return "極低潛力"
    
def check_success_within_30_days(row, policy_df):
    uuid = row['客戶UUID']
    visit_date = row['拜訪時間 年/月/日']
    
    policies = policy_df[policy_df['經紀人1-被保人CRM UUID'] == uuid]
    if policies.empty:
        return 0
    
    matched = policies[
        (policies['投保日 年/月/日'] > visit_date) &
        (policies['投保日 年/月/日'] <= visit_date + pd.Timedelta(days=30))
    ]
    return 1 if not matched.empty else 0

def evaluate_predictions(results_path, policy_df, threshold=0.6):
    # 讀取預測結果 Excel
    results_df = pd.read_excel(results_path)
    
    # 時間欄位轉型
    results_df['拜訪時間 年/月/日'] = pd.to_datetime(results_df['拜訪時間 年/月/日'], errors='coerce')
    policy_df['投保日 年/月/日'] = pd.to_datetime(policy_df['投保日 年/月/日'], errors='coerce')
    
    # 實際成交標籤
    results_df['實際是否成交'] = results_df.apply(lambda row: check_success_within_30_days(row, policy_df), axis=1)
    
    # 預測標籤
    results_df['預測是否成交'] = results_df['預測成交機率'].apply(lambda x: 1 if x >= threshold else 0)
    
    # 混淆矩陣與分類報告
    cm = confusion_matrix(results_df['實際是否成交'], results_df['預測是否成交'])
    report = classification_report(results_df['實際是否成交'], results_df['預測是否成交'], output_dict=True)
    
    # 儲存回 Excel
    with pd.ExcelWriter(results_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_df.to_excel(writer, sheet_name='預測結果_已驗證', index=False)
        
        # 建立混淆矩陣 DataFrame
        cm_df = pd.DataFrame(cm, index=["實際:未成交", "實際:成交"], columns=["預測:未成交", "預測:成交"])
        
        # 建立 classification_report DataFrame
        cr_df = pd.DataFrame(report).T
        
        # 寫入報表
        start_row = 0
        cm_df.to_excel(writer, sheet_name='模型評估', startrow=start_row)
        cr_df.to_excel(writer, sheet_name='模型評估', startrow=start_row + len(cm_df) + 3)

    print(f"✅ 評估完成，結果已寫入：{results_path}")    


def predict_with_model(df_ready, output_path, source_file=None):
    model, w2v_model, tfidf_vectorizer, scaler = load_models()

    # 數值特徵
    numerical_cols = [
        '業務客戶性別組合', '最新職級', '拜訪目的', 
        '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', # '拜訪紀錄密度', 
        '備註字數', '有意義詞數', 
        '目前年資', '營業單位_編碼', # '當年度賽季增員數', '加前一賽季增員數', '最新職級', 
        '上半年準客戶數', '今年度活動參與率', '上年度FYC', '距離晉升天數', 
        '件數', '總保費' # '業務客戶性別組合', 
    ]
    X_num = df_ready[numerical_cols].fillna(0)
    X_scaled = scaler.transform(X_num)

    # TF-IDF + Word2Vec 向量
    def vectorize_sentence_weighted(sentence):
        tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
        vecs, weights = [], []
        for word in sentence:
            if word in w2v_model.wv and word in tfidf_dict:
                vecs.append(w2v_model.wv[word] * tfidf_dict[word])
                weights.append(tfidf_dict[word])
        return np.sum(vecs, axis=0) / np.sum(weights) if vecs else np.zeros(w2v_model.vector_size)

    tokens_list = df_ready['拜訪備註_詞語']
    X_w2v = np.array([vectorize_sentence_weighted(x) for x in tokens_list])

    # 組合後預測
    X_combined = np.hstack([X_scaled, X_w2v[:, :10]])  # 若你只選用 Top10 維度
    y_pred_proba = model.predict_proba(X_combined)[:, 1]

    df_ready['預測成交機率'] = y_pred_proba
    df_ready['潛力分類'] = df_ready['預測成交機率'].apply(classify_probability)

    df_ready.to_excel(output_path, index=False)
    print(f"✅ 預測完成，已儲存至：{output_path}")
    
    if source_file is not None:
        policy_df = pd.read_excel(source_file, sheet_name="POLICY")
        evaluate_predictions(results_path=output_path, policy_df=policy_df, threshold=0.6)
    