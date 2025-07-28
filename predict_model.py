# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 09:34:53 2025

@author: Z01788
"""

import pickle
import os
import dill
import json
from gensim.models import Word2Vec
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

def load_models(model_dir="D:/備註文字探勘/models"):
    with open(os.path.join(model_dir, "latest.json"), encoding="utf-8") as f:
        latest = json.load(f)
    model = joblib.load(os.path.join(model_dir, latest["model"]))
    w2v_model = joblib.load(os.path.join(model_dir, latest["word2vec"]))
    tfidf = joblib.load(os.path.join(model_dir, latest["tfidf"]))
    scaler = joblib.load(os.path.join(model_dir, latest["scaler"]))
    features = joblib.load(os.path.join(model_dir, latest["features"]))
    print(f"✅ 已載入模型版本：{latest['timestamp']}")
    return model, w2v_model, tfidf, scaler, features


# def load_models(model_dir="D:/備註文字探勘/models"):
#     latest_path = os.path.join(model_dir, "latest.json")
#     if not os.path.exists(latest_path):
#         raise FileNotFoundError("❌ 找不到 latest.json，請先執行模型訓練。")

#     with open(latest_path, "r", encoding="utf-8") as f:
#         latest = json.load(f)

#     model = joblib.load(os.path.join(model_dir, latest["model"]))
#     w2v_model = joblib.load(os.path.join(model_dir, latest["word2vec"]))
#     tfidf = joblib.load(os.path.join(model_dir, latest["tfidf"]))
#     scaler = joblib.load(os.path.join(model_dir, latest["scaler"]))
#     with open(os.path.join(model_dir, "final_feature_names.json"), "r", encoding="utf-8") as f: 
#         final_feature_names = json.load(f)
    
    
#     print(f"✅ 已載入模型版本：{latest['timestamp']}")
#     return model, w2v_model, tfidf, scaler, final_feature_names


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
    model, w2v_model, tfidf_vectorizer, scaler, features = load_models()
    
    numerical_cols = [f for f in features if not f.startswith("w2v_")]
    w2v_indices = [int(f.split("_")[1]) for f in features if f.startswith("w2v_")]

    X_num = df_ready[numerical_cols].fillna(0)
    X_scaled = scaler.transform(X_num)

    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    def vectorize_sentence_weighted(sentence):
        vecs, weights = [], []
        for word in sentence:
            if word in w2v_model.wv and word in tfidf_dict:
                vecs.append(w2v_model.wv[word] * tfidf_dict[word])
                weights.append(tfidf_dict[word])
        return np.sum(vecs, axis=0) / np.sum(weights) if vecs else np.zeros(w2v_model.vector_size)

    tokens_list = df_ready['備註文字_處理'].dropna().apply(lambda x: x.split()).tolist()
    X_w2v_all = np.array([vectorize_sentence_weighted(x) for x in tokens_list])
    X_w2v_top = X_w2v_all[:, w2v_indices]
    X_final = np.hstack([X_w2v_top, X_scaled])

    y_pred_proba = model.predict_proba(X_final)[:, 1]
    y_pred = (y_pred_proba >= 0.6).astype(int)
    df_ready['預測成交機率'] = y_pred_proba
    df_ready['預測成交與否'] = y_pred

    # # 自動補上 label（與訓練時一致：拜訪與投保日天數差 <= 30 為成交）
    # if "label" not in df_ready.columns and "拜訪與投保日天數差" in df_ready.columns:
    #     df_ready["label"] = df_ready["拜訪與投保日天數差"].apply(lambda x: 1 if pd.notna(x) and x <= 30 else 0)

    # # 移除備註空值或斷詞為空的筆數
    # df_ready = df_ready[
    #     df_ready['備註文字_處理'].notna() & 
    #     (df_ready['備註文字_處理'].str.strip() != '')
    # ]
    
    # # 數值特徵
    # numerical_cols = [
    #     '業務客戶性別組合', '最新職級', '拜訪目的', 
    #     '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', # '拜訪紀錄密度', 
    #     '備註字數', '有意義詞數', 
    #     '目前年資', '營業單位_編碼', # '當年度賽季增員數', '加前一賽季增員數', '最新職級', 
    #     '上半年準客戶數', '今年度活動參與率', '上年度FYC', '距離晉升天數', 
    #     '件數', '總保費' # '業務客戶性別組合', 
    # ]
    # X_num = df_ready[numerical_cols].fillna(0)
    # X_scaled = scaler.transform(X_num)

    # # TF-IDF + Word2Vec 向量
    # def vectorize_sentence_weighted(sentence):
    #     tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    #     vecs, weights = [], []
    #     for word in sentence:
    #         if word in w2v_model.wv and word in tfidf_dict:
    #             vecs.append(w2v_model.wv[word] * tfidf_dict[word])
    #             weights.append(tfidf_dict[word])
    #     return np.sum(vecs, axis=0) / np.sum(weights) if vecs else np.zeros(w2v_model.vector_size)

    # tokens_list = df_ready['拜訪備註_詞語']
    # X_w2v = np.array([vectorize_sentence_weighted(x) for x in tokens_list])
    
    # w2v_feature_names = [f"w2v_{i}" for i in range(10)]
    # num_feature_names = numerical_cols
    # X_combined_df = pd.DataFrame(np.hstack([X_w2v[:, :10], X_scaled]), columns=w2v_feature_names + num_feature_names)
    # X_combined_df = X_combined_df[final_feature_names]  # 根據訓練時的順序重新排列
    # X_combined = X_combined_df.values


    # # 合併
    # # X_combined = np.hstack([X_scaled, X_w2v[:, :10]])
    # y_pred_proba = model.predict_proba(X_combined)[:, 1]
    # df_ready['預測成交機率'] = y_pred_proba
    df_ready['潛力分類'] = df_ready['預測成交機率'].apply(classify_probability)
    # df_ready['預測成交與否'] = (df_ready['預測成交機率'] >= 0.6).astype(int)

    # 評估報告產生
    def generate_model_report(y_true, y_pred, y_prob):
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        report_df["ROC AUC"] = roc_auc_score(y_true, y_prob)
        report_df["PR AUC"] = average_precision_score(y_true, y_prob)
        return report_df

    # ✅ 儲存為 Excel：包含預測結果 + 模型評估
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df_ready.to_excel(writer, index=False, sheet_name="預測結果")

        if "label" in df_ready.columns:
            eval_df = generate_model_report(
                df_ready["label"],
                df_ready["預測成交與否"],
                df_ready["預測成交機率"]
            )
            eval_df.to_excel(writer, sheet_name="模型評估")
            print("✅ 評估完成，結果已寫入 Excel：", output_path)
        else:
            print("⚠️ 無法執行模型評估：缺少 label 欄位")

    print(f"✅ 預測完成，已儲存至：{output_path}")

    # 可選的額外評估（如你有 evaluate_predictions 函數）
    if source_file is not None:
        policy_df = pd.read_excel(source_file, sheet_name="POLICY")
        evaluate_predictions(results_path=output_path, policy_df=policy_df, threshold=0.6)