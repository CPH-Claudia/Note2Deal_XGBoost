# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:41:32 2025

@author: Z01788
"""
def train_model_pipeline(df_ready, policy_df=None, save_dir="D:/備註文字探勘/models"):
    import os
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    from gensim.models import Word2Vec
    from sklearn.feature_extraction.text import TfidfVectorizer
    from xgboost import XGBClassifier
    import joblib
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import shap
    import json
    
    # # Step 0: 篩選有效樣本
    # df_ready = df_ready[df_ready['拜訪目的'].notna()]
    # df_ready = df_ready[df_ready['備註文字_處理'].str.strip() != '']

    # Step 1: 建立成交標籤
    df_model = df_ready[df_ready['平均每客戶拜訪次數'] > 4].copy()
    df_model["label"] = df_model["拜訪與投保日天數差"].apply(lambda x: 1 if pd.notna(x) and x <= 30 else 0)
    y = df_model["label"]

    # Step 2: 數值特徵
    numerical_cols = [
        '業務客戶性別組合', '最新職級', '拜訪目的', 
        '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', 
        '備註字數', '有意義詞數', 
        '目前年資', '營業單位_編碼', 
        '上半年準客戶數', '今年度活動參與率', '上年度FYC', '距離晉升天數', 
        '件數', '總保費'
    ]
    X_num = df_model[numerical_cols].fillna(0)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Step 3: Word2Vec + TF-IDF 加權向量
    def identity(x): return x

    token_lists = df_model['備註文字_處理'].dropna().apply(lambda x: x.split()).tolist()
    w2v_model = Word2Vec(sentences=token_lists, vector_size=100, window=5, min_count=2)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=identity, token_pattern=None)
    tfidf_vectorizer.fit(token_lists)
    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

    def vectorize_sentence_weighted(sentence):
        vecs, weights = [], []
        for word in sentence:
            if word in w2v_model.wv and word in tfidf_dict:
                vecs.append(w2v_model.wv[word] * tfidf_dict[word])
                weights.append(tfidf_dict[word])
        return np.sum(vecs, axis=0) / np.sum(weights) if vecs else np.zeros(w2v_model.vector_size)

    X_w2v_weighted = np.array([vectorize_sentence_weighted(s.split()) for s in df_model['備註文字_處理']])

    # Step 4: 選擇 Word2Vec Top 10 特徵
    X_all = np.hstack([X_w2v_weighted, X_num_scaled])
    model_init = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_init.fit(X_all, y)
    w2v_importances = model_init.feature_importances_[:X_w2v_weighted.shape[1]]
    top_k = 10
    w2v_top_indices = np.argsort(w2v_importances)[::-1][:top_k]
    X_w2v_top = X_w2v_weighted[:, w2v_top_indices]
    w2v_top_feature_names = [f'w2v_{i}' for i in w2v_top_indices]

    # Step 5: 合併全部特徵
    X_combined = np.hstack([X_w2v_top, X_num_scaled])
    final_feature_names = w2v_top_feature_names + numerical_cols

    # Step 6: 交叉驗證評估模型
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_scores, pr_scores = [], []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    for train_idx, val_idx in skf.split(X_combined, y):
        X_train, X_val = X_combined[train_idx], X_combined[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.6).astype(int)

        roc_scores.append(roc_auc_score(y_val, y_proba))
        pr_scores.append(average_precision_score(y_val, y_proba))
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    print("\n📊 Cross-Validation Results:")
    print(classification_report(all_y_true, all_y_pred))
    print(f"Average ROC AUC: {np.mean(roc_scores):.4f}")
    print(f"Average PR AUC : {np.mean(pr_scores):.4f}")

    # Step 7: 儲存模型與相關物件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, os.path.join(save_dir, f"model_final_{timestamp}.pkl"))
    joblib.dump(w2v_model, os.path.join(save_dir, f"word2vec_model_{timestamp}.pkl"))
    joblib.dump(tfidf_vectorizer, os.path.join(save_dir, f"tfidf_vectorizer_{timestamp}.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, f"scaler_{timestamp}.pkl"))
    with open(os.path.join(save_dir, "final_feature_names.json"), "w", encoding="utf-8") as f: 
        json.dump(final_feature_names, f, ensure_ascii=False, indent=2)

    print(f"✅ 模型與向量器已儲存（時間戳記：{timestamp}）")

    # Step 8: 匯出預測機率與斷詞的長格式
    df_model = df_model.reset_index(drop=True)
    y_proba_final = model.predict_proba(X_combined)[:, 1]

    records = []
    for idx, (uuid, text, proba, label) in enumerate(zip(
        df_model.get('客戶UUID', df_model.index),
        df_model['備註文字_處理'],
        y_proba_final,
        df_model['label']
    )):
        if pd.isna(text): continue
        for word in text.split():
            records.append({
                "客戶UUID": uuid,
                "斷詞": word,
                "預測機率": proba,
                "是否成交": label
            })

    wordcloud_df = pd.DataFrame(records)
    
    # Step 9: SHAP 解釋與區間輸出
    summary_list = []
    try:
        explainer = shap.Explainer(model, X_combined, feature_names=final_feature_names)
        shap_values = explainer(X_combined)

        mean_dict = dict(zip(numerical_cols, scaler.mean_))
        scale_dict = dict(zip(numerical_cols, scaler.scale_))

        for i, var in enumerate(numerical_cols):
            x = X_num_scaled[:, i]
            shap_val = shap_values[:, len(w2v_top_feature_names) + i].values
            df = pd.DataFrame({"值": x, "SHAP": shap_val}).sort_values("值").reset_index(drop=True)
            df["SHAP_smooth"] = df["SHAP"].rolling(window=20, min_periods=1).mean()
            df["is_neg"] = (df["SHAP_smooth"] < 0).astype(int)
            df["neg_group"] = (df["is_neg"].diff(1) != 0).cumsum()
            neg_ranges = df[df["is_neg"] == 1].groupby("neg_group")["值"].agg(["min", "max"]).values
            value_min, value_max = df["值"].min(), df["值"].max()
            positive_ranges = []
            current_start = value_min
            for lo, hi in neg_ranges:
                if current_start < lo:
                    positive_ranges.append((current_start, lo))
                current_start = max(current_start, hi)
            if current_start < value_max:
                positive_ranges.append((current_start, value_max))
            for lo, hi in positive_ranges:
                lo_raw = lo * scale_dict[var] + mean_dict[var]
                hi_raw = hi * scale_dict[var] + mean_dict[var]
                summary_list.append({
                    "變數": var,
                    "原始值區間_起": round(lo_raw, 2),
                    "原始值區間_迄": round(hi_raw, 2),
                    "標準化值_起": round(lo, 2),
                    "標準化值_迄": round(hi, 2)
                })
    except Exception as e:
        print(f"❌ SHAP 分析失敗：{e}")

    from pandas import ExcelWriter
    output_path = os.path.join("D:/備註文字探勘/results", f"retrain_output_{timestamp}.xlsx")
    with ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_model.assign(預測機率=y_proba_final).to_excel(writer, index=False, sheet_name="ModelResults")
        wordcloud_df.to_excel(writer, index=False, sheet_name="WordCloud")
        pd.DataFrame(summary_list).to_excel(writer, sheet_name="SHAP貢獻區間", index=False)

    print(f"✅ 資料已儲出至：{output_path}")
    
    # Step 10: 更新 latest.json
    latest_info = {
        "model": f"model_final_{timestamp}.pkl",
        "word2vec": f"word2vec_model_{timestamp}.pkl",
        "tfidf": f"tfidf_vectorizer_{timestamp}.pkl",
        "scaler": f"scaler_{timestamp}.pkl",
        "reference": f"train_reference_{timestamp}.csv",  # optional
        "timestamp": timestamp
    }
    
    with open(os.path.join(save_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(latest_info, f, ensure_ascii=False, indent=2)

    
def retrain_and_save_model(df_ready, policy_df=None, save_dir="D:/備註文字探勘/models"):
    # 執行整個訓練與儲存流程（內部已儲存模型與結果）
    model = train_model_pipeline(df_ready, policy_df, save_dir=save_dir)
    print("✅ retrain_and_save_model 完成執行！")
    return model
