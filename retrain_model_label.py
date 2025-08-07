# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:04:21 2025

@author: Z01788
"""

import pandas as pd
import numpy as np
import shap
import os
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from xgboost import XGBClassifier
from pandas import ExcelWriter
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model_pipeline_with_strategies(df_ready, policy_df=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "D:/備註文字探勘/results"
    os.makedirs(save_dir, exist_ok=True)
    
    strategy_ids = [0, 2, 6]
    monitoring_log = []
    shap_log = []
    
    # 建立斷詞語料（用整份 df_ready）
    df_model = df_ready.copy()
    token_lists_all = df_model['備註文字_處理'].dropna().apply(lambda x: x.split()).tolist()
    corpus_text_all = df_model["備註文字_處理"].dropna().apply(lambda x: " ".join(x.split()))
    
    # === Word2Vec 模型訓練 ===
    w2v_model = Word2Vec(sentences=token_lists_all, vector_size=100, window=5, min_count=1, workers=4)
    
    # === TF-IDF 建立 ===
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(corpus_text_all)
    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

    
    for strategy_id in strategy_ids:
        df_model = df_model[df_model['平均每客戶拜訪次數'] > 4].copy()
        
        # === 檢查必要欄位是否存在，若無則補上處理 ===
        # Label: 拜訪與投保日天數差 <= 30 判為成交（1），否則為未成交（0）
        if "label" not in df_model.columns:
            df_model["label"] = df_model["拜訪與投保日天數差"].apply(lambda x: 1 if pd.notna(x) and x >= -7 and x <= 180 else 0)

        # === 備註斷詞向量（Word2Vec + TF-IDF 加權平均）===
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

        # df_model["w2v_vector"] = df_model['備註文字_處理'].apply(lambda x: vectorize_sentence_weighted(x.split()) if pd.notna(x) else np.zeros(100))
        df_model["w2v_vector"] = df_model['備註文字_處理'].apply(
            lambda x: vectorize_sentence_weighted(x.split()) if pd.notna(x) else np.zeros(100)
        )
        df_model["tfidf_weights"] = df_model['備註文字_處理'].apply(lambda x: [tfidf_dict.get(w, 0) for w in x.split()] if pd.notna(x) else [])

        # 數值型特徵
        numerical_cols = [
            '業務客戶性別組合', '最新職級', '拜訪目的',
            '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', 
            '備註字數', '有意義詞數',
            '目前年資', '營業單位_編碼',
            '上半年準客戶數', '最近半年活動參與率', '上一個半年度FYC', '距離最近晉升天數',
            '件數', '總保費', '拜訪序號', '賽季'
        ]
        
        tag_cols = []
        if strategy_id in [1, 2, 5, 6]:
            tag_cols.extend(['個人化標籤_背景', '個人化標籤_銷售'])
        if strategy_id in [3, 4, 5, 6]:
            tag_cols.append('拜訪備註_標籤')
        if strategy_id == 7:
            tag_cols = ['個人化標籤_背景', '個人化標籤_銷售', '拜訪備註_標籤']
        
        def merge_tags(row):
            all_tags = []
            for col in tag_cols:
                if pd.notna(row[col]):
                    parts = str(row[col]).replace("，", ",").replace("、", ",").replace("  ", " ").replace(" ,", ",").split(",")
                    for tag in parts:
                        tag_clean = tag.strip().strip(".").strip("🖊️").lower()
                        if tag_clean:
                            all_tags.append(tag_clean)
            return list(set(all_tags))

        if tag_cols:
            df_model["merged_tags"] = df_model.apply(merge_tags, axis=1)
        else:
            df_model["merged_tags"] = [[] for _ in range(len(df_model))]
        
        if strategy_id in [2, 4, 6, 7]:
            valid_mask = df_model["merged_tags"].apply(lambda x: len(x) > 0)
            df_model = df_model[valid_mask]
        
        print("🔍 merged_tags 非空筆數：", df_model["merged_tags"].apply(lambda x: len(x) > 0).sum())

        # --- 2. 切分 Train / Holdout ---
        df_model = df_model.sample(frac=1, random_state=42).reset_index(drop=True)
        cutoff = int(len(df_model) * 0.8)
        df_train = df_model.iloc[:cutoff].copy()
        df_holdout = df_model.iloc[cutoff:].copy()
    
        # --- 3. 數值與向量特徵準備 ---
        X_num = df_train[numerical_cols].copy()
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
    
        w2v_vectors = df_train["w2v_vector"].to_list()
        X_w2v_weighted = []
        for vecs in w2v_vectors:
            if isinstance(vecs, list) and len(vecs) > 0:
                try:
                    weighted_vec = np.mean(vecs, axis=0)
                except Exception:
                    weighted_vec = np.zeros(100)
            else:
                weighted_vec = np.zeros(100)
            X_w2v_weighted.append(weighted_vec)
        X_w2v_weighted = np.array(X_w2v_weighted)
    
        # Step 4: 使用 XGBoost Feature Importance 選出 Word2Vec Top 10 維度
        X_all = np.hstack([X_w2v_weighted, X_num_scaled])
        y = df_train["label"]
        model_init = XGBClassifier(eval_metric='logloss', random_state=42)
        model_init.fit(X_all, y)
        w2v_importances = model_init.feature_importances_[:X_w2v_weighted.shape[1]]
        top_k = 10
        w2v_top_indices = np.argsort(w2v_importances)[::-1][:top_k]
        # X_w2v_top = X_w2v_weighted[:, w2v_top_indices]
    
        # --- 4. MultiLabelBinarizer for Tags ---
        mlb = MultiLabelBinarizer()
        mlb.fit(df_train["merged_tags"])
        
        # --- 5. Feature Preparation Function ---
        # def prepare_features(df_subset):
        #     X_num = df_subset[numerical_cols].copy()
        #     X_scaled = scaler.transform(X_num)
        #     w2v_vectors = df_subset["w2v_vector"].to_list()
        #     X_w2v_weighted = []
        #     for vecs in w2v_vectors:
        #         if isinstance(vecs, list) and len(vecs) > 0:
        #             try:
        #                 weighted_vec = np.mean(vecs, axis=0)
        #             except Exception:
        #                 weighted_vec = np.zeros(100)
        #         else:
        #             weighted_vec = np.zeros(100)
        #         X_w2v_weighted.append(weighted_vec)
        #     X_w2v_weighted = np.array(X_w2v_weighted)
        #     X_w2v_top_subset = X_w2v_weighted[:, w2v_top_indices]
    
        #     final_features = np.hstack([X_w2v_top_subset, X_scaled])
    
        #     if tag_cols:
        #         tags_trans = mlb.transform(df_subset["merged_tags"])
        #         final_features = np.hstack([final_features, tags_trans])
        #     return final_features
        def prepare_features(df_subset, return_feature_names=False):
            X_num = df_subset[numerical_cols].copy()
            X_scaled = scaler.transform(X_num)
            
            w2v_vectors = df_subset["w2v_vector"].to_list()
            X_w2v_weighted = []
            for vecs in w2v_vectors:
                if isinstance(vecs, list) and len(vecs) > 0:
                    try:
                        weighted_vec = np.mean(vecs, axis=0)
                    except Exception:
                        weighted_vec = np.zeros(100)
                else:
                    weighted_vec = np.zeros(100)
                X_w2v_weighted.append(weighted_vec)
            X_w2v_weighted = np.array(X_w2v_weighted)
            X_w2v_top_subset = X_w2v_weighted[:, w2v_top_indices]
        
            # === 合併所有特徵 ===
            final_features = np.hstack([X_w2v_top_subset, X_scaled])
        
            final_feature_names = [
                f"w2v_{i}" for i in w2v_top_indices
            ] + numerical_cols
        
            if tag_cols:
                tags_trans = mlb.transform(df_subset["merged_tags"])
                final_features = np.hstack([final_features, tags_trans])
                final_feature_names += mlb.classes_.tolist()
        
            if return_feature_names:
                return final_features, final_feature_names
            else:
                return final_features
        
        # --- 6. Train Cross-Validation ---
        X_train_full, final_feature_names = prepare_features(df_train, return_feature_names=True)
        y_train = df_train["label"]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_scores, roc_scores = [], []
        for train_idx, val_idx in skf.split(X_train_full, y_train):
            X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = XGBClassifier(random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            y_val_pred = model.predict_proba(X_val)[:, 1]
            pr_scores.append(average_precision_score(y_val, y_val_pred))
            roc_scores.append(roc_auc_score(y_val, y_val_pred))
    
        print(f"✅ CV 平均 PR AUC: {np.mean(pr_scores):.4f}, ROC AUC: {np.mean(roc_scores):.4f}")
                
        # --- 7. Final Model 訓練 & Hold-out 預測 ---
        model = XGBClassifier(random_state=42, n_jobs=-1)
        model.fit(X_train_full, y_train)
    
        df_train["is_holdout"] = 0
        df_holdout["is_holdout"] = 1
        df_combined = pd.concat([df_train, df_holdout], axis=0)
        X_combined = prepare_features(df_combined)
        y_combined = df_combined["label"]
    
        y_pred_proba = model.predict_proba(X_combined)[:, 1]
        df_combined["pred_prob"] = y_pred_proba
        
        # === 儲存訓練資料參考樣貌供 drift 檢查 ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_dir = os.path.join("D:/備註文字探勘/models", f"{timestamp}_strategy_{strategy_id}")
        os.makedirs(ref_dir, exist_ok=True)
        df_train[numerical_cols].to_csv(os.path.join(ref_dir, "train_reference.csv"), index=False)
        print(f"✅ strategy {strategy_id} 的 train_reference 已匯出至 {ref_dir}")

        
        # ref_dir = os.path.join("D:/備註文字探勘/models", timestamp)
        # os.makedirs(ref_dir, exist_ok=True)
        
        # # selected_cols = [
        # #     '備註字數', '件數', '總保費', '拜訪目的', '業務客戶年齡差距',
        # #     '上半年準客戶數', '今年度活動參與率', '上年度FYC'
        # # ]
        
        # df_train[numerical_cols].to_csv(os.path.join(ref_dir, "train_reference.csv"), index=False)
        # print(f"✅ train_reference 已匯出")

        # Step 7: 儲存模型與相關物件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(model, os.path.join(save_dir, f"model_final_{timestamp}.pkl"))
        joblib.dump(w2v_model, os.path.join(save_dir, f"word2vec_model_{timestamp}.pkl"))
        joblib.dump(tfidf_vectorizer, os.path.join(save_dir, f"tfidf_vectorizer_{timestamp}.pkl"))
        joblib.dump(scaler, os.path.join(save_dir, f"scaler_{timestamp}.pkl"))
        joblib.dump(final_feature_names, os.path.join(save_dir, f"feature_names_{timestamp}.pkl"))
    
        # Step 8: 更新 latest.json
        latest_info = {
            "timestamp": timestamp,
            "model": f"model_final_{timestamp}.pkl",
            "word2vec": f"word2vec_model_{timestamp}.pkl",
            "tfidf": f"tfidf_vectorizer_{timestamp}.pkl",
            "scaler": f"scaler_{timestamp}.pkl",
            "features": f"feature_names_{timestamp}.pkl"
        }
        with open(os.path.join(save_dir, "latest.json"), "w", encoding="utf-8") as f:
            json.dump(latest_info, f, ensure_ascii=False, indent=2)
        print(f"✅ 最新模型檔案資訊已儲存至 latest.json")
    
        # # Step 9: 匯出 train_reference.csv（供資料漂移檢查使用）
        # df_ready.to_csv(os.path.join(save_dir, f"train_reference_{timestamp}.csv"), index=False)
        # df_ready.to_csv(os.path.join(save_dir, "train_reference.csv"), index=False)
        
        # # --- 8. 簡易監控統計 ---
        # train_pos = df_combined[df_combined["is_holdout"] == 0]["label"].sum()
        # train_total = (df_combined["is_holdout"] == 0).sum()
        # holdout_pos = df_combined[df_combined["is_holdout"] == 1]["label"].sum()
        # holdout_total = (df_combined["is_holdout"] == 1).sum()
    
        # print(f"Train 資料: {train_total} 筆，正類佔比: {train_pos / train_total:.2%}")
        # print(f"Holdout 資料: {holdout_total} 筆，正類佔比: {holdout_pos / holdout_total:.2%}")

        # === 匯出 WordCloud 長格式 ===
        wordcloud_records = []
        for uuid, text, proba, label in zip(df_combined['客戶UUID'], df_combined['備註文字_處理'], df_combined["pred_prob"], df_combined['label']):
            if pd.isna(text): continue
            for word in text.split():
                wordcloud_records.append({
                    "客戶UUID": uuid,
                    "斷詞": word,
                    "預測機率": proba,
                    "是否成交": label
                })
        wordcloud_df = pd.DataFrame(wordcloud_records)
    
        # === SHAP 分析（僅策略 0, 2, 6）===
        debug_mode = False  # 若要看 SHAP 失敗原因，改為 True
        summary_list = []
        if strategy_id in [0, 2, 6]:
            try:
                explainer = shap.Explainer(model, X_train_full, feature_names=[f"W2V_{i}" for i in range(top_k)] + numerical_cols + list(mlb.classes_) if tag_cols else [])
                shap_values = explainer(X_train_full)
    
                mean_dict = dict(zip(numerical_cols, scaler.mean_))
                scale_dict = dict(zip(numerical_cols, scaler.scale_))
    
                for i, var in enumerate(numerical_cols):
                    x = X_train_full[:, top_k + i]
                    shap_val = shap_values[:, top_k + i].values
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
                            "strategy_id": strategy_id,
                            "變數": var,
                            "原始值區間_起": round(lo_raw, 2),
                            "原始值區間_迄": round(hi_raw, 2),
                            "標準化值_起": round(lo, 2),
                            "標準化值_迄": round(hi, 2)
                        })
            except Exception as e:
                if debug_mode:
                    print(f"❌ SHAP 分析失敗(strategy {strategy_id})：{e}")
            # except Exception as e:
            #     print(f"❌ SHAP 分析失敗(strategy {strategy_id})：{e}")
    
        # === 儲存成 Excel 檔 ===
        output_path = os.path.join(save_dir, f"model_strategy_{strategy_id}_{timestamp}.xlsx")
        with ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_combined.to_excel(writer, index=False, sheet_name="ModelResults")
            wordcloud_df.to_excel(writer, index=False, sheet_name="WordCloud")
            if summary_list:
                pd.DataFrame(summary_list).to_excel(writer, sheet_name="SHAP貢獻區間", index=False)
        print(f"✅ 模型 {strategy_id} 資料儲存至：{output_path}")
    
        # === 寫入監控資訊 ===
        df_train_ = df_combined[df_combined["is_holdout"] == 0]
        df_holdout_ = df_combined[df_combined["is_holdout"] == 1]
        monitoring_row = {
            "timestamp": timestamp,
            "strategy_id": strategy_id,
            "model_file": f"model_strategy_{strategy_id}_{timestamp}.pkl",
            "train_size": len(df_train_),
            "train_pos_ratio": round(df_train_["label"].mean(), 4),
            "test_size": len(df_holdout_),
            "test_pos_ratio": round(df_holdout_["label"].mean(), 4),
            "cv_roc_auc": np.mean(roc_scores),
            "cv_pr_auc": np.mean(pr_scores),
            "test_roc_auc": roc_auc_score(df_holdout_["label"], df_holdout_["pred_prob"]),
            "test_pr_auc": average_precision_score(df_holdout_["label"], df_holdout_["pred_prob"])
        }
    
        monitoring_xlsx_path = os.path.join(save_dir, "model_monitoring.xlsx")
        shap_df = pd.DataFrame(summary_list)
        monitoring_df = pd.DataFrame([monitoring_row])
        if os.path.exists(monitoring_xlsx_path):
            with ExcelWriter(monitoring_xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                pd.concat([pd.read_excel(monitoring_xlsx_path, sheet_name="summary_log"), monitoring_df]).to_excel(writer, sheet_name="summary_log", index=False)
                if not shap_df.empty:
                    pd.concat([pd.read_excel(monitoring_xlsx_path, sheet_name="shap_ranges_log"), shap_df]).to_excel(writer, sheet_name="shap_ranges_log", index=False)
        else:
            with ExcelWriter(monitoring_xlsx_path, engine="openpyxl") as writer:
                monitoring_df.to_excel(writer, sheet_name="summary_log", index=False)
                if not shap_df.empty:
                    shap_df.to_excel(writer, sheet_name="shap_ranges_log", index=False)
    
        print("📊 模型訓練與 SHAP 分析完成")
    return df_combined, model
        
    
# train_model_pipeline_with_strategies(df_ready, policy_df)
