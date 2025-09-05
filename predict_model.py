# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 09:34:53 2025

@author: Z01788
"""

import os, json, joblib, re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from openpyxl import load_workbook

# ---------- 共用小工具 ----------
def _get_latest_models_dir(models_root):
    if not os.path.exists(models_root):
        return None
    subdirs = [d for d in os.listdir(models_root)
               if os.path.isdir(os.path.join(models_root,d)) and len(d)>=8 and d[:8].isdigit()]
    if not subdirs:
        return None
    subdirs.sort(reverse=True)
    return os.path.join(models_root, subdirs[0])

def load_models(models_root="D:/備註文字探勘/models", strategy_id=0):
    # 先找 models/<timestamp>/latest.json，找不到再退回 models/latest.json
    latest_json = None
    latest_dir = _get_latest_models_dir(models_root)
    if latest_dir and os.path.exists(os.path.join(latest_dir,"latest.json")):
        latest_json = os.path.join(latest_dir,"latest.json")
    elif os.path.exists(os.path.join(models_root,"latest.json")):
        latest_json = os.path.join(models_root,"latest.json")
    else:
        raise FileNotFoundError("找不到 models/latest.json 或任何 timestamp 子目錄的 latest.json")

    with open(latest_json,encoding="utf-8") as f:
        latest = json.load(f)

    if "strategies" not in latest or str(strategy_id) not in latest["strategies"]:
        raise KeyError(f"latest.json 找不到 strategy_id={strategy_id}")

    # 以 latest.json 所在目錄為基準轉絕對路徑
    base = os.path.dirname(latest_json)
    e = latest["strategies"][str(strategy_id)]
    def _abs(p): return p if os.path.isabs(p) else os.path.join(base,p)

    model   = joblib.load(_abs(e["model"]))
    w2v = joblib.load(_abs(e["word2vec"]))
    tfidf   = joblib.load(_abs(e["tfidf"]))
    # scaler  = joblib.load(_abs(e["scaler"]))
    feats   = joblib.load(_abs(e["features"]))
    
    scaler  = None
    if "scaler" in e:
        try:
            scaler = joblib.load(_abs(e["scaler"]))
        except Exception:
            scaler = None
    
    feats   = joblib.load(_abs(e["features"]))
    
    w2v_idx = None
    if "w2v_top_indices" in e:
        try:
            w2v_idx = joblib.load(_abs(e["w2v_top_indices"]))
        except Exception:
            w2v_idx = None
    
    print(f"✅ 載入模型完成：{os.path.basename(base)}（strategy {strategy_id}）")
    return model, w2v, tfidf, scaler, feats, w2v_idx



# def _merge_tags_row(row, cols=("個人化標籤_背景","個人化標籤_銷售","拜訪備註_標籤")):
#     all_tags = []
#     for c in cols:
#         if c in row and pd.notna(row[c]):
#             parts = (str(row[c])
#                      .replace("，", ",")
#                      .replace("、", ",")
#                      .replace("  ", " ")
#                      .replace(" ,", ",")
#                      .split(","))
#             for tag in parts:
#                 t = tag.strip().strip(".").strip("🖊️").lower()
#                 if t:
#                     all_tags.append(t)
#     return list(set(all_tags))
def _merge_tags_row(row, cols=("個人化標籤_背景", "個人化標籤_銷售", "拜訪備註_標籤")):
    all_tags = []
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # 表情
        u"\U0001F300-\U0001F5FF"  # 符號&圖形
        u"\U0001F680-\U0001F6FF"  # 交通工具
        u"\U0001F1E0-\U0001F1FF"  # 國旗
        "]+", flags=re.UNICODE)

    for c in cols:
        if c in row and pd.notna(row[c]):
            clean = emoji_pattern.sub("", str(row[c]))
            parts = clean.replace("，", ",").replace("、", ",").replace("  ", " ").replace(" ,", ",").split(",")
            for tag in parts:
                t = tag.strip().strip(".").lower()
                if t:
                    all_tags.append(t)
    return list(set(all_tags))

# ---------- 載模型（改版） ----------
# def load_models(base_dir="D:/備註文字探勘/results", strategy_id=0):
#     """
#     從 results 底下找到「最新一批」(最新 timestamp 資料夾) 的 latest.json，
#     然後載入該 strategy 的模型、處理器與特徵資訊。
#     latest.json 應該長這樣：
#     {
#       "timestamp": "...",
#       "strategies": {
#         "0": {
#           "model": "D:/.../strategy_0/model_final.pkl",
#           "word2vec": "...",
#           "tfidf": "...",
#           "scaler": "...",
#           "features": "....pkl",
#           "train_reference": "...",
#           "w2v_top_indices": "....pkl"   # 可選，沒有就用 fallback
#         },
#         "2": {...},
#         "6": {...}
#       }
#     }
#     """
#     latest_run = _find_latest_run_dir(base_dir)
#     if latest_run is None:
#         raise FileNotFoundError("找不到任何 results/<timestamp>/ 資料夾")

#     latest_json = os.path.join(latest_run, "latest.json")
#     if not os.path.exists(latest_json):
#         raise FileNotFoundError(f"找不到 latest.json: {latest_json}")

#     with open(latest_json, encoding="utf-8") as f:
#         latest = json.load(f)

#     strat = latest["strategies"][str(strategy_id)]

#     # 這些路徑在重訓時就寫成「完整路徑」，直接 load 即可
#     model   = joblib.load(strat["model"])
#     w2v     = joblib.load(strat["word2vec"])
#     tfidf   = joblib.load(strat["tfidf"])
#     scaler  = joblib.load(strat["scaler"])
#     feats   = joblib.load(strat["features"])

#     # 可選：w2v_top_indices
#     w2v_idx = None
#     if "w2v_top_indices" in strat and os.path.exists(strat["w2v_top_indices"]):
#         try:
#             w2v_idx = joblib.load(strat["w2v_top_indices"])
#         except Exception:
#             w2v_idx = None

#     print(f"✅ 已載入最新模型（strategy={strategy_id}）版本：{latest['timestamp']}")
#     return model, w2v, tfidf, scaler, feats, w2v_idx


def classify_probability(p):
    if p >= 0.9: return "極高潛力"
    elif p >= 0.75: return "高潛力"
    elif p >= 0.5: return "中潛力"
    elif p >= 0.25: return "低潛力"
    else: return "極低潛力"
    
def predict_with_strategy(df_ready, strategy_id, models_root):
    model, w2v_model, tfidf, scaler, features, top_idx = load_models(models_root, strategy_id)

    w2v_feats = [f for f in features if f.upper().startswith("W2V_")]
    top_k = len(w2v_feats)
    num_feats = [f for f in features if f not in w2v_feats and f in df_ready.columns]
    tag_feats = [f for f in features if f not in w2v_feats + num_feats]

    # 數值
    X_num = df_ready[num_feats].fillna(0) if num_feats else pd.DataFrame(index=df_ready.index)
    # X_scaled = scaler.transform(X_num) if not X_num.empty else np.zeros((len(df_ready), 0))
    X_scaled = scaler.transform(X_num) if (scaler is not None and not X_num.empty) else (X_num.values if not X_num.empty else np.zeros((len(df_ready), 0)))

    # 文字向量
    tfidf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    def get_vec(tokens):
        vecs, weights = [], []
        for word in tokens:
            if word in w2v_model.wv and word in tfidf_dict:
                vecs.append(w2v_model.wv[word] * tfidf_dict[word])
                weights.append(tfidf_dict[word])
        return np.sum(vecs, axis=0) / np.sum(weights) if vecs else np.zeros(w2v_model.vector_size)

    w2v_all = np.vstack([get_vec(str(x).split()) for x in df_ready['備註文字_處理']])
    X_w2v = w2v_all[:, top_idx] if top_idx is not None else w2v_all[:, :top_k]

    # 標籤類
    def split_tags(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        if isinstance(val, (list, tuple)):
            s = ",".join(map(str, val))
        else:
            s = str(val)
        s = (s.replace("，", ",").replace("、", ",")
               .replace("  ", " ").replace(" ,", ","))
        toks = [p.strip().strip(".").strip("🖊️").strip("#").lower()
                for p in s.split(",")]
        return [t for t in toks if t]
    
    # 拜訪備註_標籤 → 向量
    if "拜訪備註_標籤" in df_ready.columns:
        df_ready["visit_tag_list"] = df_ready["拜訪備註_標籤"].apply(split_tags)
        idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
        def tags_to_vector(tags):
            if not tags: 
                return np.zeros(w2v_model.vector_size, dtype=np.float32)
            toks = [t for t in tags if t in w2v_model.wv]
            if not toks:
                return np.zeros(w2v_model.vector_size, dtype=np.float32)
            w = [idf.get(t, 0.0) for t in toks]
            if not any(x > 0 for x in w):
                w = [1.0] * len(toks)
            return np.average([w2v_model.wv[t] for t in toks], axis=0, weights=w).astype(np.float32)
        df_ready["w2v_tag_vector"] = df_ready["visit_tag_list"].apply(tags_to_vector)
    else:
        df_ready["w2v_tag_vector"] = [np.zeros(w2v_model.vector_size, dtype=np.float32) for _ in range(len(df_ready))]
    
    # if tag_feats:
    #     merged = df_ready.apply(_merge_tags_row, axis=1)
    #     tag_idx = {t: i for i, t in enumerate(tag_feats)}
    #     X_tags = np.zeros((len(df_ready), len(tag_feats)))
    #     for r, tags in enumerate(merged):
    #         for t in tags:
    #             if t in tag_idx:
    #                 X_tags[r, tag_idx[t]] = 1.0
    # else:
    #     X_tags = np.zeros((len(df_ready), 0))

    X_final = np.hstack([X_w2v, X_scaled, X_tags])
    y_prob = model.predict_proba(X_final)[:, 1]
    y_pred = (y_prob >= 0.6).astype(int)

    df_out = df_ready.copy()
    df_out["預測成交機率"] = y_prob
    df_out["預測成交與否"] = y_pred
    df_out["潛力分類"] = df_out["預測成交機率"].apply(classify_probability)
    df_out["strategy_id"] = strategy_id
    return df_out

def predict_batch(df_ready, output_path, strategy_ids=[0, 2, 6], models_root="D:/備註文字探勘/models", policy_path=None):
    all_results = []
    for sid in strategy_ids:
        try:
            print(f"🚀 Predicting strategy {sid}")
            result = predict_with_strategy(df_ready, sid, models_root)
            result["strategy_id"] = sid
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed for strategy {sid}: {e}")

    df_all = pd.concat(all_results, ignore_index=True)

    # === optional: 比對實際成交（若給定保單資料）===
    if policy_path is not None:
        try:
            policy_df = pd.read_excel(policy_path, sheet_name="POLICY")
            policy_df['投保日 年/月/日'] = pd.to_datetime(policy_df['投保日 年/月/日'], errors='coerce')

            def check_success(row):
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

            df_all['實際是否成交'] = df_all.apply(check_success, axis=1)
            df_all['預測是否成交'] = (df_all['預測成交機率'] >= 0.6).astype(int)
        except Exception as e:
            print(f"⚠️ 保單比對失敗：{e}")

    # === 斷詞長格式 ===
    tokens_long = df_all[['客戶UUID', '拜訪紀錄UUID', '備註文字_處理', '預測成交機率', 'strategy_id']].copy()
    tokens_long['備註文字_詞語'] = tokens_long['備註文字_處理'].astype(str).str.split()
    tokens_long = tokens_long.explode("備註文字_詞語").rename(columns={"備註文字_詞語": "詞語"}).dropna(subset=["詞語"])

    # === 儲存輸出 ===
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="預測結果", index=False)
        tokens_long.to_excel(writer, sheet_name="斷詞長格式", index=False)

        if '實際是否成交' in df_all.columns:
            cm = confusion_matrix(df_all['實際是否成交'], df_all['預測是否成交'])
            report = classification_report(df_all['實際是否成交'], df_all['預測是否成交'], output_dict=True)

            cm_df = pd.DataFrame(cm, index=["實際:未成交", "實際:成交"], columns=["預測:未成交", "預測:成交"])
            cr_df = pd.DataFrame(report).T

            cm_df.to_excel(writer, sheet_name='模型評估', startrow=0)
            cr_df.to_excel(writer, sheet_name='模型評估', startrow=len(cm_df) + 3)

    print(f"✅ All strategy predictions saved to {output_path}")
            

    
# def check_success_within_30_days(row, policy_df):
#     uuid = row['客戶UUID']
#     visit_date = row['拜訪時間 年/月/日']
    
#     policies = policy_df[policy_df['經紀人1-被保人CRM UUID'] == uuid]
#     if policies.empty:
#         return 0
    
#     matched = policies[
#         (policies['投保日 年/月/日'] > visit_date) &
#         (policies['投保日 年/月/日'] <= visit_date + pd.Timedelta(days=30))
#     ]
#     return 1 if not matched.empty else 0

# def evaluate_predictions(results_path, policy_df, threshold=0.6):
#     results_df = pd.read_excel(results_path)
#     results_df['拜訪時間 年/月/日'] = pd.to_datetime(results_df['拜訪時間 年/月/日'], errors='coerce')
#     policy_df['投保日 年/月/日'] = pd.to_datetime(policy_df['投保日 年/月/日'], errors='coerce')
#     results_df['實際是否成交'] = results_df.apply(lambda row: check_success_within_30_days(row, policy_df), axis=1)
#     results_df['預測是否成交'] = (results_df['預測成交機率'] >= threshold).astype(int)

#     cm = confusion_matrix(results_df['實際是否成交'], results_df['預測是否成交'])
#     report = classification_report(results_df['實際是否成交'], results_df['預測是否成交'], output_dict=True)

#     with pd.ExcelWriter(results_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#         results_df.to_excel(writer, sheet_name='預測結果_已驗證', index=False)
#         cm_df = pd.DataFrame(cm, index=["實際:未成交", "實際:成交"], columns=["預測:未成交", "預測:成交"])
#         cr_df = pd.DataFrame(report).T
#         cm_df.to_excel(writer, sheet_name='模型評估', startrow=0)
#         cr_df.to_excel(writer, sheet_name='模型評估', startrow=len(cm_df) + 3)
#     print(f"✅ 評估完成，結果已寫入：{results_path}") 


# # ---------- 預測主流程（改版） ----------
# def predict_with_model(df_ready, output_path, source_file=None, strategy_id=0, models_root="D:/備註文字探勘/models"):
#     # 1) 讀模型與處理器（最新一批 & 指定 strategy）
#     model, w2v_model, tfidf_vectorizer, scaler, features, w2v_top_indices = load_models(models_root, strategy_id)

#     # 2) 依 features 拆出 W2V / 數值 / 標籤類
#     feat_upper = [f.upper() for f in features]
#     w2v_feats   = [f for f in features if f.upper().startswith("W2V_")]
#     top_k       = len(w2v_feats)

#     # 數值欄位：以「features 中、且也出現在 df_ready.columns」者為準
#     numerical_cols = [f for f in features if (f not in w2v_feats) and (f in df_ready.columns)]

#     # 剩下的就視為 tag 類（有些策略沒有標籤，這裡就會是空 list）
#     tag_classes = [f for f in features if (f not in w2v_feats) and (f not in numerical_cols)]

#     # 3) 數值特徵
#     X_num = df_ready[numerical_cols].copy() if numerical_cols else pd.DataFrame(index=df_ready.index)
#     if not X_num.empty:
#         X_num = X_num.fillna(0)
#         X_scaled = scaler.transform(X_num)
#     else:
#         X_scaled = np.zeros((len(df_ready), 0))

#     # 4) 文字向量（加權平均）；注意要逐列處理，避免掉對齊
#     tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

#     def vectorize_sentence_weighted(tokens):
#         vecs, weights = [], []
#         for word in tokens:
#             if (word in w2v_model.wv) and (word in tfidf_dict):
#                 vecs.append(w2v_model.wv[word] * tfidf_dict[word])
#                 weights.append(tfidf_dict[word])
#         if vecs:
#             return np.sum(vecs, axis=0) / np.sum(weights)
#         return np.zeros(w2v_model.vector_size)

#     tokens_series = df_ready['備註文字_處理'].fillna("").astype(str).apply(lambda x: x.split())
#     w2v_all = np.vstack([vectorize_sentence_weighted(toks) for toks in tokens_series])

#     # 5) 只取訓練時選過的前 top_k 維（有存 indices 則用，沒有就退而求其次用 range(top_k)）
#     if w2v_top_indices is not None:
#         X_w2v_top = w2v_all[:, w2v_top_indices]
#     else:
#         X_w2v_top = w2v_all[:, :top_k] if top_k > 0 else np.zeros((len(df_ready), 0))

#     # 6) 標籤類特徵（如果 features 有列出）
#     if tag_classes:
#         # 產出 merged_tags
#         merged = df_ready.apply(_merge_tags_row, axis=1)
#         # 對應成 multi-hot，tag_classes 的名稱就是 one-hot 欄位名
#         tag_index = {t: i for i, t in enumerate(tag_classes)}
#         X_tags = np.zeros((len(df_ready), len(tag_classes)), dtype=np.float32)
#         for r, tags in enumerate(merged):
#             for t in tags:
#                 if t in tag_index:
#                     X_tags[r, tag_index[t]] = 1.0
#     else:
#         X_tags = np.zeros((len(df_ready), 0), dtype=np.float32)

#     # 7) 組合成最終特徵
#     X_final = np.hstack([X_w2v_top, X_scaled, X_tags])

#     # 8) 推論
#     y_pred_proba = model.predict_proba(X_final)[:, 1]
#     y_pred = (y_pred_proba >= 0.6).astype(int)

#     out = df_ready.copy()
#     out['預測成交機率'] = y_pred_proba
#     out['預測成交與否'] = y_pred
#     out['潛力分類'] = out['預測成交機率'].apply(classify_probability)
    
#     # === 建立斷詞長格式（包含預測機率）===
#     if '拜訪備註_詞語' in out.columns:
#         tokens_exploded = out[[
#             '客戶UUID', '拜訪紀錄UUID', '拜訪備註_詞語', '預測成交機率'
#         ]].explode('拜訪備註_詞語').rename(columns={'拜訪備註_詞語': '詞語'}).dropna(subset=['詞語'])
#     else:
#         tokens_exploded = pd.DataFrame()

#     # 9) 輸出 Excel（如有真實標籤就一起算評估）
#     def generate_model_report(y_true, y_pred, y_prob):
#         report_dict = classification_report(y_true, y_pred, output_dict=True)
#         report_df = pd.DataFrame(report_dict).T
#         report_df["ROC AUC"] = roc_auc_score(y_true, y_prob)
#         report_df["PR AUC"] = average_precision_score(y_true, y_prob)
#         return report_df

#     # 一次寫同一份 Excel（含三個 Sheet：預測結果 / 斷詞長格式 / 模型評估[如有]）
#     with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
#         out.to_excel(writer, index=False, sheet_name="預測結果")
#         tokens_exploded.to_excel(writer, index=False, sheet_name="斷詞長格式")

#         if 'label' in out.columns:
#             eval_df = generate_model_report(out['label'], out['預測成交與否'], out['預測成交機率'])
#             eval_df.to_excel(writer, sheet_name="模型評估")
#         # else: 沒有 label 就不寫「模型評估」sheet

#     print(f"✅ 預測完成，已儲存至：{output_path}")
    
#     # 10) （可選）如果你還要用保單做 30 天成交比對，就用 evaluate_predictions
#     if source_file is not None:
#         try:
#             policy_df = pd.read_excel(source_file, sheet_name="POLICY")
#             evaluate_predictions(results_path=output_path, policy_df=policy_df, threshold=0.6)
#         except Exception as e:
#             print(f"⚠️ 無法執行保單比對評估：{e}")
