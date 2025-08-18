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
import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False 


def add_shap_positive_hits(df_combined, summary_df, variables, 
                           ts_col="timestamp", sid_col="strategy_id"):
    # ... 省略前段過濾 summary_df 的邏輯 ...

    # 用來收集「所有變數的所有新欄位」
    new_cols_all = {}

    for var in variables:
        if var not in df_combined.columns:
            continue

        sub = summary_df[summary_df["變數"] == var].copy()
        if sub.empty:
            n = len(df_combined)
            new_cols_all[f"{var}_命中"]      = np.zeros(n, dtype=np.int8)
            new_cols_all[f"{var}_距離前"]    = np.full(n, np.nan)
            new_cols_all[f"{var}_距離後"]    = np.full(n, np.nan)
            new_cols_all[f"{var}_距離最近"]  = np.full(n, np.nan)
            new_cols_all[f"{var}_最近區間_起"] = np.full(n, np.nan)
            new_cols_all[f"{var}_最近區間_迄"] = np.full(n, np.nan)
            continue

        sub = sub.dropna(subset=["原始值區間_起", "原始值區間_迄"]).drop_duplicates(
            subset=["原始值區間_起", "原始值區間_迄"]
        ).sort_values(["原始值區間_起", "原始值區間_迄"])
        lo = sub["原始值區間_起"].to_numpy(float)
        hi = sub["原始值區間_迄"].to_numpy(float)

        vals = df_combined[var].to_numpy()
        n = len(vals)
        hit       = np.zeros(n, dtype=np.int8)
        dist_prev = np.full(n, np.nan)
        dist_next = np.full(n, np.nan)
        dist_near = np.full(n, np.nan)
        near_lo   = np.full(n, np.nan)
        near_hi   = np.full(n, np.nan)

        for i, v in enumerate(vals):
            if pd.isna(v):
                continue
            idx_next = np.searchsorted(lo, v, side="left")
            idx_prev = idx_next - 1
            in_prev = (idx_prev >= 0 and v >= lo[idx_prev] and v <= hi[idx_prev])
            if in_prev:
                hit[i] = 1
                dist_prev[i] = dist_next[i] = dist_near[i] = 0.0
                near_lo[i], near_hi[i] = lo[idx_prev], hi[idx_prev]
                continue
            # not hit → 距離前/後段（保留原非負距離以便比較）
            dist_prev_raw = np.nan
            dist_next_raw = np.nan
            near_lo_prev = near_hi_prev = np.nan
            near_lo_next = near_hi_next = np.nan
    
            if idx_prev >= 0:
                d = v - hi[idx_prev]           # v 位於前一段右側 → 正數
                dist_prev_raw = max(d, 0.0)
                near_lo_prev, near_hi_prev = lo[idx_prev], hi[idx_prev]
    
            if idx_next < len(lo):
                d = lo[idx_next] - v           # v 位於下一段左側 → 正數
                dist_next_raw = max(d, 0.0)
                near_lo_next, near_hi_next = lo[idx_next], hi[idx_next]
    
            # 取最近距離並加上方向符號：
            # - 靠近「下一段」→ v 偏低，應往上 → 距離最近 設為負值
            # - 靠近「前一段」→ v 偏高，應往下 → 距離最近 設為正值
            if not pd.isna(dist_prev_raw) and not pd.isna(dist_next_raw):
                if dist_next_raw < dist_prev_raw:
                    dist_near[i] = -dist_next_raw
                    near_lo[i], near_hi[i] = near_lo_next, near_hi_next
                else:
                    dist_near[i] = +dist_prev_raw
                    near_lo[i], near_hi[i] = near_lo_prev, near_hi_prev
            elif not pd.isna(dist_next_raw):
                dist_near[i] = -dist_next_raw
                near_lo[i], near_hi[i] = near_lo_next, near_hi_next
            elif not pd.isna(dist_prev_raw):
                dist_near[i] = +dist_prev_raw
                near_lo[i], near_hi[i] = near_lo_prev, near_hi_prev
            # 兩者皆 NaN 則保持 NaN（極端情況）
            # # not hit → 距離前/後段
            # if idx_prev >= 0:
            #     d = v - hi[idx_prev]
            #     dist_prev[i] = max(d, 0.0)
            #     near_lo_prev, near_hi_prev = lo[idx_prev], hi[idx_prev]
            # else:
            #     near_lo_prev = near_hi_prev = np.nan

            # if idx_next < len(lo):
            #     d = lo[idx_next] - v
            #     dist_next[i] = max(d, 0.0)
            #     near_lo_next, near_hi_next = lo[idx_next], hi[idx_next]
            # else:
            #     near_lo_next = near_hi_next = np.nan

            # # 最近
            # cand = []
            # if not pd.isna(dist_prev[i]): cand.append((dist_prev[i], near_lo_prev, near_hi_prev))
            # if not pd.isna(dist_next[i]): cand.append((dist_next[i], near_lo_next, near_hi_next))
            # if cand:
            #     d_best, l_best, h_best = min(cand, key=lambda x: x[0])
            #     dist_near[i], near_lo[i], near_hi[i] = d_best, l_best, h_best

        # 改成一次性收集欄位
        new_cols_all[f"{var}_命中"]        = hit
        new_cols_all[f"{var}_距離前"]      = dist_prev
        new_cols_all[f"{var}_距離後"]      = dist_next
        new_cols_all[f"{var}_距離最近"]    = dist_near
        new_cols_all[f"{var}_最近區間_起"] = near_lo
        new_cols_all[f"{var}_最近區間_迄"] = near_hi

    # 一次 concat，避免碎片化
    if new_cols_all:
        df_combined = pd.concat([df_combined, pd.DataFrame(new_cols_all, index=df_combined.index)], axis=1)
        # 可選：去碎片
        df_combined = df_combined.copy()

    return df_combined


def add_positive_visit_counts(summary_df, df_all, numerical_cols,
                              source_col=None, train_flag_value=0, holdout_flag_value=1):
    """
    summary_df: 由 plot_shap_bin_auto_with_summary_dual_x 回傳（含 變數、原始值區間_起/迄）
    df_all:     你要用來計數的資料（建議用 df_combined，含 train + holdout）
    numerical_cols: 只處理這些數值欄
    source_col: 若提供（如 "is_holdout"），會另外算出 train/holdout 拆分的數量
    """
    out = summary_df.copy()
    out["正向拜訪數"] = 0
    out["樣本數"] = 0
    out["正向占比"] = np.nan

    # 若要拆 train/holdout
    if source_col is not None:
        out["正向拜訪數_train"] = 0
        out["樣本數_train"] = 0
        out["正向占比_train"] = np.nan
        out["正向拜訪數_holdout"] = 0
        out["樣本數_holdout"] = 0
        out["正向占比_holdout"] = np.nan

    for idx, r in out.iterrows():
        var = r["變數"]
        if var not in numerical_cols:
            continue

        lo = r["原始值區間_起"]
        hi = r["原始值區間_迄"]

        # 容忍無窮界
        lo_eff = -np.inf if pd.isna(lo) else lo
        hi_eff =  np.inf if pd.isna(hi) else hi

        col = df_all[var]
        total = col.notna().sum()
        hit_mask = col.between(lo_eff, hi_eff, inclusive="both")
        hit = hit_mask.sum()

        out.at[idx, "正向拜訪數"] = int(hit)
        out.at[idx, "樣本數"] = int(total)
        out.at[idx, "正向占比"] = round(hit / total, 4) if total > 0 else np.nan

        if source_col is not None:
            tr = df_all[df_all[source_col] == train_flag_value][var]
            te = df_all[df_all[source_col] == holdout_flag_value][var]

            tr_tot = tr.notna().sum()
            te_tot = te.notna().sum()
            tr_hit = tr.between(lo_eff, hi_eff, inclusive="both").sum()
            te_hit = te.between(lo_eff, hi_eff, inclusive="both").sum()

            out.at[idx, "正向拜訪數_train"] = int(tr_hit)
            out.at[idx, "樣本數_train"] = int(tr_tot)
            out.at[idx, "正向占比_train"] = round(tr_hit / tr_tot, 4) if tr_tot > 0 else np.nan

            out.at[idx, "正向拜訪數_holdout"] = int(te_hit)
            out.at[idx, "樣本數_holdout"] = int(te_tot)
            out.at[idx, "正向占比_holdout"] = round(te_hit / te_tot, 4) if te_tot > 0 else np.nan

    return out

# def build_heatmap_long(df_combined, variables, id_cols=None):
#     """
#     將 df_combined 內的 {變數}_命中 / {變數}_距離 轉為直向表。
#     variables: 要展開的變數名稱清單（通常是 numerical_cols）
#     id_cols:   要保留的識別欄位，預設會自動擷取能找到的欄位
#     """
#     if id_cols is None:
#         id_cols = [c for c in ["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID",
#                                "is_holdout","label","pred_prob"]
#                    if c in df_combined.columns]

#     pieces = []
#     for var in variables:
#         hit_col = f"{var}_命中"
#         if hit_col not in df_combined.columns:
#             # 沒有命中欄位就略過（避免報錯）
#             continue
#         dist_col = f"{var}_距離最近"

#         keep = id_cols + [hit_col]
#         if dist_col in df_combined.columns:
#             keep += [dist_col]

#         tmp = df_combined[keep].copy()
#         tmp.rename(columns={hit_col:"命中"}, inplace=True)
#         if dist_col in tmp.columns:
#             tmp.rename(columns={dist_col:"距離"}, inplace=True)
#         else:
#             tmp["距離"] = np.nan

#         tmp["變數"] = var
#         tmp["命中"] = tmp["命中"].astype("Int64")
#         pieces.append(tmp)

#     if pieces:
#         return pd.concat(pieces, ignore_index=True)
#     else:
#         # 若一個都沒有，回傳空 DF（避免 None）
#         return pd.DataFrame(columns=(id_cols+["命中","距離","變數"]))

import math

def build_heatmap_long(
    df_combined: pd.DataFrame,
    variables: list,
    id_cols: list = None,
    ts_col: str = "timestamp",
    sid_col: str = "strategy_id",
    hit_suffixes=("_命中", "_距離最近", "_最近方向", "_最近區間_起", "_最近區間_迄"),
) -> pd.DataFrame:
    """
    將 df_combined 內各變數的 SHAP 命中/距離資訊轉成直向表給 Tableau 熱力圖使用。
    會先一次性補齊缺欄位以避免 DataFrame fragmentation 警告。
    """
    if id_cols is None:
        # 盡量保留能辨識該拜訪紀錄的欄位（存在才放）
        id_candidates = [ts_col, sid_col, "客戶UUID", "拜訪紀錄UUID", "is_holdout"]
        id_cols = [c for c in id_candidates if c in df_combined.columns]

    # === 1) 一次性建立缺少的欄位，避免逐欄賦值造成 fragmentation ===
    missing_cols = {}
    for var in variables:
        cols_needed = {
            "原始值": var,
            "命中": f"{var}{hit_suffixes[0]}",
            "距離最近": f"{var}{hit_suffixes[1]}",
            "最近方向": f"{var}{hit_suffixes[2]}",
            "最近區間_起": f"{var}{hit_suffixes[3]}",
            "最近區間_迄": f"{var}{hit_suffixes[4]}",
        }
        # 如果原始值欄位不存在，不補（因為這就是變數本身）
        # 其他 SHAP 衍生欄位若不存在就補預設
        for alias, realcol in cols_needed.items():
            if realcol not in df_combined.columns:
                if alias == "命中":
                    # 命中預設 0
                    missing_cols[realcol] = pd.Series(0, index=df_combined.index)
                elif alias != "原始值":
                    # 其餘預設 NaN
                    missing_cols[realcol] = pd.Series(np.nan, index=df_combined.index)
    if missing_cols:
        df_combined = pd.concat([df_combined, pd.DataFrame(missing_cols)], axis=1)

    # === 2) 建直向表 ===
    long_parts = []
    for var in variables:
        base_cols = {
            "原始值": var,
            "命中": f"{var}{hit_suffixes[0]}",
            "距離最近": f"{var}{hit_suffixes[1]}",
            "最近方向": f"{var}{hit_suffixes[2]}",
            "最近區間_起": f"{var}{hit_suffixes[3]}",
            "最近區間_迄": f"{var}{hit_suffixes[4]}",
        }
        # 確保存在（萬一 variables 有不存在於 df 的欄位就跳過）
        if base_cols["原始值"] not in df_combined.columns:
            continue

        select_cols = id_cols + list(base_cols.values())
        tmp = df_combined[select_cols].copy()

        # 攤平成統一欄名
        tmp = tmp.rename(columns={
            base_cols["原始值"]: "原始值",
            base_cols["命中"]: "命中",
            base_cols["距離最近"]: "距離最近",
            base_cols["最近方向"]: "最近方向",
            base_cols["最近區間_起"]: "最近區間_起",
            base_cols["最近區間_迄"]: "最近區間_迄",
        })
        tmp["變數"] = var
        # 命中轉成 tiny int（若你希望純 0/1）
        if "命中" in tmp.columns:
            tmp["命中"] = tmp["命中"].astype("Int8")

        long_parts.append(tmp)

    if not long_parts:
        return pd.DataFrame(columns=id_cols + ["變數", "原始值", "命中", "距離最近", "最近方向", "最近區間_起", "最近區間_迄"])

    heatmap_long = pd.concat(long_parts, ignore_index=True)

    # 欄位順序美化
    front = [c for c in [ts_col, sid_col, "客戶UUID", "拜訪紀錄UUID", "is_holdout", "label", "pred_prob"] if c in heatmap_long.columns]
    cols = front + ["變數", "原始值", "命中", "距離最近", "最近方向", "最近區間_起", "最近區間_迄"]
    heatmap_long = heatmap_long[cols]

    return heatmap_long



def plot_shap_bin_auto_with_summary_dual_x(
    X_data, shap_values, feature_names, variables,
    mean_dict, scale_dict,
    window=20, output_dir=None,
    min_range_width=0.1, merge_gap=0.05
):
    summary_list = []

    for var in variables:
        try:
            i = feature_names.index(var)
            x = X_data[:, i]
            shap_val = shap_values[:, i].values

            df = pd.DataFrame({"值": x, "SHAP": shap_val}).sort_values("值").reset_index(drop=True)
            df["SHAP_smooth"] = df["SHAP"].rolling(window=window, min_periods=1).mean()

            df["is_neg"] = (df["SHAP_smooth"] < 0).astype(int)
            df["neg_group"] = (df["is_neg"].diff(1) != 0).cumsum()
            neg_segments = df[df["is_neg"] == 1].groupby("neg_group")
            neg_ranges = [(seg["值"].min(), seg["值"].max()) for _, seg in neg_segments]

            value_min, value_max = df["值"].min(), df["值"].max()
            positive_ranges = []
            current_start = value_min
            for neg_start, neg_end in sorted(neg_ranges):
                if current_start < neg_start:
                    positive_ranges.append((current_start, neg_start))
                current_start = max(current_start, neg_end)
            if current_start < value_max:
                positive_ranges.append((current_start, value_max))

            positive_ranges = [(lo, hi) for lo, hi in positive_ranges if (hi - lo) >= min_range_width]

            # 合併破碎段
            merged_ranges = []
            for lo, hi in sorted(positive_ranges):
                if not merged_ranges:
                    merged_ranges.append([lo, hi])
                else:
                    prev_lo, prev_hi = merged_ranges[-1]
                    if lo - prev_hi <= merge_gap:
                        merged_ranges[-1][1] = hi
                    else:
                        merged_ranges.append([lo, hi])

            # 還原原始數值
            mean, scale = mean_dict[var], scale_dict[var]
            restored_ranges = [(lo * scale + mean, hi * scale + mean) for lo, hi in merged_ranges]

            # 繪圖開始
            fig, ax1 = plt.subplots(figsize=(8, 4))

            # peak_idx = df["SHAP_smooth"].idxmax()
            # peak_x = df.loc[peak_idx, "值"]
            # peak_raw = peak_x * scale + mean

            ax1.scatter(df["值"], df["SHAP"], alpha=0.3, label="原始 SHAP")
            ax1.plot(df["值"], df["SHAP_smooth"], color='blue', label="平滑趨勢")
            ax1.axhline(0, color='gray', linestyle='--')
            # ax1.axvline(peak_x, color='red', linestyle='--', 
            #             label=f'最大貢獻點 = {peak_raw:,.2f}')  # ⭐ 貨幣格式)

            for idx, ((lo, hi), (lo_raw, hi_raw)) in enumerate(zip(merged_ranges, restored_ranges)):
                ax1.axvspan(lo, hi, color='lightgreen', alpha=0.3)
                summary_list.append({
                    '變數': var,
                    '原始值區間_起': round(lo_raw, 2),
                    '原始值區間_迄': round(hi_raw, 2),
                    '標準化值_起': round(lo, 2),
                    '標準化值_迄': round(hi, 2)
                })

            # 設定主 X 軸（標準化值）標籤在上方
            ax1.xaxis.set_label_position('top')
            ax1.xaxis.tick_top()
            ax1.set_xlabel("標準化", labelpad=10)
            ax1.set_ylabel("SHAP 值")
            ax1.set_title(f"{var} 對成交的 SHAP 趨勢")
            
            # 雙 X 軸：下方顯示原始數值，對齊標準化 X 軸
            def to_raw(x): return x * scale + mean
            def to_std(x): return (x - mean) / scale
            
            # 替代 ax2 = ax1.twiny()
            secax = ax1.secondary_xaxis('bottom', functions=(to_raw, to_std))
            secax.set_xlabel("原始數值")
            
            # 平均與標準差說明
            mean_text = f"原始平均值: {mean:,.2f}\n原始標準差: {scale:,.2f}"  # ⭐ 貨幣格式
            ax1.text(0.98, 0.95, mean_text, transform=ax1.transAxes,
                     ha='right', va='top', fontsize=8, color='dimgray',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))

            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            if output_dir:
                filename = os.path.join(output_dir, f"{var}_shap_plot2.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ 已儲存：{filename}")
            else:
                plt.tight_layout()
            
        except Exception as e:
            print(f"❌ {var} 失敗：{e}")

    return pd.DataFrame(summary_list)


def train_model_pipeline_with_strategies(df_ready, policy_df=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_dir = os.path.join("D:/備註文字探勘/results", timestamp)
    # os.makedirs(save_dir, exist_ok=True)
    models_root  = "D:/備註文字探勘/models"
    results_root = "D:/備註文字探勘/results"
    models_dir   = os.path.join(models_root, timestamp)         # 模型這次訓練的目錄
    results_dir  = os.path.join(results_root, timestamp)        # 結果這次輸出的目錄
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    strategy_ids = [0, 2, 6]
    all_monitoring_rows = []      # 累積每個 strategy 的 monitoring_row
    all_summary_dfs = []          # 累積每個 strategy 的 SHAP 區間 DataFrame（summary_df）
    all_heatmap_longs = []


    for strategy_id in strategy_ids:
        df_model = df_ready.copy()
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

        # ==========================================
        # 依 strategy_id 決定 tag 欄位與過濾條件
        # ==========================================
        
        
        # 依 strategy_id 決定要合併的欄位
        if strategy_id == 0:
            tag_cols = []
        elif strategy_id == 1:
            tag_cols = ['個人化標籤_背景', '個人化標籤_銷售']
        elif strategy_id == 2:
            tag_cols = ['個人化標籤_背景', '個人化標籤_銷售']
        elif strategy_id == 3:
            tag_cols = ['拜訪備註_標籤']
        elif strategy_id == 4:
            tag_cols = ['拜訪備註_標籤']
        elif strategy_id == 5:
            tag_cols = ['個人化標籤_背景', '個人化標籤_銷售', '拜訪備註_標籤']
        elif strategy_id == 6:
            tag_cols = ['個人化標籤_背景', '個人化標籤_銷售', '拜訪備註_標籤']
        elif strategy_id == 7:
            tag_cols = ['個人化標籤_背景', '個人化標籤_銷售', '拜訪備註_標籤']
        else:
            tag_cols = []
        
        # 標籤清理函式
        def split_tags(val):
            if pd.isna(val):
                return []
            s = str(val).replace("，", ",").replace("、", ",").replace("  ", " ").replace(" ,", ",")
            parts = [p.strip().strip(".").strip("🖊️").lower() for p in s.split(",")]
            return [p for p in parts if p]
        
        # 合併標籤
        if tag_cols:
            df_model["merged_tags"] = df_model.apply(
                lambda r: sorted(set(sum([split_tags(r.get(c, None)) for c in tag_cols], []))),
                axis=1
            )
        else:
            df_model["merged_tags"] = [[] for _ in range(len(df_model))]
        
        # 篩選條件
        if strategy_id in [2, 4, 6]:
            # 需要任一欄位非空
            df_model = df_model[df_model["merged_tags"].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        
        elif strategy_id == 7:
            # 個人化標籤 與 拜訪備註_標籤 都要有值
            has_personal = df_model[['個人化標籤_背景','個人化標籤_銷售']].apply(
                lambda r: any(len(split_tags(r.get(c))) > 0 for c in ['個人化標籤_背景','個人化標籤_銷售']),
                axis=1
            )
            has_visit = df_model['拜訪備註_標籤'].apply(lambda v: len(split_tags(v)) > 0)
            df_model = df_model[has_personal & has_visit].reset_index(drop=True)
        
        # Debug 訊息
        print(f"[strategy {strategy_id}] merged_tags 非空筆數：{(df_model['merged_tags'].apply(len) > 0).sum()} / 總樣本數：{len(df_model)}")

        
        # --- 2. 切分 Train / Holdout ---
        df_model = df_model.sample(frac=1, random_state=42).reset_index(drop=True)
        cutoff = int(len(df_model) * 0.8)
        df_train = df_model.iloc[:cutoff].copy()
        df_holdout = df_model.iloc[cutoff:].copy()
    
        # --- 3. 數值與向量特徵準備 ---
        # 數值型特徵
        numerical_cols = [
            '業務客戶性別組合', '最新職級', '拜訪目的',
            '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務客戶年齡差距', 
            '備註字數', '有意義詞數',
            '目前年資', '營業單位_編碼',
            '上半年準客戶數', '最近半年活動參與率', '上一個半年度FYC', '距離最近晉升天數',
            '件數', '總保費', '拜訪序號', '賽季'
        ]


        X_num = df_train[numerical_cols].copy()
        # scaler = StandardScaler()
        # X_num_scaled = scaler.fit_transform(X_num)
        
        # 取訓練集統計量（跳過 NaN）
        means = df_train[numerical_cols].mean(skipna=True)
        stds  = df_train[numerical_cols].std(skipna=True).replace(0, 1)  # 避免除以 0
        
        def scale_keep_nan(df_subset):
            X_num = df_subset[numerical_cols].copy()
            # 手動標準化：NaN 會原樣保留
            X_scaled = (X_num - means) / stds
            return X_scaled.values  # 之後和 W2V / tag 特徵 hstack
        X_num_scaled = scale_keep_nan(X_num)
    
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
        def prepare_features(df_subset, return_feature_names=False):
            X_num = df_subset[numerical_cols].copy()
            # X_scaled = scaler.transform(X_num)
            X_scaled = scale_keep_nan(X_num)
            
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
            final_features = final_features.astype(np.float32, copy=False)
        
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
        # --- 存模型到 models/<timestamp>/strategy_{sid}/ ---
        strategy_dir=os.path.join(models_dir, f"strategy_{strategy_id}")
        os.makedirs(strategy_dir, exist_ok=True)
        joblib.dump(model,              os.path.join(strategy_dir,"model_final.pkl"))
        # 若你有 w2v / tfidf 物件可一起 dump；這裡假設以「平均後向量」為主不需再存
        # joblib.dump(scaler,             os.path.join(strategy_dir,"scaler.pkl"))
        joblib.dump(w2v_model,    os.path.join(strategy_dir,"word2vec_model.pkl"))
        # 存 feature_names（含 w2v_前綴 + 數值 +（可選）tag 名稱）
        feat_names=[f"w2v_{i}" for i in w2v_top_indices]+numerical_cols
        if tag_cols: feat_names+=list(mlb.classes_)
        joblib.dump(final_feature_names,         os.path.join(strategy_dir,"feature_names.pkl"))
        # train_reference（數值特徵分佈參考）
        df_train[numerical_cols].to_csv(os.path.join(strategy_dir, "train_reference.csv"), index=False)

        
        
        
        # Step 8: 更新 latest.json（寫在 save_dir 下，記錄所有策略的最新位置）
        latest_info = {
            "timestamp": timestamp,
            "strategies": {}
        }
        for sid in strategy_ids:
            sd = os.path.join(models_dir, f"strategy_{sid}")
            latest_info["strategies"][str(sid)] = {
                # 這裡用「相對於 latest.json 所在目錄」的相對路徑，predict 端會自動轉絕對
                "model":           os.path.relpath(os.path.join(sd,"model_final.pkl"), models_dir),
                "scaler":          os.path.relpath(os.path.join(sd,"scaler.pkl"), models_dir),
                "features":        os.path.relpath(os.path.join(sd,"feature_names.pkl"), models_dir),
                "w2v_top_indices": os.path.relpath(os.path.join(sd,"w2v_top_indices.pkl"), models_dir),
                "train_reference": os.path.relpath(os.path.join(sd,"train_reference.csv"), models_dir)
            }
            
            # 寫在本次 timestamp 目錄
        with open(os.path.join(models_dir,"latest.json"),"w",encoding="utf-8") as f:
            json.dump(latest_info,f,ensure_ascii=False,indent=2)
        # 也在 models 根目錄寫一份
        with open(os.path.join(models_root,"latest.json"),"w",encoding="utf-8") as f:
            json.dump(latest_info,f,ensure_ascii=False,indent=2)
    
        print(f"✅ latest.json 已更新：{models_root} 與 {models_dir}")
        

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
        summary_df = pd.DataFrame(columns=["變數","原始值區間_起","原始值區間_迄"]) 
        if strategy_id in [0, 2, 6]:
            try:
                explainer = shap.TreeExplainer(model, feature_names=final_feature_names)
                shap_values = explainer(X_train_full)
                # explainer = shap.Explainer(model, X_train_full, feature_names=[f"W2V_{i}" for i in range(top_k)] + numerical_cols + list(mlb.classes_) if tag_cols else [])
                # shap_values = explainer(X_train_full)
    
                # mean_dict = dict(zip(numerical_cols, scaler.mean_))
                # scale_dict = dict(zip(numerical_cols, scaler.scale_))
                
                output_dir = os.path.join(results_dir, f"shap_plots_strategy_{strategy_id}")
                os.makedirs(output_dir, exist_ok=True)
        
                summary_df = plot_shap_bin_auto_with_summary_dual_x(
                    X_train_full, shap_values, final_feature_names,
                    variables=numerical_cols,  # 可調整是否包含 tag
                    mean_dict = dict(zip(numerical_cols, means)), 
                    scale_dict = dict(zip(numerical_cols, stds)), 
                    output_dir=output_dir,
                    window=20, min_range_width=0.1, merge_gap=0.05
                )
                summary_df.insert(0, "timestamp", timestamp)
                summary_df.insert(1, "strategy_id", strategy_id)
                
                # ⭐ 在這裡加：用 train+holdout 的 df_combined 去數每段區間的拜訪數
                summary_df = add_positive_visit_counts(
                    summary_df, df_combined, numerical_cols,
                    source_col="is_holdout",  # 有這欄就會同時計算 train/holdout 拆分
                    train_flag_value=0, holdout_flag_value=1
                )
                
                all_summary_dfs.append(summary_df)
                print(f"✅ SHAP 圖片與區間已儲存：strategy {strategy_id}")
        
            except Exception as e:
                if debug_mode:
                    print(f"❌ SHAP 分析失敗(strategy {strategy_id})：{e}")
                    
        # === 依 SHAP 正向區間，為每筆資料標註「正向」欄位 ===
        df_combined = add_shap_positive_hits(df_combined, summary_df, numerical_cols)
        
        # 3. 一次性新增 timestamp & strategy_id（避免多次 insert）
        df_combined = df_combined.assign(
            timestamp=timestamp,
            strategy_id=strategy_id
        ).copy() 
                    

        # === strategy 迴圈內，原本計算完 df_combined 後加這段 ===
        # heatmap_long = build_heatmap_long(df_combined, numerical_cols)
        
        heatmap_long = build_heatmap_long(
            df_combined=df_combined,
            variables=numerical_cols,          # 你要畫熱力圖的變數清單
            id_cols=["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"]
        )
        print(f"[debug] strategy={strategy_id} heatmap_long.shape={heatmap_long.shape}")
        print("[debug] build_heatmap_long dtypes:", heatmap_long.dtypes.to_dict())

        # 收集起來
        all_heatmap_longs.append(heatmap_long)
        print(f"[debug] collected so far: {[df['strategy_id'].iloc[0] if not df.empty and 'strategy_id' in df.columns else 'EMPTY' for df in all_heatmap_longs]}")
        
        # === 單一 strategy 檔案（留存原始）+ 彙整檔（累積所有 strategy） ===
        # # 先加上識別欄位
        # df_combined = df_combined.copy()
        # df_combined["timestamp"] = timestamp
        # df_combined["strategy_id"] = strategy_id
    
        wordcloud_df = wordcloud_df.copy()
        wordcloud_df["timestamp"] = timestamp
        wordcloud_df["strategy_id"] = strategy_id
    
        # 1) 單一 strategy 檔案
        single_path = os.path.join(results_dir, f"model_strategy_{strategy_id}_{timestamp}.xlsx")
        with pd.ExcelWriter(single_path, engine='xlsxwriter') as writer:
            df_combined.to_excel(writer, index=False, sheet_name="ModelResults")
            wordcloud_df.to_excel(writer, index=False, sheet_name="WordCloud")
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name="SHAP貢獻區間", index=False)
        print(f"✅ 模型 {strategy_id} 資料儲存至：{single_path}")
    
        # 2) 全部策略彙整檔（追加寫入同一份）
        aggregate_path = os.path.join(results_dir, f"ALL_strategies_results_{timestamp}.xlsx")
    
        def _append_sheet(agg_path, sheet_name, new_df):
            if os.path.exists(agg_path):
                try:
                    old = pd.read_excel(agg_path, sheet_name=sheet_name)
                    merged = pd.concat([old, new_df], ignore_index=True)
                except Exception:
                    merged = new_df
                with pd.ExcelWriter(agg_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
                    merged.to_excel(w, index=False, sheet_name=sheet_name)
            else:
                with pd.ExcelWriter(agg_path, engine="openpyxl", mode="w") as w:
                    new_df.to_excel(w, index=False, sheet_name=sheet_name)
    
        _append_sheet(aggregate_path, "ModelResults", df_combined)
        _append_sheet(aggregate_path, "WordCloud",  wordcloud_df)
        if not summary_df.empty:
            _append_sheet(aggregate_path, "SHAP貢獻區間", summary_df)
    
        print(f"📚 已彙整到：{aggregate_path}")
    
        # === 寫入監控資訊（先累積，迴圈外統一寫歷史檔）===
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
            "test_roc_auc": roc_auc_score(df_holdout_["label"], df_holdout_["pred_prob"]),  # 你可改回 roc_auc_score
            "test_pr_auc": average_precision_score(df_holdout_["label"], df_holdout_["pred_prob"])
        }
        all_monitoring_rows.append(monitoring_row)
        
        
    # ====== 迴圈結束後：一次寫入歷史檔 ======
    # 1) 先把本次訓練的資料組起來
    new_monitoring_df = pd.DataFrame(all_monitoring_rows)
    new_shap_df = pd.concat(all_summary_dfs, ignore_index=True) if all_summary_dfs else pd.DataFrame()
    
    # 2) 歷史檔放在 results 根目錄（不是 timestamp 子資料夾）
    history_path = os.path.join("D:/備註文字探勘/results", "model_monitoring.xlsx")
    
    # 3) 如果歷史檔存在，先讀舊資料（若某個 sheet 不存在，就用空DF）
    if os.path.exists(history_path):
        xls = pd.ExcelFile(history_path)
        if "summary_log" in xls.sheet_names:
            old_summary_df = pd.read_excel(history_path, sheet_name="summary_log")
        else:
            old_summary_df = pd.DataFrame()
        if "shap_ranges_log" in xls.sheet_names:
            old_shap_df = pd.read_excel(history_path, sheet_name="shap_ranges_log")
        else:
            old_shap_df = pd.DataFrame()
    else:
        old_summary_df = pd.DataFrame()
        old_shap_df = pd.DataFrame()
    
    # 4) 合併舊+新
    summary_out = pd.concat([old_summary_df, new_monitoring_df], ignore_index=True)
    if not new_shap_df.empty:
        shap_out = pd.concat([old_shap_df, new_shap_df], ignore_index=True)
    else:
        shap_out = old_shap_df
    
    # 5) 寫回（覆寫 sheet 內容，但保留同一個檔案）
    with pd.ExcelWriter(history_path, engine="openpyxl") as writer:
        summary_out.to_excel(writer, sheet_name="summary_log", index=False)
        if not shap_out.empty:
            shap_out.to_excel(writer, sheet_name="shap_ranges_log", index=False)
    
    print(f"📈 歷史監控檔已更新：{history_path}")
    
    # === strategy 迴圈結束後，統一合併與輸出 ===
    if all_heatmap_longs:
        merged_heatmap_long = pd.concat(all_heatmap_longs, ignore_index=True)
        heatmap_out_path = os.path.join(results_dir, "heatmap_long_all_strategies.csv")
        merged_heatmap_long.to_csv(heatmap_out_path, index=False, encoding="utf-8-sig")
        print(f"✅ heatmap_long 輸出完成：{heatmap_out_path}")
    else:
        print("⚠️ 沒有任何 heatmap_long 資料可輸出")
    
    return df_combined, model
        
    
# train_model_pipeline_with_strategies(df_ready, policy_df)
