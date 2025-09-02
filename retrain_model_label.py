# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:04:21 2025

@author: Z01788
"""
# packages
import pandas as pd
import numpy as np
import shap
import os, csv, json, re, gc
from pathlib import Path
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

# df functions
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

# import math

# def build_heatmap_long(
#     df_combined: pd.DataFrame,
#     variables: list,
#     id_cols: list = None,
#     ts_col: str = "timestamp",
#     sid_col: str = "strategy_id",
#     hit_suffixes=("_命中", "_距離最近", "_最近區間_起", "_最近區間_迄"), # "_最近方向", 
# ) -> pd.DataFrame:
#     """
#     將 df_combined 內各變數的 SHAP 命中/距離資訊轉成直向表給 Tableau 熱力圖使用。
#     會先一次性補齊缺欄位以避免 DataFrame fragmentation 警告。
#     """
#     if id_cols is None:
#         # 盡量保留能辨識該拜訪紀錄的欄位（存在才放）
#         id_candidates = [ts_col, sid_col, "客戶UUID", "拜訪紀錄UUID", "is_holdout"]
#         id_cols = [c for c in id_candidates if c in df_combined.columns]

#     # === 1) 一次性建立缺少的欄位，避免逐欄賦值造成 fragmentation ===
#     missing_cols = {}
#     for var in variables:
#         cols_needed = {
#             "原始值": var,
#             "命中": f"{var}{hit_suffixes[0]}",
#             "距離最近": f"{var}{hit_suffixes[1]}",
#             # "最近方向": f"{var}{hit_suffixes[2]}",
#             "最近區間_起": f"{var}{hit_suffixes[2]}",
#             "最近區間_迄": f"{var}{hit_suffixes[3]}",
#         }
#         # 如果原始值欄位不存在，不補（因為這就是變數本身）
#         # 其他 SHAP 衍生欄位若不存在就補預設
#         for alias, realcol in cols_needed.items():
#             if realcol not in df_combined.columns:
#                 if alias == "命中":
#                     # 命中預設 0
#                     missing_cols[realcol] = pd.Series(0, index=df_combined.index)
#                 elif alias != "原始值":
#                     # 其餘預設 NaN
#                     missing_cols[realcol] = pd.Series(np.nan, index=df_combined.index)
#     if missing_cols:
#         df_combined = pd.concat([df_combined, pd.DataFrame(missing_cols)], axis=1)

#     # === 2) 建直向表 ===
#     long_parts = []
#     for var in variables:
#         base_cols = {
#             "原始值": var,
#             "命中": f"{var}{hit_suffixes[0]}",
#             "距離最近": f"{var}{hit_suffixes[1]}",
#             # "最近方向": f"{var}{hit_suffixes[2]}",
#             "最近區間_起": f"{var}{hit_suffixes[2]}",
#             "最近區間_迄": f"{var}{hit_suffixes[3]}",
#         }
#         # 確保存在（萬一 variables 有不存在於 df 的欄位就跳過）
#         if base_cols["原始值"] not in df_combined.columns:
#             continue

#         select_cols = id_cols + list(base_cols.values())
#         tmp = df_combined[select_cols].copy()

#         # 攤平成統一欄名
#         tmp = tmp.rename(columns={
#             base_cols["原始值"]: "原始值",
#             base_cols["命中"]: "命中",
#             base_cols["距離最近"]: "距離最近",
#             # base_cols["最近方向"]: "最近方向",
#             base_cols["最近區間_起"]: "最近區間_起",
#             base_cols["最近區間_迄"]: "最近區間_迄",
#         })
#         tmp["變數"] = var
#         # 命中轉成 tiny int（若你希望純 0/1）
#         if "命中" in tmp.columns:
#             tmp["命中"] = tmp["命中"].astype("Int8")

#         long_parts.append(tmp)

#     if not long_parts:
#         return pd.DataFrame(columns=id_cols + ["變數", "原始值", "命中", "距離最近", "最近區間_起", "最近區間_迄"]) # "最近方向", 

#     heatmap_long = pd.concat(long_parts, ignore_index=True)

#     # 欄位順序美化
#     front = [c for c in [ts_col, sid_col, "客戶UUID", "拜訪紀錄UUID", "is_holdout", "label", "pred_prob"] if c in heatmap_long.columns]
#     cols = front + ["變數", "原始值", "命中", "距離最近", "最近區間_起", "最近區間_迄"] # "最近方向", 
#     heatmap_long = heatmap_long[cols]

#     return heatmap_long

def _ensure_shap_array(shap_values):
    """把 shap.Explanation 或 ndarray → 2D ndarray (n_samples, n_features)。"""
    if hasattr(shap_values, "values"):  # shap.Explanation
        arr = np.array(shap_values.values)
    else:
        arr = np.array(shap_values)
    if arr.ndim == 3:
        # 多類別時取正類別 (index 1)；若只有 1 維就取 0
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr

def _build_heatmap_long_only(
    df_combined: pd.DataFrame,
    variables: list,
    id_cols: list,
    hit_suffixes=("_命中", "_距離最近", "_最近區間_起", "_最近區間_迄"),
):
    """只做命中/距離的長表（你的 build_heatmap_long 精簡後的核心）。"""
    # 一次性補齊缺欄位（避免 fragmentation）
    missing_cols = {}
    for var in variables:
        col_hit = f"{var}{hit_suffixes[0]}"
        col_dnr = f"{var}{hit_suffixes[1]}"
        col_lo  = f"{var}{hit_suffixes[2]}"
        col_hi  = f"{var}{hit_suffixes[3]}"
        if col_hit not in df_combined.columns:
            missing_cols[col_hit] = pd.Series(0, index=df_combined.index)   # 預設未命中=0
        if col_dnr not in df_combined.columns:
            missing_cols[col_dnr] = pd.Series(np.nan, index=df_combined.index)
        if col_lo not in df_combined.columns:
            missing_cols[col_lo] = pd.Series(np.nan, index=df_combined.index)
        if col_hi not in df_combined.columns:
            missing_cols[col_hi] = pd.Series(np.nan, index=df_combined.index)
    if missing_cols:
        df_combined = pd.concat([df_combined, pd.DataFrame(missing_cols)], axis=1)

    parts = []
    for var in variables:
        cols_map = {
            "原始值": var,
            "命中": f"{var}{hit_suffixes[0]}",
            "距離最近": f"{var}{hit_suffixes[1]}",
            "最近區間_起": f"{var}{hit_suffixes[2]}",
            "最近區間_迄": f"{var}{hit_suffixes[3]}",
        }
        if cols_map["原始值"] not in df_combined.columns:
            continue

        use_cols = id_cols + list(cols_map.values())
        tmp = df_combined[use_cols].rename(columns={
            cols_map["原始值"]: "原始值",
            cols_map["命中"]: "命中",
            cols_map["距離最近"]: "距離最近",
            cols_map["最近區間_起"]: "最近區間_起",
            cols_map["最近區間_迄"]: "最近區間_迄",
        }).copy()
        tmp["變數"] = var
        # 命中轉 tiny int
        if "命中" in tmp.columns:
            tmp["命中"] = tmp["命中"].astype("Int8")
        parts.append(tmp)

    if not parts:
        return pd.DataFrame(columns=id_cols + ["變數", "原始值", "命中", "距離最近", "最近區間_起", "最近區間_迄"])

    heatmap_long = pd.concat(parts, ignore_index=True)
    # 欄位順序
    front = [c for c in ["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"] if c in heatmap_long.columns]
    cols  = front + ["變數", "原始值", "命中", "距離最近", "最近區間_起", "最近區間_迄"]
    return heatmap_long[cols]

def _build_shap_long_only(
    df_base: pd.DataFrame,
    shap_values,
    feature_names: list,
    id_cols: list,
    shap_prefix="SHAP_",
    shap_value_col="SHAP值",
):
    """只做 SHAP 長表（含原始值）。"""
    arr = _ensure_shap_array(shap_values)  # (n, m)
    if arr.shape[1] != len(feature_names):
        raise ValueError(f"SHAP 特徵數不符：{arr.shape[1]} vs {len(feature_names)}")

    # 組成 SHAP 寬表
    shap_cols = {f"{shap_prefix}{fn}": arr[:, j] for j, fn in enumerate(feature_names)}
    shap_wide = pd.DataFrame(shap_cols, index=df_base.index).reset_index(drop=True)
    base_ids  = df_base[id_cols].reset_index(drop=True)
    shap_wide = pd.concat([base_ids, shap_wide], axis=1)

    # 寬 → 長
    melt_cols = [c for c in shap_wide.columns if c.startswith(shap_prefix)]
    long = shap_wide.melt(id_vars=id_cols, value_vars=melt_cols,
                          var_name="SHAP欄位", value_name=shap_value_col)
    long["變數"] = long["SHAP欄位"].str.replace(f"^{shap_prefix}", "", regex=True)
    long.drop(columns=["SHAP欄位"], inplace=True)

    # 串原始值（存在才 merge）
    keep_vars = [v for v in feature_names if v in df_base.columns]
    if keep_vars:
        raw_long = df_base[id_cols + keep_vars].melt(id_vars=id_cols, var_name="變數", value_name="原始值")
        long = pd.merge(long, raw_long, on=id_cols + ["變數"], how="left")
    else:
        long["原始值"] = np.nan

    # 欄位順序
    front = [c for c in ["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"] if c in long.columns]
    return long[front + ["變數","原始值", shap_value_col]]

def build_tableau_long(
    df_combined: pd.DataFrame,
    variables: list,
    id_cols: list = None,
    shap_values=None,          # 可選：有給就會合併 SHAP 長表
    feature_names: list = None,# shap_values 對應的特徵名稱
    hit_suffixes=("_命中", "_距離最近", "_最近區間_起", "_最近區間_迄"),
    shap_value_col="SHAP值",
):
    """
    『整合版』：回傳一張同時包含
      - 命中/距離（Heatmap 用）
      - SHAP值（有提供 shap_values/feature_names 時）
    的長表，方便在 Tableau 一次使用。
    """
    # 預設 id 欄位
    if id_cols is None:
        id_cols = [c for c in ["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"]
                   if c in df_combined.columns]

    # 1) 先做 Heatmap 長表（命中/距離/原始值）
    heat_long = _build_heatmap_long_only(
        df_combined=df_combined,
        variables=variables,
        id_cols=id_cols,
        hit_suffixes=hit_suffixes,
    )

    # 2) 可選：合併 SHAP 長表
    if shap_values is not None and feature_names is not None:
        shap_long = _build_shap_long_only(
            df_base=df_combined,
            shap_values=shap_values,
            feature_names=feature_names,
            id_cols=id_cols,
            shap_value_col=shap_value_col
        )
        # 合併：以 id_cols + 變數 為 key
        out = pd.merge(
            heat_long,
            shap_long[id_cols + ["變數", shap_value_col]],  # 避免重複的原始值欄
            on=id_cols + ["變數"],
            how="left"
        )
    else:
        out = heat_long

    return out

# def build_shap_wide(df_base: pd.DataFrame,
#                     shap_values,
#                     feature_names: list,
#                     id_cols: list = None,
#                     prefix: str = "SHAP_") -> pd.DataFrame:
#     """
#     產出『寬格式』：每列=拜訪，每個特徵對應一欄 SHAP_變數名。
#     df_base：至少要跟 shap_values 對齊（同樣的列順序！）
#     """
#     arr = _ensure_shap_array(shap_values)  # (n, m)
#     if arr.shape[1] != len(feature_names):
#         raise ValueError(f"SHAP 特徵數不符：{arr.shape[1]} vs {len(feature_names)}")

#     shap_cols = {f"{prefix}{fn}": arr[:, j] for j, fn in enumerate(feature_names)}
#     shap_wide = pd.DataFrame(shap_cols, index=df_base.index)

#     # 額外掛上識別欄位（存在才加）
#     if id_cols is None:
#         id_cols = [c for c in ["timestamp", "strategy_id", "客戶UUID", "拜訪紀錄UUID", "pred_prob", "label", "is_holdout"] if c in df_base.columns]
#     shap_wide = pd.concat([df_base[id_cols].reset_index(drop=True), shap_wide.reset_index(drop=True)], axis=1)
#     return shap_wide

def build_shap_wide(
    df_base: pd.DataFrame,
    shap_values,
    feature_names: list,
    id_cols: list = None,
    shap_prefix: str = "SHAP_",
    include_raw: bool = False,
    raw_cols: list = None,
    raw_prefix: str = "RAW_",
) -> pd.DataFrame:
    """
    產出『寬格式』：每列=拜訪，每個特徵對應一欄 SHAP_<特徵>。
    需要時，可同時附上原始值欄位（RAW_<變數>）。
    - df_base：與 shap_values 對齊（相同列順序）
    - feature_names：與 shap_values 的第二維對齊
    - include_raw=True：會自動挑 raw_cols（或用指定的 raw_cols）
    """
    import numpy as np
    import pandas as pd

    # --- 1) 轉成 numpy，並檢查維度 ---
    arr = np.asarray(getattr(shap_values, "values", shap_values))
    if arr.ndim == 3:  # 有些 tree explainer 會回傳 (n, m, classes)
        # 取正類別那一層或加總，依你的二分類邏輯調整
        arr = arr[:, :, -1]
    if arr.shape[1] != len(feature_names):
        raise ValueError(f"SHAP 特徵數不符：{arr.shape[1]} vs {len(feature_names)}")

    # --- 2) 做 SHAP 寬表 ---
    shap_cols = {f"{shap_prefix}{fn}": arr[:, j] for j, fn in enumerate(feature_names)}
    shap_wide = pd.DataFrame(shap_cols, index=df_base.index)

    # --- 3) 掛上識別欄位 ---
    if id_cols is None:
        id_cols = [c for c in ["timestamp", "strategy_id", "客戶UUID", "拜訪紀錄UUID",
                               "pred_prob", "label", "is_holdout"] if c in df_base.columns]
    out = pd.concat([df_base[id_cols].reset_index(drop=True),
                     shap_wide.reset_index(drop=True)], axis=1)

    # --- 4) 需要原始值時，附上 RAW_ 欄位 ---
    if include_raw:
        # 若未指定 raw_cols，就「自動推斷」：
        # 取 feature_names 中同時存在於 df_base 的欄位，且排除明顯不是原始數值的特徵（如 w2v_*, 一熱後的 tag）
        if raw_cols is None:
            # 你也可以把判斷條件換成「在 numerical_cols 內」
            candidates = [fn for fn in feature_names
                          if (fn in df_base.columns)
                          and (not fn.startswith("w2v_"))]
            # 僅保留數值型欄位，避免把類別/字串塞進 RAW
            raw_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df_base[c])]

        if raw_cols:
            raw_df = df_base[raw_cols].copy()
            raw_df.columns = [f"{raw_prefix}{c}" for c in raw_df.columns]
            out = pd.concat([out, raw_df.reset_index(drop=True)], axis=1)

    return out

import shutil
def append_csv_fast(src_csv: str, dst_csv: str, header_written: bool) -> bool:
    """把 src_csv 直接拷貝到 dst_csv；若已寫過 header，會跳過來源第一行。返回是否已寫入 header。"""
    with open(src_csv, 'r', encoding='utf-8-sig', newline='') as fin, \
         open(dst_csv, 'a', encoding='utf-8-sig', newline='') as fout:
        if header_written:
            next(fin, None)  # 跳過來源標題列
        shutil.copyfileobj(fin, fout)
    return True  # 代表 header 已經存在



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



def append_csv(path: str, df: pd.DataFrame): 
    """將 df 追加寫到 CSV；如果檔案不存在就寫入表頭。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')
    

def clean_object_col(s: pd.Series) -> pd.Series:
    """只處理 object 欄位：list/dict/tuple/set/ndarray → JSON；字串移除換行與制表符。"""
    def _clean(x):
        # 第一層防呆：如果是 list、array 等，不直接對它做 pd.isna()
        if isinstance(x, (list, dict, tuple, set, np.ndarray)):
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return json.dumps(str(x), ensure_ascii=False)
        
        # 第二層：確認非 above 類型後再判斷 isna
        if pd.isna(x):
            return ""

        x = str(x)
        x = re.sub(r'[\r\n\t]+', ' ', x)
        return x

    return s.map(_clean)

def safe_write_csv(df: pd.DataFrame, out_path: str, force_schema: list = None, mode: str = "w"):
    """
    將 DataFrame 以「安全格式」輸出 CSV：
    - 僅清理 object 欄位（JSON 化、去換行）
    - QUOTE_ALL、utf-8-sig、單一換行符
    - 可選 force_schema 固定欄順序
    - mode="w" 覆蓋, mode="a" 追加（會自動補上 header 與否）
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 只清 object 欄位
    df_out = df.copy()
    obj_cols = df_out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df_out[c] = clean_object_col(df_out[c])

    # 欄位順序（可選）
    if force_schema:
        # 缺的補空，多的丟掉
        for c in force_schema:
            if c not in df_out.columns:
                df_out[c] = ""
        df_out = df_out[force_schema]

    # 追加模式時：若檔案不存在，視同寫新檔；存在才不寫 header
    write_header = True
    if mode == "a" and p.exists():
        write_header = False

    df_out.to_csv(
        p, index=False, encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL, lineterminator="\n",
        mode=mode, header=write_header
    )

def safe_append_csv(out_path: str, df: pd.DataFrame, force_schema: list = None):
    """等同 append_csv，但用安全寫法（含清理與 QUOTE_ALL）。"""
    safe_write_csv(df, out_path, force_schema=force_schema, mode="a")



# main function
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
    all_personal_recos = []
    all_shap_wide_list = []


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
            '職級_代碼', '拜訪目的', '業務男_要保男', '業務女_要保女', '業務男_要保女', '業務女_要保男', # '業務要保人性別組合', 
            '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務要保人年齡差', 
            '備註字數', '有意義詞數', 'hashtag_count', 
            '截至該月年資', '營業單位_編碼', '上半年準客戶數', '最近半年活動參與率', '上一個半年度FYC', 
            '距離最近晉升天數', '截至當日累積件數', '截至當日累積保費', 
            '拜訪序號', '賽季'
        ]
        
        # # ✅ 只保留真正數值 dtype 欄位
        # num_cols = [
        #     '平均拜訪間隔天數', '每週平均拜訪客戶數', '業務要保人年齡差', 
        #     '備註字數', '有意義詞數', 
        #     '截至該月年資', '上半年準客戶數', '最近半年活動參與率', '上一個半年度FYC', 
        #     '距離最近晉升天數', '截至當日累積件數', '截至當日累積保費'
        #     ]
        # # num_cols = [c for c in numerical_cols
        # #             if c in df_train.columns and pd.api.types.is_numeric_dtype(df_train[c])]
        
        # # if len(num_cols) < len(numerical_cols):
        # #     missing = [c for c in numerical_cols if c not in num_cols]
        # #     print("⚠️ 已自動排除非數值欄位：", missing)
            
        # cat_cols = [c for c in numerical_cols if c not in num_cols]


        # X_num = df_train[numerical_cols].copy()
        # # scaler = StandardScaler()
        # # X_num_scaled = scaler.fit_transform(X_num)
        
        # # 取訓練集統計量（跳過 NaN）
        # means = df_train[numerical_cols].mean(skipna=True)
        # stds  = df_train[numerical_cols].std(skipna=True).replace(0, 1)  # 避免除以 0
        
        # def scale_keep_nan(df_subset):
        #     X_num = df_subset[numerical_cols].copy()
        #     # 手動標準化：NaN 會原樣保留
        #     X_scaled = (X_num - means) / stds
        #     return X_scaled.values  # 之後和 W2V / tag 特徵 hstack
        # X_num_scaled = scale_keep_nan(X_num)
        
        # --- 只對數值欄做 scale ---
        means = df_train[numerical_cols].mean(skipna=True)
        stds  = df_train[numerical_cols].std(skipna=True).replace(0, 1)
        
        def scale_keep_nan(df_subset):
            X_num = df_subset[numerical_cols].copy()
            X_num = X_num.apply(pd.to_numeric, errors="coerce")  # 再次保險
            X_scaled = (X_num - means.reindex(numerical_cols)) / stds.reindex(numerical_cols)
            return X_scaled.values
        
        X_num_scaled = scale_keep_nan(df_train)
    
        # --- Word2Vec 特徵 ---
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
        
        known_tags = set(mlb.classes_)

        def safe_mlb_transform(mlb, tags_series):
            # 只保留訓練看過的標籤
            filtered = tags_series.apply(lambda lst: [t for t in (lst or []) if t in known_tags])
            return mlb.transform(filtered)
        
        # --- 5. Feature Preparation Function ---
        def prepare_features(df_subset, return_feature_names=False):
            X_num = df_subset[numerical_cols].copy()
            # X_scaled = scaler.transform(X_num)
            X_scaled = scale_keep_nan(X_num)
            # 數值欄位 (標準化後)
            # X_scaled = scale_keep_nan(df_subset)
            
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
            
            # 在 prepare_features 裡使用：
            if tag_cols:
                tags_trans = safe_mlb_transform(mlb, df_subset["merged_tags"])
                final_features = np.hstack([final_features, tags_trans])
                final_feature_names += list(mlb.classes_)  # 若你有維持名稱
            
            # if tag_cols:
            #     tags_trans = mlb.transform(df_subset["merged_tags"])
            #     final_features = np.hstack([final_features, tags_trans])
            #     final_feature_names += mlb.classes_.tolist()
        
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
        
        # heatmap_long = build_heatmap_long(
        #     df_combined=df_combined,
        #     variables=numerical_cols,          # 你要畫熱力圖的變數清單
        #     id_cols=["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"]
        # )
        # print(f"[debug] strategy={strategy_id} heatmap_long.shape={heatmap_long.shape}")
        # print("[debug] build_heatmap_long dtypes:", heatmap_long.dtypes.to_dict())

        # # 收集起來
        # all_heatmap_longs.append(heatmap_long)
        # print(f"[debug] collected so far: {[df['strategy_id'].iloc[0] if not df.empty and 'strategy_id' in df.columns else 'EMPTY' for df in all_heatmap_longs]}")
        
        # 先保證 shap 值與 df_combined 對齊
        explainer = shap.TreeExplainer(model, feature_names=final_feature_names)
        shap_values_combined = explainer(X_combined)
        
        # heatmap_shap_long = build_tableau_long(
        #     df_combined=df_combined,
        #     variables=numerical_cols,  # 只把「數值型」放進 heatmap 命中/距離
        #     id_cols=["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"],
        #     shap_values=shap_values_combined,
        #     feature_names=final_feature_names,     # 這會把所有特徵的 SHAP 一起展開（包含 W2V/one-hot 等）
        #     # 若你只想在長表中保留數值變數的 SHAP，可以事後再過濾 out[out["變數"].isin(numerical_cols)]
        # )
        # heatmap_shap_long = heatmap_shap_long[heatmap_shap_long["變數"].isin(numerical_cols)]
        # all_heatmap_longs.append(heatmap_shap_long)
        
        
        
        # === 逐筆 SHAP（寬表：原始值 + SHAP 值） ===
        # 1) 對本 strategy 的 df_combined 做 SHAP
        X_shap = prepare_features(df_combined)  # 與訓練一致
        try:
            explainer  # 若前面已建立 explainer 就重用
        except NameError:
            explainer = shap.TreeExplainer(model, feature_names=final_feature_names)
        shap_vals = explainer(X_shap)  # shap.Explanation
        
        # 2) 準備欄位
        id_cols   = [c for c in ["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout"] if c in df_combined.columns]
        meta_cols = id_cols + [c for c in ["label","pred_prob"] if c in df_combined.columns]
        
        # 只輸出「真的在 df_combined 有原始值的變數」；通常是數值欄（避免 w2v_*、one-hot）
        raw_export_cols = [c for c in numerical_cols if c in df_combined.columns]
        
        # 3) 組寬表：原始值 + 各變數 SHAP 值
        # shap_wide = pd.DataFrame(shap_vals.values, columns=final_feature_names, index=df_combined.index)
        # # 只取對應原始值的那一批 SHAP（避免把 w2v_* / tag one-hot 也帶出去）
        # shap_wide_part = shap_wide[[c for c in raw_export_cols if c in shap_wide.columns]].add_prefix("SHAP_")
        
        # shap_with_raw_wide = pd.concat(
        #     [
        #         df_combined[meta_cols].reset_index(drop=True),
        #         df_combined[raw_export_cols].reset_index(drop=True),
        #         shap_wide_part.reset_index(drop=True),
        #     ],
        #     axis=1
        # )
        
        # shap_wide = build_shap_wide(
        #     df_base=df_combined,
        #     shap_values=shap_values_combined,
        #     feature_names=final_feature_names,
        #     id_cols=["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"]
        # )
        
        shap_wide = build_shap_wide(
            df_base=df_combined,                 # 與 shap 計算對齊的 DataFrame
            shap_values=shap_values_combined,
            feature_names=final_feature_names,   # 你的 SHAP 特徵名
            id_cols=["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","pred_prob","label","is_holdout"],
            shap_prefix="SHAP_",
            include_raw=True,                    # ← 開這個就會附上原始值
            # raw_cols=numerical_cols,           # （可選）若你只想放數值原始欄位，直接指定
            raw_prefix="RAW_",
        )
        
        # 4) 單一 strategy 檔案
        wide_path = os.path.join(results_dir, f"shap_wide_{timestamp}_strategy_{strategy_id}.csv")
        shap_wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
        # shap_with_raw_wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
        print(f"✅ SHAP 寬表：{wide_path}")
        
        all_shap_wide_list.append(shap_wide)
        
        # # 5) 追加到本批次 ALL 檔案（每個 strategy 追加一次）
        # def _append_csv(path, df):
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        #     write_header = not os.path.exists(path)
        #     df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")
        
        # agg_wide_csv = os.path.join(results_dir, f"ALL_SHAP_wide_{timestamp}.csv")
        # _append_csv(agg_wide_csv, shap_with_raw_wide)
        # print(f"📚 已追加到（CSV）：{agg_wide_csv}")
        
        
        
        # ========= 個人化 uplift 建議：工具函式 =========
        def _predict_proba_rows(df_rows: pd.DataFrame) -> np.ndarray:
            """使用現有的 prepare_features 與已訓練 model，回傳每列的預測機率。"""
            X = prepare_features(df_rows)
            return model.predict_proba(X)[:, 1]

        def _nearest_positive_targets(v, intervals: np.ndarray):
            """
            v 可能是數字/字串/pd.NA，這裡先安全轉成數值；intervals 也轉成 float 陣列。
            回傳候選目標點（在區間內則回傳 [(v, 'in')]）。
            """
            # 1) 安全把 v 轉數值；非數值變 NaN
            v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            if pd.isna(v):
                return []
        
            # 2) 確保 intervals 是 float 的 numpy 陣列，並去掉 NaN 行
            intervals = np.asarray(intervals, dtype=float)
            if intervals.size == 0:
                return []
            if intervals.ndim != 2 or intervals.shape[1] != 2:
                return []  # 形狀不對就放棄
            # 去掉有 NaN 的區間
            mask = ~np.isnan(intervals).any(axis=1)
            intervals = intervals[mask]
            if intervals.size == 0:
                return []
        
            # 3) 排序（以起點 lo 由小到大）
            intervals = intervals[np.argsort(intervals[:, 0])]
        
            # 4) 已在任一區間
            for lo, hi in intervals:
                if lo <= v <= hi:
                    return [(float(v), "in")]
        
            # 5) 不在任何區間 → 找最近段
            los = intervals[:, 0]
            his = intervals[:, 1]
            idx_next = np.searchsorted(los, v, side="left")

            def points_of(lo, hi):
                mid = (lo + hi) / 2.0
                q75 = lo + 0.75 * (hi - lo)
                return [(lo, "edge"), (mid, "mid"), (q75, "q75"), (hi, "edge")]

            if idx_next == 0:
                # v 在所有區間左側 → 目標用第一段
                lo, hi = intervals[0]
                cand = points_of(lo, hi)
            elif idx_next >= len(intervals):
                # v 在所有區間右側 → 目標用最後一段
                lo, hi = intervals[-1]
                cand = points_of(lo, hi)
            else:
                # v 在兩段之間，挑最近的一段
                lo_prev, hi_prev = intervals[idx_next - 1]
                lo_next, hi_next = intervals[idx_next]
                d_prev = max(v - hi_prev, 0.0)   # v 比上一段右端點大多少
                d_next = max(lo_next - v, 0.0)   # 下一段左端點離 v 多遠
                if d_next < d_prev:
                    cand = points_of(lo_next, hi_next)
                else:
                    cand = points_of(lo_prev, hi_prev)

            # 移除重複的 target
            seen = set()
            uniq = []
            for t, tag in cand:
                key = round(t, 10)
                if key not in seen:
                    uniq.append((t, tag))
                    seen.add(key)
            return uniq

        def recommend_by_interval_jump(row: pd.Series, var: str, summary_df: pd.DataFrame,
                                       uplift_pp: float = 0.05) -> dict:
            """
            依 SHAP 正向區間，對單一 row+變數 var 給建議：
              - 命中：建議維持（或回傳 None 代表不用調）
              - 未命中：嘗試將 var 調到就近正向區間（邊界/中位/上四分位），取能讓機率最大、且 >= uplift_pp 的第一個點
            回傳：dict（包含 current_value, target_value, current_proba, new_proba, delta 等）
            """
            # 取該變數的正向區間
            sub = summary_df[summary_df["變數"] == var].dropna(subset=["原始值區間_起", "原始值區間_迄"])
            sub = sub.drop_duplicates(subset=["原始值區間_起", "原始值區間_迄"]).sort_values(["原始值區間_起", "原始值區間_迄"])
            if sub.empty:
                return {"var": var, "reason": "no_positive_interval"}

            intervals = sub[["原始值區間_起", "原始值區間_迄"]].to_numpy(dtype=float)
            v0 = row[var]
            row_df = pd.DataFrame([row])

            cur_p = float(_predict_proba_rows(row_df)[0])
            targets = _nearest_positive_targets(v0, intervals)
            if not targets:
                return {"var": var, "reason": "no_target"}

            # 若已命中
            if targets[0][1] == "in":
                return {
                    "var": var,
                    "current_value": float(v0) if pd.notna(v0) else None,
                    "target_value": float(v0) if pd.notna(v0) else None,
                    "current_proba": cur_p,
                    "new_proba": cur_p,
                    "uplift": 0.0,
                    "met_uplift": True,
                    "note": "already in positive interval"
                }

            # 未命中 → 試多個候選點（邊界/中位/上四分位/右邊界）
            best = None
            for tgt, tag in targets:
                trial = row.copy()
                trial[var] = tgt
                p_new = float(_predict_proba_rows(pd.DataFrame([trial]))[0])
                upl = p_new - cur_p
                if best is None or p_new > best["new_proba"]:
                    best = {
                        "var": var,
                        "current_value": float(v0) if pd.notna(v0) else None,
                        "target_value": float(tgt),
                        "current_proba": cur_p,
                        "new_proba": p_new,
                        "uplift": upl,
                        "met_uplift": upl >= uplift_pp,
                        "picked_point": tag
                    }

            if best is None:
                return {"var": var, "reason": "no_improvement"}

            return best

        def recommend_by_grid(row: pd.Series, var: str, direction: str,
                              step: float = 1.0, steps: int = 30,
                              uplift_pp: float = 0.05,
                              lo_cap: float = None, hi_cap: float = None) -> dict:
            """
            依方向做細步長 grid search：
              direction: 'up'（往上加），'down'（往下減）
              step: 每次變更多少
              steps: 最多嘗試幾步
              lo_cap/hi_cap: 可選的上下界（建議用訓練集 1%/99% 分位數）
            """
            v0 = row[var]
            cur_p = float(_predict_proba_rows(pd.DataFrame([row]))[0])
            best = {"var": var, "current_value": float(v0) if pd.notna(v0) else None,
                    "target_value": None, "current_proba": cur_p,
                    "new_proba": cur_p, "uplift": 0.0, "met_uplift": False}

            if pd.isna(v0):
                return {**best, "reason": "nan_value"}

            for k in range(1, steps + 1):
                v_try = v0 + (step * k if direction == "up" else -step * k)
                if lo_cap is not None and v_try < lo_cap:
                    break
                if hi_cap is not None and v_try > hi_cap:
                    break

                trial = row.copy()
                trial[var] = v_try
                p_new = float(_predict_proba_rows(pd.DataFrame([trial]))[0])
                upl = p_new - cur_p
                if p_new > best["new_proba"]:
                    best.update({"target_value": float(v_try), "new_proba": p_new,
                                 "uplift": upl, "met_uplift": upl >= uplift_pp})
                    if upl >= uplift_pp:
                        break

            return best

        # def make_personal_recos(df_source: pd.DataFrame,
        #                         variables: list,
        #                         summary_df: pd.DataFrame,
        #                         uplift_pp: float = 0.05,
        #                         mode: str = "interval",
        #                         grid_cfg: dict = None) -> pd.DataFrame:
        #     """
        #     針對 df_source（例如 df_combined 或其中一部分）的每一列 × 每個變數，產生個人化建議。
        #     mode = "interval" 使用 SHAP 正向區間跳轉；"grid" 使用細步長搜尋。
        #     grid_cfg：{"step":1.0,"steps":30,"direction_map":{var:"up"/"down"},"caps":{var:(lo,hi)}}
        #     """
        #     recs = []
        #     for idx, row in df_source.iterrows():
        #         for var in variables:
        #             if var not in row.index:
        #                 continue
        #             if mode == "interval":
        #                 r = recommend_by_interval_jump(row, var, summary_df, uplift_pp=uplift_pp)
        #             else:
        #                 # 推方向：若你前面已在 df_combined 有 var_最近方向（↑/↓），可用來決定方向；否則預設 'up'
        #                 direction = "up"
        #                 if f"{var}_最近方向" in row.index and pd.notna(row[f"{var}_最近方向"]):
        #                     direction = "up" if row[f"{var}_最近方向"] == "↑" else "down"
        #                 step = (grid_cfg or {}).get("step", 1.0)
        #                 steps = (grid_cfg or {}).get("steps", 30)
        #                 lo_cap, hi_cap = None, None
        #                 if (grid_cfg or {}).get("caps") and var in grid_cfg["caps"]:
        #                     lo_cap, hi_cap = grid_cfg["caps"][var]
        #                 r = recommend_by_grid(row, var, direction=direction, step=step, steps=steps,
        #                                       uplift_pp=uplift_pp, lo_cap=lo_cap, hi_cap=hi_cap)

        #             base = {
        #                 "idx": idx,
        #                 "客戶UUID": row["客戶UUID"] if "客戶UUID" in row.index else None,
        #                 "拜訪紀錄UUID": row["拜訪紀錄UUID"] if "拜訪紀錄UUID" in row.index else None,
        #                 "變數": var,
        #                 "當前值": None if pd.isna(row[var]) else float(row[var]),
        #                 "當前機率": r.get("current_proba"),
        #                 "目標值": r.get("target_value"),
        #                 "新機率": r.get("new_proba"),
        #                 "提升(百分點)": r.get("uplift"),
        #                 "是否達標": r.get("met_uplift"),
        #                 "說明": r.get("reason", r.get("picked_point", "")),
        #             }
        #             recs.append(base)

        #     return pd.DataFrame(recs)
        
        def _make_note(var_name, target_value, uplift):
            """
            產生「建議說明」文字：
              - 目標值四捨五入至小數第2位
              - 提升幅度 <= 0% 則不顯示（回傳 None）
              - 例：將【平均拜訪間隔天數】調整到 7.50，成交率可能提升 3.2%
            """
            try:
                if target_value is None or uplift is None or float(uplift) <= 0.0:
                    return None
                return f"將【{var_name}】調整到 {float(target_value):.2f}，成交率可能提升 {float(uplift):.1%}"
            except Exception:
                return None
        
        def make_personal_recos_streaming(
            df_source: pd.DataFrame,
            variables: list,
            summary_df: pd.DataFrame,
            uplift_pp: float = 0.05,
            mode: str = "interval",                 # "interval" or "grid"
            grid_cfg: dict = None,
            out_csv: str = None,
            flush_every: int = 20000,
            extra_id_cols: list = None              # 例如 ["timestamp","strategy_id"]
        ):
            """
            串流產生個人化建議到 CSV，避免一次塞滿記憶體。
            每筆建議會即時加入「建議說明」欄位（四捨五入與門檻已處理）。
            """
            assert out_csv is not None, "請提供 out_csv 路徑"
        
            # 準備輸出
            write_header = True
            buf = []
            extra_id_cols = extra_id_cols or []
        
            # 逐列 × 逐變數
            for ridx, row in df_source.iterrows():
                for var in variables:
                    if var not in row.index:
                        continue
        
                    # 產生建議（沿用你現有兩種模式）
                    if mode == "interval":
                        r = recommend_by_interval_jump(row, var, summary_df, uplift_pp=uplift_pp)
                    else:
                        direction = "up"
                        if f"{var}_最近方向" in row.index and pd.notna(row[f"{var}_最近方向"]):
                            direction = "up" if row[f"{var}_最近方向"] == "↑" else "down"
                        step  = (grid_cfg or {}).get("step", 1.0)
                        steps = (grid_cfg or {}).get("steps", 30)
                        lo_cap, hi_cap = None, None
                        if (grid_cfg or {}).get("caps") and var in grid_cfg["caps"]:
                            lo_cap, hi_cap = grid_cfg["caps"][var]
                        r = recommend_by_grid(row, var, direction=direction, step=step, steps=steps,
                                              uplift_pp=uplift_pp, lo_cap=lo_cap, hi_cap=hi_cap)
        
                    # 基本欄位
                    rec = {
                        "idx": ridx,
                        "客戶UUID": row["客戶UUID"] if "客戶UUID" in row.index else None,
                        "拜訪紀錄UUID": row["拜訪紀錄UUID"] if "拜訪紀錄UUID" in row.index else None,
                        "變數": var,
                        "當前值": None if pd.isna(row[var]) else float(row[var]),
                        "當前機率": r.get("current_proba"),
                        "目標值": r.get("target_value"),
                        "新機率": r.get("new_proba"),
                        "提升(百分點)": r.get("uplift"),
                        "是否達標": r.get("met_uplift"),
                        "說明": r.get("reason", r.get("picked_point", "")),
                    }
        
                    # 加進額外識別欄位（如 timestamp/strategy_id）
                    for cid in extra_id_cols:
                        rec[cid] = row[cid] if cid in row.index else None
        
                    # ✅ 直接加上「建議說明」
                    rec["建議說明"] = _make_note(
                        var_name=var,
                        target_value=rec["目標值"],
                        uplift=rec["提升(百分點)"]
                    )
        
                    buf.append(rec)
        
                    # 批次落檔
                    if len(buf) >= flush_every:
                        df_chunk = pd.DataFrame(buf)
                        mode_flag = "w" if write_header else "a"
                        df_chunk.to_csv(out_csv, mode=mode_flag, index=False, encoding="utf-8-sig",
                                        header=write_header)
                        write_header = False
                        buf.clear()
        
            # 收尾落檔
            if buf:
                df_chunk = pd.DataFrame(buf)
                mode_flag = "w" if write_header else "a"
                df_chunk.to_csv(out_csv, mode=mode_flag, index=False, encoding="utf-8-sig",
                                header=write_header)
                buf.clear()

        
        first_write = True  # 控制是否寫入表頭
        
        def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
            """盡量壓小記憶體：數值轉 float32/int32；字串改 category（若高重複）。"""
            for c in df.columns:
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype('float32')
                elif pd.api.types.is_integer_dtype(df[c]):
                    # 留一點空間避免溢位
                    maxv = df[c].max(skipna=True)
                    minv = df[c].min(skipna=True)
                    if minv >= -2**31 and maxv < 2**31:
                        df[c] = df[c].astype('int32')
                elif pd.api.types.is_object_dtype(df[c]) and df[c].nunique(dropna=False) / max(len(df),1) < 0.5:
                    df[c] = df[c].astype('category')
            return df
        
        # 你的 strategy_id 迴圈內，把下段替換掉
        heatmap_shap_long = build_tableau_long(
            df_combined=df_combined,
            variables=numerical_cols, 
            id_cols=["timestamp","strategy_id","客戶UUID","拜訪紀錄UUID","is_holdout","label","pred_prob"],
            shap_values=shap_values_combined,
            feature_names=final_feature_names
        )
        # 只保留數值變數（在 _build_shap_long_only 裡先做更省 RAM，見下方進階優化）
        heatmap_shap_long = heatmap_shap_long[heatmap_shap_long["變數"].isin(numerical_cols)]
        
        # 壓縮欄位型別減少 I/O 與 RAM
        heatmap_shap_long = _downcast_df(heatmap_shap_long)
        
        # 直接寫檔，避免保留在記憶體
        if first_write:
            heatmap_shap_long.to_csv(os.path.join(results_dir, f"ALL_heatmap_shap_long_{timestamp}.csv"), 
                                     index=False, encoding="utf-8-sig", mode='w', header=True)
            first_write = False
        else:
            heatmap_shap_long.to_csv(os.path.join(results_dir, f"ALL_heatmap_shap_long_{timestamp}.csv"), 
                                     index=False, encoding="utf-8-sig", mode='a', header=False)
        
        # 釋放中間物件
        del heatmap_shap_long, shap_values_combined
        gc.collect()

        
        
        
        # === 單一 strategy 檔案（留存原始）+ 彙整檔（累積所有 strategy） ===
        # # 先加上識別欄位
        # df_combined = df_combined.copy()
        # df_combined["timestamp"] = timestamp
        # df_combined["strategy_id"] = strategy_id
    
        wordcloud_df = wordcloud_df.copy()
        wordcloud_df["timestamp"] = timestamp
        wordcloud_df["strategy_id"] = strategy_id
    
        # 1) 單一 strategy 檔案
        # single_path = os.path.join(results_dir, f"model_strategy_{strategy_id}_{timestamp}.xlsx")
        # with pd.ExcelWriter(single_path, engine='xlsxwriter') as writer:
        #     df_combined.to_excel(writer, index=False, sheet_name="ModelResults")
        #     if not summary_df.empty:
        #         summary_df.to_excel(writer, index=False, sheet_name="SHAP貢獻區間")
        # print(f"✅ 模型 {strategy_id} 資料儲存至：{single_path}")
        
        # WordCloud 改存 CSV（超大表）
        wc_csv = os.path.join(results_dir, f"WordCloud_strategy_{strategy_id}_{timestamp}.csv")
        wordcloud_df.to_csv(wc_csv, index=False, encoding="utf-8-sig")
        print(f"✅ WordCloud 改存 CSV：{wc_csv}")
    
        # 2) 全部策略彙整檔（追加寫入同一份）
        # aggregate_path = os.path.join(results_dir, f"ALL_strategies_results_{timestamp}.xlsx")
    
        # def _append_sheet(agg_path, sheet_name, new_df):
        #     if os.path.exists(agg_path):
        #         try:
        #             old = pd.read_excel(agg_path, sheet_name=sheet_name)
        #             merged = pd.concat([old, new_df], ignore_index=True)
        #         except Exception:
        #             merged = new_df
        #         with pd.ExcelWriter(agg_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        #             merged.to_excel(w, index=False, sheet_name=sheet_name)
        #     else:
        #         with pd.ExcelWriter(agg_path, engine="openpyxl", mode="w") as w:
        #             new_df.to_excel(w, index=False, sheet_name=sheet_name)
    
        # _append_sheet(aggregate_path, "ModelResults", df_combined)
        # _append_sheet(aggregate_path, "WordCloud",  wordcloud_df)
        # if not summary_df.empty:
        #     _append_sheet(aggregate_path, "SHAP貢獻區間", summary_df)
    
        # print(f"📚 已彙整到：{aggregate_path}")
        
        # 當次 run 的彙整 CSV（統一放在 results/<timestamp>/ 下）
        agg_model_csv    = os.path.join(results_dir, f"ALL_ModelResults_{timestamp}.csv")
        agg_wordcloud_csv= os.path.join(results_dir, f"ALL_WordCloud_{timestamp}.csv")
        agg_shap_csv     = os.path.join(results_dir, f"ALL_SHAP_ranges_{timestamp}.csv")
        
        safe_append_csv(agg_model_csv, df_combined)
        safe_append_csv(agg_wordcloud_csv, wordcloud_df)
        if 'summary_df' in locals() and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            safe_append_csv(agg_shap_csv, summary_df)
        
        print(f"📚 已彙整到（CSV）：\n  - {agg_model_csv}\n  - {agg_wordcloud_csv}")
        if 'summary_df' in locals() and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            print(f"  - {agg_shap_csv}")

    
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
        
        
        # # retrain_model_label.py 822
        # 個人化建議
        # sample_df = df_combined.sample(n=100, random_state=42)
        # personal_recos_sample = make_personal_recos(sample_df,               # 也可挑 top-N 或特定名單
        #                                 variables=numerical_cols,            # 想要提供建議的變數清單
        #                                 summary_df=summary_df,               # 你剛跑出的 SHAP 正向區間
        #                                 uplift_pp=0.05,                      # 目標提升 5 個百分點
        #                                 mode="interval"                      # "interval" or "grid")
        #                                 )
        
        # === 產生個人化建議（每列 × 每個變數）===
        # 用 SHAP 正向區間直接跳（建議做法）
        # personal_recos = make_personal_recos(
        #     df_source=df_combined,               # 也可挑 top-N 或特定名單
        #     variables=numerical_cols,            # 想要提供建議的變數清單
        #     summary_df=summary_df,               # 你剛跑出的 SHAP 正向區間
        #     uplift_pp=0.05,                      # 目標提升 5 個百分點
        #     mode="interval"                      # "interval" or "grid"
        # )
        # 也可選 grid 搜尋寫法（範例）：
        # personal_recos = make_personal_recos(
        #     df_source=df_combined,
        #     variables=["件數","每週平均拜訪客戶數"],
        #     summary_df=summary_df,
        #     uplift_pp=0.05,
        #     mode="grid",
        #     grid_cfg={"step":1.0, "steps":30}
        # )
        
        # # === 增加建議說明欄位 ===
        # def make_note(row):
        #     try:
        #         if pd.notna(row["目標值"]) and pd.notna(row["提升(百分點)"]):
        #             uplift = float(row["提升(百分點)"])
        #             return f"將【{row['變數']}】調整到 {row['目標值']}，成交率可能提升 {uplift:.1%}"
        #         else:
        #             return None
        #     except Exception:
        #         return None
        
        # personal_recos["建議說明"] = personal_recos.apply(make_note, axis=1)
        
        # # 加上識別欄位並輸出
        # personal_recos["timestamp"] = timestamp
        # personal_recos["strategy_id"] = strategy_id
        
        # all_personal_recos.append(personal_recos)
        
        # out_csv = os.path.join(results_dir, f"personal_recommendations_strategy_{strategy_id}.csv")
        # personal_recos.to_csv(out_csv, index=False, encoding="utf-8-sig")
        # print(f"✅ 個人化建議已輸出：{out_csv}")
        
        out_csv = os.path.join(results_dir, f"personal_recommendations_strategy_{strategy_id}.csv")

        # df_combined = df_combined.copy()
        # if "timestamp" not in df_combined.columns:
        #     df_combined["timestamp"] = timestamp
        # if "strategy_id" not in df_combined.columns:
        #     df_combined["strategy_id"] = strategy_id
        
        make_personal_recos_streaming(
            df_source=df_combined,
            variables=raw_export_cols,          # 或你的「可調變數白名單」
            summary_df=summary_df,
            uplift_pp=0.05,
            mode="interval",
            out_csv=out_csv,
            flush_every=20000,
            extra_id_cols=["timestamp","strategy_id"]
        )
        print(f"✅ 個人化建議（含建議說明）已串流輸出：{out_csv}")
        
        
    # ====== 迴圈結束後：一次寫入歷史檔 ======
    # 1) 先把本次訓練的資料組起來
    new_monitoring_df = pd.DataFrame(all_monitoring_rows)
    new_shap_df = pd.concat(all_summary_dfs, ignore_index=True) if all_summary_dfs else pd.DataFrame()
    
    # # 2) 歷史檔放在 results 根目錄（不是 timestamp 子資料夾）
    # history_path = os.path.join("D:/備註文字探勘/results", "model_monitoring.xlsx")
    
    # # 3) 如果歷史檔存在，先讀舊資料（若某個 sheet 不存在，就用空DF）
    # if os.path.exists(history_path):
    #     xls = pd.ExcelFile(history_path)
    #     if "summary_log" in xls.sheet_names:
    #         old_summary_df = pd.read_excel(history_path, sheet_name="summary_log")
    #     else:
    #         old_summary_df = pd.DataFrame()
    #     if "shap_ranges_log" in xls.sheet_names:
    #         old_shap_df = pd.read_excel(history_path, sheet_name="shap_ranges_log")
    #     else:
    #         old_shap_df = pd.DataFrame()
    # else:
    #     old_summary_df = pd.DataFrame()
    #     old_shap_df = pd.DataFrame()
    
    # # 4) 合併舊+新
    # summary_out = pd.concat([old_summary_df, new_monitoring_df], ignore_index=True)
    # if not new_shap_df.empty:
    #     shap_out = pd.concat([old_shap_df, new_shap_df], ignore_index=True)
    # else:
    #     shap_out = old_shap_df
    
    # # 5) 寫回（覆寫 sheet 內容，但保留同一個檔案）
    # with pd.ExcelWriter(history_path, engine="openpyxl") as writer:
    #     summary_out.to_excel(writer, sheet_name="summary_log", index=False)
    #     if not shap_out.empty:
    #         shap_out.to_excel(writer, sheet_name="shap_ranges_log", index=False)
    
    # print(f"📈 歷史監控檔已更新：{history_path}")
    
    history_summary_csv = os.path.join("D:/備註文字探勘/results", "summary_log.csv")
    history_shap_csv    = os.path.join("D:/備註文字探勘/results", "shap_ranges_log.csv")
    
    if not new_monitoring_df.empty:
        append_csv(history_summary_csv, new_monitoring_df)
    if not new_shap_df.empty:
        append_csv(history_shap_csv, new_shap_df)
    
    print(f"📈 歷史監控 CSV 已更新：\n  - {history_summary_csv}")
    if not new_shap_df.empty:
        print(f"  - {history_shap_csv}")
    
    
    # === strategy 迴圈結束後，統一合併與輸出 ===
    # if all_heatmap_longs:
    #     merged_heatmap_long = pd.concat(all_heatmap_longs, ignore_index=True)
    #     heatmap_out_path = os.path.join(results_dir, f"heatmap_long_all_{timestamp}.csv")
    #     merged_heatmap_long.to_csv(heatmap_out_path, index=False, encoding="utf-8-sig")
    #     print(f"✅ heatmap_long 輸出完成：{heatmap_out_path}")
    # else:
    #     print("⚠️ 沒有任何 heatmap_long 資料可輸出")
    # 迴圈外
    if all_shap_wide_list:
        all_wide = pd.concat(all_shap_wide_list, ignore_index=True)
        all_wide.to_csv(os.path.join(results_dir, f"ALL_strategies_SHAP_wide_{timestamp}.csv"),
                        index=False, encoding="utf-8-sig")
    
    # if all_heatmap_longs:
    #     merged = pd.concat(all_heatmap_longs, ignore_index=True)
    #     merged.to_csv(os.path.join(results_dir, f"ALL_heatmap_shap_long_{timestamp}.csv"),
    #                   index=False, encoding="utf-8-sig")
    
    # if all_personal_recos:
    #     df_all_recos = pd.concat(all_personal_recos, ignore_index=True)
    #     out_path = os.path.join(results_dir, f"ALL_personal_recommendations_{timestamp}.csv")
    #     df_all_recos.to_csv(out_path, index=False, encoding="utf-8-sig")
    #     print(f"✅ 個人化建議（全部策略）輸出完成：{out_path}")
    # else:
    #     print("⚠️ 本批未產出任何個人化建議")
    
    import glob
    
    all_out = os.path.join(results_dir, f"ALL_personal_recommendations_{timestamp}.csv")
    if os.path.exists(all_out):
        os.remove(all_out)
    
    header_written = False
    for fp in sorted(glob.glob(os.path.join(results_dir, "personal_recommendations_strategy_*.csv"))):
        header_written = append_csv_fast(fp, all_out, header_written)
    
    print(f"📦 已彙整 personal_recommendations → {all_out}")
    
    return df_combined, model
        
    
# train_model_pipeline_with_strategies(df_ready, policy_df)