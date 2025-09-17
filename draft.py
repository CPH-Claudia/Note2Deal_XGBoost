# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 09:38:01 2025

@author: Z01788
"""
# %% Heatmap_Long.csv
import os
import numpy as np
import pandas as pd

def make_heatmap_file_from_results(results_path, sheet="ModelResults", out_csv_path=None):
    """
    從已存在的預測結果 Excel（含 ModelResults）生成直向 heatmap 檔。
    - 讀取 results_path 的 `sheet`（預設 ModelResults）
    - 依據 *_命中 / *_距離最近 / *_最近方向 / *_最近區間_起 / *_最近區間_迄 組裝直向表
    - 直向表欄位：timestamp, strategy_id, 客戶UUID, 拜訪紀錄UUID, is_holdout, label, pred_prob,
                 變數, 原始值, 命中, 距離最近, 最近方向, 最近區間_起, 最近區間_迄
    - 會同時：
        1) 在原 Excel 內覆寫/新增工作表 `Heatmap_Long`
        2) 輸出一份同名 *_heatmap_long.csv（或 out_csv_path 指定的路徑）
    """
    df = pd.read_csv(results_path)

    # 識別欄位（存在才帶）
    id_candidates = ["timestamp", "strategy_id", "客戶UUID", "拜訪紀錄UUID", "is_holdout", "label", "pred_prob"]
    id_cols = [c for c in id_candidates if c in df.columns]

    # 找出所有 *_命中 欄位 → 推斷變數
    hit_suffix = "_命中"
    dist_suffix = "_距離最近"
    dir_suffix  = "_最近方向"
    lo_suffix   = "_最近區間_起"
    hi_suffix   = "_最近區間_迄"

    hit_cols = [c for c in df.columns if c.endswith(hit_suffix)]
    if not hit_cols:
        raise ValueError("此結果檔沒有任何 *_命中 欄位可以轉為直向結構。")

    variables = [c[:-len(hit_suffix)] for c in hit_cols]

    # === 一次性補齊缺少的衍生欄位，避免 fragmentation ===
    need_cols = {}
    for var in variables:
        c_hit = f"{var}{hit_suffix}"
        c_dst = f"{var}{dist_suffix}"
        c_dir = f"{var}{dir_suffix}"
        c_lo  = f"{var}{lo_suffix}"
        c_hi  = f"{var}{hi_suffix}"

        # 命中預設 0，其他預設 NaN（若不存在）
        if c_hit not in df.columns:
            need_cols[c_hit] = pd.Series(0, index=df.index, dtype="Int8")
        if c_dst not in df.columns:
            need_cols[c_dst] = pd.Series(np.nan, index=df.index, dtype="float")
        if c_dir not in df.columns:
            need_cols[c_dir] = pd.Series(np.nan, index=df.index, dtype="object")
        if c_lo not in df.columns:
            need_cols[c_lo]  = pd.Series(np.nan, index=df.index, dtype="float")
        if c_hi not in df.columns:
            need_cols[c_hi]  = pd.Series(np.nan, index=df.index, dtype="float")

    if need_cols:
        df = pd.concat([df, pd.DataFrame(need_cols)], axis=1)

    # === 建直向表 ===
    long_parts = []
    for var in variables:
        cols_map = {
            "原始值": var,
            "命中": f"{var}{hit_suffix}",
            "距離最近": f"{var}{dist_suffix}",
            "最近方向": f"{var}{dir_suffix}",
            "最近區間_起": f"{var}{lo_suffix}",
            "最近區間_迄": f"{var}{hi_suffix}",
        }
        # 原始值欄位一定要存在（即變數本身）
        if cols_map["原始值"] not in df.columns:
            # 假如不存在，就跳過這個變數
            continue

        select_cols = id_cols + list(cols_map.values())
        tmp = df[select_cols].copy()
        tmp = tmp.rename(columns={
            cols_map["原始值"]: "原始值",
            cols_map["命中"]: "命中",
            cols_map["距離最近"]: "距離最近",
            cols_map["最近方向"]: "最近方向",
            cols_map["最近區間_起"]: "最近區間_起",
            cols_map["最近區間_迄"]: "最近區間_迄",
        })
        tmp["變數"] = var
        # 命中轉 tiny int
        if "命中" in tmp.columns:
            try:
                tmp["命中"] = tmp["命中"].astype("Int8")
            except Exception:
                tmp["命中"] = pd.to_numeric(tmp["命中"], errors="coerce").fillna(0).astype("Int8")

        long_parts.append(tmp)

    if not long_parts:
        print("⚠️ 沒有可用變數能建立直向表。")
        return pd.DataFrame()

    heatmap_long = pd.concat(long_parts, ignore_index=True)

    # 欄位順序美化
    front = [c for c in ["timestamp", "strategy_id", "客戶UUID", "拜訪紀錄UUID", "is_holdout", "label", "pred_prob"] if c in heatmap_long.columns]
    cols = front + ["變數", "原始值", "命中", "距離最近", "最近方向", "最近區間_起", "最近區間_迄"]
    heatmap_long = heatmap_long[cols]

    # === 輸出 CSV ===
    if out_csv_path is None:
        base, ext = os.path.splitext(results_path)
        out_csv_path = f"{base}_heatmap_long.csv"
    heatmap_long.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出 CSV：{out_csv_path}")

    # # === 回寫 Excel 的 Heatmap_Long 工作表 ===
    # try:
    #     with pd.ExcelWriter(results_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    #         heatmap_long.to_excel(w, sheet_name="Heatmap_Long", index=False)
    #     print(f"✅ 已在 {os.path.basename(results_path)} 內覆蓋/新增工作表：Heatmap_Long")
    # except Exception as e:
    #     # 有些舊版 xlsx 或被保護、或檔案開啟中，可能寫不回
    #     print(f"⚠️ 無法寫回 Excel（已產出 CSV）：{e}")

    return heatmap_long

# 單檔轉出
make_heatmap_file_from_results(r"D:\備註文字探勘\results\20250828_184607\ALL_ModelResults_20250828_184607.csv")



# %% rebuild_all_personal_recos.py
import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer



# === 把你「目前使用中的」 make_personal_recos 貼到這裡 ===
# === 直接貼上，取代原本 NotImplementedError 那段 ===
# 近鄰正向目標點（依 SHAP 區間）
def _nearest_positive_targets(v, intervals: np.ndarray):
    """
    v: 原始值（可為 float 或 NaN/NA）
    intervals: shape (k,2) 的 ndarray，每列 [lo, hi]（皆為 float）
    回傳：若 v 在任一區間 → [(v, 'in')]
         否則回傳就近區間的候選點（左/中/75%/右邊界），去重後依序列出
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if intervals.size == 0:
        return []

    # 命中任何區間
    for lo, hi in intervals:
        if lo <= v <= hi:
            return [(float(v), "in")]

    # 找 v 位於各區間的相對位置
    los = intervals[:, 0]
    his = intervals[:, 1]
    idx_next = np.searchsorted(los, v, side="left")

    def points_of(lo, hi):
        mid = (lo + hi) / 2.0
        q75 = lo + 0.75 * (hi - lo)
        return [(lo, "edge"), (mid, "mid"), (q75, "q75"), (hi, "edge")]

    if idx_next == 0:
        lo, hi = intervals[0]
        cand = points_of(lo, hi)
    elif idx_next >= len(intervals):
        lo, hi = intervals[-1]
        cand = points_of(lo, hi)
    else:
        lo_prev, hi_prev = intervals[idx_next - 1]
        lo_next, hi_next = intervals[idx_next]
        d_prev = max(v - hi_prev, 0.0)
        d_next = max(lo_next - v, 0.0)
        cand = points_of(*( (lo_next, hi_next) if d_next < d_prev else (lo_prev, hi_prev) ))

    # 去重（避免重複點）
    out, seen = [], set()
    for t, tag in cand:
        key = round(float(t), 10)
        if key not in seen:
            out.append((float(t), tag))
            seen.add(key)
    return out

def recommend_by_interval_jump(row: pd.Series, var: str, summary_df: pd.DataFrame,
                               uplift_pp: float = 0.05) -> dict:
    """
    依 SHAP 正向區間，對單一 row+變數 var 提出建議：
      - 若當前值命中 → 建議維持（uplift=0）
      - 若未命中 → 試就近區間的幾個代表點（左/中/75%/右），取能讓機率最高、且 >= uplift_pp 的第一個點
    需要：外部已宣告 global model_predict_proba_fn(df_rows)->np.ndarray[proba]
    """
    sub = summary_df[summary_df["變數"] == var].dropna(subset=["原始值區間_起", "原始值區間_迄"])
    sub = sub.drop_duplicates(subset=["原始值區間_起", "原始值區間_迄"]).sort_values(["原始值區間_起", "原始值區間_迄"])
    if sub.empty:
        return {"var": var, "reason": "no_positive_interval"}

    intervals = sub[["原始值區間_起", "原始值區間_迄"]].to_numpy(dtype=float)
    v0 = row.get(var, np.nan)
    v0f = None if pd.isna(v0) else float(v0)

    # 當前機率
    cur_p = float(model_predict_proba_fn(pd.DataFrame([row]))[0])

    # 找目標候選
    targets = _nearest_positive_targets(v0f, intervals)
    if not targets:
        return {"var": var, "current_proba": cur_p, "reason": "no_target"}

    # 命中：維持現值
    if targets[0][1] == "in":
        return {
            "var": var,
            "current_value": v0f,
            "target_value": v0f,
            "current_proba": cur_p,
            "new_proba": cur_p,
            "uplift": 0.0,
            "met_uplift": True,
            "note": "already_in_positive_interval",
        }

    # 未命中 → 嘗試候選點
    best = None
    for tgt, tag in targets:
        trial = row.copy()
        trial[var] = tgt
        p_new = float(model_predict_proba_fn(pd.DataFrame([trial]))[0])
        upl = p_new - cur_p
        if (best is None) or (p_new > best["new_proba"]):
            best = {
                "var": var,
                "current_value": v0f,
                "target_value": float(tgt),
                "current_proba": cur_p,
                "new_proba": p_new,
                "uplift": upl,
                "met_uplift": upl >= uplift_pp,
                "picked_point": tag,
            }

    return best or {"var": var, "current_proba": cur_p, "reason": "no_improvement"}

def recommend_by_grid(row: pd.Series, var: str, direction: str = "up",
                      step: float = 1.0, steps: int = 30,
                      uplift_pp: float = 0.05,
                      lo_cap: float = None, hi_cap: float = None) -> dict:
    """
    細步長搜尋（備用方案）：
      - direction: 'up' or 'down'
      - 逐步嘗試 var += step / -= step，最多 steps 次；若越界(lo_cap/hi_cap)就停止
    """
    v0 = row.get(var, np.nan)
    v0f = None if pd.isna(v0) else float(v0)
    cur_p = float(model_predict_proba_fn(pd.DataFrame([row]))[0])
    best = {
        "var": var, "current_value": v0f, "target_value": None,
        "current_proba": cur_p, "new_proba": cur_p,
        "uplift": 0.0, "met_uplift": False
    }
    if v0f is None:
        return {**best, "reason": "nan_value"}

    for k in range(1, steps + 1):
        v_try = v0f + (step * k if direction == "up" else -step * k)
        if lo_cap is not None and v_try < lo_cap:
            break
        if hi_cap is not None and v_try > hi_cap:
            break
        trial = row.copy()
        trial[var] = v_try
        p_new = float(model_predict_proba_fn(pd.DataFrame([trial]))[0])
        upl = p_new - cur_p
        if p_new > best["new_proba"]:
            best.update({"target_value": float(v_try), "new_proba": p_new,
                         "uplift": upl, "met_uplift": upl >= uplift_pp})
            if upl >= uplift_pp:
                break
    return best

def make_personal_recos(df_source: pd.DataFrame,
                        variables: list,
                        summary_df: pd.DataFrame,
                        uplift_pp: float = 0.05,
                        mode: str = "interval",
                        grid_cfg: dict = None) -> pd.DataFrame:
    """
    針對 df_source 的每一列 × variables，產生個人化建議。
    需在外部先定義好：model_predict_proba_fn(df_rows) → ndarray 機率。
    mode = "interval"：用 SHAP 正向區間跳轉（建議）
    mode = "grid"：用細步長搜尋（備選）
    """
    if not isinstance(df_source, pd.DataFrame):
        df_source = pd.DataFrame(df_source)

    # 只保留 df_source 中存在的變數
    variables = [v for v in variables if v in df_source.columns]

    recs = []
    for idx, row in df_source.iterrows():
        for var in variables:
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

            recs.append({
                "idx": idx,
                "客戶UUID": row.get("客戶UUID", None),
                "拜訪紀錄UUID": row.get("拜訪紀錄UUID", None),
                "變數": var,
                "當前值": None if pd.isna(row.get(var, np.nan)) else float(row[var]),
                "當前機率": r.get("current_proba"),
                "目標值": r.get("target_value"),
                "新機率": r.get("new_proba"),
                "提升(百分點)": r.get("uplift"),
                "是否達標": r.get("met_uplift"),
                "說明": r.get("reason", r.get("picked_point", "")),
            })

    return pd.DataFrame(recs)


# ！！！唯一定義要改：函數內部所有預測一律呼叫 model_predict_proba_fn(df_rows)
# （例如把 _predict_proba_rows(...) 改成 model_predict_proba_fn(...)）
def make_personal_recos(df_source, variables, summary_df, uplift_pp=0.05, mode="interval", grid_cfg=None):
    raise NotImplementedError("請貼上你現用的 make_personal_recos，並把預測改叫 model_predict_proba_fn(df_rows)")

def _try_read_table(path_without_ext: str, sheet=None) -> pd.DataFrame:
    xlsx = path_without_ext + ".xlsx"
    csv  = path_without_ext + ".csv"
    if os.path.exists(xlsx):
        if sheet:
            return pd.read_excel(xlsx, sheet_name=sheet)
        return pd.read_excel(xlsx)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"找不到 {xlsx} 或 {csv}")

def _recompute_w2v_mean_vectors(df: pd.DataFrame, w2v_model, text_col="備註文字_處理") -> np.ndarray:
    dim = w2v_model.vector_size
    out = np.zeros((len(df), dim), dtype="float32")
    texts = df.get(text_col, "").fillna("").astype(str).values
    for i, txt in enumerate(texts):
        toks = txt.split()
        if not toks: 
            continue
        vecs = [w2v_model.wv[t] for t in toks if t in w2v_model.wv]
        if vecs:
            out[i, :] = np.mean(vecs, axis=0)
    return out

def rebuild_all_personal_recos(
    results_root: str,
    models_root: str,
    timestamp: str,
    strategy_ids=None,          # None = 從 ALL 檔自動偵測
    uplift_pp: float = 0.05,
    mode: str = "interval",
    out_name: str = None        # 預設：ALL_strategies_personal_recommendations_<ts>.csv
):
    results_dir = os.path.join(results_root, timestamp)
    models_dir  = os.path.join(models_root,  timestamp)
    assert os.path.isdir(results_dir), f"找不到 results 目錄：{results_dir}"
    assert os.path.isdir(models_dir),  f"找不到 models 目錄：{models_dir}"

    # 讀合併檔
    df_all   = _try_read_table(os.path.join(results_dir, f"ALL_strategies_ModelResults_{timestamp}"))
    shap_all = _try_read_table(os.path.join(results_dir, f"ALL_strategies_SHAP_ranges_{timestamp}"))

    # 目標策略
    if strategy_ids is None:
        if "strategy_id" not in df_all.columns:
            raise ValueError("ALL ModelResults 缺少 strategy_id 欄位。")
        strategy_ids = tuple(sorted(df_all["strategy_id"].dropna().unique().tolist()))

    recos_all = []

    for sid in strategy_ids:
        print(f"\n=== 補跑 personal_recos：strategy {sid} ===")
        df_combined = df_all[df_all["strategy_id"] == sid].copy()
        summary_df  = shap_all[shap_all["strategy_id"] == sid].copy()
        if df_combined.empty:
            print("  ⚠️ 沒有資料，跳過")
            continue
        if summary_df.empty or "變數" not in summary_df.columns:
            print("  ⚠️ 沒有 SHAP 正向區間，跳過")
            continue

        # 變數以 SHAP 的「變數」且實際存在於 df_combined、且為數值 dtype 為準
        cand_vars = sorted(set(summary_df["變數"].dropna()) & set(df_combined.columns))
        num_vars  = [c for c in cand_vars if pd.api.types.is_numeric_dtype(df_combined[c])]
        if not num_vars:
            print("  ⚠️ 找不到數值型變數，跳過")
            continue

        # 載模型與向量
        sdir   = os.path.join(models_dir, f"strategy_{sid}")
        f_model= os.path.join(sdir, "model_final.pkl")
        f_w2v  = os.path.join(sdir, "word2vec_model.pkl")
        f_feat = os.path.join(sdir, "feature_names.pkl")
        f_idx  = os.path.join(sdir, "w2v_top_indices.pkl")
        if not (os.path.exists(f_model) and os.path.exists(f_w2v)):
            print("  ⚠️ 缺模型或 W2V，跳過")
            continue
        model      = joblib.load(f_model)
        w2v_model  = joblib.load(f_w2v)
        feat_names = joblib.load(f_feat) if os.path.exists(f_feat) else None
        if os.path.exists(f_idx):
            w2v_top_indices = joblib.load(f_idx)
        else:
            if feat_names is None:
                print("  ⚠️ 沒有 w2v_top_indices / feature_names，跳過")
                continue
            w2v_top_indices = np.array([int(f.split("_")[1]) for f in feat_names if str(f).startswith("w2v_")])

        # 用 is_holdout==0 的切片估計 means/stds
        df_train = df_combined[df_combined["is_holdout"].eq(0)] if "is_holdout" in df_combined.columns else df_combined
        means = df_train[num_vars].mean(numeric_only=True, skipna=True)
        stds  = df_train[num_vars].std(numeric_only=True,  skipna=True).replace(0, 1)

        # MLBin（若有）
        mlb = MultiLabelBinarizer()
        if "merged_tags" in df_train.columns:
            mlb.fit(df_train["merged_tags"].apply(lambda x: x if isinstance(x, list) else []))
        else:
            mlb.fit([[]])

        # 本地特徵：和訓練一致
        # def prepare_features_local(df_subset: pd.DataFrame) -> np.ndarray:
        #     X_num = df_subset[num_vars].copy()
        #     X_scaled = ((X_num - means) / stds).to_numpy()

        #     W = _recompute_w2v_mean_vectors(df_subset, w2v_model, text_col="備註文字_處理")
        #     W_top = W[:, w2v_top_indices]

        #     feat = np.hstack([W_top, X_scaled]).astype(np.float32, copy=False)

        #     if "merged_tags" in df_subset.columns and hasattr(mlb, "classes_"):
        #         tags = df_subset["merged_tags"].apply(lambda x: x if isinstance(x, list) else [])
        #         tags_bin = mlb.transform(tags)
        #         feat = np.hstack([feat, tags_bin])

        #     return feat
        
        def prepare_features_local(df_rows: pd.DataFrame) -> np.ndarray:
            """
            依照 models/<ts>/strategy_<sid> 內保存的 feature_names.pkl / w2v_top_indices.pkl / scaler.pkl
            來重建與訓練時完全相同「順序與長度」的特徵向量。
            依賴外層已載入：
              - feature_names: List[str]
              - w2v_top_indices: np.ndarray (或 list[int])
              - scaler (可為 None：若不存在就不做標準化)
              - w2v_model: gensim Word2Vec（若當初推論是用詞向量平均）
            並假設 df_rows 具備欄：
              - '備註文字_處理'（已斷詞、以空白分隔）
              - 所有數值欄（與訓練相同名稱）
              - 'merged_tags'（list[str]；若沒有，會以空 list 補）
            """
            # 1) 解析最終欄位：分成 w2v_* / 數值 / tag 三群
            final_feats = list(feat_names)  # 由模型目錄讀到的最終特徵順序（長度=expected）
            w2v_cols = [c for c in final_feats if c.startswith("w2v_")]
        
            # 數值欄：訓練時就是 numerical_cols；從 final_feats 扣掉 w2v_ 與 tag 即可
            # 在我們的保存策略裡，tag 欄就是剩下的那些不是 w2v_ 且不是數值欄名的字串。
            # 但補跑時你不一定知道訓練時的 numerical_cols 名單，
            # 因此我們要根據 df_rows 的 dtype 來判斷「在 final_feats 裡同時也存在於 df_rows 且為 numeric 的」當整數值欄。
            def is_numeric_series(s):
                return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        
            num_cols = []
            tag_cols = []
            for c in final_feats:
                if c in w2v_cols:
                    continue
                if c in df_rows.columns and is_numeric_series(df_rows[c]):
                    num_cols.append(c)
                elif c.startswith("w2v_"):
                    # 已處理
                    pass
                else:
                    # 其餘視為 tag one-hot 欄位名稱（= 當初的 mlb.classes_）
                    tag_cols.append(c)
        
            # 2) 準備 W2V（取訓練時的 top indices）
            def vectorize_sentence_weighted(sentence_tokens: list) -> np.ndarray:
                # TF-IDF 權重若當初訓練時沒保存，補跑就用平均向量；重點是長度要對且 indices 要一致
                vecs = []
                for w in sentence_tokens:
                    if w in w2v_model.wv:
                        vecs.append(w2v_model.wv[w])
                if vecs:
                    arr = np.mean(np.vstack(vecs), axis=0)
                else:
                    arr = np.zeros(w2v_model.vector_size, dtype=np.float32)
                return arr
        
            tokens_list = df_rows.get('備註文字_處理', pd.Series([""]*len(df_rows))).fillna("").astype(str).apply(lambda s: s.split())
            w2v_all = np.vstack([vectorize_sentence_weighted(toks) for toks in tokens_list])
            # 只取訓練時的 top indices
            w2v_top = w2v_all[:, np.array(w2v_top_indices, dtype=int)]
        
            # 3) 數值欄：照名稱抓，缺的補 NaN
            X_num = pd.DataFrame(index=df_rows.index)
            for c in num_cols:
                if c in df_rows.columns:
                    X_num[c] = df_rows[c]
                else:
                    X_num[c] = np.nan
        
            X_num_scaled = X_num.values
        
            # 4) tag one-hot：根據「final_feats 中的 tag_cols 名」與 df_rows['merged_tags'] 做一致 one-hot
            tags_series = df_rows.get("merged_tags", pd.Series([[]]*len(df_rows)))
            # 容錯：把非 list 的都轉成 list
            tags_series = tags_series.apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [str(x)]))
            tag_mat = np.zeros((len(df_rows), len(tag_cols)), dtype=np.int8)
            if len(tag_cols) > 0:
                tag_index = {t: j for j, t in enumerate(tag_cols)}
                for i, tags in enumerate(tags_series):
                    for t in tags:
                        j = tag_index.get(str(t).lower())
                        if j is not None:
                            tag_mat[i, j] = 1
        
            # 5) 依照 final_feats 的順序組合
            parts = []
            # w2v
            parts.append(w2v_top)
            # 數值
            if len(num_cols) > 0:
                parts.append(X_num_scaled)
            # tag
            if len(tag_cols) > 0:
                parts.append(tag_mat)
        
            X_final = np.hstack(parts).astype(np.float32, copy=False)
        
            # 安全檢查：長度是否完全吻合
            if X_final.shape[1] != len(final_feats):
                print("[!] feature length mismatch after assemble:",
                      "built =", X_final.shape[1], "expected =", len(final_feats))
                # 額外 debug：列出缺失/多出的欄位（可開發時用）
                # 這裡就不展開以避免太長
                raise ValueError(f"Feature shape mismatch after assemble: got {X_final.shape[1]}, expected {len(final_feats)}")
        
            return X_final
        

        # 注入：讓 make_personal_recos 叫這個做預測
        global model_predict_proba_fn
        def model_predict_proba_fn(df_rows: pd.DataFrame) -> np.ndarray:
            X = prepare_features_local(df_rows)
            return model.predict_proba(X)[:, 1]

        recos = make_personal_recos(
            df_source=df_combined,
            variables=num_vars,
            summary_df=summary_df,
            uplift_pp=uplift_pp,
            mode=mode
        )
        recos["timestamp"]   = timestamp
        recos["strategy_id"] = sid
        recos_all.append(recos)

    if not recos_all:
        print("\n⚠️ 沒有任何可輸出的個人化建議")
        return

    df_out = pd.concat(recos_all, ignore_index=True)
    out_name = out_name or f"ALL_strategies_personal_recommendations_{timestamp}.csv"
    out_path = os.path.join(results_dir, out_name)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 合併輸出完成：{out_path}")

# === 實際執行：改成你的路徑與這批 timestamp ===
rebuild_all_personal_recos(
    results_root="D:/備註文字探勘/results",
    models_root="D:/備註文字探勘/models",
    timestamp="20250822_183531",
    strategy_ids=None,            # None=自動偵測 ALL 裡有哪些 strategy_id
    uplift_pp=0.05,
    mode="interval",
)


# 路徑換成你實際 personal_recos 的 CSV 檔案
csv_path = r"D:/備註文字探勘/results/20250822_183531/ALL_strategies_personal_recommendations_20250822_183531.csv"
df = pd.read_csv(csv_path)

# 新增建議說明欄位
def make_note(row):
    try:
        if pd.notna(row["目標值"]) and pd.notna(row["提升(百分點)"]):
            uplift = float(row["提升(百分點)"])
            return f"將【{row['變數']}】調整到 {row['目標值']}，成交率可能提升 {uplift:.1%}"
        else:
            return None
    except Exception:
        return None

df["建議說明"] = df.apply(make_note, axis=1)

# 覆寫或另存新檔
out_path = os.path.splitext(csv_path)[0] + "_with_note.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")


# %% 合併同一拜訪的所有建議
""" 
只保留業務可控制的變數（例如：['每週平均拜訪客戶數','平均拜訪間隔天數']）
依照提升百分比排序，拼接建議文字 
"""
import pandas as pd
import csv

# 載入結果
csv_path = r"D:/備註文字探勘/results/20250916_113038/ALL_personal_recommendations_20250916_113038.csv"
long = pd.read_csv(csv_path)

# 定義業務可控制的變數
controllable_vars = ['每週平均拜訪客戶數', '平均拜訪間隔天數', '備註字數', '有意義詞數', 'hashtag_count']

# 過濾出可控變數
df_ctrl = long[long["變數"].isin(controllable_vars)].copy()

# 轉成數值 & 過濾掉 uplift <= 0
df_ctrl["提升(百分點)"] = pd.to_numeric(df_ctrl["提升(百分點)"], errors="coerce")
df_ctrl = df_ctrl[df_ctrl["提升(百分點)"] > 0].copy()

# 建議文字：四捨五入到小數點2位
def format_suggestion(row):
    target_val = None if pd.isna(row["目標值"]) else round(float(row["目標值"]), 2)
    uplift_pct = row["提升(百分點)"] * 100  # 轉百分比
    return f"將【{row['變數']}】調整到 {target_val}，成交率可能提升 {uplift_pct:.2f}%"

df_ctrl["建議文字_fmt"] = df_ctrl.apply(format_suggestion, axis=1)

# 依照 拜訪紀錄UUID 分組，排序後合併建議（換行拼接）
df_summary = (
    df_ctrl.sort_values(["拜訪紀錄UUID", "strategy_id", "提升(百分點)"], ascending=[True, True, False])
    .groupby(["拜訪紀錄UUID", "strategy_id"])["建議文字_fmt"]
    .apply(lambda x: "\n".join(x.dropna().tolist()))
    .reset_index()
)

# 合併回原始拜訪紀錄層級
df_visit = long.drop_duplicates(subset=["拜訪紀錄UUID", "strategy_id"])[
    ["客戶UUID", "拜訪紀錄UUID", "timestamp", "strategy_id", "當前機率"]
]

df_final = df_visit.merge(df_summary, on=["拜訪紀錄UUID", "strategy_id"], how="left")

out_path = r"D:/備註文字探勘/results/20250916_113038/personal_recos_summary1.csv"
df_final.to_csv(out_path, quoting=csv.QUOTE_ALL, index=False, encoding="utf-8-sig")
print(f"✅ 已輸出合併後檔案：{out_path}")


    
# safe_write_csv(df_final, out_path = out_path)
# df_t = df_final.tail(10)

# df_ready = pd.read_csv(r"D:/備註文字探勘/results/20250822_183531/ALL_strategies_ModelResults_20250822_183531.csv")
# heatmap = pd.read_csv(r"D:/備註文字探勘/results/20250822_183531/heatmap_long_all_strategies.csv")
# heatmap.tail(10)

# %% 重整 csv 格式
import pandas as pd
import json

# === 1. 讀取原始 CSV ===
path = r"D:/備註文字探勘/results/20250822_183531/ALL_strategies_ModelResults_20250822_183531.csv"
df = pd.read_csv(path, encoding="utf-8-sig")

# === 2. 確定欄位順序 (schema) ===
# 你可以用第一批欄位當模板，也可以自己手動指定
schema = list(df.columns)  # 假設現有的就是正確欄位
# schema = ["timestamp", "strategy_id", "客戶UUID", "拜訪紀錄UUID", "label", "pred_prob", ...] # 你可手動定義

# === 3. 修正所有資料 (萬一多批合併過) ===
# 讀進來的 df 如果欄位有出入，這裡會自動對齊
df_aligned = df.reindex(columns=schema)

# === 4. 輸出乾淨檔案 ===
out_path = path.replace(".csv", "_clean.csv")
df_aligned.to_csv(out_path, index=False, encoding="utf-8-sig")



import pandas as pd
import numpy as np
import json, re, csv
from pathlib import Path

def clean_object_col(s):
    """只處理 object 欄位：list/dict→JSON；字串移除換行與制表符。"""
    def _clean(x):
        # 保留 NaN
        if pd.isna(x):
            return ""
        # numpy arrays / list / dict / tuple / set → JSON 字串（保證有引號且不亂切欄）
        if isinstance(x, (list, dict, tuple, set, np.ndarray)):
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return json.dumps(str(x), ensure_ascii=False)
        # 其他轉字串並清掉 \r \n \t
        x = str(x)
        x = re.sub(r'[\r\n\t]+', ' ', x)
        return x
    return s.map(_clean)

def fix_csv_file(in_path, out_path=None, force_schema=None):
    """
    讀舊 CSV → 清理 → 重新輸出安全 CSV
    - force_schema: 若你想固定欄位順序，傳入一個欄位名列表
    """
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_clean.csv")

    # 讀檔（試著自動偵測分隔與編碼，若你都用逗號+utf-8-sig，可以固定）
    df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str)  # 先都讀成字串最安全

    # 如果你想保留數值型態，可以在這裡轉回（可選）
    # 例如：for c in some_numeric_cols: df[c] = pd.to_numeric(df[c], errors='coerce')

    # 只清 object 欄位（目前全部是 str，就會全清一次）
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = clean_object_col(df[c])

    # 對齊欄位順序（可選）
    if force_schema:
        # 缺的補空，多的丟掉
        df = df.reindex(columns=force_schema)

    # 寫出：強制每格加引號，避免任何逗號/特殊字元再度切欄
    df.to_csv(out_path, index=False, encoding="utf-8-sig",
              quoting=csv.QUOTE_ALL, lineterminator="\n")

    print(f"✅ 已輸出乾淨 CSV：{out_path}")
    # 驗證一下欄位數是否穩定
    test = pd.read_csv(out_path, encoding="utf-8-sig")
    print(f"    讀回檢查：shape={test.shape}, columns={len(test.columns)}")
    return out_path

# === 用法：對你那幾個彙整檔逐一清理 ===
files = [
    r"D:/備註文字探勘/results/20250822_183531/ALL_strategies_ModelResults_20250822_183531.csv"#,
    # r"D:/備註文字探勘/results/ALL_strategies_WordCloud_20250822_183531.csv",
    # r"D:/備註文字探勘/results/ALL_strategies_SHAP_ranges_20250822_183531.csv",
    # r"D:/備註文字探勘/results/personal_recommendations_ALL_20250822_183531.csv",  # 若有
]
for p in files:
    try:
        fix_csv_file(p)
    except FileNotFoundError:
        print(f"⚠️ 找不到檔案：{p}（略過）")
        
# %% 結果檔以 stratey id 分批
import pandas as pd

src = r"D:\備註文字探勘\results\20250828_184607\ALL_ModelResults_20250828_184607.csv"
df = pd.read_csv(src, encoding="utf-8-sig")

df2 = df[df["strategy_id"]==2]
df6 = df[df["strategy_id"]==6]

df2.to_csv(r"D:\tags\20250828_184607\strategy2.csv", index=False, encoding="utf-8-sig")
df6.to_csv(r"D:\tags\20250828_184607\strategy6.csv", index=False, encoding="utf-8-sig")

