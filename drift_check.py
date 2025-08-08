# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 09:42:34 2025

@author: Z01788
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# === 1) 尋找最新的 train_reference（支援多 strategy）===
def get_latest_train_reference_path(
    base_dir="D:/備註文字探勘/models",
    strategy_ids=(0, 2, 6),
    require_all=True,
    use_mtime=False,
):
    """
    回傳:
      - require_all=True  且三組都在同一個 timestamp：{sid: path}
      - require_all=False 每個 sid 各自找最近的一份：{sid: path}
      - 找不到則回傳 None
    """
    if not os.path.isdir(base_dir):
        return None

    # 蒐集類似 YYYYMMDD_* 的子資料夾
    entries = []
    with os.scandir(base_dir) as it:
        for e in it:
            if e.is_dir():
                name = e.name
                if len(name) >= 8 and name[:8].isdigit():
                    entries.append((e.path, e.stat().st_mtime))

    if not entries:
        return None

    # 排序：最新在前
    if use_mtime:
        entries.sort(key=lambda x: x[1], reverse=True)
    else:
        entries.sort(key=lambda x: os.path.basename(x[0]), reverse=True)

    # 嚴格模式：三組都在同一個 timestamp
    if require_all:
        for folder, _ in entries:
            ok = True
            refs = {}
            for sid in strategy_ids:
                p = os.path.join(folder, f"strategy_{sid}", "train_reference.csv")
                if os.path.exists(p):
                    refs[sid] = p
                else:
                    ok = False
                    break
            if ok:
                return refs
        return None

    # 鬆綁模式：每個 sid 自己找最近的一份
    latest_per_sid = {}
    for sid in strategy_ids:
        for folder, _ in entries:
            p = os.path.join(folder, f"strategy_{sid}", "train_reference.csv")
            if os.path.exists(p):
                latest_per_sid[sid] = p
                break
    return latest_per_sid if latest_per_sid else None


# === 2) 單批 KS 檢定 ===
def detect_feature_drift(new_df, ref_df, columns, p_threshold=0.05, min_n=50):
    """
    對 columns 逐欄做 KS 檢定，回傳 {col: {'p': pval, 'n_new': n1, 'n_ref': n2}} 或 None。
    僅對數值欄做檢定；樣本不足則回 None。
    """
    out = {}
    for col in columns:
        if col not in new_df.columns or col not in ref_df.columns:
            out[col] = None
            continue

        new_vals = pd.to_numeric(new_df[col], errors="coerce").dropna().values
        ref_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna().values

        # 樣本太少，不檢
        if len(new_vals) < min_n or len(ref_vals) < min_n:
            out[col] = {"p": None, "n_new": len(new_vals), "n_ref": len(ref_vals)}
            continue

        stat, pval = ks_2samp(new_vals, ref_vals)
        out[col] = {"p": float(pval), "n_new": len(new_vals), "n_ref": len(ref_vals)}
    return out


# === 3) 高階入口：支援單一路徑 / 多 strategy / 自動尋找 ===
def check_drift_and_warn(
    new_df,
    ref_source=None,           # None | str | dict
    selected_cols=None,
    stop_if_drift=False,
    p_threshold=0.05,
    require_all=True,          # 當 ref_source=None 時，是否強制同一 timestamp 下三組都齊
    strategy_ids=(0, 2, 6),
):
    """
    - ref_source=None：自動找最新 train_reference（看 require_all）。
    - ref_source=str：單一路徑，視為 'default' 策略。
    - ref_source=dict：{sid: path}，多策略路徑。

    回傳：
      True/False （是否允許繼續）
    """
    # 預設檢查欄位（建議傳入你實際用於模型的數值欄）
    if not selected_cols:
        selected_cols = [
            '備註字數', '件數', '總保費', '拜訪目的', '業務客戶年齡差距',
            '上半年準客戶數', '最近半年活動參與率', '上一個半年度FYC'
        ]

    # 取得參考資料來源
    refs = None
    if ref_source is None:
        refs = get_latest_train_reference_path(
            base_dir="D:/備註文字探勘/models",
            strategy_ids=strategy_ids,
            require_all=require_all,
        )
        if refs is None:
            print("⚠️ 找不到任何 train_reference.csv，略過資料漂移檢查（允許繼續）")
            return True
    elif isinstance(ref_source, str):
        refs = {"default": ref_source}
    elif isinstance(ref_source, dict):
        refs = ref_source
    else:
        print("⚠️ ref_source 型別不支援，略過資料漂移檢查（允許繼續）")
        return True

    # 逐個策略做檢定
    overall_alert = False
    for sid, path in refs.items():
        try:
            ref_df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ 無法讀取參考資料（strategy={sid}）：{e}，略過此策略")
            continue

        results = detect_feature_drift(new_df, ref_df, selected_cols, p_threshold=p_threshold)

        print(f"\n📊 [Drift 檢查] strategy={sid} 參考檔：{path}")
        drift_alerts = []
        for col, info in results.items():
            if info is None:
                print(f"  - {col}: 缺欄位或樣本不足")
                continue
            p = info["p"]
            n_new, n_ref = info["n_new"], info["n_ref"]
            if p is None:
                print(f"  - {col}: 樣本不足（new={n_new}, ref={n_ref}）→ 略過")
                continue
            flag = (p < p_threshold)
            print(f"  - {col}: {'❌' if flag else '✅'} p={p:.4f} (new n={n_new}, ref n={n_ref})")
            if flag:
                drift_alerts.append(col)

        if drift_alerts:
            overall_alert = True
            print(f"⚠️ strategy={sid} 偵測到 {len(drift_alerts)} 個特徵分布顯著不同：{drift_alerts}")
        else:
            print(f"✅ strategy={sid} 未偵測到顯著資料漂移")

    if overall_alert and stop_if_drift:
        print("🚫 停止：因偵測到顯著資料漂移（stop_if_drift=True）")
        return False

    return True

