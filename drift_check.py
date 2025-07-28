# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 09:42:34 2025

@author: Z01788
"""

# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import ks_2samp

def detect_feature_drift(new_df, ref_df, columns, p_threshold=0.05):
    drift_flags = {}
    for col in columns:
        if col in new_df.columns and col in ref_df.columns:
            stat, pval = ks_2samp(new_df[col].dropna(), ref_df[col].dropna())
            drift_flags[col] = pval
        else:
            drift_flags[col] = None
    return drift_flags

def check_drift_and_warn(new_df, ref_path, selected_cols=None, stop_if_drift=False):
    if not selected_cols:
        selected_cols = [
            '備註字數', '件數', '總保費', '拜訪目的', '業務客戶年齡差距',
            '上半年準客戶數', '今年度活動參與率', '上年度FYC'
        ]
    try:
        ref_df = pd.read_csv(ref_path)
        drift_results = detect_feature_drift(new_df, ref_df, selected_cols)
        drift_alerts = [col for col, p in drift_results.items() if p is not None and p < 0.05]

        print("📊 資料漂移檢查 P 值：")
        for col, p in drift_results.items():
            print(f"  - {col}: {'❌' if p is not None and p < 0.05 else '✅'} p={p:.4f}" if p is not None else f"  - {col}: 缺欄位")

        if drift_alerts:
            print(f"⚠️ 偵測到 {len(drift_alerts)} 個特徵與訓練資料分布顯著不同：{drift_alerts}")
            if stop_if_drift:
                print("🚫 自動中止預測流程")
                return False
        else:
            print("✅ 未偵測到顯著資料漂移，可安全預測")
        return True
    except Exception as e:
        print("⚠️ 無法執行資料漂移檢查：", e)
        return True  # 若讀不到就允許繼續執行
