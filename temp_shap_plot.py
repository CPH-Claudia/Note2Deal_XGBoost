# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:00:06 2025

@author: Z01788
"""

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

            peak_idx = df["SHAP_smooth"].idxmax()
            peak_x = df.loc[peak_idx, "值"]
            peak_raw = peak_x * scale + mean

            ax1.scatter(df["值"], df["SHAP"], alpha=0.3, label="原始 SHAP")
            ax1.plot(df["值"], df["SHAP_smooth"], color='blue', label="平滑趨勢")
            ax1.axhline(0, color='gray', linestyle='--')
            ax1.axvline(peak_x, color='red', linestyle='--', label=f'最大貢獻點 = {peak_raw:.2f}')

            for idx, ((lo, hi), (lo_raw, hi_raw)) in enumerate(zip(merged_ranges, restored_ranges)):
                ax1.axvspan(lo, hi, color='lightgreen', alpha=0.3)
                summary_list.append({
                    '變數': var,
                    '原始值區間_起': round(lo_raw, 2),
                    '原始值區間_迄': round(hi_raw, 2),
                    '標準化值_起': round(lo, 2),
                    '標準化值_迄': round(hi, 2)
                })

            ax1.set_xlabel(f"{var} (標準化)")
            ax1.set_ylabel("SHAP 值")
            ax1.set_title(f"{var} 對成交的 SHAP 趨勢")

            # # ➤ 新增副 X 軸（放在下方、顯示原始數值）
            # from matplotlib.transforms import blended_transform_factory
            
            # def to_raw(x): return x * scale + mean
            # def to_std(x): return (x - mean) / scale
            
            # secax = ax1.secondary_xaxis('top', functions=(to_raw, to_std))
            # secax.set_xlabel("原始數值")
            # secax.set_xticks(ax1.get_xticks())
            # secax.set_xticklabels([f"{to_raw(t):.2f}" for t in ax1.get_xticks()])
            
            # 雙 X 軸：下方顯示原始數值，對齊標準化 X 軸
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            
            # 抓主 X 軸（標準化）上的實際刻度
            tick_pos = ax1.get_xticks()
            tick_raw = [t * scale + mean for t in tick_pos]
            tick_labels = [f"{v:.2f}" for v in tick_raw]
            
            # 套用至下軸
            ax2.set_xticks(tick_pos)
            ax2.set_xticklabels(tick_labels)
            ax2.set_xlabel(f"{var} (原始數值)", labelpad=25)
            


            # 平均與標準差說明
            mean_text = f"原始平均值: {mean:.2f}\n原始標準差: {scale:.2f}"
            ax1.text(0.98, 0.95, mean_text, transform=ax1.transAxes,
                     ha='right', va='top', fontsize=8, color='dimgray',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))

            ax1.legend(loc='upper left')
            ax1.grid(True)

            # if output_dir:
            #     filename = os.path.join(output_dir, f"{var}_shap_trend.png")
            #     plt.savefig(filename, dpi=300, bbox_inches='tight')
            #     plt.close()
            #     print(f"✅ 已儲存：{filename}")
            # else:
            #     plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ {var} 失敗：{e}")

    return pd.DataFrame(summary_list)

df_summary = plot_shap_bin_auto_with_summary_dual_x(
    X_data=X_trainval,
    shap_values=shap_values,
    feature_names=final_feature_names,
    variables=num_vars,
    mean_dict=mean_dict,
    scale_dict=scale_dict,
    window=20,
    output_dir="D:/備註文字探勘/shap_2024",
    min_range_width=0.1,
    merge_gap=0.05
)