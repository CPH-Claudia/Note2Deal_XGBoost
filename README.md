# 成交預測模型

利用 CRM 拜訪備註文字、拜訪行為、業務／客戶特徵與歷史保單資訊，建立 XGBoost 模型，預測每筆拜訪與後續成交的關聯程度，並將拜訪依預測機率排序。

模型主要定位為 高潛力拜訪排序工具，而非直接預測某位客戶一定成交。

## 專案目標
- 將非結構化的 CRM 拜訪備註轉換為可供模型使用的文字特徵
- 整合拜訪、保單、業務與客戶等不同粒度資料
- 建立拜訪層級的成交預測模型
- 透過資料漂移檢查判斷是否需要重新訓練
- 使用模型建立後的新資料進行 時間外驗證（Out-of-Time Validation, OOT）
- 評估模型在未來資料上的排序能力與泛化表現

## 有效拜訪定義

若同一客戶在：

拜訪日前 7 天 ～ 拜訪日後 180 天

至少有一筆保單成交，則該筆拜訪標記為有效拜訪。

此定義衡量的是拜訪與成交的時間關聯，不代表該次拜訪直接造成成交。

## 🔧 主要模組與函數說明

| 功能分類     | 檔案                             | 主要函數                           | 說明                             |
|--------------|----------------------------------|------------------------------------|----------------------------------|
| 資料清理     | `insurance_data_clean.py`        | `prepare_model_dataset`           | 清理拜訪資料與保單欄位，合併成模型訓練格式 |
| 資料偏移檢查 | `drift_check.py`                 | `check_drift_and_warn`            | 使用 KS 檢定確認新資料是否與訓練資料分布不同 |
|              |                                  | `get_latest_train_reference_path` | 取得最新訓練資料作為參考比對基準         |
| 模型訓練與預測 | `retrain_model_label_changed.py`         | `train_model_pipeline_with_strategies` | 執行多策略模型訓練與 SHAP 解釋、模型儲存 |
| 預測與輸出   | `predict_model.py`               | `predict_with_model`              | 載入最新模型進行預測與長格式詞語輸出     |
| 主流程入口   | `full_pipeline_with_ckip.py`     | `full_pipeline_with_ckip`         | 整合斷詞、清理、偏移檢查、預測/重訓邏輯   |

---

## ⚙️ 執行流程簡介

### Step 1: 斷詞與清理
使用 `CKIPTagger` 對「拜訪備註」欄位斷詞，分離純文字與 #標籤，並計算文字長度、有意義詞數、詞向量等資訊。

### Step 2: 資料準備
- 整併數值欄位（件數、總保費、客戶年齡差等）
- 清理缺失值與極端值
- 加入過往保單資料欄位

### Step 3: 資料漂移檢查
透過 `Kolmogorov-Smirnov 檢定` 檢查新資料與訓練資料之間的分布差異：
- 若發現漂移，進行模型重訓。
- 若無明顯漂移，使用既有模型預測。

### Step 4: 模型訓練與儲存
使用 XGBoost 訓練多組策略模型，並儲存以下物件：
- 模型本體 `model_final_*.pkl`
- TF-IDF 權重器
- Word2Vec 向量
- 數值標準化器
- 特徵名稱
- `latest.json` 紀錄當前版本
- 訓練資料 `train_reference.csv` 留作漂移對照

### Step 5: 模型預測與 SHAP 解釋
- 執行預測，輸出預測機率
- SHAP 解釋重要特徵與正向貢獻區間
- 匯出斷詞長格式 (word-level feature map)

---

## 🧪 輸出內容

每次執行預測流程後將匯出：

- `預測_<timestamp>.xlsx`
  - Sheet1: 模型預測結果（含 UUID、預測機率）
  - Sheet2: WordCloud 特徵統計
  - Sheet3: SHAP 正向貢獻區間
  - Sheet4: 拜訪備註的詞語斷詞長格式

---

## 🧰 主要依賴套件

- `pandas`, `numpy`
- `xgboost`, `scikit-learn`
- `joblib`, `openpyxl`, `xlsxwriter`
- `gensim`（Word2Vec）
- `CKIPTagger`（中文斷詞）
- `shap`
- `matplotlib`, `seaborn`

快速執行：
```bash
python39 D:\備註文字探勘\repeater\full_pipeline_with_ckip.py
Please enter the full path to the new Excel data file:
D:\備註文字探勘\repeater\新資料_0731.xlsx
```


時間外驗證（OOT Validation）
為什麼另外做 OOT？

模型開發階段原先使用 random holdout / cross-validation 評估模型。

但隨機切分的 train / test 仍來自相近的歷史期間，因此無法完全回答：

模型建立完成後，遇到真正較晚的新資料是否仍然有效？

因此額外進行 Out-of-Time Validation。

OOT 驗證設計

本次使用：

Model version：
20251031_193703 / Strategy 0
資料時間切割
資料	時間
訓練拜訪資料	2024/03/27 ～ 2025/09/25
訓練保單資料	2024/04/27 ～ 2025/10/25
模型建立	2025/10/31
OOT 拜訪資料	2025/11/01 ～ 2025/12/31
保單 outcome 觀察截止	2026/06/30

OOT 拜訪全部發生於模型建立之後，未參與模型訓練。

OOT 驗證流程
Historical Data
2024/03 ～ 2025/09
        ↓
Train Strategy 0
        ↓
Model Built
2025/10/31
        ↓
Frozen Model
        ↓
2025/11 ～ 2025/12
Unseen Visit Data
        ↓
Build the Same 86 Features
        ↓
Predict Probability
        ↓
Observe Policy Outcome
through 2026/06/30
        ↓
Compare Prediction
with Actual Outcome
驗證原則

OOT 階段：

不重新訓練 XGBoost
不重新訓練 Word2Vec
不重新 fit TF-IDF
沿用原模型 feature schema
沿用原模型營業單位 mapping
使用相同的 86 個模型特徵

執行時確認：

model.n_features_in_ : 86
feature_names.pkl    : 86
X_oot                : 86

三者一致後才進行預測。

📊 OOT 驗證結果

OOT 共包含：

Sample Size    : 2,692
Positive Count : 1,228
Base Rate      : 45.62%

主要模型指標：

Metric	Result
ROC-AUC	0.792
PR-AUC	0.782
Brier Score	0.187
Mean Predicted Probability	41.75%
高分拜訪的實際有效率
Model Ranking	N	Observed Positive Rate	Lift
Overall	2,692	45.62%	1.00
Top 20%	539	85.71%	1.88×
Top 10%	270	92.59%	2.03×
Top 5%	135	94.07%	2.06×
OOT 結果解讀

全部時間外拜訪中，約：

45.6%

符合有效拜訪定義。

但依模型預測分數排序後：

Top 10% → 92.6%

的拜訪符合有效拜訪定義。

因此：

Lift@10% = 2.03

表示前 10% 高分拜訪的有效率，約為全部拜訪平均的 2.03 倍。

ROC-AUC 約 0.79、PR-AUC 約 0.78，也顯示模型在模型建立後的新資料中仍具有良好的 discrimination / ranking ability。

OOT 結論

Strategy 0 在未參與模型訓練、且發生於模型建立後的新時間區段中，仍能有效將較可能與成交相關的拜訪集中於高分區。

因此目前模型較適合定位為：

High-potential Visit Ranking Model

而非：

Exact Conversion Probability
or
Causal Conversion Model
⚠️ 模型限制與後續改善

本次 OOT 已將模型訓練期間與驗證拜訪期間分開，但原始 feature engineering 中仍有部分歷史統計特徵可能使用完整資料期間計算，例如：

平均每客戶拜訪次數
平均拜訪間隔天數
每週平均拜訪客戶數
距離最近晉升天數

因此目前結果支持模型具有 時間外排序能力，但若要正式投入 production，下一版建議將所有 predictor 改為嚴格的：

Point-in-Time Features

即每筆拜訪只能使用該次拜訪發生當下以前已知的資訊。

其他可改善方向：

建立完整 point-in-time feature pipeline
加入 feature schema validation
保存所有 Encoder / Transformer
Rolling temporal validation
Model / data drift monitoring
Calibration analysis
不同 outcome window（30 / 90 / 180 days）Sensitivity Analysis
📁 輸出內容
一般模型預測

每次執行預測流程後輸出：

預測_<timestamp>.xlsx

內容包含：

模型預測結果與 UUID
預測機率
WordCloud 特徵統計
SHAP 正向貢獻區間
拜訪備註斷詞長格式
OOT Validation

時間外驗證輸出：

oot_validation_20251031_193703_strategy0.xlsx

主要包含：

metrics：ROC-AUC、PR-AUC、Brier Score、Lift
deciles：依模型分數分群後的實際有效率
detail：每筆拜訪的 prediction / actual label

