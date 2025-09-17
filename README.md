# 成交預測模型

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
python full_pipeline_with_ckip.py
Please enter the full path to the new Excel data file:
D:\備註文字探勘\repeater\新資料_0731.xlsx

