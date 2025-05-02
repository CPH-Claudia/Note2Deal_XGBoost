# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:52:47 2025

@author: Z01788
"""

import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, average_precision_score
# from imblearn.over_sampling import SMOTE
# import tensorflow_addons as tfa
# from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf    
# import tensorflow.keras.backend as K



# Data processing
policy = pd.read_excel('D:/增員/增員對像推薦資料.xlsx', sheet_name=0) #, dtype={"保單申請案號": str}
policy["投保日"] = pd.to_datetime(policy["投保日"])
policy["單位_姓名"] = policy["經紀人營業單位"].astype(str) + "_" + policy["被保人"].astype(str)
policy_data = policy[
    (policy['被保人目前年齡'] > 20) &
    (policy['被保人目前年齡'] < 70) &
    (policy['要保人目前年齡'] > 20) &
    (policy['要保人目前年齡'] < 70) &
    (~policy['保額new'].isna()) &
    (policy['繳款保費new'] > 0) &
    (~policy['繳款保費new'].isna())]

# 匯率對照表
exchange_rates = {
    "NTD": 1,
    "AUD": 20.77,
    "EUR": 35.96,
    "GBP": 42.69,
    "JPY": 0.226,
    "NZD": 18.86,
    "RMB": 4.52,
    "USD": 32.995,
    "ZAR": 1.761
}
# 依照幣別換算保額為台幣
policy_data["保額new"] = policy_data.apply(lambda row: row["保額new"] * exchange_rates.get(row["幣別"], 1), axis=1)


agent = pd.read_excel('D:/增員/增員對像推薦資料.xlsx', sheet_name=1)

signing = pd.read_excel('D:/增員/增員對像推薦資料.xlsx', sheet_name=3)
signing["簽約日"] = pd.to_datetime(signing["簽約日"])

merged_df = policy_data.merge(agent, on="單位_姓名", how="left").merge(signing, on="單位_姓名", how="left")
merged_df["是否為業務"] = (merged_df["標籤名稱"] == "錠嵂業務").astype(int)

# 篩選符合條件的數據：簽約日 > 投保日 或 簽約日為空值
filtered_df = merged_df[(merged_df['簽約日'].isna()) | (merged_df['簽約日'] > merged_df['投保日'])]

df = filtered_df.copy()


# df['平均投保頻率'] = df.groupby('經紀人1-被保人CRM UUID')['投保日'].apply(
#     lambda x: abs(x.diff().dt.days.mean()) if len(x) > 1 else np.nan
# ).reset_index(drop=True)

# # 計算首次與最近投保日
# df['首次投保日'] = df.groupby('經紀人1-被保人CRM UUID')['投保日'].min().reset_index(drop=True)
# df['最近投保日'] = df.groupby('經紀人1-被保人CRM UUID')['投保日'].max().reset_index(drop=True)

# df['首次與最近投保時間差'] = (df['最近投保日'] - df['首次投保日']).dt.days
# df['平均保費'] = df['繳款保費new_sum'] / df['受理件數_sum']


# # 計算當前日期
# current_date = datetime.today()

# # 計算時間差
# df['最近投保距今天數'] = (current_date - df['最近投保日']).dt.days
# df['首次投保距今天數'] = (current_date - df['首次投保日']).dt.days



# %% 1 讀取並預處理數據
# 假設你的數據在 `filtered_df`
# 選擇重要的數據列
time_series_cols = ['經紀人1-被保人CRM UUID', '投保日', '繳款保費new', '保額new', '受理件數', '是否為業務']

# 只保留必要的數據
df = df[time_series_cols].copy()

# **計算每位客戶的累積投保次數**
df['投保次數'] = df.groupby('經紀人1-被保人CRM UUID').cumcount() + 1  # 按照客戶ID計算累積次數
current_date = datetime.today()
df['投保日距今天數'] = (current_date - df['投保日']).dt.days

# %% 要餵給模型的資料
df = df.sort_values(['經紀人1-被保人CRM UUID', '投保日'])

# %% 2 轉換成 LSTM 需要的格式
# 正規化數據（MinMaxScaler）
scaler = MinMaxScaler()
df[['繳款保費new', '保額new', '受理件數', '投保次數', '投保日距今天數']] = scaler.fit_transform(
    df[['繳款保費new', '保額new', '受理件數', '投保次數', '投保日距今天數']])

# 創建時間窗口
def create_time_series(df, time_steps=5):
    X, y, uuids = [], [], []
    customer_groups = df.groupby('經紀人1-被保人CRM UUID')

    for uuid, group in customer_groups:
        values = group[['繳款保費new', '保額new', '受理件數', '投保次數', '投保日距今天數']].values
        labels = group['是否為業務'].values
        
        for i in range(len(values) - time_steps):
            X.append(values[i:i+time_steps])
            y.append(labels[i+time_steps])
            uuids.append(uuid)  # 保留 UUID

    return np.array(X), np.array(y), np.array(uuids)


# 設定時間窗口大小
time_steps = 5
X, y, uuids = create_time_series(df, time_steps)


# %% 3 拆分訓練集與測試集
X_train, X_test, y_train, y_test, uuids_train, uuids_test = train_test_split(
    X, y, uuids, test_size=0.2, random_state=42, stratify=y)

# # 只對訓練集做 SMOTE
# # **1️⃣ 把 LSTM 時序數據轉回 2D**
# X_train_2D = X_train.reshape(X_train.shape[0], -1)  # 轉換為 (samples, features)
# y_train_2D = y_train  # y 仍然是一維數據

# # **2️⃣ 使用 SMOTE**
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_2D, y_train_2D)

# # **3️⃣ 轉回 LSTM 需要的 3D 格式**
# X_train_resampled = X_train_resampled.reshape(-1, time_steps, X.shape[2])

# %% 4 建立 LSTM 模型
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(time_steps, X.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # 二分類輸出
])

# %% 5 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# def focal_loss(alpha=0.75, gamma=2.0):
#     def loss(y_true, y_pred):
#         y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)  # 避免 log(0) 問題
#         loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) - \
#                (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
#         return K.mean(loss)
#     return loss

# # 使用 Focal Loss 訓練 LSTM
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss=focal_loss(alpha=0.75, gamma=2.0),
#               metrics=['accuracy'])

# %% 6 訓練模型
# 計算類別權重
def train_model():
    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
    return history

# **執行訓練**
history = train_model()

history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# # 訓練 LSTM
# history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),
#                     class_weight=class_weight_dict)  # 加入類別權重

# %% 7 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n📈 LSTM 測試集準確率: {test_acc:.5f}")

# %% 8 預測客戶潛力
# 預測測試集
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int) # 降低閾值，讓模型更容易預測為業務員

# %% 9 顯示預測結果
results_df = pd.DataFrame({'實際值': y_test, '預測值': y_pred.flatten()})

# 1️⃣ **顯示分類報告**
print("\n📊 **模型評估結果**")
print(classification_report(y_test, y_pred))

# 2️⃣ **計算 AUC**
auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {auc:.5f}")

# 3️⃣ **計算 Precision-Recall AUC**
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Precision: {precision:.5f}, Recall: {recall:.5f}, F1-score: {f1:.5f}")

# 3️⃣ **計算 AUC-PR分數**
auc_pr = average_precision_score(y_test, y_pred)
print(f"AUC-PR Score: {auc_pr:.5f}")



# %% 預測客戶成為業務機率
X_test1 = X_test.copy()
X_test_2D = X_test1.reshape(X_test1.shape[0], -1)  # (samples, features)
X_test_df = pd.DataFrame(X_test_2D)  # 轉回 DataFrame
X_test_df["經紀人1-被保人CRM UUID"] = uuids_test  # 保留 UUID
X_test_df["是否成為業務"] = y_test  # 加入標籤
X_test_df["成為業務機率"] = y_pred_proba  # 先預設為 NaN



# **2️⃣ 取出非業務的客戶
# X_test_non_agents = X_test_df[X_test_df["是否成為業務"] == 0].drop(columns=["是否成為業務", "成為業務機率"], errors="ignore")
X_test_non_agents = X_test_df[X_test_df["是否成為業務"] == 0].copy()


# **3️⃣ 確保 `X_test_non_agents` 是 3D
# X_test_non_agents = X_test_non_agents.values.reshape(X_test_non_agents.shape[0], time_steps, X_test.shape[2])
X_test_non_agents_values = X_test_non_agents.drop(columns=["經紀人1-被保人CRM UUID", "是否成為業務", "成為業務機率"]).values
X_test_non_agents_3D = X_test_non_agents_values.reshape(X_test_non_agents_values.shape[0], time_steps, X_test.shape[2])

# **4️⃣ 預測非業務員的成為業務機率
# X_test_df.loc[X_test_df["是否成為業務"] == 0, "成為業務機率"] = model.predict(X_test_non_agents).flatten()
X_test_non_agents["成為業務機率"] = model.predict(X_test_non_agents_3D).flatten()

# **5️⃣ 篩選高潛力客戶
# customer = X_test_df[X_test_df["是否成為業務"] == 0].copy()
# customer["高潛力客戶"] = customer["成為業務機率"].apply(lambda x: "是" if pd.notna(x) and x > 0.6 else "否")
# customer["高潛力客戶"].value_counts()
X_test_non_agents["高潛力客戶"] = X_test_non_agents["成為業務機率"].apply(lambda x: "是" if pd.notna(x) and x > 0.7 else "否")
X_test_non_agents["高潛力客戶"].value_counts()

# 測試客戶有1126
# 60%以上成為業務的機率 有180人
# 70%以上成為業務的機率 有142人

# %% 最終資料
# 可再去分析這些客戶有甚麼特質 
customer = X_test_non_agents.copy()



# from tensorflow.keras.models import Model
# from tensorflow.keras import models
# from tensorflow.keras import layers
# from tensorflow.compat.v1 import *
# tf.compat.v1.disable_v2_behavior()

# tf.keras.backend.clear_session()

# model_1 = Sequential([
#     LSTM(64, activation='relu', return_sequences=True, input_shape=(time_steps, X.shape[2])),
#     Dropout(0.2),
#     LSTM(32, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')  # 二分類輸出
# ])

# model_1.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# history_1 = model_1.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))




import shap
# X_train_2D = X_train.reshape(X_train.shape[0], -1)  # 轉換為 2D (samples, features)

# **使用 SHAP Kernel Explainer 來解釋 LSTM**
explainer = shap.GradientExplainer(model, X_train[:100])  # 訓練好的 LSTM 模型
shap_values = explainer.shap_values(X_test)  # 計算 SHAP 值

import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')

# shap.summary_plot(shap_values, X_test[:50])

# shap.plots.waterfall(shap_values[0])

X_test_reshaped = X_test[:, -1, :]
shap_values_reshaped = np.array(shap_values)[:, -1, :]  # 取最後一個時間步的 SHAP 值
shap_values_reshaped = shap_values_reshaped.reshape(shap_values_reshaped.shape[0], -1)
# **定義特徵名稱**
feature_names = ["受理件數", "繳款保費", "保額", "投保次數", "投保日距今天數"]

# **繪製 SHAP Summary Plot**
shap.summary_plot(shap_values_reshaped, X_test_reshaped, feature_names=feature_names)


# **選擇第一筆資料**
idx = 55  # 第一筆資料

# **取第一筆測試數據的 SHAP 值**
shap_values_single = shap_values_reshaped[idx]

base_value = np.mean(shap_values_reshaped[idx])  # 計算所有 SHAP 值的均值，確保是單一數值

# **繪製 waterfall 圖**
shap.plots.waterfall(shap.Explanation(
    values=shap_values_single, 
    base_values=base_value, 
    data=X_test_reshaped[idx], 
    feature_names=feature_names
))

# from lime.lime_tabular import LimeTabularExplainer
# time_steps = 3
# original_features = ["繳款保費new", "保額new", "受理件數", "投保次數", "投保日距今天數"]

# # 生成正確的特徵名稱
# feature_names = [f"{feat}_T-{t}" for t in range(time_steps, 0, -1) for feat in original_features]

# print("Feature names:", feature_names)
# print("Feature count:", len(feature_names))
# # 建立 LIME 解釋器
# explainer = LimeTabularExplainer(X_train_2D,
#                                  mode="classification",
#                                  feature_names = feature_names)

# # **選擇一筆新客戶來解釋**
# i = 0
# exp = explainer.explain_instance(X_test[i].reshape(-1),
#                                  model.predict,
#                                  num_features=5)

# # **顯示解釋結果**
# exp.show_in_notebook()
