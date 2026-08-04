# -*- coding: utf-8 -*-
"""
2025/10/31 模型 OOT（時間外）驗證

用途：
1. 使用與舊流程相同的原始 Excel。
2. 以 CKIP 建立 OOT 斷詞結果。
3. 呼叫 insurance_data_clean.prepare_model_dataset 建立基礎特徵。
4. 覆寫 OOT 不可重新 fit 的欄位：有意義詞數、營業單位編碼。
5. 載入 2025/10/31 舊模型，先完整支援 strategy 0。
6. 對 OOT 新拜訪產生 pred_prob。
7. 以 -7～180 天內任一保單建立 actual_label。
8. 計算 ROC-AUC、PR-AUC、Brier、Lift@5/10/20 與 Decile。

重要：
- 本程式不會重新訓練 XGBoost、TF-IDF 或模型用 Word2Vec。
- strategy 2、6 需要舊 MultiLabelBinarizer；原訓練片段沒有保存 mlb，
  所以本版先完成 strategy 0，避免特徵欄位錯位。
- insurance_data_clean.py 必須與本程式位於同一資料夾，或在 PYTHONPATH 中。
"""

from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path
from time import perf_counter
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import sys
import types


def register_pickle_compatibility_modules():
    """
    提供舊 pickle 載入時需要的函式名稱，
    避免必須執行原本整支 retrain_model_label_changed.py。
    """
    module_name = "retrain_model_label_changed"

    if module_name not in sys.modules:
        compatibility_module = types.ModuleType(module_name)

        def identity(x):
            return x

        # 讓 pickle 認為此函式屬於舊模組
        identity.__module__ = module_name

        compatibility_module.identity = identity
        sys.modules[module_name] = compatibility_module


# ============================================================
# 0. 設定區：請先修改這裡
# ============================================================
OOT_RAW_XLSX = Path(r"D:\備註文字探勘\model_validation\時間外測試資料_0728.xlsx")
MODEL_ROOT = Path(r"D:\備註文字探勘\models\20251031_193703")
STRATEGY_ID = 0

TRAIN_PROCESSED_FILE = Path(
    r"D:\備註文字探勘\results\20251031_193703\ALL_ModelResults_20251031_193703.csv"
)
TRAIN_RAW_XLSX = Path(
    r"D:\備註文字探勘\repeater\新資料_1031.xlsx"
)
CKIP_DATA_DIR = Path(r"C:\Users\Z01788\data")
REFERENCE_DIR = Path(r"D:\備註文字探勘\model_validation\reference")
MEANINGFUL_WORDS_PKL = REFERENCE_DIR / "meaningful_words_20251031.pkl"
UNIT_MAP_PKL = REFERENCE_DIR / "unit_map_20251031.pkl"
STOPWORDS_CACHE = REFERENCE_DIR / "stopwords_trad.txt"
OUTPUT_DIR = Path(r"D:\備註文字探勘\model_validation\output")

OOT_START = pd.Timestamp("2025-11-01")
OOT_END = pd.Timestamp("2025-12-31")
POLICY_DATA_END = pd.Timestamp("2026-06-30")
LABEL_MIN_DAY = -7
LABEL_MAX_DAY = 180
REQUIRE_SAME_AGENT = False
MIN_AVG_VISITS_PER_CUSTOMER = 4
ALLOW_SEED_ONLY_FALLBACK = False

NUMERICAL_COLS = [
    "職級_代碼", "拜訪目的",
    "業務男_要保男", "業務女_要保女", "業務男_要保女", "業務女_要保男",
    "平均拜訪間隔天數", "每週平均拜訪客戶數", "業務要保人年齡差",
    "備註字數", "有意義詞數", "hashtag_count",
    "截至該月年資", "營業單位_編碼", "上半年準客戶數",
    "最近半年活動參與率", "上一個半年度FYC",
    "距離最近晉升天數", "截至當日累積件數", "截至當日累積保費",
    "拜訪序號", "賽季",
]

GENDER_DUMMY_COLS = [
    "業務男_要保男", "業務女_要保女", "業務男_要保女", "業務女_要保男"
]

SEED_WORDS = [
    "保障", "保單", "理賠", "投保", "保費", "變更", "簽名", "保單健診", "華南產",
    "癌症險", "旅平險", "新安東京", "還本型保單", "富邦", "安達", "續保", "富邦產",
    "定期壽險", "重大傷病", "實支實付", "住院醫療", "分紅躉繳", "轉介紹", "保險",
    "行銷活動", "防疫險", "保險經紀人", "健診", "籃子理論", "錠嵂", "意外險",
    "資產規劃", "車險需求", "儲蓄險", "中壽", "中國人壽", "旅平", "三商美邦",
    "簽約", "成交", "失能險", "保經", "見面三講", "開門三講", "退休規劃",
    "車險", "醫療險", "火險", "壽險", "新光", "遠雄", "Toyota", "機車險",
    "寵物險", "自動化工程師", "六大保障", "建議書", "台灣人壽", "失智症",
    "app", "OPP", "保險存摺", "國泰", "照會", "三照", "遞送", "市調表", "解約",
    "美元保單", "美元儲蓄", "送保單", "照會單", "台壽", "保誠", "癌症", "不動產",
    "問卷", "轉介", "簽收", "建立關係", "強制險", "永達", "觀念溝通", "需求分析",
    "終身", "萬事利達", "友邦", "寒暄", "關心", "年金", "PHB", "宏泰", "南山",
    "長照", "XHB", "HNRC", "新生兒", "約訪", "年繳", "美金", "phb", "探班",
    "要保人", "企管副會長", "意外險需求", "double鑫", "下週",
]


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description}不存在：{path}")


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def extract_non_sharp_text(note: object) -> str:
    if pd.isna(note):
        return ""
    lines = str(note).replace("_x000D_", "\n").replace("\r", "").splitlines()
    return "\n".join(
        line.strip()
        for line in lines
        if line.strip() and not (line.startswith("#") or line.startswith("＃"))
    )


def pkl_path_for_excel(excel_path: Path) -> Path:
    return excel_path.with_suffix(".pkl")


def load_stopwords(cache_path: Path) -> set[str]:
    if cache_path.exists():
        print(f"📦 使用固定停用詞快取：{cache_path}")
        return {x.strip() for x in cache_path.read_text(encoding="utf-8").splitlines() if x.strip()}

    print("⏬ 停用詞快取不存在，首次下載並轉繁體。")
    import requests
    from opencc import OpenCC

    urls = [
        "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt",
        "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt",
        "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt",
        "https://raw.githubusercontent.com/goto456/stopwords/master/scu_stopwords.txt",
    ]
    cc = OpenCC("s2t")
    stopwords: set[str] = set()
    for url in urls:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        for line in response.text.splitlines():
            word = cc.convert(line.strip())
            if word:
                stopwords.add(word)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(sorted(stopwords)), encoding="utf-8")
    return stopwords


def merge_custom_terms(ws_result: list[list[str]], custom_terms: Iterable[str]) -> list[list[str]]:
    term_set = set(custom_terms)
    if not term_set:
        return ws_result
    max_len = max(len(term) for term in term_set)
    merged_result: list[list[str]] = []
    for sentence in ws_result:
        merged_sentence: list[str] = []
        i = 0
        while i < len(sentence):
            match = None
            for length in range(min(max_len, len(sentence) - i), 0, -1):
                phrase = "".join(sentence[i:i + length])
                if phrase in term_set:
                    match = phrase
                    i += length
                    break
            if match is not None:
                merged_sentence.append(match)
            else:
                merged_sentence.append(sentence[i])
                i += 1
        merged_result.append(merged_sentence)
    return merged_result


def create_ckip_pickle(excel_path: Path, *, force: bool = False) -> Path:
    require_file(excel_path, "原始 Excel")
    output_pkl = pkl_path_for_excel(excel_path)
    if output_pkl.exists() and not force:
        print(f"✅ 已有斷詞檔，直接使用：{output_pkl}")
        return output_pkl
    require_file(CKIP_DATA_DIR, "CKIP data 資料夾")
    from ckiptagger import WS

    print(f"🧩 CKIP 斷詞：{excel_path.name}")
    ws = WS(str(CKIP_DATA_DIR))
    visit_raw = pd.read_excel(excel_path, sheet_name="VISIT")
    if "拜訪備註" not in visit_raw.columns:
        raise ValueError("VISIT 缺少欄位：拜訪備註")
    visit_raw["拜訪備註_文字"] = visit_raw["拜訪備註"].apply(extract_non_sharp_text)
    text_list = visit_raw["拜訪備註_文字"].fillna("").tolist()
    ws_segments = ws(text_list)
    word_list = merge_custom_terms(ws_segments, set(SEED_WORDS))
    stopwords = load_stopwords(STOPWORDS_CACHE)
    filtered_words = [
        [
            word for word in sentence
            if word not in stopwords
            and len(word) > 1
            and not re.fullmatch(r"[\W_]+", word)
            and not re.fullmatch(r"\d+", word)
        ]
        for sentence in word_list
    ]
    with output_pkl.open("wb") as f:
        pickle.dump(filtered_words, f)
    if len(filtered_words) != len(visit_raw):
        raise RuntimeError("斷詞結果筆數與 VISIT 筆數不一致")
    print(f"✅ 斷詞完成：{output_pkl}，共 {len(filtered_words):,} 筆")
    return output_pkl


def build_meaningful_words_from_training(training_raw_xlsx: Path) -> set[str]:
    require_file(training_raw_xlsx, "舊訓練原始 Excel")
    create_ckip_pickle(training_raw_xlsx)
    visit = pd.read_excel(training_raw_xlsx, sheet_name="VISIT")
    with pkl_path_for_excel(training_raw_xlsx).open("rb") as f:
        filtered_words = pickle.load(f)
    visit = visit.reset_index(drop=True)
    visit["備註文字_處理"] = [
        " ".join(words) if isinstance(words, list) else ""
        for words in filtered_words[:len(visit)]
    ]
    token_lists = visit["備註文字_處理"].dropna().apply(str.split).tolist()
    meaningful_words = set(SEED_WORDS)
    if token_lists:
        model_w2v = Word2Vec(
            sentences=token_lists, vector_size=150, window=5,
            min_count=2, workers=4, seed=42
        )
        for word in SEED_WORDS:
            if word in model_w2v.wv:
                for sim_word, score in model_w2v.wv.most_similar(word, topn=20):
                    if score > 0.6:
                        meaningful_words.add(sim_word)
    MEANINGFUL_WORDS_PKL.parent.mkdir(parents=True, exist_ok=True)
    with MEANINGFUL_WORDS_PKL.open("wb") as f:
        pickle.dump(meaningful_words, f)
    print(f"✅ meaningful words 已固定：{len(meaningful_words):,} 詞")
    return meaningful_words


def load_meaningful_words() -> set[str]:
    if MEANINGFUL_WORDS_PKL.exists():
        with MEANINGFUL_WORDS_PKL.open("rb") as f:
            words = pickle.load(f)
        print(f"📦 載入固定 meaningful words：{len(words):,} 詞")
        return set(words)
    if TRAIN_RAW_XLSX.exists():
        return build_meaningful_words_from_training(TRAIN_RAW_XLSX)
    if ALLOW_SEED_ONLY_FALLBACK:
        warnings.warn("只使用 seed words，屬近似還原。")
        return set(SEED_WORDS)
    raise FileNotFoundError("缺少 meaningful words pkl 與舊訓練原始 Excel")


def build_unit_map_from_processed(processed_file: Path) -> dict[str, int]:
    require_file(processed_file, "舊訓練處理結果")
    old = pd.read_excel(processed_file) if processed_file.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(processed_file)
    required = {"營業單位", "營業單位_編碼"}
    missing = required - set(old.columns)
    if missing:
        raise ValueError(f"舊訓練處理結果缺少欄位：{sorted(missing)}")
    pairs = old[["營業單位", "營業單位_編碼"]].dropna().drop_duplicates()
    inconsistent = pairs.groupby("營業單位")["營業單位_編碼"].nunique().loc[lambda s: s > 1]
    if not inconsistent.empty:
        raise ValueError(f"相同營業單位對應多個編碼：{inconsistent.to_dict()}")
    unit_map = {str(r["營業單位"]): int(r["營業單位_編碼"]) for _, r in pairs.iterrows()}
    UNIT_MAP_PKL.parent.mkdir(parents=True, exist_ok=True)
    with UNIT_MAP_PKL.open("wb") as f:
        pickle.dump(unit_map, f)
    print(f"✅ 固定營業單位 mapping 已建立：{len(unit_map):,} 單位")
    return unit_map


def load_unit_map() -> dict[str, int]:
    if UNIT_MAP_PKL.exists():
        with UNIT_MAP_PKL.open("rb") as f:
            mapping = pickle.load(f)
        print(f"📦 載入固定營業單位 mapping：{len(mapping):,} 單位")
        return dict(mapping)
    return build_unit_map_from_processed(TRAIN_PROCESSED_FILE)


def ensure_gender_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in GENDER_DUMMY_COLS:
        if col not in result.columns:
            result[col] = 0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result


def prepare_oot_features(raw_xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_file(raw_xlsx, "OOT 原始 Excel")
    create_ckip_pickle(raw_xlsx)
    from insurance_data_clean import prepare_model_dataset

    print("🧱 執行原 insurance_data_clean.prepare_model_dataset...")
    df_ready, policy_df = prepare_model_dataset(str(raw_xlsx))

    meaningful_words = load_meaningful_words()
    unit_map = load_unit_map()

    def count_meaningful(text: object) -> int:
        if pd.isna(text):
            return 0
        return sum(1 for word in str(text).split() if word in meaningful_words)

    df_ready["有意義詞數"] = df_ready["備註文字_處理"].fillna("").apply(count_meaningful)
    df_ready["營業單位"] = df_ready["營業單位"].astype("string")
    df_ready["營業單位_編碼"] = df_ready["營業單位"].map(unit_map)
    unknown_units = df_ready.loc[df_ready["營業單位_編碼"].isna(), "營業單位"].dropna().value_counts()
    if not unknown_units.empty:
        print("⚠️ 舊模型未見營業單位：")
        print(unknown_units.head(20).to_string())
    df_ready = ensure_gender_columns(df_ready)

    df_ready = df_ready[
        df_ready["備註文字_處理"].notna()
        & (df_ready["備註文字_處理"].str.strip() != "")
    ].copy()
    df_ready["拜訪時間 年/月/日"] = pd.to_datetime(df_ready["拜訪時間 年/月/日"], errors="coerce")

    oot = df_ready[df_ready["拜訪時間 年/月/日"].between(OOT_START, OOT_END, inclusive="both")].copy()
    oot = oot[pd.to_numeric(oot["平均每客戶拜訪次數"], errors="coerce") > MIN_AVG_VISITS_PER_CUSTOMER].copy()
    oot = oot.dropna(subset=["拜訪紀錄UUID", "客戶UUID", "拜訪時間 年/月/日"])

    duplicate_count = oot["拜訪紀錄UUID"].duplicated().sum()
    if duplicate_count:
        warnings.warn(f"OOT 有 {duplicate_count:,} 筆重複拜訪 UUID，暫保留第一筆。")
        oot = oot.drop_duplicates(subset=["拜訪紀錄UUID"], keep="first")

    missing_numerical = sorted(set(NUMERICAL_COLS) - set(oot.columns))
    if missing_numerical:
        raise ValueError(f"OOT 特徵缺少模型數值欄位：{missing_numerical}")

    print(f"✅ OOT 特徵完成：{len(oot):,} 筆")
    return oot, policy_df


def vectorize_sentence_weighted(text: object, *, w2v_model, tfidf_dict: dict[str, float]) -> np.ndarray:
    vector_size = int(w2v_model.vector_size)
    if pd.isna(text):
        return np.zeros(vector_size, dtype=np.float32)
    vectors, weights = [], []
    for word in str(text).split():
        if word in w2v_model.wv and word in tfidf_dict:
            weight = float(tfidf_dict[word])
            vectors.append(w2v_model.wv[word] * weight)
            weights.append(weight)
    if not vectors or np.sum(weights) == 0:
        return np.zeros(vector_size, dtype=np.float32)
    return (np.sum(vectors, axis=0) / np.sum(weights)).astype(np.float32)


def predict_strategy_0(oot: pd.DataFrame) -> pd.DataFrame:
    register_pickle_compatibility_modules()
    
    if STRATEGY_ID != 0:
        raise NotImplementedError("本版先完整支援 strategy 0。")
    strategy_dir = MODEL_ROOT / f"strategy_{STRATEGY_ID}"
    for name in [
        "model_final.pkl", "word2vec_model.pkl", "tfidf_vectorizer.pkl",
        "w2v_top_indices.pkl", "feature_names.pkl", "train_reference.csv"
    ]:
        require_file(strategy_dir / name, "模型資產")

    model = joblib.load(strategy_dir / "model_final.pkl")
    w2v_model = joblib.load(strategy_dir / "word2vec_model.pkl")
    tfidf_vectorizer = joblib.load(strategy_dir / "tfidf_vectorizer.pkl")
    w2v_top_indices = np.asarray(joblib.load(strategy_dir / "w2v_top_indices.pkl"))
    feature_names = list(joblib.load(strategy_dir / "feature_names.pkl"))
    train_reference = pd.read_csv(strategy_dir / "train_reference.csv")

    train_num = train_reference[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce")
    means = train_num.mean(skipna=True)
    stds = train_num.std(skipna=True).replace(0, 1)
    oot_num = oot[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce")
    x_num = ((oot_num - means.reindex(NUMERICAL_COLS)) / stds.reindex(NUMERICAL_COLS)).to_numpy()

    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    oot_vectors = [
        vectorize_sentence_weighted(text, w2v_model=w2v_model, tfidf_dict=tfidf_dict)
        for text in oot["備註文字_處理"]
    ]
    x_w2v_all = np.vstack(oot_vectors)
    x_w2v_top = x_w2v_all[:, w2v_top_indices]
    x_w2vtag_top = np.zeros((len(oot), len(w2v_top_indices)), dtype=np.float32)
    x_oot = np.hstack([x_w2v_top, x_num, x_w2vtag_top])

    constructed_names = (
        [f"w2v_{index}" for index in w2v_top_indices]
        + NUMERICAL_COLS
        + [f"w2vtag_{index}" for index in w2v_top_indices]
    )

    print(f"模型 n_features_in_：{model.n_features_in_}")
    print(f"feature_names.pkl：{len(feature_names)}")
    print(f"本次 X_oot：{x_oot.shape[1]}")

    if x_oot.shape[1] != model.n_features_in_:
        raise ValueError(f"OOT 特徵數與舊模型不一致：{x_oot.shape[1]} vs {model.n_features_in_}")
    if constructed_names != feature_names:
        differences = [
            {"position": i, "constructed": a, "saved": b}
            for i, (a, b) in enumerate(zip(constructed_names, feature_names))
            if a != b
        ]
        raise ValueError(f"OOT 特徵名稱／順序與舊模型不一致。前 10 個差異：{differences[:10]}")

    result = oot.copy()
    result["pred_prob"] = model.predict_proba(x_oot)[:, 1]
    result["model_timestamp"] = MODEL_ROOT.name
    result["strategy_id"] = STRATEGY_ID
    result["is_oot"] = 1
    print("✅ OOT 預測完成")
    return result


def build_actual_labels(predictions: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    policy = policy_df.copy()
    policy_customer_candidates = ["經紀人1-被保人CRM UUID", "被保人UUID", "客戶UUID"]
    policy_customer_col = next((c for c in policy_customer_candidates if c in policy.columns), None)
    if policy_customer_col is None:
        raise ValueError("POLICY 找不到客戶鍵")
    policy_date_candidates = ["投保日 年/月/日", "投保日"]
    policy_date_col = next((c for c in policy_date_candidates if c in policy.columns), None)
    if policy_date_col is None:
        raise ValueError("POLICY 找不到投保日")

    result["_customer_key"] = normalize_id(result["客戶UUID"])
    policy["_customer_key"] = normalize_id(policy[policy_customer_col])
    result["拜訪時間 年/月/日"] = pd.to_datetime(result["拜訪時間 年/月/日"], errors="coerce")
    policy[policy_date_col] = pd.to_datetime(policy[policy_date_col], errors="coerce")

    dedup_cols = ["_customer_key", policy_date_col]
    if "保單申請案號" in policy.columns:
        dedup_cols.append("保單申請案號")
    policy = policy.drop_duplicates(subset=dedup_cols, keep="first")

    merged = result.merge(policy, on="_customer_key", how="left", suffixes=("_visit", "_policy"))
    visit_date_col = "拜訪時間 年/月/日_visit" if "拜訪時間 年/月/日_visit" in merged.columns else "拜訪時間 年/月/日"
    merged_policy_date_col = f"{policy_date_col}_policy" if f"{policy_date_col}_policy" in merged.columns else policy_date_col
    merged["day_diff"] = (merged[merged_policy_date_col].dt.normalize() - merged[visit_date_col].dt.normalize()).dt.days
    in_window = merged["day_diff"].between(LABEL_MIN_DAY, LABEL_MAX_DAY, inclusive="both")

    if REQUIRE_SAME_AGENT:
        policy_agent_candidates = ["招攬業代", "經紀人1", "業代"]
        policy_agent_col = next((c for c in policy_agent_candidates if c in merged.columns or f"{c}_policy" in merged.columns), None)
        if policy_agent_col is None:
            raise ValueError("找不到成交業代欄位")
        p_col = f"{policy_agent_col}_policy" if f"{policy_agent_col}_policy" in merged.columns else policy_agent_col
        v_col = "業代_visit" if "業代_visit" in merged.columns else "業代"
        in_window = in_window & (normalize_id(merged[v_col]) == normalize_id(merged[p_col]))

    merged["policy_in_window"] = in_window.fillna(False).astype(int)
    merged["matched_policy_date"] = merged[merged_policy_date_col].where(merged["policy_in_window"] == 1)

    actual = merged.groupby("拜訪紀錄UUID", dropna=False, as_index=False).agg(
        actual_label=("policy_in_window", "max"),
        first_matched_policy_date=("matched_policy_date", "min"),
        matched_policy_rows=("policy_in_window", "sum"),
    )
    result = result.merge(actual, on="拜訪紀錄UUID", how="left")
    result["actual_label"] = result["actual_label"].fillna(0).astype(int)
    result["label_matured"] = (
        result["拜訪時間 年/月/日"].dt.normalize() + pd.Timedelta(days=LABEL_MAX_DAY)
        <= POLICY_DATA_END
    ).astype(int)
    return result


def top_metrics(df: pd.DataFrame, fraction: float) -> dict[str, float]:
    ranked = df.sort_values("pred_prob", ascending=False)
    top_n = max(1, int(np.ceil(len(ranked) * fraction)))
    top = ranked.head(top_n)
    base_rate = float(ranked["actual_label"].mean())
    top_rate = float(top["actual_label"].mean())
    lift = top_rate / base_rate if base_rate > 0 else np.nan
    pct = int(fraction * 100)
    return {f"top{pct}_n": top_n, f"top{pct}_rate": top_rate, f"lift_at_{pct}": lift}


def evaluate_predictions(detail: pd.DataFrame) -> pd.DataFrame:
    eval_df = detail[detail["label_matured"] == 1].dropna(subset=["pred_prob", "actual_label"]).copy()
    if eval_df.empty:
        raise ValueError("沒有成熟 OOT 資料可評估")
    result: dict[str, object] = {
        "model_timestamp": MODEL_ROOT.name,
        "strategy_id": STRATEGY_ID,
        "oot_start": OOT_START.date(),
        "oot_end": OOT_END.date(),
        "policy_data_end": POLICY_DATA_END.date(),
        "sample_size": len(eval_df),
        "positive_count": int(eval_df["actual_label"].sum()),
        "base_rate": float(eval_df["actual_label"].mean()),
        "mean_pred_prob": float(eval_df["pred_prob"].mean()),
    }
    if eval_df["actual_label"].nunique() == 2:
        result["roc_auc"] = roc_auc_score(eval_df["actual_label"], eval_df["pred_prob"])
        result["pr_auc"] = average_precision_score(eval_df["actual_label"], eval_df["pred_prob"])
        result["brier_score"] = brier_score_loss(eval_df["actual_label"], eval_df["pred_prob"])
    else:
        result["roc_auc"] = result["pr_auc"] = result["brier_score"] = np.nan
    for fraction in (0.05, 0.10, 0.20):
        result.update(top_metrics(eval_df, fraction))
    return pd.DataFrame([result])


def create_decile_table(detail: pd.DataFrame) -> pd.DataFrame:
    data = detail[detail["label_matured"] == 1].copy()
    if len(data) < 10:
        return pd.DataFrame()
    rank_order = data["pred_prob"].rank(method="first", ascending=False)
    data["decile"] = pd.qcut(rank_order, q=10, labels=range(1, 11))
    summary = data.groupby("decile", observed=True, as_index=False).agg(
        visit_count=("拜訪紀錄UUID", "count"),
        positive_count=("actual_label", "sum"),
        actual_rate=("actual_label", "mean"),
        avg_pred_prob=("pred_prob", "mean"),
        min_pred_prob=("pred_prob", "min"),
        max_pred_prob=("pred_prob", "max"),
    )
    base_rate = float(data["actual_label"].mean())
    summary["lift"] = summary["actual_rate"] / base_rate if base_rate > 0 else np.nan
    return summary


def main() -> None:
    t0 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    oot_features, policy_df = prepare_oot_features(OOT_RAW_XLSX)
    feature_file = OUTPUT_DIR / f"oot_features_{MODEL_ROOT.name}_strategy{STRATEGY_ID}.xlsx"
    oot_features.to_excel(feature_file, index=False)

    predictions = predict_strategy_0(oot_features)
    # prediction_file = OUTPUT_DIR / f"oot_predictions_{MODEL_ROOT.name}_strategy{STRATEGY_ID}.xlsx"
    # predictions.to_excel(prediction_file, index=False)

    detail = build_actual_labels(predictions, policy_df)
    metrics = evaluate_predictions(detail)
    deciles = create_decile_table(detail)

    report_file = OUTPUT_DIR / f"oot_validation_{MODEL_ROOT.name}_strategy{STRATEGY_ID}.xlsx"
    with pd.ExcelWriter(report_file) as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        deciles.to_excel(writer, sheet_name="deciles", index=False)
        detail.to_excel(writer, sheet_name="detail", index=False)

    print("\n========== OOT 績效 ==========")
    print(metrics.T.to_string(header=False))
    print(f"\n💾 報告：{report_file}")
    print(f"✅ 完成，耗時 {(perf_counter() - t0) / 60:.1f} 分鐘")


if __name__ == "__main__":
    main()
