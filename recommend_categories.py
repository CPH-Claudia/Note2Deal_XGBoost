# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:53:05 2025

@author: Z01788
"""

"""
recommend_categories.py
用 Word2Vec / FastText / Chinese-BERT 之一，對「未分類標籤」做語意比對，輸出推薦類別與近鄰依據。

輸入：
- tag_classification_results.csv   # 每行拜訪紀錄 + "標籤分類" JSON
- tag_token_map.csv               # 已有的 標籤→分類 對照
- (可選) category_dictionary.csv, custom_activity_map.csv, agent_overrides.csv

輸出：
- uncategorized_recommendations.csv  # 未分類標籤 → 推薦類別/相似度/近鄰依據
"""

import os
import re
import json
import math
import argparse
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# ===== Optional deps (graceful fallback) =====
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from gensim.models import Word2Vec, FastText as FTModel, KeyedVectors
        GENSIM_AVAILABLE = True
    except Exception:
        GENSIM_AVAILABLE = False

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        HF_AVAILABLE = True
    except Exception:
        HF_AVAILABLE = False


CATS = ["業務行為", "客戶回饋", "下一步"]  # 你的三大類（不含「其他」）
ALL_CATS = ["業務行為", "客戶回饋", "下一步", "其他"]

def read_json_pairs(s: str):
    try:
        pairs = json.loads(s)
        # 確保格式 [[tok, cat], ...]
        out = []
        for it in pairs:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                out.append((str(it[0]).strip(), str(it[1]).strip()))
        return out
    except Exception:
        return []

def load_seed_tokens(tag_token_map_path: str,
                     extra_rule_csvs: list = None):
    """
    讀已標好類別的「種子詞庫」：
    - 來源一：tag_token_map.csv 的 (標籤, 分類)
    - 來源二：三張規則 csv（pattern, category, ...）（僅取 pattern 視為 '標籤'）
    回傳：dict: {category: set(tokens)}
    """
    seeds = {c: set() for c in ALL_CATS}

    if os.path.exists(tag_token_map_path):
        m = pd.read_csv(tag_token_map_path)
        if {"標籤", "分類"}.issubset(m.columns):
            for _, r in m.iterrows():
                tok = str(r["標籤"]).strip()
                cat = str(r["分類"]).strip()
                if tok and cat in seeds:
                    seeds[cat].add(tok)

    extra_rule_csvs = extra_rule_csvs or []
    for p in extra_rule_csvs:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if {"pattern", "category"}.issubset(df.columns):
            for _, r in df.iterrows():
                pat = str(r["pattern"]).strip()
                cat = str(r["category"]).strip()
                if pat and cat in seeds:
                    # 僅把 pattern 視為一個候選 token（活動名/關鍵字）
                    seeds[cat].add(pat)

    # 不把「其他」當做種子類別（它是緩衝區），但保留集合結構
    return seeds

def extract_all_tokens_from_results(results_csv: str):
    """
    從 tag_classification_results.csv 取出：
    - 全部標籤 tokens
    - 未分類 tokens（分類 = '其他'）
    - 同時蒐集原始文字作為語料
    """
    df = pd.read_csv(results_csv)
    all_tokens = []
    labeled_pairs = []  # (token, category)
    raw_lines = []
    uncat = Counter()

    if "原始文字" in df.columns:
        raw_lines = df["原始文字"].astype(str).tolist()

    for _, r in df.iterrows():
        pairs = read_json_pairs(str(r.get("標籤分類", "")))
        for tok, cat in pairs:
            all_tokens.append(tok)
            labeled_pairs.append((tok, cat))
            if cat == "其他":
                uncat[tok] += 1

    all_tokens = list(dict.fromkeys(all_tokens))  # 去重保序
    uncat_tokens = [t for t, _ in uncat.most_common()]
    return all_tokens, labeled_pairs, uncat_tokens, raw_lines

# ===== Embedding backends =====

def build_w2v_embeddings(corpus_texts, tokens, vector_size=200, window=5, min_count=1, epochs=20):
    """
    用你的「原始文字」+ 所有 tokens 先拼一份小語料，快速訓練一個 Word2Vec。
    """
    if not GENSIM_AVAILABLE:
        raise RuntimeError("需要 gensim 才能訓練 Word2Vec。請先安裝 gensim。")

    def simple_cut(s):
        # 這裡用最簡分詞（空白 + 逐字），你可以換成 CKIP/Jieba。
        # 對短標籤，直接當作一個 token；對長句，拆成字。
        s = re.sub(r"\s+", " ", s.strip())
        if " " in s:
            return s.split(" ")
        # 以字元為單位
        return list(s)

    sentences = []
    for t in corpus_texts:
        if not isinstance(t, str):
            continue
        seg = simple_cut(t)
        if seg:
            sentences.append(seg)

    # 再把 tokens 放進語料（每個 token 當一個句子）
    for tok in tokens:
        sentences.append(list(tok) if tok and " " not in tok else tok.split(" "))

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # skip-gram
        epochs=epochs
    )
    return model.wv  # KeyedVectors

def load_fasttext(path_or_bin):
    """
    載入 fastText 預訓練向量（本機），支援 .vec 或 .bin（gensim FTModel）
    """
    if not GENSIM_AVAILABLE:
        raise RuntimeError("需要 gensim 才能載入 FastText。")
    ext = os.path.splitext(path_or_bin)[-1].lower()
    if ext == ".vec":
        kv = KeyedVectors.load_word2vec_format(path_or_bin, binary=False)
        return kv
    elif ext == ".bin":
        ft = FTModel.load_fasttext_format(path_or_bin)
        return ft.wv
    else:
        raise ValueError("fastText 檔案需為 .vec 或 .bin")

class BertEncoder:
    def __init__(self, model_name_or_path="bert-base-chinese", device=None):
        if not HF_AVAILABLE:
            raise RuntimeError("需要 transformers/torch 才能使用 BERT。")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts, batch_size=64):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=32)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs).last_hidden_state  # [B, L, H]
            # 取 CLS 或 mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B,L,1]
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean = summed / counts
            embs.append(mean.cpu().numpy())
        return np.vstack(embs)

# ===== Similarity / Centroid =====

def cos_sim(a: np.ndarray, b: np.ndarray):
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T

def build_centroids(embeddings: dict, labeled: list):
    """
    embeddings: {token: vec}
    labeled: [(token, cat), ...] 只用 CATS 三大類
    回傳：centroids {cat: vec}, also cat_members {cat: [token]}
    """
    cat_members = {c: [] for c in CATS}
    for tok, cat in labeled:
        if cat in CATS and tok in embeddings:
            cat_members[cat].append(embeddings[tok])

    centroids = {}
    for c in CATS:
        if cat_members[c]:
            centroids[c] = np.mean(np.vstack(cat_members[c]), axis=0)
    return centroids, cat_members

def nearest_labeled_tokens(embeddings, token_vec, labeled_embeddings, topk=5):
    """
    回傳最接近的已標籤 tokens 清單（含類別與相似度）
    labeled_embeddings: [(tok, cat, vec)]
    """
    vecs = np.vstack([v for _, _, v in labeled_embeddings])
    sims = cos_sim(token_vec[None, :], vecs).ravel()
    idx = np.argsort(-sims)[:topk]
    out = []
    for i in idx:
        tok, cat, vec = labeled_embeddings[i]
        out.append((tok, cat, float(sims[i])))
    return out

# ===== Main pipeline =====

def recommend(
    results_csv="tag_classification_results.csv",
    tag_map_csv="tag_token_map.csv",
    extra_rule_csvs=None,
    method="w2v",                 # "w2v" | "fasttext" | "bert"
    fasttext_path=None,           # 本機 fastText 權重路徑
    bert_name_or_path="bert-base-chinese",
    topk_tokens=5,
    out_csv="uncategorized_recommendations.csv"
):
    # 1) 蒐集資料
    all_tokens, labeled_pairs, uncat_tokens, raw_lines = extract_all_tokens_from_results(results_csv)
    seeds = load_seed_tokens(tag_map_csv, extra_rule_csvs)

    # 若 tag_token_map 沒涵蓋到，從 results 的 labeled_pairs 也補充為 seed
    for tok, cat in labeled_pairs:
        if cat in seeds:
            seeds[cat].add(tok)

    # 2) 建立嵌入
    token_list = list(dict.fromkeys(all_tokens))
    token_to_vec = {}

    if method == "w2v":
        if not GENSIM_AVAILABLE:
            raise RuntimeError("請安裝 gensim 才能使用 Word2Vec。")
        # 用原始句子 + tokens 快速自訓
        wv = build_w2v_embeddings(raw_lines, token_list, vector_size=200, window=5, min_count=1, epochs=20)
        for t in token_list:
            if t in wv:
                token_to_vec[t] = wv[t]

    elif method == "fasttext":
        if not fasttext_path or not os.path.exists(fasttext_path):
            raise FileNotFoundError("請提供本機 fastText 權重路徑（.vec 或 .bin）。")
        kv = load_fasttext(fasttext_path)
        for t in token_list:
            # fastText 對 OOV 有 subword 能力（bin），vec 沒有；這裡用 try/except
            try:
                token_to_vec[t] = kv[t]
            except KeyError:
                pass

    elif method == "bert":
        if not HF_AVAILABLE:
            raise RuntimeError("請安裝 transformers/torch，並提供可用中文 BERT 模型。")
        encoder = BertEncoder(bert_name_or_path)
        # 直接對 tokens 做句向量（短詞也可）
        token_vecs = encoder.encode(token_list, batch_size=64)
        for t, v in zip(token_list, token_vecs):
            token_to_vec[t] = v

    else:
        raise ValueError("method 必須是 'w2v' | 'fasttext' | 'bert'")

    # 3) 準備 labeled 向量與類別 centroid
    labeled = []
    for tok, cat in labeled_pairs:
        if tok in token_to_vec and cat in CATS:  # 只三大類
            labeled.append((tok, cat, token_to_vec[tok]))

    centroids, _ = build_centroids(token_to_vec, labeled_pairs)

    # 4) 未分類 → 推薦
    rows = []
    for tok in uncat_tokens:
        if tok not in token_to_vec:
            # 這個 token 沒向量（OOV），跳過或標註
            rows.append({
                "未分類標籤": tok,
                "推薦類別": "",
                "最大相似度": "",
                "各類相似度(業務行為/客戶回饋/下一步)": "",
                "近鄰1": "",
                "近鄰2": "",
                "近鄰3": "",
                "備註": "無向量(OOV)"
            })
            continue

        v = token_to_vec[tok]

        # 與三大類 centroid 相似度
        cat_sims = {}
        for c, cen in centroids.items():
            cat_sims[c] = float(cos_sim(v[None, :], cen[None, :]).ravel()[0]) if c in CATS else -1.0

        # 推薦類別 = 最高相似度的類別
        if cat_sims:
            best_cat = max(cat_sims.items(), key=lambda x: x[1])[0]
            best_sim = cat_sims[best_cat]
        else:
            best_cat, best_sim = "", ""

        # 找已標籤 token 的近鄰（當依據）
        neigh = nearest_labeled_tokens(token_to_vec, v, labeled, topk=3) if labeled else []
        n1 = f"{neigh[0][0]}({neigh[0][1]};{neigh[0][2]:.3f})" if len(neigh) > 0 else ""
        n2 = f"{neigh[1][0]}({neigh[1][1]};{neigh[1][2]:.3f})" if len(neigh) > 1 else ""
        n3 = f"{neigh[2][0]}({neigh[2][1]};{neigh[2][2]:.3f})" if len(neigh) > 2 else ""

        rows.append({
            "未分類標籤": tok,
            "推薦類別": best_cat,
            "最大相似度": round(best_sim, 3) if best_sim != "" else "",
            "各類相似度(業務行為/客戶回饋/下一步)": "; ".join([f"{c}:{cat_sims.get(c, float('nan')):.3f}" for c in CATS]),
            "近鄰1": n1,
            "近鄰2": n2,
            "近鄰3": n3,
            "備註": ""
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="tag_classification_results.csv")
    parser.add_argument("--tag_map_csv", default="tag_token_map.csv")
    parser.add_argument("--category_dictionary_csv", default="category_dictionary.csv")
    parser.add_argument("--custom_activity_map_csv", default="custom_activity_map.csv")
    parser.add_argument("--agent_overrides_csv", default="agent_overrides.csv")
    parser.add_argument("--method", choices=["w2v", "fasttext", "bert"], default="w2v")
    parser.add_argument("--fasttext_path", default=None, help="本機 fastText 權重（.vec 或 .bin）")
    parser.add_argument("--bert_name_or_path", default="bert-base-chinese", help="可填本機 BERT 模型資料夾")
    parser.add_argument("--out_csv", default="uncategorized_recommendations.csv")
    args = parser.parse_args()

    extras = []
    for p in [args.category_dictionary_csv, args.custom_activity_map_csv, args.agent_overrides_csv]:
        if os.path.exists(p):
            extras.append(p)

    out_csv = recommend(
        results_csv=args.results_csv,
        tag_map_csv=args.tag_map_csv,
        extra_rule_csvs=extras,
        method=args.method,
        fasttext_path=args.fasttext_path,
        bert_name_or_path=args.bert_name_or_path,
        out_csv=args.out_csv
    )
    print(f"完成：{out_csv}")


# # Word2Vec（離線快速）
# python recommend_categories.py --method w2v \
#   --results_csv tag_classification_results.csv \
#   --tag_map_csv tag_token_map.csv \
#   --out_csv uncategorized_recommendations.csv

# # FastText（有本機權重）
# python recommend_categories.py --method fasttext \
#   --fasttext_path /path/to/cc.zh.300.bin \
#   --results_csv tag_classification_results.csv \
#   --tag_map_csv tag_token_map.csv
  
# # Chinese-BERT（有 transformers/torch 與本機/可下載模型）
# python recommend_categories.py --method bert \
#   --bert_name_or_path bert-base-chinese \
#   --results_csv tag_classification_results.csv \
#   --tag_map_csv tag_token_map.csv

