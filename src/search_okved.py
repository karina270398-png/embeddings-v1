#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


def main():
    parser = argparse.ArgumentParser(description="Search OKVED embeddings")
    parser.add_argument("--index", default="data", help="Directory with okved_embeddings.npy, okved_metadata.parquet")
    parser.add_argument("--query", required=True, help="User query in Russian (human-readable)")
    parser.add_argument("--k", type=int, default=10, help="Top-K results to show")
    args = parser.parse_args()

    idx_dir = Path(args.index).expanduser().resolve()
    emb_path = idx_dir / "okved_embeddings.npy"
    meta_path = idx_dir / "okved_metadata.parquet"
    meta_json = idx_dir / "okved_index_meta.json"

    # Load artifacts
    embs = np.load(emb_path)
    meta = pd.read_parquet(meta_path)
    if meta_json.exists():
        info = json.loads(meta_json.read_text(encoding="utf-8"))
    else:
        info = {"model": "intfloat/multilingual-e5-base", "dimension": int(embs.shape[1]), "count": int(embs.shape[0])}

    # Build query embedding (no special prefix for paraphrase-multilingual)
    model = TextEmbedding(model_name=MODEL_NAME)
    q_text = args.query.strip()
    q_vec = np.asarray(list(model.embed([q_text]))[0], dtype=np.float32)
    q_vec = l2_normalize(q_vec)

    # Cosine sim via dot product on L2-normalized vectors
    scores = embs @ q_vec
    topk_idx = np.argsort(scores)[-args.k:][::-1]

    print(f"Model: {info.get('model')} | records: {info.get('count')} | dim: {info.get('dimension')}")
    print("==== Top results ====")
    for rank, i in enumerate(topk_idx, start=1):
        row = meta.iloc[int(i)]
        code = (row.get("code") or "").strip()
        name = (row.get("name") or "").strip()
        score = float(scores[int(i)])
        print(f"{rank:2d}. [{score: .4f}] {code} — {name}")


if __name__ == "__main__":
    main()
