#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Replace custom bullet markers "^-" with normal dashes and newlines
    s = re.sub(r"\^\-", "\n- ", s)
    # Collapse excessive whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text_for_embed(row: pd.Series) -> str:
    parts = []
    code = (row.get("code") or "").strip()
    name = normalize_text(row.get("name") or "")
    desc = normalize_text(row.get("description") or "")
    if code:
        parts.append(f"ОКВЭД {code}")
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    text = ". ".join(p for p in parts if p)
    # For paraphrase-multilingual models no special prefix is needed
    return text if text else ""


def main():
    parser = argparse.ArgumentParser(description="Build OKVED embeddings using FastEmbed (multilingual e5-base)")
    parser.add_argument("--csv", required=True, help="Path to source OKVED CSV (cp1251, ';' separated, no header)")
    parser.add_argument("--out", default="data", help="Output directory for artifacts")
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    src_csv = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV. The file is Windows-1251 encoded, semicolon-separated, quotes present, no header
    df = pd.read_csv(
        src_csv,
        sep=";",
        header=None,
        names=["section", "code", "name", "description", "field5", "field6", "date_from", "date_to"],
        encoding="cp1251",
        engine="python",
        dtype=str,
        quoting=csv.QUOTE_MINIMAL,
    )

    # Trim whitespace
    for col in ["section", "code", "name", "description"]:
        df[col] = df[col].astype(str).str.strip()

    # Build text payloads
    df["text_for_embed"] = df.apply(build_text_for_embed, axis=1)

    # Persist metadata as Parquet
    meta_path = out_dir / "okved_metadata.parquet"
    df[["section", "code", "name", "description", "date_from", "date_to", "text_for_embed"]].to_parquet(meta_path, index=False)

    # Prepare embeddings
    model = TextEmbedding(model_name=MODEL_NAME)

    texts = df["text_for_embed"].tolist()
    # Generate embeddings as a list to keep order aligned with metadata
    embs = []
    for vec in track(model.embed(texts, batch_size=args.batch), total=len(texts), description="Embedding"):
        embs.append(np.asarray(vec, dtype=np.float32))
    if embs:
        emb_array = np.vstack(embs)
    else:
        emb_array = np.zeros((0, 0), dtype=np.float32)

    # L2 normalize
    if emb_array.size:
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True) + 1e-12
        emb_array = emb_array / norms

    # Save artifacts
    npy_path = out_dir / "okved_embeddings.npy"
    np.save(npy_path, emb_array)

    # Also save a lightweight JSON with model + dims
    meta_json = out_dir / "okved_index_meta.json"
    model_dim = int(emb_array.shape[1]) if emb_array.size else 0
    meta_json.write_text(
        (
            '{"model":"%s","dimension":%d,"count":%d}'
            % (MODEL_NAME, model_dim, int(emb_array.shape[0]))
        ),
        encoding="utf-8",
    )

    print(f"Saved metadata -> {meta_path}")
    print(f"Saved embeddings -> {npy_path}")
    print(f"Saved index meta -> {meta_json}")


if __name__ == "__main__":
    main()
