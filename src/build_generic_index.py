#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from rich import print
from rich.progress import track

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def try_read_table(path: Path, sep: str = ";", encoding: Optional[str] = None, sheet: Optional[str] = None, no_header: bool = False) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".csv"}:
        for enc in ([encoding] if encoding else ["utf-8", "cp1251", "windows-1251", "latin-1"]):
            try:
                return pd.read_csv(p, sep=sep, encoding=enc, engine="python", dtype=str, quoting=csv.QUOTE_MINIMAL, header=None if no_header else "infer")
            except Exception as e:
                last_e = e
        raise last_e  # type: ignore
    elif p.suffix.lower() in {".json"}:
        data = json.loads(p.read_text(encoding=encoding or "utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            # pick first list-like value
            for v in data.values():
                if isinstance(v, list):
                    return pd.DataFrame(v)
        return pd.json_normalize(data)
    elif p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p, sheet_name=sheet or 0, dtype=str)
        except Exception:
            # fallback engine if needed
            return pd.read_excel(p, sheet_name=sheet or 0, engine="openpyxl", dtype=str)
    elif p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_cols:
            return lower_cols[name]
    # try contains
    for c in df.columns:
        cl = c.lower()
        if any(name in cl for name in candidates):
            return c
    return None


def build_text_row(row: pd.Series, code_col: Optional[str], name_col: Optional[str], desc_col: Optional[str]) -> str:
    parts: List[str] = []
    if code_col and isinstance(row.get(code_col), str) and row.get(code_col).strip():
        parts.append(f"КОД {row.get(code_col).strip()}")
    if name_col and isinstance(row.get(name_col), str) and row.get(name_col).strip():
        parts.append(row.get(name_col).strip())
    if desc_col and isinstance(row.get(desc_col), str) and row.get(desc_col).strip():
        parts.append(row.get(desc_col).strip())
    return ". ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Build generic embeddings for a dictionary-like table")
    ap.add_argument("--input", required=True, help="Path to input file (csv/json/xlsx/parquet)")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g., data/catalogs/okfs)")
    ap.add_argument("--sep", default=";", help="CSV separator")
    ap.add_argument("--encoding", default=None, help="Override encoding (default: auto)")
    ap.add_argument("--sheet", default=None, help="Excel sheet name or index")
    ap.add_argument("--no-header", action="store_true", help="Treat first row as data (CSV)")
    ap.add_argument("--code-cols", default=None, help="Comma-separated candidate columns for code")
    ap.add_argument("--name-cols", default=None, help="Comma-separated candidate columns for name/title")
    ap.add_argument("--desc-cols", default=None, help="Comma-separated candidate columns for description")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = try_read_table(inp, sep=args.sep, encoding=args.encoding, sheet=args.sheet, no_header=args.no_header)

    # Normalize column names map
    df.columns = [str(c).strip() for c in df.columns]

    code_candidates = [s.strip().lower() for s in (args.code_cols.split(",") if args.code_cols else [])] or [
        "code", "код", "id", "номер", "окпд", "оквэд", "оквэд2", "окфс", "окопф"
    ]
    name_candidates = [s.strip().lower() for s in (args.name_cols.split(",") if args.name_cols else [])] or [
        "name", "наименование", "наим", "title", "label"
    ]
    desc_candidates = [s.strip().lower() for s in (args.desc_cols.split(",") if args.desc_cols else [])] or [
        "description", "описание", "desc", "details"
    ]

    code_col = pick_first_present(df, code_candidates)
    name_col = pick_first_present(df, name_candidates)
    desc_col = pick_first_present(df, desc_candidates)

    if not name_col and not code_col:
        # Heuristic fallback: assume first columns as code/name/desc
        cols = list(df.columns)
        if len(cols) >= 2:
            code_col = cols[0]
            name_col = cols[1]
            if len(cols) >= 3:
                desc_col = cols[2]
        else:
            raise ValueError("Не нашли подходящих колонок для code/name — укажите явным образом через --code-cols/--name-cols")

    texts: List[str] = []
    for _, row in df.iterrows():
        texts.append(build_text_row(row, code_col, name_col, desc_col))

    # Persist metadata parquet
    meta_cols = []
    for c in [code_col, name_col, desc_col]:
        if c and c not in meta_cols:
            meta_cols.append(c)
    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame({"text": texts})
    meta_df["text_for_embed"] = texts
    meta_path = outdir / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    # Embeddings
    model = TextEmbedding(model_name=MODEL_NAME)
    embs = []
    for vec in track(model.embed(texts, batch_size=args.batch), total=len(texts), description="Embedding"):
        embs.append(np.asarray(vec, dtype=np.float32))
    emb_array = np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)
    if emb_array.size:
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True) + 1e-12
        emb_array = emb_array / norms

    np.save(outdir / "embeddings.npy", emb_array)
    (outdir / "index_meta.json").write_text(
        json.dumps({
            "model": MODEL_NAME,
            "dimension": int(emb_array.shape[1]) if emb_array.size else 0,
            "count": int(emb_array.shape[0]),
            "source": str(inp),
            "columns": {"code": code_col, "name": name_col, "description": desc_col},
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved -> {outdir}")


if __name__ == "__main__":
    main()
