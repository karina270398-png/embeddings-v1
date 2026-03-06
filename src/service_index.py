#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Tuple, Union, Optional
import json

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


class OkvedIndex:
    def __init__(self, index_dir: Union[str, Path]):
        p = Path(index_dir).expanduser().resolve()
        emb_path = p / "okved_embeddings.npy"
        meta_path = p / "okved_metadata.parquet"
        if not emb_path.exists():
            emb_path = p / "embeddings.npy"
        if not meta_path.exists():
            meta_path = p / "metadata.parquet"
        self.embs = np.load(emb_path)
        self.meta = pd.read_parquet(meta_path)
        # Optional: load column mapping
        self.cols = {"code": "code", "name": "name", "description": "description", "section": "section"}
        imeta = p / "index_meta.json"
        if not imeta.exists():
            imeta = p / "okved_index_meta.json"
        if imeta.exists():
            try:
                m = json.loads(imeta.read_text(encoding="utf-8"))
                cols = m.get("columns") or {}
                # Only override if present
                for k in ["code", "name", "description", "section"]:
                    v = cols.get(k)
                    if v is None:
                        continue
                    # normalize to string labels since parquet may coerce to str
                    v_str = str(v)
                    if v_str in self.meta.columns:
                        self.cols[k] = v_str
            except Exception:
                pass
        self.model = TextEmbedding(model_name=MODEL_NAME)

    def embed_query(self, q: str) -> np.ndarray:
        vec = np.asarray(list(self.model.embed([q.strip()]))[0], dtype=np.float32)
        return l2_normalize(vec)

    def search(self, q: str, k: int = 10) -> List[Tuple[int, float]]:
        q_vec = self.embed_query(q)
        scores = self.embs @ q_vec
        topk = np.argsort(scores)[-k:][::-1]
        return [(int(i), float(scores[int(i)])) for i in topk]

    def row(self, idx: int) -> dict:
        r = self.meta.iloc[int(idx)]
        # Try mapped cols then fallbacks
        def getv(key: str) -> str:
            col = self.cols.get(key)
            if col and col in r and isinstance(r.get(col), str):
                return r.get(col).strip()
            if key in r and isinstance(r.get(key), str):
                return r.get(key).strip()
            return ""
        # Fallback: if empty code/name, try first columns
        code = getv("code")
        name = getv("name")
        desc = getv("description")
        if not code or not name:
            cols = list(self.meta.columns)
            if not code and cols:
                v = r.get(cols[0])
                code = v.strip() if isinstance(v, str) else code
            if not name and len(cols) > 1:
                v = r.get(cols[1])
                name = v.strip() if isinstance(v, str) else name
        return {
            "section": getv("section"),
            "code": code or "",
            "name": name or "",
            "description": desc or "",
        }

    @staticmethod
    def rollup_code_2digit(code: str) -> Optional[str]:
        # Extract leading digits, take first two
        num = []
        for ch in code:
            if ch.isdigit():
                num.append(ch)
            elif ch == ".":
                break
            else:
                break
        if not num:
            return None
        return ("".join(num))[:2] if len(num) >= 2 else "".join(num)