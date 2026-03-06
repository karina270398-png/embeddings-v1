#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict

from service_index import OkvedIndex

DATA_DIR = Path(os.getenv("OKVED_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))).resolve()
CATALOGS_DIR = DATA_DIR / "catalogs"


class IndexRegistry:
    def __init__(self):
        self._cache: Dict[str, OkvedIndex] = {}
        # Preload default OKVED
        if (DATA_DIR / "okved_embeddings.npy").exists():
            self._cache["okved"] = OkvedIndex(DATA_DIR)
        # Discover catalogs
        if CATALOGS_DIR.exists():
            for sub in CATALOGS_DIR.iterdir():
                if not sub.is_dir():
                    continue
                # Each catalog dir expected to have embeddings.npy and metadata.parquet
                if (sub / "embeddings.npy").exists() and (sub / "metadata.parquet").exists():
                    self._cache[sub.name] = OkvedIndex(sub)

    def list(self):
        return sorted(self._cache.keys())

    def get(self, name: str) -> OkvedIndex:
        if name not in self._cache:
            raise KeyError(f"Unknown catalog: {name}")
        return self._cache[name]
