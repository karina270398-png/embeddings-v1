#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from service_index import OkvedIndex
from multi_index import IndexRegistry

INDEX_DIR = os.getenv("OKVED_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
CARDS_API_BASE = os.getenv("CARDS_API_BASE", "http://10.0.61.88:8092")
CARDS_API_PATH = os.getenv("CARDS_API_PATH", "/api/v1/batchCardsByFiltersPreview")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

app = FastAPI(title="OKVED Search API")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Human-readable query in Russian")
    top_k: int = Field(20, ge=1, le=200)
    rollup: str = Field("2digit", description="'2digit' or 'full'")
    only_main_okveds: bool = True
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    catalog: str = Field("okved", description="Catalog name: 'okved' or entries from data/catalogs")


class SearchResponse(BaseModel):
    okveds: List[str]
    matches: List[dict]
    external_url: str
    external_status: int
    external_response: Optional[dict] = None


# Load indexes at startup
registry = IndexRegistry()


@app.post("/search_and_preview", response_model=SearchResponse)
async def search_and_preview(req: SearchRequest):
    try:
        idx = registry.get(req.catalog)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown catalog: {req.catalog}. Available: {', '.join(registry.list())}")
    pairs = idx.search(req.query, k=req.top_k)
    matches = []
    codes = []
    for pos, (i, score) in enumerate(pairs, start=1):
        r = idx.row(i)
        code = r.get("code") or ""
        roll = idx.rollup_code_2digit(code) if req.rollup == "2digit" else code
        if roll and roll not in codes:
            codes.append(roll)
        matches.append({
            "rank": pos,
            "score": round(float(score), 6),
            "code": code,
            "name": r.get("name"),
            "rollup": roll,
        })

    url = f"{CARDS_API_BASE.rstrip('/')}{CARDS_API_PATH}?limit={req.limit}&offset={req.offset}"
    payload = {"okveds": codes, "only_main_okveds": bool(req.only_main_okveds)}

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers={"accept": "application/json"})
            data = None
            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text[:1000]}
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    return SearchResponse(
        okveds=codes,
        matches=matches,
        external_url=url,
        external_status=resp.status_code,
        external_response=data,
    )

@app.get("/catalogs")
async def list_catalogs():
    return {"catalogs": registry.list()}
