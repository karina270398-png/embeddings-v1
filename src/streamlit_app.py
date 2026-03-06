#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from functools import lru_cache

import httpx
import pandas as pd
import streamlit as st

from service_index import OkvedIndex
from multi_index import IndexRegistry

st.set_page_config(page_title="OKVED Search", layout="wide")

INDEX_DIR = os.getenv("OKVED_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
CARDS_API_BASE = os.getenv("CARDS_API_BASE", "http://10.0.61.88:8092")
CARDS_API_PATH = os.getenv("CARDS_API_PATH", "/api/v1/batchCardsByFiltersPreview")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))


@st.cache_resource(show_spinner=False)
def load_registry() -> IndexRegistry:
    return IndexRegistry()

reg = load_registry()

st.title("Поиск по справочникам + предпросмотр карт")
cols = st.columns([2,1,1,1,1])
with cols[0]:
    query = st.text_input("Запрос", value="перевозка грузов автомобильным транспортом")
with cols[1]:
    catalog = st.selectbox("Каталог", options=reg.list(), index=0 if 'okved' in reg.list() else 0)
with cols[2]:
    k = st.slider("Top-K", min_value=5, max_value=100, value=25, step=5)
with cols[3]:
    rollup = st.selectbox("Коды на отправку", options=["2digit", "full"], index=0)
with cols[4]:
    only_main = st.checkbox("Только основной ОКВЭД", value=True)

limit = st.number_input("Limit", min_value=1, max_value=1000, value=50, step=1)
offset = st.number_input("Offset", min_value=0, value=0, step=10)

if st.button("Найти и получить предпросмотр"):
    idx = reg.get(catalog)
    pairs = idx.search(query, k=k)
    rows = []
    codes = []
    for rank, (i, score) in enumerate(pairs, start=1):
        r = idx.row(i)
        code = r.get("code") or ""
        roll = idx.rollup_code_2digit(code) if rollup == "2digit" else code
        if roll and roll not in codes:
            codes.append(roll)
        rows.append({
            "rank": rank,
            "score": round(float(score), 6),
            "code": code,
            "name": r.get("name"),
            "rollup": roll,
        })

    st.subheader("Подобранные коды (отправка в API)")
    st.write(", ".join(codes) if codes else "—")

    st.subheader("Соответствия")
    st.dataframe(pd.DataFrame(rows))

    url = f"{CARDS_API_BASE.rstrip('/')}{CARDS_API_PATH}?limit={int(limit)}&offset={int(offset)}"
    payload = {"okveds": codes, "only_main_okveds": bool(only_main)}

    st.subheader("Ответ внешнего API")
    with st.spinner("Запрос..."):
        try:
            r = httpx.post(url, json=payload, headers={"accept": "application/json"}, timeout=HTTP_TIMEOUT)
            if r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
                st.json(data)
            else:
                st.code(r.text[:2000])
            st.caption(f"{r.status_code} {url}")
        except httpx.RequestError as e:
            st.error(f"Ошибка обращения к API: {e}")

st.divider()
with st.expander("Настройки"):
    st.text_input("INDEX_DIR", value=INDEX_DIR)
    st.text_input("Доступные каталоги", value=", ".join(reg.list()))
    st.text_input("CARDS_API_BASE", value=CARDS_API_BASE)
    st.text_input("CARDS_API_PATH", value=CARDS_API_PATH)
