# okved-embeddings

Поиск по ОКВЭД и другим справочникам через эмбеддинги (`fastembed`) с интерфейсами:
- CLI
- FastAPI
- Streamlit

Проект умеет:
- строить индекс для ОКВЭД из исходного CSV;
- строить индексы для произвольных справочников (CSV/JSON/XLSX/Parquet);
- искать наиболее релевантные коды по текстовому запросу;
- отправлять найденные коды во внешний API предпросмотра карточек.

## Стек

- Python 3.9+
- `fastembed`
- `numpy`, `pandas`, `pyarrow`
- `fastapi`, `uvicorn`
- `streamlit`

## Структура

- `src/build_index.py` — построение индекса ОКВЭД из CSV.
- `src/build_generic_index.py` — построение индекса для произвольного справочника.
- `src/search_okved.py` — локальный CLI-поиск по индексу.
- `src/service_index.py` — загрузка индекса и семантический поиск.
- `src/multi_index.py` — реестр каталогов индексов.
- `src/api.py` — FastAPI endpoint `search_and_preview`.
- `src/streamlit_app.py` — Streamlit UI.
- `data/` — артефакты индексов.
- `data/catalogs/<name>/` — дополнительные каталоги.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Быстрый старт

### 1) Построить индекс ОКВЭД

```bash
./build.sh
```

Или вручную:

```bash
python src/build_index.py \
  --csv "/path/to/okved.csv" \
  --out data \
  --batch 64
```

### 2) Выполнить локальный поиск (CLI)

```bash
python src/search_okved.py \
  --index data \
  --query "перевозка грузов автомобильным транспортом" \
  --k 10
```

### 3) Запустить API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:

```bash
curl -X POST "http://localhost:8000/search_and_preview" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "оптовая торговля пищевыми продуктами",
    "top_k": 20,
    "rollup": "2digit",
    "only_main_okveds": true,
    "limit": 50,
    "offset": 0,
    "catalog": "okved"
  }'
```

Доступные каталоги:

```bash
curl "http://localhost:8000/catalogs"
```

### 4) Запустить Streamlit

```bash
streamlit run src/streamlit_app.py
```

## Переменные окружения

- `OKVED_INDEX_DIR` — путь к директории с индексом (по умолчанию `../data`).
- `CARDS_API_BASE` — базовый URL внешнего API (по умолчанию `http://10.0.61.88:8092`).
- `CARDS_API_PATH` — путь внешнего API (по умолчанию `/api/v1/batchCardsByFiltersPreview`).
- `HTTP_TIMEOUT` — timeout HTTP-запросов в секундах (по умолчанию `30`).

## Формат индекса

Для базового ОКВЭД в `data/`:
- `okved_embeddings.npy` — нормализованные эмбеддинги;
- `okved_metadata.parquet` — метаданные записей;
- `okved_index_meta.json` — модель, размерность, количество.

Для произвольных каталогов в `data/catalogs/<catalog>/`:
- `embeddings.npy`
- `metadata.parquet`
- `index_meta.json`

## Добавление нового каталога

```bash
python src/build_generic_index.py \
  --input "/path/to/catalog.xlsx" \
  --outdir "data/catalogs/my_catalog" \
  --batch 64
```

После этого каталог автоматически появится в API и Streamlit (через `IndexRegistry`).

