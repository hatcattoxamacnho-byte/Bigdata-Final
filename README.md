# Big-Data

Superstore Analytics Dashboard — Big Data Final Report

## Project structure (required for Streamlit Cloud)

```
Finalbigdata/
├── app.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── charts.py
│   ├── helpers.py
│   └── insight_charts.py
├── data/
│   └── Superstore.csv
└── .streamlit/
    └── config.toml
```

**Important:** Do not put `charts.py`, `helpers.py`, or `Superstore.csv` in the repo root.
`app.py` imports `from utils import charts` and loads data from `data/Superstore.csv`.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push the folder structure above to GitHub (`hatcattoxamacnho-byte/Finalbigdata`)
2. Open [share.streamlit.io](https://share.streamlit.io) → **Create app**
3. Repository: `hatcattoxamacnho-byte/Finalbigdata` · Branch: `main` · Main file: `app.py`
4. Click **Deploy**

If you see `ModuleNotFoundError: No module named 'utils'`, your files are in the wrong folders — move Python files into `utils/` and CSV into `data/`.
