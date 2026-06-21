# Mom's English Vocabulary Cards

A mobile-friendly flashcard app for Mom's personal English vocabulary list (~2,006 words).

## Features

Same as the main vocabulary app:

- Alphabetical category cards (A–Z + 기타)
- Flashcard study with flip animation
- Learned tracking, points, and quiz mode
- Mobile-first dark theme UI

## Run locally

```bash
cd C:\Users\ungdu\mom_vocabulary_app
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

Open [http://localhost:8502](http://localhost:8502)

## Data source

Words are loaded from `Mom's word.txt` (tab-separated English ↔ Korean pairs).

## Deploy

Push to GitHub and deploy on [Streamlit Cloud](https://share.streamlit.io) with `app.py` as the entry point.
