"""Mobile-friendly English vocabulary flashcard app powered by Streamlit."""

from __future__ import annotations

import hashlib
import html
import json
import random
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_TITLE = "Mom's English Vocabulary Cards"
VOCAB_PATH = Path(__file__).parent / "vocabulary.json"
PROGRESS_PATH = Path(__file__).parent / "progress.json"

CARD_GRADIENTS = [
    ("#667eea", "#764ba2"),
    ("#f093fb", "#f5576c"),
    ("#4facfe", "#00f2fe"),
    ("#43e97b", "#38f9d7"),
    ("#fa709a", "#fee140"),
    ("#a18cd1", "#fbc2eb"),
    ("#ff9a9e", "#fecfef"),
    ("#ffecd2", "#fcb69f"),
    ("#89f7fe", "#66a6ff"),
    ("#f6d365", "#fda085"),
]


@st.cache_data
def load_vocabulary(vocab_mtime: float) -> dict:
    del vocab_mtime  # cache key only — reload when vocabulary.json changes
    with VOCAB_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def default_progress() -> dict:
    return {"learned": {}, "learned_log": [], "points_log": []}


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return default_progress()
    try:
        with PROGRESS_PATH.open(encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return default_progress()
    base = default_progress()
    base.update({key: data.get(key, base[key]) for key in base})
    return base


def save_progress(progress: dict) -> None:
    with PROGRESS_PATH.open("w", encoding="utf-8") as file:
        json.dump(progress, file, ensure_ascii=False, indent=2)


def sync_learned_status(progress: dict) -> None:
    st.session_state.learned_status = {
        key: "learned" for key in progress.get("learned", {})
    }


def init_session_state() -> None:
    defaults = {
        "page": "home",
        "selected_category": None,
        "card_index": 0,
        "flipped": False,
        "random_mode": False,
        "card_order": [],
        "learned_status": {},
        "category_query": "",
        "category_page": 0,
        "quiz_category": "Random",
        "quiz_source": "All words",
        "quiz_question": None,
        "quiz_feedback": None,
        "quiz_total_points": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "progress" not in st.session_state:
        st.session_state.progress = load_progress()
        sync_learned_status(st.session_state.progress)


def category_display_name(name: str) -> str:
    if name == "#":
        return "기타 / 문장"
    return name


def category_palette(name: str) -> tuple[str, str]:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(CARD_GRADIENTS)
    return CARD_GRADIENTS[index]


def words_for_category(vocabulary: list[dict], category: str) -> list[dict]:
    return [item for item in vocabulary if item["category"] == category]


def status_key(word: dict) -> str:
    return f"{word['category']}::{word['word']}"


def learned_count(progress: dict) -> int:
    return len(progress.get("learned", {}))


def total_points(progress: dict) -> int:
    return sum(entry.get("points", 0) for entry in progress.get("points_log", []))


def daily_counts(log: list[dict], value_key: str = "count") -> pd.DataFrame:
    if not log:
        return pd.DataFrame(columns=["date", value_key])
    counter = Counter(entry["date"] for entry in log)
    rows = [{"date": day, value_key: count} for day, count in sorted(counter.items())]
    return pd.DataFrame(rows)


def daily_points(log: list[dict]) -> pd.DataFrame:
    if not log:
        return pd.DataFrame(columns=["date", "points"])
    counter: Counter[str] = Counter()
    for entry in log:
        counter[entry["date"]] += entry.get("points", 0)
    rows = [{"date": day, "points": pts} for day, pts in sorted(counter.items())]
    return pd.DataFrame(rows)


def mark_learned(word: dict) -> None:
    progress = st.session_state.progress
    key = status_key(word)
    today = date.today().isoformat()
    now = datetime.now().isoformat(timespec="seconds")

    progress["learned"][key] = {"date": today, "marked_at": now}
    progress["learned_log"].append(
        {
            "date": today,
            "marked_at": now,
            "word_key": key,
            "word": word["word"],
            "category": word["category"],
        }
    )
    save_progress(progress)
    st.session_state.learned_status[key] = "learned"


def add_quiz_point(word: dict) -> None:
    progress = st.session_state.progress
    today = date.today().isoformat()
    progress["points_log"].append(
        {
            "date": today,
            "marked_at": datetime.now().isoformat(timespec="seconds"),
            "points": 1,
            "word_key": status_key(word),
            "word": word["word"],
            "category": word["category"],
        }
    )
    save_progress(progress)


def reset_card_order(category: str, vocabulary: list[dict]) -> None:
    words = words_for_category(vocabulary, category)
    order = list(range(len(words)))
    if st.session_state.random_mode:
        random.shuffle(order)
    st.session_state.card_order = order
    st.session_state.card_index = 0
    st.session_state.flipped = False


def current_word(vocabulary: list[dict]) -> dict | None:
    category = st.session_state.selected_category
    if not category:
        return None
    words = words_for_category(vocabulary, category)
    if not words or not st.session_state.card_order:
        return None
    index = st.session_state.card_order[st.session_state.card_index]
    return words[index]


def learned_word_items(vocabulary: list[dict], progress: dict) -> list[dict]:
    learned_keys = set(progress.get("learned", {}))
    return [item for item in vocabulary if status_key(item) in learned_keys]


def quiz_pool(vocabulary: list[dict], progress: dict) -> list[dict]:
    if st.session_state.quiz_source == "Learned only":
        pool = learned_word_items(vocabulary, progress)
    else:
        pool = list(vocabulary)

    category = st.session_state.quiz_category
    if category != "Random":
        pool = [item for item in pool if item["category"] == category]
    return pool


def build_quiz_question(pool: list[dict], vocabulary: list[dict]) -> dict | None:
    if not pool:
        return None

    correct = random.choice(pool)
    others = [item for item in vocabulary if item["word"] != correct["word"]]
    distractor_count = min(3, len(others))
    distractors = random.sample(others, distractor_count) if distractor_count else []
    options = [correct, *distractors]
    random.shuffle(options)
    return {
        "correct": correct,
        "options": options,
        "prompt": correct["meaning"],
        "part_of_speech": correct.get("part_of_speech", ""),
    }


def render_calendar_heatmap(counts_by_date: dict[str, int], title: str) -> None:
    today = date.today()
    first_day = today.replace(day=1)
    start_offset = first_day.weekday()
    month_days = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    days_in_month = month_days.day

    st.markdown(f"**{title} — {today.strftime('%B %Y')}**")
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    header = "".join(f'<div class="cal-head">{label}</div>' for label in weekday_labels)
    cells = ['<div class="cal-empty"></div>' for _ in range(start_offset)]

    for day_num in range(1, days_in_month + 1):
        day_str = date(today.year, today.month, day_num).isoformat()
        count = counts_by_date.get(day_str, 0)
        level = min(count, 4)
        cells.append(
            f'<div class="cal-cell level-{level}" title="{day_str}: {count}">'
            f'<span>{day_num}</span><small>{count or ""}</small></div>'
        )

    st.markdown(
        f'<div class="calendar-grid">{header}{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def render_stats_detail_page(
    *,
    title: str,
    log: list[dict],
    value_label: str,
    aggregate_fn,
) -> None:
    st.markdown(f"### {title}")
    if st.button("← Home", use_container_width=False):
        st.session_state.page = "home"
        st.rerun()

    if not log:
        st.info("No data yet.")
        return

    df = aggregate_fn(log)
    total = int(df[value_label].sum()) if not df.empty else 0
    st.metric(f"Total {value_label}", total)

    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    st.markdown("**Daily trend**")
    st.line_chart(trend_df.set_index("date")[value_label], height=220)
    st.markdown("**Daily bar chart**")
    st.bar_chart(trend_df.set_index("date")[value_label], height=220)

    counts_by_date = dict(zip(df["date"], df[value_label]))
    render_calendar_heatmap(counts_by_date, title)

    st.markdown("**Daily breakdown**")
    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            :root {
                --text: #f3f4f6;
                --muted: #9ca3af;
                --accent: #818cf8;
                --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            }

            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                color: var(--text);
            }

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"] {
                background-color: #0b0f14;
                background-image:
                    linear-gradient(180deg, rgba(8, 12, 18, 0.72) 0%, rgba(10, 16, 26, 0.86) 100%),
                    url("https://images.unsplash.com/photo-1456513080920-11a948d24615?auto=format&fit=crop&w=1800&q=80");
                background-size: cover;
                background-position: center top;
                background-attachment: fixed;
                background-repeat: no-repeat;
            }

            #MainMenu, footer, header { visibility: hidden; }

            .block-container {
                max-width: 720px;
                padding-top: 1.25rem;
                padding-bottom: 5rem;
            }

            .app-header { text-align: center; margin-bottom: 0.75rem; }
            .app-header h1 {
                font-size: 1.55rem;
                font-weight: 700;
                margin: 0;
                color: #ffffff;
                text-shadow: 0 2px 16px rgba(0, 0, 0, 0.45);
                line-height: 1.3;
            }
            .app-header p {
                margin: 0.35rem 0 0;
                color: var(--muted);
                font-size: 0.95rem;
            }

            .stats-row {
                display: flex;
                justify-content: center;
                gap: 0.55rem;
                flex-wrap: wrap;
                margin: 1rem 0 1.25rem;
            }

            .stat-pill {
                background: rgba(17, 24, 39, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 999px;
                padding: 0.45rem 0.9rem;
                font-size: 0.82rem;
                color: #d1d5db;
                box-shadow: var(--shadow);
                backdrop-filter: blur(8px);
            }

            .stat-pill.learned-pill {
                border-color: rgba(52, 211, 153, 0.35);
                color: #a7f3d0;
            }

            .cat-style-marker { display: none; }

            div[data-testid="stVerticalBlock"] > div:has(.cat-style-marker) + div [data-testid="stButton"] {
                margin-bottom: 0.85rem !important;
            }

            div[data-testid="stVerticalBlock"] > div:has(.cat-style-marker) + div [data-testid="stButton"] button {
                border-radius: 22px !important;
                min-height: 118px !important;
                height: 118px !important;
                white-space: pre-line !important;
                line-height: 1.35 !important;
                font-size: 0.95rem !important;
                font-weight: 700 !important;
                box-shadow: 0 14px 28px rgba(15, 23, 42, 0.16) !important;
                border: none !important;
                color: #ffffff !important;
                transition: transform 0.22s ease, box-shadow 0.22s ease !important;
            }

            div[data-testid="stVerticalBlock"] > div:has(.cat-style-marker) + div [data-testid="stButton"] button:hover {
                transform: translateY(-4px) scale(1.02);
                box-shadow: 0 18px 34px rgba(15, 23, 42, 0.22) !important;
                border: none !important;
            }

            .category-title {
                font-size: 1.05rem;
                font-weight: 700;
                line-height: 1.35;
                margin-bottom: 0.35rem;
                word-break: keep-all;
            }

            .category-subtitle {
                font-size: 0.78rem;
                opacity: 0.92;
                font-weight: 500;
            }

            .study-category {
                font-size: 1.35rem;
                font-weight: 700;
                color: var(--text);
                margin: 0;
            }

            .progress-wrap {
                background: rgba(17, 24, 39, 0.78);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 18px;
                padding: 0.85rem 1rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
                backdrop-filter: blur(10px);
            }

            .progress-label {
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                color: var(--muted);
                margin-bottom: 0.45rem;
            }

            .progress-bar {
                height: 8px;
                background: rgba(255, 255, 255, 0.12);
                border-radius: 999px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #6366f1, #8b5cf6);
                border-radius: 999px;
            }

            .pager-row {
                display: flex;
                justify-content: center;
                gap: 0.5rem;
                margin-top: 0.75rem;
                color: #cbd5e1;
                font-size: 0.85rem;
            }

            .calendar-grid {
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 0.35rem;
                margin: 0.75rem 0 1rem;
            }

            .cal-head, .cal-cell, .cal-empty {
                text-align: center;
                border-radius: 10px;
                min-height: 42px;
                font-size: 0.75rem;
            }

            .cal-head { color: #94a3b8; font-weight: 600; padding-top: 0.2rem; }
            .cal-empty { background: transparent; }

            .cal-cell {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.08);
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 0.1rem;
            }

            .cal-cell small { color: #cbd5e1; font-size: 0.65rem; }
            .cal-cell.level-1 { background: rgba(52, 211, 153, 0.25); }
            .cal-cell.level-2 { background: rgba(52, 211, 153, 0.4); }
            .cal-cell.level-3 { background: rgba(52, 211, 153, 0.58); }
            .cal-cell.level-4 { background: rgba(16, 185, 129, 0.78); color: #052e1a; }

            .quiz-prompt {
                background: rgba(17, 24, 39, 0.78);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 22px;
                padding: 1.4rem 1.1rem;
                margin: 0.75rem 0 1rem;
                text-align: center;
            }

            .quiz-prompt .label {
                color: #818cf8;
                font-size: 0.8rem;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }

            .quiz-prompt .meaning {
                font-size: 1.25rem;
                line-height: 1.55;
                color: #f9fafb;
                word-break: keep-all;
            }

            div[data-testid="stHorizontalBlock"] button,
            div[data-testid="column"] button {
                min-height: 3rem;
                border-radius: 16px !important;
                font-weight: 600 !important;
            }

            div[data-testid="stButton"] button[kind="primary"] {
                background: linear-gradient(135deg, #10b981, #059669) !important;
                border: none !important;
            }

            div[data-testid="stTextInput"] input,
            div[data-testid="stSelectbox"] > div > div {
                background: rgba(17, 24, 39, 0.78) !important;
                color: #f9fafb !important;
                border: 1px solid rgba(255, 255, 255, 0.14) !important;
                border-radius: 14px !important;
            }

            div[data-testid="stAlert"] {
                background: rgba(17, 24, 39, 0.85) !important;
                color: #e5e7eb !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
            }

            @media (max-width: 640px) {
                .block-container { padding-left: 0.85rem; padding-right: 0.85rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def open_category(name: str) -> None:
    st.session_state.page = "study"
    st.session_state.selected_category = name
    reset_card_order(name, st.session_state.vocabulary)
    st.rerun()


def reset_category_page() -> None:
    st.session_state.category_page = 0


def render_category_tile(name: str, count: int, button_key: str) -> bool:
    start, end = category_palette(name)
    marker = hashlib.md5(button_key.encode()).hexdigest()[:12]
    st.markdown(
        f"""
        <div id="cat-{marker}" class="cat-style-marker"></div>
        <style>
        div[data-testid="stVerticalBlock"] > div:has(#cat-{marker}) + div [data-testid="stButton"] button {{
            background: linear-gradient(135deg, {start}, {end}) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    label = category_display_name(name)
    return st.button(
        f"{label}\n{count} words",
        key=button_key,
        use_container_width=True,
        help=f"Study {count} words",
    )


def render_home_top_bar(meta: dict, progress: dict) -> None:
    _, test_col, points_col = st.columns([2.2, 1, 1])
    with test_col:
        if st.button("📝 Test", use_container_width=True):
            st.session_state.page = "test"
            st.session_state.quiz_question = None
            st.session_state.quiz_feedback = None
            st.rerun()
    with points_col:
        points = total_points(progress)
        if st.button(f"🏆 {points:,} pts", use_container_width=True):
            st.session_state.page = "points_stats"
            st.rerun()

    st.markdown(
        f"""
        <div class="app-header">
            <h1>{APP_TITLE}</h1>
            <p>의미별 50개 주제 — 카드를 눌러 학습을 시작하세요</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    learned = learned_count(progress)
    total_words = meta["total_words"]
    stat_a, stat_b, stat_c = st.columns([1.2, 1.2, 1.3])
    with stat_a:
        st.markdown(
            f'<div class="stats-row" style="margin:0;"><span class="stat-pill">{total_words:,} words</span></div>',
            unsafe_allow_html=True,
        )
    with stat_b:
        st.markdown(
            f'<div class="stats-row" style="margin:0;"><span class="stat-pill">{meta["total_categories"]:,} categories</span></div>',
            unsafe_allow_html=True,
        )
    with stat_c:
        if st.button(f"✅ Learned {learned}/{total_words}", use_container_width=True):
            st.session_state.page = "learned_stats"
            st.rerun()


def render_category_cards(categories: dict[str, int]) -> None:
    query = st.session_state.category_query.strip().lower()
    filtered = [
        (name, count)
        for name, count in categories.items()
        if not query or query in name.lower()
    ]

    if not filtered:
        st.info("No categories match your search.")
        return

    page_size = 48
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if st.session_state.category_page >= total_pages:
        st.session_state.category_page = 0

    page = st.session_state.category_page
    page_items = filtered[page * page_size : (page + 1) * page_size]

    for row_start in range(0, len(page_items), 2):
        row_items = page_items[row_start : row_start + 2]
        columns = st.columns(2)
        for column, (name, count) in zip(columns, row_items):
            with column:
                button_key = f"cat_{page}_{row_start}_{name}"
                if render_category_tile(name, count, button_key):
                    open_category(name)

    st.markdown(
        f'<div class="pager-row">Page {page + 1} of {total_pages} · '
        f"{len(filtered)} categories</div>",
        unsafe_allow_html=True,
    )
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("← Prev", use_container_width=True, disabled=page == 0):
            st.session_state.category_page -= 1
            st.rerun()
    with next_col:
        if st.button("Next →", use_container_width=True, disabled=page >= total_pages - 1):
            st.session_state.category_page += 1
            st.rerun()


def render_home(meta: dict, categories: dict[str, int], progress: dict) -> None:
    render_home_top_bar(meta, progress)
    st.text_input(
        "Search categories",
        placeholder="예: 칭찬, 분노, 행복...",
        key="category_query",
        label_visibility="collapsed",
        on_change=reset_category_page,
    )
    render_category_cards(categories)


def render_flashcard(word: dict) -> None:
    status = st.session_state.learned_status.get(status_key(word))
    status_html = ""
    if status == "learned":
        status_html = '<span class="status-badge status-learned">✅ Learned</span>'
    elif status == "review":
        status_html = '<span class="status-badge status-review">🔄 Review Again</span>'

    safe = {
        key: html.escape(str(word.get(key, "") or ""))
        for key in ("word", "meaning", "part_of_speech", "example")
    }
    pos = safe["part_of_speech"] or "—"
    example = safe["example"] or "—"

    card_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
            body {{ margin: 0; font-family: Inter, sans-serif; background: transparent; }}
            .scene {{ perspective: 1200px; width: 100%; max-width: 520px; margin: 0 auto; }}
            .card {{
                position: relative; width: 100%; min-height: 320px;
                transform-style: preserve-3d;
                transition: transform 0.65s cubic-bezier(0.4, 0.2, 0.2, 1);
                cursor: pointer;
            }}
            .card.is-flipped {{ transform: rotateY(180deg); }}
            .face {{
                position: absolute; inset: 0; backface-visibility: hidden;
                border-radius: 28px; background: #ffffff;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.14);
                padding: 1.5rem 1.25rem; box-sizing: border-box;
            }}
            .front {{ display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }}
            .back {{ transform: rotateY(180deg); text-align: left; }}
            .word {{ font-size: clamp(2rem, 8vw, 2.8rem); font-weight: 700; color: #111827; margin: 0; }}
            .hint {{ margin-top: 1rem; font-size: 0.82rem; color: #9ca3af; }}
            .row {{ margin-bottom: 0.9rem; }}
            .label {{ font-size: 0.78rem; font-weight: 700; color: #6366f1; margin-bottom: 0.2rem; }}
            .text {{ font-size: 1rem; color: #111827; line-height: 1.55; word-break: keep-all; }}
        </style>
    </head>
    <body>
        <div class="scene">
            <div class="card" onclick="this.classList.toggle('is-flipped')">
                <div class="face front">
                    <p class="word">{safe['word']}</p>
                    <p class="hint">Tap card to flip</p>
                    {status_html}
                </div>
                <div class="face back">
                    <p class="word" style="font-size:1.8rem;margin-bottom:1rem;">{safe['word']}</p>
                    <div class="row"><div class="label">뜻</div><div class="text">{safe['meaning']}</div></div>
                    <div class="row"><div class="label">품사</div><div class="text">{pos}</div></div>
                    <div class="row"><div class="label">예문</div><div class="text">{example}</div></div>
                    {status_html}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    components.html(card_html, height=360, scrolling=False)


def render_study(vocabulary: list[dict]) -> None:
    category = st.session_state.selected_category
    words = words_for_category(vocabulary, category)
    total = len(words)

    if total == 0:
        st.warning("No words found in this category.")
        if st.button("← Back to categories", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        return

    if not st.session_state.card_order:
        reset_card_order(category, vocabulary)

    current_index = st.session_state.card_index
    display_number = current_index + 1
    progress_pct = (display_number / total) * 100
    word = current_word(vocabulary)

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(f'<p class="study-category">{category}</p>', unsafe_allow_html=True)
    with top_right:
        if st.button("Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    st.markdown(
        f"""
        <div class="progress-wrap">
            <div class="progress-label">
                <span>Card {display_number} / {total}</span>
                <span>{progress_pct:.0f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct:.2f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if word:
        render_flashcard(word)

    control_a, control_b, control_c = st.columns([1, 1, 1])
    with control_a:
        if st.button("←", use_container_width=True, disabled=current_index == 0):
            st.session_state.card_index -= 1
            st.session_state.flipped = False
            st.rerun()
    with control_b:
        random_label = "Random ✓" if st.session_state.random_mode else "Random"
        if st.button(random_label, use_container_width=True):
            st.session_state.random_mode = not st.session_state.random_mode
            reset_card_order(category, vocabulary)
            st.rerun()
    with control_c:
        if st.button("→", use_container_width=True, disabled=current_index >= total - 1):
            st.session_state.card_index += 1
            st.session_state.flipped = False
            st.rerun()

    learn_col, review_col = st.columns(2)
    with learn_col:
        if st.button("✅ Learned", use_container_width=True, type="primary"):
            mark_learned(word)
            if current_index < total - 1:
                st.session_state.card_index += 1
                st.session_state.flipped = False
            st.rerun()
    with review_col:
        if st.button("🔄 Review Again", use_container_width=True):
            st.session_state.learned_status[status_key(word)] = "review"
            if current_index < total - 1:
                st.session_state.card_index += 1
                st.session_state.flipped = False
            st.rerun()


def render_test(vocabulary: list[dict], categories: dict[str, int], progress: dict) -> None:
    st.markdown("### 📝 Vocabulary Test")

    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("← Home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.quiz_question = None
            st.session_state.quiz_feedback = None
            st.rerun()
    with top_right:
        st.markdown(
            f"<div style='text-align:right;padding-top:0.5rem;color:#fbbf24;font-weight:700;'>"
            f"🏆 {total_points(progress):,} pts</div>",
            unsafe_allow_html=True,
        )

    category_options = ["Random", *categories.keys()]
    st.selectbox("Category", category_options, key="quiz_category")
    st.radio("Word pool", ["All words", "Learned only"], key="quiz_source", horizontal=True)

    pool = quiz_pool(vocabulary, progress)
    st.caption(f"{len(pool)} words available for this test setup.")

    if len(pool) < 4:
        st.warning("Need at least 4 words for a multiple-choice test with this setup.")
        return

    if st.button("Start / Next Question", use_container_width=True, type="primary"):
        st.session_state.quiz_question = build_quiz_question(pool, vocabulary)
        st.session_state.quiz_feedback = None
        st.rerun()

    question = st.session_state.quiz_question
    if not question:
        st.info("Choose options above, then start the test.")
        return

    pos = html.escape(question.get("part_of_speech") or "")
    meaning = html.escape(question["prompt"])
    st.markdown(
        f"""
        <div class="quiz-prompt">
            <div class="label">Choose the correct English word</div>
            <div class="meaning">뜻: {meaning}</div>
            {f'<div style="margin-top:0.5rem;color:#94a3b8;font-size:0.85rem;">품사: {pos}</div>' if pos else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.quiz_feedback:
        st.markdown(st.session_state.quiz_feedback)

    option_cols = st.columns(2)
    for index, option in enumerate(question["options"]):
        col = option_cols[index % 2]
        with col:
            if st.button(option["word"], key=f"quiz_opt_{option['word']}_{index}", use_container_width=True):
                if option["word"] == question["correct"]["word"]:
                    add_quiz_point(question["correct"])
                    st.session_state.quiz_feedback = (
                        f"✅ Correct! **{option['word']}** (+1 point)"
                    )
                else:
                    st.session_state.quiz_feedback = (
                        f"❌ Wrong. The answer was **{question['correct']['word']}**."
                    )
                st.session_state.quiz_question = build_quiz_question(pool, vocabulary)
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    init_session_state()
    inject_global_css()

    data = load_vocabulary(VOCAB_PATH.stat().st_mtime)
    st.session_state.vocabulary = data["vocabulary"]
    meta = data["meta"]
    categories = meta["categories"]
    progress = st.session_state.progress
    page = st.session_state.page

    if page == "home":
        render_home(meta, categories, progress)
    elif page == "study":
        render_study(st.session_state.vocabulary)
    elif page == "test":
        render_test(st.session_state.vocabulary, categories, progress)
    elif page == "learned_stats":
        render_stats_detail_page(
            title="Learned Progress",
            log=progress.get("learned_log", []),
            value_label="count",
            aggregate_fn=lambda log: daily_counts(log, "count"),
        )
    elif page == "points_stats":
        render_stats_detail_page(
            title="Points History",
            log=progress.get("points_log", []),
            value_label="points",
            aggregate_fn=daily_points,
        )
    else:
        st.session_state.page = "home"
        st.rerun()


if __name__ == "__main__":
    main()
