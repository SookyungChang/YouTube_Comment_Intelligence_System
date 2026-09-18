# app.py — Gradio version for Azure

import os
import gradio as gr
import pandas as pd

from ycis.config import Config
from ycis.data.fetchData import get_comments
from ycis.data.preprocess import filter_english_comments
from ycis.data.storage import save_to_db_with_embedding
from ycis.topic_modeling import update_topics_in_db
from ycis.inference import Predictor
from ycis.llm import get_comments_from_db, summarize_topic_comments
config = Config()

labels = {0: "negative", 1: "positive"}

# ─────────────────────────────────────────
# Model loading (once at startup)
# ─────────────────────────────────────────

print("⏳ Loading BERT model...", flush=True)

print("before BERT", flush=True)
bert_path = config.TRAINED_MODEL_PATH
predictor = Predictor(bert_path)
print("after BERT", flush=True)

DB_NAME = "comments_cache.db"
DB_PATH = config.DB_DIR / DB_NAME

print("✅ BERT model loaded!", flush=True)


# ─────────────────────────────────────────
# Shared state (replaces st.session_state)
# ─────────────────────────────────────────

# Holds the last processed results so Tab 2 can access them.
# _session: dict = {}


# ─────────────────────────────────────────
# Helper — URL → video ID
# ─────────────────────────────────────────


def extract_video_id(raw: str) -> str:
    raw = raw.strip()
    if "youtube.com" in raw:
        from urllib.parse import urlparse, parse_qs

        params = parse_qs(urlparse(raw).query)
        return params.get("v", [raw])[0]
    if "youtu.be" in raw:
        return raw.split("youtu.be/")[-1].split("?")[0]
    return raw


# ─────────────────────────────────────────
# Tab 1 — Analyze Video
# ─────────────────────────────────────────


def analyze_video(video_input: str, session_state: dict, progress=None):
    """
    Fetch comments → sentiment → save to db (with embeddings) → topic modeling.

    Returns:
        status_md    : str            — status / error message
        metrics_md   : str            — markdown table (total / pos / neg)
        chart_df     : pd.DataFrame   — for gr.BarPlot
        pos_topics   : pd.DataFrame   — positive topic table
        neg_topics   : pd.DataFrame   — negative topic table
        session_state: dict           — Individual storage
    """
    EMPTY = (
        "",
        "",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        session_state,
    )  ## Error return

    if not video_input or not video_input.strip():
        video_input = "https://www.youtube.com/watch?v=Od6M0AXpcxQ"
        video_id = extract_video_id(video_input)
    else:
        video_id = extract_video_id(video_input)

    # ── Fetch comments ──────────────────
    if progress is not None:
        progress(0.1, desc="⏳ 1/4 Fetching YouTube comments...")
    try:
        df = get_comments(video_id, max_results=100, max_pages=15)
        df = filter_english_comments(df)
    except Exception as e:
        return (f"❌ Failed to fetch comments: {e}",) + EMPTY[1:]

    if df.empty:
        return ("⚠️ No English comments found for this video.",) + EMPTY[1:]

    # ── Sentiment ───────────────────────
    if progress is not None:
        progress(0.4, desc="🧠 2/4 Running BERT sentiment analysis...")
    try:
        # predict_df mutates `df` in place, adding "bert_preds" (0/1 label)
        # plus "scores_0"/"scores_1" columns, and also returns the same object.
        df_sen = predictor.predict_df(df)
    except Exception as e:
        return (f"❌ Sentiment prediction failed: {e}",) + EMPTY[1:]

    # ── Save to DB (with embeddings for RAG) ─────
    if progress is not None:
        progress(0.6, desc="💾 3/4 Saving comments + embeddings to DB...")
    try:
        save_to_db_with_embedding(video_id, df, DB_NAME)
    except Exception as e:
        return (f"❌ Failed to save comments to DB: {e}",) + EMPTY[1:]

    # ── Topic modeling ──────────────────
    if progress is not None:
        progress(0.85, desc="🔮 4/4 Topic modeling...")
    topic_summary: dict = {}
    try:
        for label in [0, 1]:
            label_name = "✅ Positive" if label == 1 else "❌ Negative"
            topic_info, _ = update_topics_in_db(
                DB_PATH, video_id, sentiment_label=label
            )
            if topic_info is not None and not topic_info.empty and "Topic" in topic_info.columns:
                topic_info = topic_info[topic_info["Topic"] != -1]
            topic_summary[label_name] = topic_info
    except Exception as e:
        return (f"❌ Topic modeling failed: {e}",) + EMPTY[1:]

    if progress is not None:
        progress(1.0, desc="✅ Analysis completed.")

    # ── Persist for RAG tab ─────────────
    session_state["video_id"] = video_id
    session_state["topic_summary"] = topic_summary

    # ── Build outputs ───────────────────
    counts = df_sen["bert_preds"].value_counts()
    total = len(df_sen)
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))

    metrics_md = (
        f"| 📊 Total | ❌ Negative | ✅ Positive |\n"
        f"|---------|------------|------------|\n"
        f"| **{total}** | **{neg}** | **{pos}** |"
    )

    chart_df = pd.DataFrame(
        {
            "Sentiment": ["Negative", "Positive"],
            "Count": [neg, pos],
        }
    )

    pos_topics = topic_summary.get("✅ Positive", pd.DataFrame())
    neg_topics = topic_summary.get("❌ Negative", pd.DataFrame())

    def trim_cols(tdf):
        if tdf is not None and not tdf.empty:
            cols = [c for c in ["Topic", "Count", "Name"] if c in tdf.columns]
            return tdf[cols].reset_index(drop=True)
        return pd.DataFrame()

    status = f"✅ Processed **{total}** comments! Head over to the **💬 Summarize Topic** tab to explore them."

    return (
        status,
        metrics_md,
        chart_df,
        trim_cols(pos_topics),
        trim_cols(neg_topics),
        session_state,
    )


# ─────────────────────────────────────────
# Topic summarization (RAG)
# ─────────────────────────────────────────


def summarize_topic(video_id_rag: str, sentiment_val, topic_id: int):
    """Pull comments for a given sentiment+topic bucket from the DB and summarize them."""
    if not video_id_rag or not video_id_rag.strip():
        return "⚠️ Please analyze a video first, or enter a video ID.", ""

    try:
        comments = get_comments_from_db(
            db_path=DB_PATH,
            table_name=video_id_rag.strip(),
            sentiment_label=int(sentiment_val),
            topic=int(topic_id),
        )

        if not comments:
            return "⚠️ No comments found for that sentiment/topic combination.", ""

        result = summarize_topic_comments(comments)

        context_parts = [f"**Comment {i}:**\n\n{c}\n\n---" for i, c in enumerate(comments, 1)]
        context_md = "\n".join(context_parts)

        return result, context_md

    except Exception as e:
        return f"❌ Error: {e}", ""


# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="emerald",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
)

with gr.Blocks(theme=theme, title="🎬 YouTube Comment Analyzer") as demo:
    session_state = gr.State(value={})
    gr.Markdown(
        """
        # 🎬 YouTube Comment Sentiment & Topic Analyzer
        Analyze YouTube comments with **BERT** sentiment analysis and discover **topics** —
        then click any topic to get an AI summary of those comments.
        """
    )

    gr.Markdown("### Enter a YouTube video to process its comments.")

    with gr.Row():
        video_input = gr.Textbox(
            label="YouTube Video ID or URL",
            placeholder="e.g. XqYTfpxFuDM or https://www.youtube.com/watch?v=XqYTfpxFuDM",
            scale=5,
        )
        analyze_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)

    status_out = gr.Markdown()
    metrics_out = gr.Markdown()

    chart_out = gr.BarPlot(
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_map={"Negative": "#e74c3c", "Positive": "#2ecc71"},
        title="Sentiment Distribution",
        height=300,
        visible=False,
        y_lim=[0, None],
    )

    gr.Markdown("#### 🔍 Topics Found — click a row to summarize that topic")
    with gr.Row():
        with gr.Column():
            gr.Markdown("##### ✅ Positive Topics")
            pos_topics_out = gr.DataFrame(interactive=False)
        with gr.Column():
            gr.Markdown("##### ❌ Negative Topics")
            neg_topics_out = gr.DataFrame(interactive=False)

    gr.Markdown("#### 🧠 Topic Summary")
    selected_topic_out = gr.Markdown()
    summary_answer_out = gr.Markdown()
    with gr.Accordion("📄 Comments Used", open=False):
        summary_context_out = gr.Markdown()

    def run_analysis(video_input, current_session, progress=gr.Progress()):
        status, metrics, chart_df, pos_t, neg_t, updated_session = (
            analyze_video(video_input, current_session, progress)
        )
        show_chart = not chart_df.empty
        return (
            status,
            metrics,
            gr.BarPlot(value=chart_df, visible=show_chart, y_lim=[0, None]),
            pos_t,
            neg_t,
            updated_session,
        )

    analyze_btn.click(
        fn=run_analysis,
        inputs=[video_input, session_state],
        show_progress="full",
        outputs=[
            status_out,
            metrics_out,
            chart_out,
            pos_topics_out,
            neg_topics_out,
            session_state,
        ],
    )

    # ── Click a topic row → auto-summarize that topic ──────
    def make_topic_click_handler(sentiment_value: int):
        def handler(current_session, table_df, evt: gr.SelectData):
            row_idx = evt.index[0]
            if table_df is None or table_df.empty or row_idx >= len(table_df):
                return "", "⚠️ Could not identify the selected topic.", ""

            topic_id = int(table_df.iloc[row_idx]["Topic"])
            video_id = current_session.get("video_id", "")
            if not video_id:
                return "", "⚠️ Please analyze a video first.", ""

            label = "✅ Positive" if sentiment_value == 1 else "❌ Negative"
            header = f"📌 {label} · Topic {topic_id}"

            summary, context_md = summarize_topic(
                video_id, sentiment_value, topic_id
            )
            return header, summary, context_md

        return handler

    pos_topics_out.select(
        fn=make_topic_click_handler(1),
        inputs=[session_state, pos_topics_out],
        outputs=[selected_topic_out, summary_answer_out, summary_context_out],
        show_progress="full",
    )
    neg_topics_out.select(
        fn=make_topic_click_handler(0),
        inputs=[session_state, neg_topics_out],
        outputs=[selected_topic_out, summary_answer_out, summary_context_out],
        show_progress="full",
    )

    gr.Markdown(
        """
        ---
        <div style='text-align:center; color:#aaa; font-size:13px;'>
        Built with 🤗 Gradio &nbsp;·&nbsp; BERT Sentiment &nbsp;·&nbsp; Topic Modeling &nbsp;·&nbsp; RAG
        </div>
        """
    )


print("🔥 BEFORE LAUNCH", flush=True)

port = int(os.environ.get("PORT", 7860))

print("🔥 UI CREATED", flush=True)

demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    show_error=True,
)

if __name__ == "__main__":
    pass