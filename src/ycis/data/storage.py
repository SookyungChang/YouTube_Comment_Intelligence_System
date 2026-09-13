# data/storage.py # df -> db file

import sqlite3
import json
import pandas as pd
from datetime import datetime, timezone

labels = {0: "negative", 1: "positive"}


def save_to_db(
    video_id: str, 
    df: pd.DataFrame, 
    # topic_info: pd.DataFrame, 
    # sentiment_label: int
):
    """Save full prediction DataFrame to SQLite, skipping duplicate commentIds."""
    df = df.copy()
    df["video_id"] = video_id
    # df["sentiment_label"] = sentiment_label
    df["analyzed_at"] = datetime.now(timezone.utc).isoformat()

    # Serialize list columns in topic_info for SQLite compatibility
    topic_info = topic_info.copy()
    if "Representation" in topic_info.columns:
        topic_info["Representation"] = topic_info["Representation"].apply(json.dumps)
    if "Representative_Docs" in topic_info.columns:
        topic_info["Representative_Docs"] = topic_info["Representative_Docs"].apply(
            json.dumps
        )

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(
            f"comment_results_{labels[sentiment_label]}",
            con=conn,
            if_exists="replace",
            index=False,
        )

        topic_info.to_sql(
            f"topic_info_{labels[sentiment_label]}",
            con=conn,
            if_exists="replace",
            index=False,
        )

    print(f"✅ Saved {len(df)} comments for video {video_id}")
