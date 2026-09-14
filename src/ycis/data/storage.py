# data/storage.py # df -> db file

import sqlite3
import numpy as np
import pandas as pd
from ycis.config import Config
config = Config()

def save_to_db_with_embedding(video_id: str, df: pd.DataFrame, db_name: str):
    db_path = config.DB_DIR / db_name
    table_name = video_id.replace("-", "_")

    # Embedding computation and binary conversion (BLOB)
    texts = df["text"].fillna("").astype(str).tolist()
    embeddings = config.SHARED_EMBEDDING_MODEL.embed_documents(texts)
    vec_bytes_list = [
        np.array(emb, dtype=np.float32).tobytes() for emb in embeddings
    ]

    save_df = pd.DataFrame()

    if "commentId" in df.columns:
        save_df["id"] = df["commentId"].astype(str)
    else:
        save_df["id"] = df.index.astype(str)

    save_df["author"] = df['author'].astype(str)
    save_df["text"] = texts
    save_df["like_count"] = df['likeCount'].astype(int)
    save_df["published_at"] = df['publishedAt'].astype(str)
    save_df["sentiment_label"] = df["bert_preds"].astype(int)
    save_df["embedding"] = vec_bytes_list

    with sqlite3.connect(db_path) as conn:
        save_df.to_sql(
            name=table_name, con=conn, if_exists="append", index=False
        )

        cursor = conn.cursor()

        cursor.execute(f"""
            DELETE FROM {table_name}
            WHERE rowid NOT IN (
                SELECT MAX(rowid)
                FROM {table_name}
                GROUP BY id
            )
        """)
    print(f"saved to db file: {db_name}")



