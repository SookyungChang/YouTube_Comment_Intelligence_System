import sqlite3
import hdbscan
import numpy as np
import pandas as pd
from bertopic import BERTopic


def update_topics_in_db(
    db_path: str,
    table_name: str,
    sentiment_label: int,
    min_cluster_size: int = 10,
    hdbscan_model=None,
):
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT id, text, embedding FROM {table_name} WHERE sentiment_label = ?"
        df_sentiment = pd.read_sql_query(
            query, conn, params=(sentiment_label,)
        )

    if len(df_sentiment) < min_cluster_size:
        print(
            f"Not enough comments for topic modeling ({len(df_sentiment)} comments)."
        )
        return None, None

    # BLOB -> float32
    embeddings = np.array([
        np.frombuffer(emb, dtype=np.float32) for emb in df_sentiment["embedding"]
    ])
    texts = df_sentiment["text"].tolist()
    ids = df_sentiment["id"].tolist()

    # BERTopic 
    if hdbscan_model is None:
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            prediction_data=True,
        )

    topic_model = BERTopic(hdbscan_model=hdbscan_model)
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

    # Outlier
    if -1 in topics:
        new_topics = topic_model.reduce_outliers(
            texts, topics, strategy="c-tf-idf"
        )
        topic_model.update_topics(texts, topics=new_topics)
    else:
        new_topics = topics

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN topic INTEGER"
            )
        except sqlite3.OperationalError:
            pass

        # (topic_number, id) 
        update_data = list(zip([int(t) for t in new_topics], ids))

        cursor.executemany(
            f"""
            UPDATE {table_name}
            SET topic = ?
            WHERE id = ?
        """,
            update_data,
        )

    # topic_info: Topic ID, Count, Name, Representation...
    topic_info = topic_model.get_topic_info()

    print(
        f"update topic numbers to DB [{table_name}] in case sentiment_label={sentiment_label}."
    )
    return topic_info, topic_model
