# llm.py # summarize comments 
import sqlite3
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ycis.config import Config
config = Config()


def get_comments_from_db(
    db_path,
    table_name: str,
    sentiment_label: int,
    topic: int,
):
    query = f"""
        SELECT text
        FROM {table_name}
        WHERE sentiment_label = ?
          AND topic = ?
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(sentiment_label, topic),
        )

    return df["text"].dropna().astype(str).tolist()


def summarize_topic_comments(
    comments: list[str]):
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        temperature=0,
    )

    comments_text = "\n".join(
        f"- {comment}" for comment in comments
    )

    prompt = ChatPromptTemplate.from_messages([
            (
                "system",
            """
            You are analyzing a collection of YouTube comments.

            Read ALL comments carefully and independently identify
            the main topic or topics discussed.

            Do not assume that the comments belong to any predefined
            topic or sentiment category.

            Provide:

            1. Main topic:
            Describe the central topic of the discussion in one
            concise sentence.

            2. Key themes:
            Identify the main recurring themes, issues, or subjects.

            3. Summary:
            Give a concise summary of what the commenters are
            discussing and saying overall.

            4. Secondary themes:
            Mention important but less frequent themes if they exist.

            Base your analysis ONLY on the comments provided.
            Do not invent information that is not supported by the comments.
            """
            ),
            (
                "human",
                "{comments}"
            ),
        ])

    chain = prompt | llm

    response = chain.invoke({
        "comments": comments_text
    })

    return response.content