import pandas as pd
from datasets import Dataset, DatasetDict
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
import re

def remove_urls(text: str) -> str:
    """Remove any URLs from a comment string."""
    # Regex pattern that matches http/https links and bare www. addresses
    url_pattern = r"https?://\S+|www\.\S+"
    # Replace any matched URLs with an empty string, then strip leftover whitespace
    return re.sub(url_pattern, "", str(text)).strip()


def safe_detect(text: str) -> str:
    """Try to detect the language of a comment. Returns result or None if uncertain."""

    # Skip empty or very short texts — too little content to detect reliably
    if not text or len(str(text)) > 5:
        try:
                # detect_langs() returns a list of language guesses with probabilities
                # We take the top guess [0]
                res = detect_langs(text)[0]
        
                # Only accept if it's English AND the model is highly confident (>90%)
                if res.lang == "en" and res.prob > 0.9:
                    return res
                else:
                    if res.lang != "en":
                        print(f"Skipping non-English: {text})")
                    else:
                        print(f"Skipping low-confidence comment: {text} (detected: {res})")
                    return None  # Return None for non-English or low-confidence results
        
        except LangDetectException:
            # If detection fails entirely (e.g. unrecognizable characters), just skip it
            print("Language detection failed for text:", text)
            return None 

      
def filter_english_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a DataFrame down to high-confidence English comments only."""

    # Step 1: Remove URLs from all comments before processing
    df["text"] = df["text"].apply(remove_urls)

    # Step 2: Only run language detection on comments longer than 5 characters
    # (avoids wasting API calls on very short strings)
    mask = df["text"].str.len() > 5

    # Step 3: Apply safe_detect only to the rows that pass the length check
    df.loc[mask, "language"] = df.loc[mask, "text"].apply(safe_detect)

    # Step 4: Drop rows where language is null (non-English, low confidence, or too short)
    return df[df["language"].notnull()].reset_index(drop=True)