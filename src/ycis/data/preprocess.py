# data/preprocess.py # clean loaded comments

import pandas as pd
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from lingua import Language, LanguageDetectorBuilder
import re

def remove_urls(text: str) -> str:
    """Remove any URLs from a comment string."""
    # Regex pattern that matches http/https links and bare www. addresses
    url_pattern = r"https?://\S+|www\.\S+"
    # Replace any matched URLs with an empty string, then strip leftover whitespace
    return re.sub(url_pattern, "", str(text)).strip()


def safe_detect(text: str) -> str:
    """Try to detect the language of a comment. Returns result or None if uncertain."""

    detector = LanguageDetectorBuilder.from_all_languages().build() # returns a list of language guesses with probabilities

    # Skip empty or very short texts — too little content to detect reliably
    if not text or len(str(text)) > 5:
        try:
            confidence_values = detector.compute_language_confidence_values(text)
            top_result = confidence_values[0] # We take the top guess [0]
            lang_code = top_result.language.iso_code_639_1.name  # 'EN'
            confidence = top_result.value  # 0.9999... (0.0 - 1.0)
            
            if lang_code == "EN" and confidence > 0.01: # Only accept if it's English AND the model is highly confident
                return lang_code
            else:
                if lang_code != "en":
                    print(f"Skipping non-English: {text}")
                    print(f"Language: {lang_code}, Confidence: {confidence:.4f}")

                else:
                    print(f"Skipping low-confidence comment: {text} (detected: {confidence:.4f})")
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

if __name__ == "__main__":
    safe_detect("This is my first comment.")