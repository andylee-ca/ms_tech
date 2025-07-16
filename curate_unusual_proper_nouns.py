"""
Functions for curating unusual proper nouns from text.
"""

import nltk
import pandas as pd
from utils import strip_html_tags


def find_proper_nouns(text: str, debug: bool = False) -> list[str]:
    """
    Extract all proper nouns (NNP, NNPS) from the input text.

    Args:
        text (str): The input string to analyze.

    Returns:
        list[str]: A list of proper nouns found in the text.
    """
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    if debug:
        print(f"Input text: {text}")
        for word, tag in tagged:
            print(f"Word: {word}, Tag: {tag}")

    return [word for word, tag in tagged if tag in ("NNP", "NNPS")]


def generate_word_frequency(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Generate a DataFrame containing the frequency of each word
    in the specified column of the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the questions.
        column (str): The column name to analyze for word frequency.

    Returns:
        pd.DataFrame: A DataFrame with 'word' and 'frequency'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in DataFrame")

    # Concatenate all questions into a single string
    all_text = " ".join(df[column].astype(str).tolist())

    # Convert to lowercase and remove HTML tags
    all_text = all_text.lower()
    all_text = strip_html_tags(all_text)

    # Tokenize the text into words
    words = nltk.word_tokenize(all_text)
    # Create a frequency distribution of the words
    freq_dist = nltk.FreqDist(words)

    # Convert the frequency distribution to a DataFrame
    freq_df = pd.DataFrame(freq_dist.items(), columns=["word", "frequency"])

    # Sort the DataFrame by frequency in descending order
    freq_df = freq_df.sort_values(by="frequency", ascending=False).reset_index(
        drop=True
    )

    return freq_df
