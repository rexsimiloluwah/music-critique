"""Helper functions"""
import string
from typing import Dict

from constants import ENGLISH_STOPWORDS


def clean_text(text: str) -> str:
    """Clean a text input.

    The cleaning process involves removing stopwords, removing punctuation,
    and normalizing the text.

    Args:
        text (str): The raw text input.

    Returns:
        cleaned_text (str): The clean text.
    """
    words = text.split(" ")
    stop_words = set(ENGLISH_STOPWORDS)
    cleaned_words = [
        word.lower()
        for word in words
        if word.lower() not in stop_words and word not in string.punctuation
    ]

    cleaned_text = " ".join(cleaned_words)
    return cleaned_text


def count_word_freq(text: str) -> Dict[str, int]:
    """Count word frequencies in text.

    Args:
        text (str): The input text.

    Returns:
        word_freq (dict[str, int]): A dictionary containing each word and their respective frequencies.
    """
    words = text.split(" ")
    word_freq = {}
    for word in words:
        word = word.lower()
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

    return word_freq
