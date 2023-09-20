"""Utility functions for working with wordclouds"""
from typing import Dict, Optional

from wordcloud import WordCloud


def generate_word_cloud(
    data: Dict[str, int],
    width: Optional[int] = 700,
    height: Optional[int] = 400,
    max_words: Optional[int] = 100,
    max_font_size: Optional[int] = 25,
    bg_color: Optional[str] = "white",
    repeat: Optional[bool] = False,
):
    """Generate word cloud.

    Args:
        data (dict[str, int]): A dictionary of words and their frequencies.
        width (Optional[int]): The width of the word cloud.
        height (Optional[int]): The height of the word cloud.
        repeat (Optional[int]): A flag which controls whether words should be repeated or not.
        max_words (Optional[int]): The max number of words in the word cloud.
        max_font_size (Optional[int]): The max font size.
        bg_color (Optional[str]): The background color.

    Returns:
        wordcloud: The generated wordcloud.
    """
    wordcloud = WordCloud(
        width=width,
        height=height,
        repeat=repeat,
        max_words=max_words,
        max_font_size=max_font_size,
        background_color=bg_color,
    ).generate_from_frequencies(data)

    return wordcloud
