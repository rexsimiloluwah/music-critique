"Utilities for working with LangChain and LLM-related stuff"
from typing import Optional

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain, LLMChain


content_criticism_prompt = ChatPromptTemplate.from_template(
    """Analyze the lyrics of the song with lyrics below and provide a constructive critical
    evaluation of the following aspects:\n
    1. Lyricism and Creativity
    2. Theme and Message
    3. Tone and Moode
    4. Imagery and Metaphors
    5. Social and Cultural Implications
    6. Originality and Uniqueness
    7. Linguistic and Grammatical Quality
    8. Artistic Integrity
    9. Impact and Reception
    \n\n

    Lyrics: `{lyrics_content}`\n

    The output should be in markdown format with the analysis points well enumerated.
    """
)

recommendations_for_improvement_prompt = ChatPromptTemplate.from_template(
    """Based on your critical analysis of the song below, provide suggestions for how the 
    content of the song could be improved or refined\n\n

    Critical Analysis: `{critical_analysis}`\n

    The output should be in markdown format with the recommendations well enumerated.
    """
)


@st.cache_data
def music_critique(
    lyrics_content: str, openai_api_key: str, temperature: Optional[float] = 0.35
):
    """Generates a critical analysis and recommendations based on the content.

    Uses a `SequentialChain`.

    Args:
        lyrics_content (str): The content of the lyrics.
        openai_api_key (str): The OpenAI API key.
        temperature (Optional[float]): The temperature value between 0 and 1.
    """
    llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)

    content_criticism_chain = LLMChain(
        llm=llm, prompt=content_criticism_prompt, output_key="critical_analysis"
    )

    recommendations_chain = LLMChain(
        llm=llm,
        prompt=recommendations_for_improvement_prompt,
        output_key="recommendations",
    )

    music_critique_chain = SequentialChain(
        chains=[content_criticism_chain, recommendations_chain],
        input_variables=["lyrics_content"],
        output_variables=["critical_analysis", "recommendations"],
        verbose=True,
    )

    result = music_critique_chain(lyrics_content)
    return result


@st.cache_data
def music_theme_generator(
    lyrics_content: str, openai_api_key: str, temperature: Optional[float] = 0.35
):
    """Generate a theme for the music based on the lyrics.

    Args:
        lyrics_content (str): The content of the lyrics.
        openai_api_key (str): The OpenAI API key.
        temperature (Optional[float]): The temperature value between 0 and 1.
    """
    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
    theme_generator_prompt_template = ChatPromptTemplate.from_template(
        """Based on the content of the song lyrics below, generate a theme that represents the context of the song.
        This theme will be passed as a prompt to an image generator to generate a cover image. Hence, keep the theme brief
        with not more than 12 words.\n\n

        Here is an example output:\n
        Output: A song on life and depression

        Lyrics content: `{lyrics_content}`
        Output: 
        """
    )

    theme_generator_prompt = theme_generator_prompt_template.format_messages(
        lyrics_content=lyrics_content
    )
    response = chat(theme_generator_prompt)
    return response.content
