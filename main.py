import io
import os 
import json
import warnings
from typing import List, Dict
from collections import Counter

import pandas as pd 
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from st_clickable_images import clickable_images

from constants import *
from utils.aai import *
from utils.image_gen import generate_image
from utils.wordcloud import generate_word_cloud
from utils.helpers import clean_text, count_word_freq
from utils.langchain_utils import music_critique, music_theme_generator

warnings.filterwarnings("ignore")
st.set_page_config(page_title="🎶 Music Critique")

@st.cache_data
def save_file(file):
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    file_path = os.path.join(UPLOADS_DIR, file.name)
    file_ext = file.name.split(".")[-1]
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    
    return file_path, file_ext

@st.cache_data
def load_examples():
    examples = json.load(open("./examples.json"))
    return examples 

@st.cache_data 
def read_example_file(filename: str):
    EXAMPLES_DIR = "./examples"

    file_path = os.path.join(EXAMPLES_DIR, filename)

    with open(file_path, 'rb') as file:
        file_bytes = file.read()

    # Create a BytesIO object from the file bytes
    file_obj = io.BytesIO(file_bytes)
    return file_obj

def plot_word_frequency_analysis(word_freqs: Dict[str, int]) -> None:
    """Create the plot for visualizing the words and their respective frequencies."""
    st.subheader("Word Frequency Analysis")
    fig = px.bar(
        x=list(word_freqs.keys()),
        y=list(word_freqs.values()),
        labels={"x": "Word", "y": "Frequency"},
        title="Word Frequency Analysis"
    )
    st.plotly_chart(fig)

def plot_word_cloud(word_freqs: Dict[str, int]) -> None:
    """Create the plot for visualizing the word cloud."""
    st.subheader("Word Cloud")
    wordcloud = generate_word_cloud(word_freqs)
    wc_fig = plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(wc_fig)

def plot_top_five_words(words: List[str]) -> None:
    """Create the plot for visualizing the top five words."""
    st.subheader("Top Five Most Frequent Words")
    count_freq = Counter(words)
    top_five_words = count_freq.most_common(5)
    fig = px.bar(
        x=[word[0] for word in top_five_words],
        y=[word[1] for word in top_five_words],
        labels={"x": "Word", "y": "Frequency"},
        title="Top Five Most Frequent Words Analysis"
    )

    # Add labels to create an infographic look
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, title_text="Frequency")
    )

    # Customize bar colors and hover text
    fig.update_traces(marker_color="skyblue", hoverinfo="y")
    st.plotly_chart(fig)



def main():
    with st.sidebar:
        openai_api_key = st.text_input("Enter OpenAI API Key", key="openai_api_key", type="password")
        aai_api_key = st.text_input("Enter AssemblyAI API Key", key="aai_api_key", type="password")
        st.markdown("[Get an OpenAI API Key](https://platform.openai.com/account/api-keys)")
        st.markdown("[Get an AssemblyAI API Key](https://www.assemblyai.com/app/account)")

        st.divider()

        language = st.selectbox("Language", tuple(map(lambda x: x["name"], SUPPORTED_LANGUAGES)))
        st.markdown("**Select Features to Enable**")
        enable_content_moderation = st.checkbox("Content Moderation", value=True)
        enable_summarization = st.checkbox("Summarization", value=True)
        enable_topic_detection = st.checkbox("Topic Detection", value=True)
        enable_sentiment_analysis = st.checkbox("Sentiment Analysis", value=True)
        enable_entity_detection = st.checkbox("Entity Detection", value=True)
        enable_text_analytics = st.checkbox("Text Analytics", value=True)
        enable_criticism = st.checkbox("Content Criticism", value=True)
        enable_suggestions_for_improvement = st.checkbox("Suggestions for Improvement", value=True)
        enable_cover_image_generation = st.checkbox("Cover Image Generation", value=True)


    st.header("🎶 Music Critique")
    st.markdown("""
MusicCritique is a revolutionary app that unlocks the untapped potential
of audio transcription and LLMs to empower artists and managers with features
that aid data-driven understanding of their recorded songs, to make more informed, strategic
decisions before releasing their musical masterpieces to the world.
""")
    file = None
    use_example_file = st.checkbox("Use an example file")

    if use_example_file:
        examples = load_examples()
        thumbnails = list(map(lambda x: x["thumbnail"], examples))
        titles = list(map(lambda x: f"{x['artist']}-{x['title']} ({x['filename'].split('.')[-1]})", examples))

        selected_example_idx = clickable_images(
            thumbnails,
            titles=titles,
            div_style={
                "max-height": "400px",
                "display": "flex",
                "justify-content": "center",
                "flex-wrap": "wrap",
                "overflow-y": "auto",
            },
            img_style={"margin": "5px", "height": "150px"},
        )

        if selected_example_idx > -1:
            selected_example = examples[selected_example_idx]
            file = read_example_file(selected_example["filename"])
            st.info(f"Using Example file: {selected_example['filename']}", icon="ℹ")
    else:
        file = st.file_uploader(label="Please upload your audio file", type=[*SUPPORTED_AUDIO_FILE_TYPES, *SUPPORTED_VIDEO_FILE_TYPES])

    if file is not None:
        if use_example_file:
            file_path = os.path.join("./examples", selected_example["filename"])
            file_ext = file_path.split(".")[-1]
        else:
            file_path, file_ext = save_file(file)

        if file_ext in SUPPORTED_AUDIO_FILE_TYPES:
            st.audio(file_path)
        elif file_ext in SUPPORTED_VIDEO_FILE_TYPES:
            st.video(file_path)

        submit_btn = st.button("Submit   🚀")

        if submit_btn:
            if not openai_api_key or not aai_api_key:
                st.warning("Please enter your OpenAI and AssemblyAI API keys to get started.", icon='⚠')
            else:
                aai_settings = {
                    "content_moderation": enable_content_moderation,
                    "summarization": enable_summarization,
                    "topic_detection": enable_topic_detection,
                    "sentiment_analysis": enable_sentiment_analysis,
                    "entity_detection": enable_entity_detection
                }

                language_code = list(filter(lambda x: x["name"]==language, SUPPORTED_LANGUAGES))[0]["code"]
                
                # Upload file to AssemblyAI
                upload_url, error = upload_to_assemblyai(file_path, aai_api_key)

                if error:
                    st.error(error, icon="❌")
                else:
                    # Submit the transcription job
                    polling_endpoint, error = submit_transcription_job(upload_url, aai_api_key, language_code, aai_settings)
                    if error:
                        st.error(error, icon="❌")
                    else:
                        # Poll the endpoint for the transcription results
                        transcription_response, error = get_transcription_results(polling_endpoint, aai_api_key)
                        if error:
                            st.error(error, icon="❌")
                        else:
                            transcription_results = transcription_response.json()
                            with st.expander("Lyrics / Transcription"):
                                st.write(transcription_results["text"])
                            
                            if enable_summarization:
                                with st.expander("Summary"):
                                    st.write(transcription_results["summary"])

                            if enable_content_moderation:
                                with st.expander("Sensitive Content"):
                                    sensitive_topics = transcription_results["content_safety_labels"]["summary"]
                                    if sensitive_topics != {}:
                                        st.warning("Mention of the following sensitive topics detected: ", icon="⚠")
                                        sensitive_topics_df = pd.DataFrame(sensitive_topics.items())
                                        sensitive_topics_df.columns = ["Topic", "Confidence"]
                                        st.table(sensitive_topics_df)
                                    else:
                                        st.success("All clear! No sensitive content detected.", icon="✅")

                            if enable_text_analytics:
                                cleaned_text = clean_text(transcription_results["text"])
                                words = cleaned_text.split()
                                word_freqs = count_word_freq(cleaned_text)

                                with st.expander("Text Analytics"):
                                    plot_word_frequency_analysis(word_freqs)
                                    plot_word_cloud(word_freqs)
                                    plot_top_five_words(words)

                            if enable_topic_detection:
                                with st.expander("Detected Topics"):
                                    st.info("The following topics were detected with their respective relevance scores: ", icon="ℹ")
                                    topics = transcription_results["iab_categories_result"]["summary"]
                                    topics_df = pd.DataFrame(topics.items())
                                    topics_df.columns = ["Topic", "Relevance"]
                                    topics_df["Topic"] = topics_df["Topic"].str.split(">")
                                    topics_df["Relevance"] = np.round(topics_df["Relevance"], 5)
                                    expanded_topics = topics_df["Topic"].apply(pd.Series).add_prefix("Topic_Level_")
                                    topics_df = (
                                        topics_df.join(expanded_topics)
                                        .drop("Topic", axis=1)
                                        .sort_values(["Relevance"], ascending=False)
                                        .fillna("")
                                    )

                                    st.table(topics_df)

                            if enable_sentiment_analysis:
                                with st.expander("Sentiment Analysis Results"):
                                    st.info("Here are the detected sentiments for parts of your content with their respective confidence scores: ", icon="ℹ")
                                    sentiment_analysis_results = transcription_results["sentiment_analysis_results"]
                                    sentiment_analysis_results = [
                                        {"text": result["text"], "sentiment": result["sentiment"], "confidence": round(result["confidence"], 3)}
                                        for result in sentiment_analysis_results
                                    ]
                                    sentiment_analysis_df = pd.DataFrame(sentiment_analysis_results)
                                    sentiment_analysis_df.columns = ["Text", "Sentiment", "Confidence"]
                                    st.table(sentiment_analysis_df)

                            if enable_entity_detection:
                                with st.expander("Entity Recognition Results"):
                                    unique_entities = {}
                                    for entity in transcription_results["entities"]:
                                        if entity["text"].lower().strip() not in unique_entities:
                                            unique_entities[entity["text"].lower().strip()] = entity["entity_type"]
                                    
                                    entities = [
                                        {"text": name, "entity_type": entity_type}
                                        for name, entity_type in unique_entities.items()
                                    ]
                                    if len(entities)==0:
                                        st.info("Oops! No entities detected.")
                                    else:
                                        st.info("Here are the detected entities: ", icon="ℹ")
                                        entities_df = pd.DataFrame(entities)
                                        entities_df.columns = ["Text", "Entity Type"]
                                        st.table(entities_df)

                            if enable_criticism or enable_suggestions_for_improvement:
                                music_critique_results = music_critique(
                                    lyrics_content=transcription_results["text"],
                                    openai_api_key=openai_api_key
                                )

                                if enable_criticism:
                                    with st.expander("Criticism Results"):
                                        st.markdown(music_critique_results["critical_analysis"])

                                if enable_suggestions_for_improvement:
                                    with st.expander("Suggestions for Improvement"):
                                        st.markdown(music_critique_results["recommendations"])
                            
                            if enable_cover_image_generation:
                                with st.expander("Cover Image Suggestion"):
                                    huggingface_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
                                    if not huggingface_api_token:
                                        st.warning("Hugging Face API Token not set.", icon="⚠")
                                    else:
                                        music_theme_response = music_theme_generator(
                                            lyrics_content=transcription_results["text"],
                                            openai_api_key=openai_api_key
                                        )
                                        st.markdown(f"Based on the theme: **{music_theme_response}**")
                                        st.divider()
                                        cover_image_bytes = generate_image(query=music_theme_response, api_key=huggingface_api_token)
                                        st.image(io.BytesIO(cover_image_bytes)) 


if __name__ == "__main__":
    main()