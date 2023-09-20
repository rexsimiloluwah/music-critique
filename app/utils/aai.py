"""Utility Functions for AssemblyAI"""
import time
from typing import Optional, Dict

import requests
import streamlit as st

AAI_BASE_URL = "https://api.assemblyai.com/v2"
TRANSCRIBE_ENDPOINT = f"{AAI_BASE_URL}/transcript"
UPLOAD_ENDPOINT = f"{AAI_BASE_URL}/upload"
TRANSCRIPTION_SETTINGS = {
    "content_moderation": True,
    "summarization": True,
    "topic_detection": True,
    "sentiment_analysis": True,
    "entity_detection": True,
}


def create_aai_headers(api_key: str, accept_json: Optional[bool] = False):
    """Create the headers for the AssemblyAI endpoint

    Args:
        api_key (str): AssemblyAI API key
        accept_json (Optional[bool]): A flag that controls whether you are sending
            JSON data in the body of the HTTP request

    Returns:
        aai_headers (dict): A dictionary containing the headers
    """
    aai_headers = {"authorization": api_key}

    if accept_json:
        aai_headers["content-type"] = "application/json"

    return aai_headers


def upload_to_assemblyai(
    file_path: str, api_key: str
) -> (Optional[str], Optional[str]):
    """Upload Audio File to Assembly AI

    Args:
        file_path (str): The path of the uploaded file.
        api_key (str): The AssemblyAI API key.

    Returns:
        tuple: A tuple containing:
            - upload_url (Optional[str]): The URL of the uploaded file.
            - error (Optional[str]): An error message.
    """
    CHUNK_SIZE = 5242880

    def read_file(filename: str):
        with open(filename, "rb") as _file:
            while True:
                data = _file.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    try:
        response = requests.post(
            UPLOAD_ENDPOINT,
            headers=create_aai_headers(api_key=api_key),
            data=read_file(file_path),
        )

        if "error" in response.json():
            return None, response.json()["error"]

        upload_url = response.json()["upload_url"]
        return upload_url, None
    except requests.exceptions.HTTPError as e:
        error_msg = f"API returned HTTP error: {e}"
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unknown error occured: {e}"
        print(error_msg)
        return None, error_msg


def submit_transcription_job(
    audio_url: str,
    api_key: str,
    language_code: Optional[str] = "en_us",
    settings: Optional[Dict[str, bool]] = TRANSCRIPTION_SETTINGS,
) -> (Optional[str], Optional[str]):
    """Submit a transcription job to AssemblyAI

    Args:
        audio_url (str): The URL of the uploaded file.
        settings (Optional[dict]): A dictionary where the keys are the transcription settings
            names and the values are boolean values indicating whether the
            setting is enabled or not.
        api_key (str): The AssemblyAI API key.
        language_code (str): The language code.

    Returns:
        tuple: A tuple containing:
            - polling_endpoint (Optional[str]): The endpoint for polling the transcription results.
            - error (Optional[str]): The error message
    """
    try:
        data = {
            "audio_url": audio_url,
            "iab_categories": settings.get("topic_detection", True),
            "content_safety": settings.get("content_moderation", True),
            "summarization": settings.get("summarization", True),
            "summary_model": "informative",
            "summary_type": "bullets",
            "sentiment_analysis": settings.get("sentiment_analysis", True),
            "entity_detection": settings.get("entity_detection", True),
        }

        headers = create_aai_headers(api_key, True)
        response = requests.post(TRANSCRIBE_ENDPOINT, json=data, headers=headers)

        if "error" in response.json():
            return None, response.json()["error"]

        job_id = response.json()["id"]
        polling_endpoint = f"{TRANSCRIBE_ENDPOINT}/{job_id}"
        return polling_endpoint, None
    except requests.exceptions.HTTPError as e:
        error_msg = f"API returned HTTP error: {e}"
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unknown error occured: {e}"
        print(error_msg)
        return None, error_msg


@st.cache_data
def get_transcription_results(polling_endpoint: str, api_key: str):
    """Get the transcription results.

    This uses the polling technique to fetch the results from the polling endpoint.

    Args:
        polling_endpoint (str): The polling endpoint.
        api_key (str): The AssemblyAI API key

    Returns:
        tuple: A tuple containing:
            - response (Optional[dict]): The transcription result object.
            - error (Optional[str]):
    """
    status = "submitted"
    polling_duration = 3
    headers = create_aai_headers(api_key)

    while True:
        response = requests.get(polling_endpoint, headers=headers)
        status = response.json()["status"]

        if status == "submitted" or status == "processing":
            time.sleep(polling_duration)
        elif status == "completed":
            return response, None
        else:
            return None, response.json()["error"] or "An unknown error occurred"
