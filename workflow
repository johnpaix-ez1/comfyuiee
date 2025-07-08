## UPLOAD OR PASTE LINK OF ANY VIDEO YOU LIKE TO GET TRANSCRIPTS AND CREATE A VIDEO
   # Clear all variables



!pip install -U yt-dlp

!sudo rm /usr/local/bin/ffmpeg
# !cp /home/ubuntu/crewgooglegemini/CAPTACITY/ffmpeg /usr/local/bin/
# !chmod +x /usr/local/bin/ffmpeg

import os

# # Set the FFMPEG_BINARY environment variable to the location of ffmpeg
##os.environ['FFMPEG_BINARY'] = '/usr/bin/ffmpeg'

#os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'


# Navigate to your project folder in Drive
%cd /home/ubuntu/crewgooglegemini/CAPTACITY/captacity


from moviepy.editor import *
from tqdm.notebook import tqdm
import shutil
from moviepy.editor import ImageClip, vfx
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.fx.resize import resize  # Import resize for zooming
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, CompositeAudioClip
import subprocess
import tempfile
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import time
import requests
import asyncio
import ast
import ssl
import re
import json
import random
from telegram import Bot
import torch
import cv2
import websocket
import uuid
import urllib.request
import urllib.parse
from PIL import Image
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
import io
import edge_tts
from termcolor import colored
import datetime
import os
import glob
import shutil
from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException
from jsonschema import validate, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from groq import Groq
import yt_dlp  # For downloading YouTube videos
import moviepy.editor as mp
from datetime import datetime
#from ultralytics import YOLO
from urllib.parse import urlparse, unquote
import numpy as np
from telegram import Bot, InputMediaPhoto
import google.generativeai as genai
from dotenv import load_dotenv
from script_generator import generate_script
from transcriber import transcribe_locally
from moviepy.video.fx.all import fadeout
from moviepy.audio.fx.all import audio_fadeout
from pexelsapi.pexels import Pexels
from pydantic import ValidationError
from jsonschema import validate
import wave
from IPython.display import clear_output
from pydub import AudioSegment
from unidecode import unidecode
# import segment_parser # Removed: Used by add_captions
import transcriber
import soundfile as sf
# from text_drawer import ( # Removed: Used by add_captions and its helpers
#     get_text_size_ex,
#     create_text_ex,
#     blur_text_clip,
#     Word,
# )

# shadow_cache = {} # Removed as create_shadow was removed
# lines_cache = {} # Removed as calculate_lines was removed
#from ._init_ import detect_local_whisper


# # Groq API setup
load_dotenv()

# Ensure Gemini API key is set in environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")  # Change this to your actual variable name if different
if gemini_api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Set up the API key
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyDZzoZ9xAox4XKRo0c9l5RLBOLTGU3UcAc'
# genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Configure the Gemini API client
# genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")  # Change model as needed

# Ensure folder exists utility
def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def select_random_duration(min_time=3, max_time=7):
    """
    Select a random duration between min_time and max_time seconds.
    """
    return random.randint(min_time, max_time)



def extract_transcript_segments(transcript):
    """
    Extract segments of the transcript based on start times.
    If a segment duration is less than 3 seconds, merge it with the next valid segment.
    """
    segments = []
    durations = []  # To store the calculated durations
    i = 0
    total_duration = 0

    # Get the total voice-over duration from the END of the last segment
    voice_over_duration = transcript[-1]['end'] if transcript else 0

    while i < len(transcript):
        current_start_time = transcript[i]['start']
        next_start_time = None
        j = i + 1

        # Find the next valid start time after merging short segments
        while j < len(transcript):
            next_start_time = transcript[j]['start']
            duration = next_start_time - current_start_time
            if duration >= 3.5:  # Valid duration found
                break
            j += 1  # Merge this short segment and move to the next one

        if next_start_time is None:  # If no more valid segments, stop
            # Calculate the duration until the END of the transcript
            final_duration = voice_over_duration - current_start_time
            if final_duration > 0:
                segments.append(transcript[i:])
                durations.append(final_duration)
                total_duration += final_duration
            break

        # Calculate the valid duration
        duration = next_start_time - current_start_time

        # Store the segment and duration
        segment = transcript[i:j]  # Includes the merged segments
        segments.append(segment)
        durations.append(duration)
        total_duration += duration

        # Move to the next segment after merging
        i = j

    # Final check: Compare total duration with voice-over duration and adjust the last duration if necessary
    if total_duration < voice_over_duration:
        difference = voice_over_duration - total_duration
        print(f"Total segment duration is {difference:.2f} seconds shorter than the voice-over.")
        durations[-1] += difference
        print(f"Adjusted the last segment duration. New total duration: {sum(durations):.2f} seconds")
    elif total_duration > voice_over_duration:
        difference = total_duration - voice_over_duration
        print(f"Total segment duration is {difference:.2f} seconds longer than the voice-over.")
        durations[-1] = max(durations[-1] - difference, 0)
        print(f"Adjusted the last segment duration. New total duration: {sum(durations):.2f} seconds")
    else:
        print("Total segment duration matches the voice-over length.")

    #print(f"Voice-over duration: {voice_over_duration:.2f} seconds")
    print(f"Final total segment duration: {sum(durations):.2f} seconds")

    return segments,


    # Set the maximum allowed total video duration (in seconds)
#MAX_TOTAL_DURATION = 55


def get_video_duration(video_path):
    """Returns the duration of the video in seconds."""
    try:
        video = mp.VideoFileClip(video_path)
        return video.duration
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return 0


def get_videos_from_custom_folder(custom_folder):
    """Retrieve and sort videos from the custom folder by creation time."""
    video_files = []
    for filename in os.listdir(custom_folder):
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            file_path = os.path.join(custom_folder, filename)
            try:
                creation_time = os.path.getctime(file_path)
                video_files.append((creation_time, file_path))
            except Exception as e:
                print(f"Error accessing {file_path}: {e}")
    # Sort videos by creation time (oldest first)
    video_files.sort(key=lambda x: x[0])
    return [file_path for _, file_path in video_files]



def search_and_download_best_videos(download_dir, custom_folder, use_pexels=True):
    downloaded_videos = []
    video_titles = []
    video_durations = []
    total_duration = 0

    # Step 1: Check the custom folder for videos
    custom_videos = get_videos_from_custom_folder(custom_folder)
    for video_path in custom_videos:
        video_duration = get_video_duration(video_path)
        downloaded_videos.append(video_path)
        video_titles.append(os.path.splitext(os.path.basename(video_path))[0])
        video_durations.append(video_duration)
        total_duration += video_duration
        print(f"Selected '{os.path.basename(video_path)}' from custom folder, total duration now {total_duration:.2f} seconds.")

    # We're not using Step 2 anymore, as we're only using videos from the custom folder

    return downloaded_videos, video_titles, video_durations



def extract_title_from_url(video_url):
    """
    Extracts the video title from the video URL.
    """
    path = urlparse(video_url).path
    # Split the path and get the last segment, then remove the video ID
    title_with_id = path.split('/')[-2]
    # Replace hyphens with spaces and decode any URL-encoded characters
    title = unquote(title_with_id.replace('-', ' '))
    return title

def remove_trailing_numbers(title):
    """
    Removes trailing numbers from a string.
    """
    return re.sub(r'\d+$', '', title).strip()


# Removed fits_frame, calculate_lines, create_shadow, get_font_path as they were helpers for add_captions

def ffmpeg(command):
    return subprocess.run(command, capture_output=True)

# Removed create_shadow

# Removed get_font_path

def detect_local_whisper(print_info):
    try:
        import whisper
        use_local_whisper = True
        if print_info:
            print("Using local whisper model...")
    except ImportError:
        use_local_whisper = False
        if print_info:
            print("Using OpenAI Whisper API...")

    return use_local_whisper


# Removed add_captions function

# Removed load_video_segments function



# Initialize the YOLO model
def load_yolo_model(model_path='yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Load custom model if needed
    model.eval()  # Set model to evaluation mode
    return model

def analyze_video_with_yolo(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    detections = []
    object_durations = {}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 object detection
        results = model(frame)

        # Process results
        for det in results.xyxy[0]:  # det is (x1, y1, x2, y2, conf, cls)
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if conf > 0.7:  # Confidence threshold
                class_name = results.names[int(cls)]
                detections.append({
                    "frame": frame_idx,
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2]
                })

                # Update object durations
                if class_name not in object_durations:
                    object_durations[class_name] = {"start_frame": frame_idx, "duration": 1}
                else:
                    object_durations[class_name]["duration"] += 1

        frame_idx += 1

        if frame_idx >= frame_count:
            break

    cap.release()

    preprocessed_data = preprocess_yolo_output(detections, object_durations, fps)
    print("Preprocessed Data:", preprocessed_data)
    return preprocessed_data



def preprocess_yolo_output(detections, object_durations, fps):
    preprocessed = {
        "scene_description": [],
        "objects": []
    }

    # Create a scene description and summarize detections
    for class_name, duration_info in object_durations.items():
        start_time = duration_info["start_frame"] / fps
        duration = duration_info["duration"] / fps

        # Only include objects with a duration of 1 second or more
        if duration >= 1.0:
            # Find the first detection for this class to get a representative bbox
            representative_detection = next((d for d in detections if d["class"] == class_name), None)
            bbox = representative_detection["bbox"] if representative_detection else None

            # Add to summary
            preprocessed["objects"].append({
                "class": class_name,
                "start_time": start_time,
                "duration": duration,
                "bbox": bbox
            })

            # Scene description (can be customized)
            description = f"A {class_name} appeared at {start_time:.2f} seconds for {duration:.2f} seconds."
            preprocessed["scene_description"].append(description)

    return preprocessed


# prep data with numpy
def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


# preparing the data for script generation
def prepare_data_for_script_generation(all_video_data):
    prepared_data = []
    for video in all_video_data:
        prepared_video = {
            "title": video["title"]
        }
        prepared_data.append(prepared_video)
    return prepared_data


# Utility function to ensure a folder exists
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# JSON validation function
def validate_json_structure(data, schema):
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False



def normalize_script_for_kokoro(script):
    import re

    # Remove asterisks (Markdown noise)
    script = script.replace("*", "")

    # Remove actual emojis (U+1F300–U+1FAFF) but KEEP punctuation like … — etc.
    script = re.sub(r'[\U0001F300-\U0001F5FF]', '', script)

    # OPTIONAL: remove a limited set of decorative symbols
    script = re.sub(r'[✅👍🚨🔥]', '', script)

    # DO NOT strip parentheticals entirely – keep them for tone
    # script = re.sub(r'\((?!\[)[^)]*\)', '', script)  # ← removed

    # Normalize whitespace but preserve newlines
    script = re.sub(r'[^\S\r\n]+', ' ', script)

    # Cleanup each line
    script = "\n".join(line.strip() for line in script.splitlines())

    # Collapse only excessive blank lines
    script = re.sub(r'\n{3,}', '\n\n', script)

    return script


def extract_json_from_response(content):
    # Extract JSON from triple-backtick json code block if possible
    json_match = re.search(r'```json\s*\n(.*?)```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Fallback: extract what looks like a JSON object
        json_match = re.search(r'(\{[\s\S]+\})', content)
        json_str = json_match.group(1).strip() if json_match else content.strip()
    return json_str

def fix_llm_json(json_str):
    # Escape unescaped newlines/tabs in JSON string values
    def fix_newlines(m):
        return '"' + m.group(1).replace('\n', '\\n').replace('\t', '\\t') + '"'
    return re.sub(r'"((?:[^"\\]|\\.)*)"', fix_newlines, json_str)

def generate_script_from_video(transcript):
    prompt = f"""You are a master storyteller and creative screenwriter who transforms transcripts into fully animated cinematic stories.

Instructions:
- Analyze the transcript's core emotional or intellectual ideas.
- Turn it into an **animated short film treatment** (2–10 minutes).
- Characters may be humans, animals, mythical creatures, or hybrids.
- Use **high drama, emotion, humor, or surprise** to entertain.
- Every scene must include setting, action, and character dialogue.
- Use creative freedom — characters can fly, shapeshift, talk to stars, etc.

Generate a JSON object with this structure:

{{
  "title": "Movie-worthy title teasing the plot",
  "genre": "Fantasy | Drama | Sci-fi | etc",
  "keywords": ["short", "animated", "hero", "betrayal", "hope"],
  "synopsis": "One paragraph summary of the story's arc.",
  "core_theme": "What the story is really about (e.g., forgiveness, courage, identity)",
  "characters": [
    {{
      "name": "Character Name",
      "type": "Species or type",
      "role": "Hero / Villain / Sidekick / etc",
      "trait": "Notable personality or ability"
    }}
  ],
  "scene_sequence": [
    {{
      "scene_title": "Opening Scene Name",
      "setting": "Where the scene takes place",
      "action": "What happens in the scene (visually described)",
      "dialogue": [
        "CharacterName: Line of dialogue",
        "AnotherCharacter: Response..."
      ]
    }},
    ...
  ],
  "ending_note": "What twist or emotional payoff ends the story"
}}

Transcript input:
\"\"\"{transcript}\"\"\"

Return only the JSON object — no explanation, no extra comments.
    """

    # Generate with your LLM
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    print("Raw API Response:", response.text)
    content = response.text

    # Extract and fix JSON from the content
    json_str = extract_json_from_response(content)
    fixed_json = fix_llm_json(json_str)

    try:
        result = json.loads(fixed_json)
        return result
    except Exception as e:
        print(f"Failed to parse movie JSON: {e}")
        return None


def generate_video_prompts_from_script(script_json, batch_size=5):
    """
    Processes a script_json to generate rich, descriptive video prompts for each scene
    using an LLM, in batches with delays.

    Args:
        script_json (dict): The script object from generate_script_from_video.
        batch_size (int): Number of scenes to process in each LLM call.

    Returns:
        list: A list of generated video prompts (strings), or None if an error occurs.
    """
    if not script_json or "scene_sequence" not in script_json:
        print("Error: Invalid script_json or missing 'scene_sequence'.")
        return None

    scenes = script_json["scene_sequence"]
    all_scene_data = [] # Will store dicts: {video_prompt, audio_prompt, negative_audio_prompt, vocals_instruction}
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    for i in range(0, len(scenes), batch_size):
        batch_scenes = scenes[i:i + batch_size]
        print(f"Processing batch of {len(batch_scenes)} scenes for video and audio prompts (scenes {i+1} to {i+len(batch_scenes)})...")

        scenes_for_llm_prompt = []
        for idx, scene in enumerate(batch_scenes):
            scene_info = {
                "original_scene_number": i + idx + 1,
                "scene_title": scene.get("scene_title", "Untitled Scene"),
                "setting": scene.get("setting", "No setting described"),
                "action": scene.get("action", "No action described"),
                "dialogue": scene.get("dialogue", []) # Important for deciding on vocals
            }
            scenes_for_llm_prompt.append(scene_info)

        prompt = f"""You are an expert multimedia prompt engineer for AI content generation.
Given the following batch of scenes from a larger script, generate a JSON list. Each item in the list should be a JSON object corresponding to one scene, containing:
1.  `video_prompt`: A rich, descriptive prompt for an AI video generation model. This should vividly describe visual elements, atmosphere, camera angles (if appropriate), character actions/expressions, and overall mood.
2.  `audio_prompt`: A descriptive prompt for an AI audio generation model like MMAudio (e.g., "a calm ambient soundscape with gentle rain", "energetic electronic music with a strong beat").
3.  `negative_audio_prompt`: A negative prompt for MMAudio (e.g., "no vocals, no distortion, no loud noises", "avoid harsh sounds").
4.  `vocals_instruction`: A brief instruction on how dialogue/vocals from the original script should be handled for this scene. Consider the scene's dialogue. Examples: "dialogue_primary", "ambient_sound_primary", "mix_dialogue_with_ambient_sound", "no_dialogue_ambient_only".

The script's overall title is: "{script_json.get('title', 'N/A')}"
The script's genre is: "{script_json.get('genre', 'N/A')}"
The script's synopsis is: "{script_json.get('synopsis', 'N/A')}"

Batch of scenes to process:
{json.dumps(scenes_for_llm_prompt, indent=2)}

Return your response as a single JSON list, where each element is an object containing the four keys: `video_prompt`, `audio_prompt`, `negative_audio_prompt`, and `vocals_instruction`.
The list should contain exactly {len(scenes_for_llm_prompt)} JSON objects, one for each scene in the input batch.

Example for a single scene object in the list:
{{
  "video_prompt": "Dynamic shot of a character running through a neon-lit cyberpunk alleyway during a heavy rainstorm, puddles reflecting the city lights.",
  "audio_prompt": "tense synthwave music with driving electronic beat, sound of heavy rain and distant sirens",
  "negative_audio_prompt": "no cheerful music, no birdsong",
  "vocals_instruction": "dialogue_primary"
}}

Ensure prompts are detailed and guide AI models effectively. Return only the JSON list of objects.
"""
        try:
            print(f"Sending request to LLM for video/audio prompts for batch starting at scene {i+1}...")
            response = gemini_model.generate_content(prompt)

            response_text = response.text.strip()
            # print(f"DEBUG: Raw LLM response for batch {i+1}: {response_text}") # For debugging

            match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL) # Look for a list of objects
            json_str = ""
            if match:
                json_str = match.group(0)
            else: # Fallback if ```json ... ``` is used
                match_markdown = re.search(r'```json\s*\n(.*?)```', response_text, re.DOTALL)
                if match_markdown:
                    json_str = match_markdown.group(1).strip()
                else:
                    json_str = response_text # Last resort

            # print(f"DEBUG: Extracted JSON string for batch {i+1}: {json_str}") # For debugging

            generated_data_for_batch = json.loads(json_str)

            if isinstance(generated_data_for_batch, list) and \
               all(isinstance(item, dict) and \
                   "video_prompt" in item and \
                   "audio_prompt" in item and \
                   "negative_audio_prompt" in item and \
                   "vocals_instruction" in item \
                   for item in generated_data_for_batch):

                if len(generated_data_for_batch) == len(batch_scenes):
                    all_scene_data.extend(generated_data_for_batch)
                    print(f"Successfully generated {len(generated_data_for_batch)} video/audio prompt sets for the batch.")
                else:
                    print(f"Warning: LLM returned {len(generated_data_for_batch)} prompt sets, but expected {len(batch_scenes)} for the batch. Using what was returned.")
                    all_scene_data.extend(generated_data_for_batch) # Or handle error more strictly
            else:
                print(f"Error: LLM response for batch {i+1} was not a list of valid prompt data objects as expected. Response: {generated_data_for_batch}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response for batch {i+1}: {e}. Response text was: '{json_str}'")
        except Exception as e:
            print(f"An error occurred during LLM call for video/audio prompts for batch {i+1}: {e}")

        if i + batch_size < len(scenes):
            print(f"Waiting for 4 seconds before the next LLM call...")
            time.sleep(4)

    if not all_scene_data:
        print("No video/audio prompt data was generated.")
        return None

    return all_scene_data


def download_youtube_video(youtube_link, output_path="/content"):
    # Define an output template with a shorter filename
    output_template = os.path.join(output_path, "%(title).50s-%(id)s.%(ext)s")
    
    # Build the yt-dlp command as a list of arguments
    command = [
        "yt-dlp",
        youtube_link,
        "-o", output_template,
        "--restrict-filenames",  # Avoid problematic characters
        "--max-filesize", "200G"   # Optional: limit file size to 2GB
    ]

    # Run the command and capture its output
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error during yt-dlp execution:", result.stderr)
        return None

    # After download, try to locate the downloaded file
    try:
        files = [os.path.join(output_path, f) for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
        if not files:
            print("No files found in the output directory.")
            return None
        latest_file = max(files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print("Error retrieving the downloaded file:", e)
        return None


def generate_transcript_with_retries(model, myfile, prompt, retries=10, delay=10):
    for attempt in range(1, retries+1):
        try:
            result = model.generate_content([myfile, prompt])
            return result
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise


def extract_transcript_segments_image_vid(transcript, segment_duration=9, min_last_segment=5):
    """
    Extracts segments of a given duration from the transcript.
    Adds a segment_id to each segment: segment_YYYYMMDD_HHMM

    Args:
        transcript (list): List of dictionaries containing 'start', 'end', and 'text'.
        segment_duration (int): Target duration of each segment in seconds.
        min_last_segment (int): Minimum duration of the last segment in seconds.

    Returns:
            Adds a segment_id to each segment: segment_YYYYMMDD_HHMMSS_count

    """
    segments = []
    current_start = 0.0
    current_text = ""
    segment_counter = 1

    # Generate timestamp string once for this run
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    i = 0
    while i < len(transcript):
        entry = transcript[i]
        end = entry['end']
        text = entry['text']

        if end - current_start > segment_duration:
            if len(segments) > 0 and end - segments[-1]['start'] < segment_duration:
                if i + 1 < len(transcript):
                    next_entry = transcript[i + 1]
                    next_end = next_entry['end']
                    next_text = next_entry['text']

                    segments[-1]['end'] = next_end
                    segments[-1]['duration'] = round(next_end - segments[-1]['start'], 2)
                    segments[-1]['text'] += " " + text + " " + next_text
                    i += 2
                else:
                    segments.append({
                        "segment_id": f"segment_{timestamp}_{segment_counter}",
                        "start": current_start,
                        "end": end,
                        "duration": round(end - current_start, 2),
                        "text": current_text.strip()
                    })
                    segment_counter += 1
                    current_start = end
                    current_text = ""
                    i += 1
            else:
                duration = end - current_start
                segments.append({
                    "segment_id": f"segment_{timestamp}_{segment_counter}",
                    "start": current_start,
                    "end": end,
                    "duration": round(duration, 2),
                    "text": current_text.strip()
                })
                segment_counter += 1
                current_start = end
                current_text = text
                i += 1
        else:
            current_text += " " + text
            i += 1

    # Handle the last segment
    if current_text:
        duration = transcript[-1]['end'] - current_start
        if duration < min_last_segment and len(segments) > 0:
            prev_segment = segments[-1]
            prev_segment['end'] = transcript[-1]['end']
            prev_segment['duration'] = round(prev_segment['end'] - prev_segment['start'], 2)
            prev_segment['text'] += " " + current_text.strip()
        else:
            segments.append({
                "segment_id": f"segment_{timestamp}_{segment_counter}",
                "start": current_start,
                "end": transcript[-1]['end'],
                "duration": round(duration, 2),
                "text": current_text.strip()
            })

    return segments

def parse_transcript_file(file_path):
    """
    Parses the transcript file and returns a list of dictionaries.
    """
    transcript = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split('  ', 1)
        if len(parts) == 2:
            time_range, text_data = parts
            start_time, end_time = map(float, time_range.split())
            
            # Extract the text part, removing JSON if present
            text = text_data.split('[', 1)[0].strip()
            
            transcript.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
    return transcript


# Your existing code...
expected_schema2 = {
    "type": "object",
    "properties": {
        "image_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "theme_keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["thumbnail_prompt", "negative_prompt", "theme_keywords"]
}


def generate_thumbnail_prompt_purple_cow(full_transcript):
    """
    Generates a fully expressive, psychology-driven, niche-agnostic YouTube thumbnail prompt for portrait (720x1280) thumbnails.
    Runs AFTER segment prompts. No video title is required.
    """
    prompt = f"""
You are an elite AI image prompt engineer and YouTube thumbnail strategist.

You are provided with the full transcript of the video below.  
**Carefully analyze it** to extract the key themes, emotional tone, metaphors, and story tension.  
Your thumbnail must be based on this transcript — the overlay text, objects, emotions, and entire visual story must align with the core message of the video.

**Video Transcript:**  
{full_transcript}

Your task is to create a single, **scroll-stopping, psychologically irresistible YouTube thumbnail prompt** — designed specifically for a portrait (vertical) 720×1280 resolution layout (ideal for YouTube Shorts and mobile viewers).

**Strict Requirements for Portrait Thumbnail (720x1280):**

- The thumbnail must have a 2–5 word, ultra-large, bold overlay text using white, yellow, orange, or blue colors.
- Use at least one curved arrow in yellow, orange, or any contrasting color that **points directly at the object of attention or emotional focus** in the thumbnail.
- **The overlay text must be extremely large and visually dominant, covering at least 80 to 90 percent of the total image height and width, formatted vertically if necessary for portrait proportions. It must appear oversized, unavoidable, and clearly the first thing the viewer notices—even in small mobile previews.**
- **Encourage the AI to describe fonts as “giant block lettering,” “extra-large display font,” or “cinematic poster text.”**
- **Text layout should prioritize vertical stacking, with bold, centered, and large lettering that naturally fits the portrait frame and leaves minimal empty space.**
- Text must be a meaningful, curiosity-driven phrase directly related to the video concept and transcript — not generic or random.
- Ensure very high color contrast between text and background for mobile readability.
- Include subtle glow, outline, or shadow directly behind the text to enhance readability against varied backgrounds.

**Background and Visual Composition:**

- Do not use child images for thumbnails.
- Use a muted, vintage, or retro-inspired color palette to prevent background distraction.
- Ensure background colors stay desaturated enough to let the bright bold text pop.
- Avoid busy, noisy, or overcomplicated backgrounds.
- Must include at least one **symbolic or metaphorical object** (examples: brain, hourglass, lightbulb, maze, broken chain, DNA, clock, etc.), relevant to the video's story or emotion.
- Symbolic objects must integrate naturally into the background without competing with the text.
- Must include a human face or silhouette showing a clear, readable emotion (fear, awe, surprise, focus, resolve, etc.), easily visible at small scale and fitting the vertical frame.
- The face can be semi-transparent, ghosted, or background-blended but the emotion must remain readable.

**Overall Style:**

- Keep it simple: clear, bold, fast-to-grasp within 1 second.
- Prioritize dramatic emotional hook and story tension.
- Visual style should feel cinematic or like a movie poster — avoid bland stock image look.
- Absolutely avoid the words: unlock, dive, or diving anywhere in the text.

**Output format:** JSON block, nothing else.

Your output:
- "thumbnail_prompt": A vivid, cinematic, vertically optimized image prompt following the above rules.
- "negative_prompt": No facial distortions, no text errors, no blurring, no clutter, no duplicate faces, no disfigured limbs, no extra limbs, no bad anatomy.
- "theme_keywords": 3–6 keywords from the prompt.

**Example output:**
{{
  "thumbnail_prompt": "Portrait layout showing a dramatic close-up of a person looking shocked, with glowing light beams behind them. A giant, vertically stacked, bold yellow block letter overlay says: 'WHAT HAPPENED NEXT' filling most of the frame. A large curved orange arrow points to the person’s eyes. Background is muted teal with soft light flare.",
  "negative_prompt": "no facial distortions, no text errors, no blurring, no clutter, no duplicate faces, no disfigured limbs, no extra limbs, no bad anatomy.",
  "theme_keywords": ["shock", "reveal", "portrait", "emotion", "arrow", "yellow text"]
}}
"""



    
    # Set the Gemini API key you want to use here (overrides any .env key for this block)
    genai.configure(api_key="AIzaSyAmrfaRZ0EsoeTAb05K8sNPrTyBd7ncNag")
    
    # Now create the model and generate content as usual
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    response_text = response.text
    
    json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
    if json_match:
        cleaned_response = json_match.group(1)
    else:
        cleaned_response = response_text
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON from API response")
        return {}


#=====================================================================================
# **Background and Visual Composition:**
# - Don't use child images for thumbnails
# - Background must match the mood of the niche — large bold vintage or retro tones for history, psychology, or mystery; then large bold and bright, glossy tones for food, tech, or pop culture. Avoid overly saturated or clashing schemes unless intentional for comedic or shocking effect.
# - Ensure background colors are muted or desaturated to maximize contrast with bright, rugged, very large bold text.
# - Avoid busy or cluttered backgrounds that compete with the text.
# - Must include at least one or more **metaphorical or symbolic objects** (example: brain, tree, robot, gold coin, DNA, clock, hourglass, lightbulb, key, maze, globe, etc.) relevant to the video’s emotional story or concept.
# - For visual niches like food, use ingredients (e.g., lemon, knife, plate, spilled oil) as symbols. For tech, use items like chips, wires, QR codes, virtual HUDs. For emotional drama, use abstract forms (e.g. shattered glass, half-heart, empty chair). Choose symbols that immediately click with the niche’s viewer psychology.
# - Symbolic objects’ colors and placement must harmonize with the background and text, never competing or causing visual noise.
# - Must include a human face or silhouette that **expresses an emotion matching the subline’s context** (example: calm, awe, focus, fear, surprise, joy, resolve).
# - The human face or silhouette must be clearly visible and the emotion must be easily recognizable at thumbnail size.
# - The human face can be semi-transparent, subtle, or blended in background — but must clearly convey the emotion.
# - The emotion expressed must support the implied story tension or hook in the thumbnail.
# - Adjust emotional expression and overall composition to match the emotional tone and energy of the video’s niche — whether calm, chaotic, comedic, informative, or suspenseful.
#=======================================================================================================

    # # Example API call (replace with your LLM client)
    # client = Groq(api_key="apikey here")
    # chat_completion = client.chat.completions.create(
    #     messages=[{"role": "user", "content": prompt}],
    #     model="llama-3.3-70b-versatile",
    # )
    # response_text = chat_completion.choices[0].message.content
    # json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
    # if json_match:
    #     cleaned_response = json_match.group(1)
    # else:
    #     cleaned_response = response_text
    # try:
    #     return json.loads(cleaned_response)
    # except json.JSONDecodeError:
    #     print("Error: Unable to parse JSON from API response")
    #     return {}




# Your existing code...
expected_schema2 = {
    "type": "object",
    "properties": {
        "image_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "theme_keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["image_prompt", "negative_prompt", "theme_keywords"]
}

def generate_image_prompts_batch(segment_batch, full_transcript):
    """
    Generates image prompts for a batch of segments using the full transcript as context.
    
    Args:
    segment_batch (list): A list of segment dictionaries, each containing 'duration' and 'text'.
    full_transcript (str): The full transcript text for context.

    Returns:
    list: A list of dictionaries containing generated prompts for each segment.
    """
    prompt = f"""You are an imaginative prompt generator for AI-based image systems like DALL-E, MidJourney, and Stable Diffusion without including nudity whether half or full.
Analyze the provided full transcript and segment details to creatively craft for each segment:

- A **dramatically compelling and uniquely conceptualized positive prompt** for generating an AI image that is instantly memorable and thought-provoking, directly relevant to the segment's core message. The image should aim for an unexpected yet fitting visual metaphor or an intensified, captivating representation of the scene/concept. Incorporate:
  - **No images with people fully nude or half. Prompts should not include people wearing underwears, or photorealistic images with naked individuals. they must be fully clothed.** 
  - **Intense and Evocative Visuals:** Focus on unusual perspectives, dynamic compositions, symbolic elements, and a heightened sense of atmosphere or emotion. Think beyond the literal to create something that sparks curiosity.
  - **Striking Lighting & Textures:** Employ dramatic lighting (e.g., chiaroscuro, strong rim lighting, ethereal glows, moody silhouettes) and emphasize tactile or visually rich textures that add depth and intrigue.
  - **Cinematic or Artistic Flair:** Lean into styles that enhance drama – epic cinematic framing, surrealism (if appropriate to the context), painterly aesthetics with bold strokes, or hyperrealism focused on an unusual detail. Specify professional camera (e.g., Canon EOS R5, Hasselblad X2D) and lens choices that would achieve a unique or high-impact look when relevant.
  - **Anatomical and Textual Integrity:** Crucially, ensure any depicted human figures have well-formed faces, anatomically accurate fingers (ideally minimized or naturally obscured if not central to the concept), natural toe positioning, and properly proportioned body parts. Any text included should be short, simple, correctly spelled, and perfectly integrated without distortion.
  - **Sensory & Environmental Amplification:** Heighten sensory and environmental details to create an immersive and unforgettable scene. Examples (adapt to the segment's content):
    - *Instead of just 'misty morning', consider: "An almost sentient, pearlescent mist coiling through stark silhouettes at dawn."*
    - *Instead of 'rain-kissed reflections', consider: "Neon-drenched, fractured reflections in a rain-slicked alley, hinting at a story untold."*
    - *Instead of 'dappled sunlight', consider: "Cathedral-like shafts of golden light piercing a dense, ancient canopy, illuminating a single, significant object."*
    - *Consider elements like: swirling particles of an unusual nature (e.g., glowing embers, crystallized thoughts, time fragments), exaggerated weather phenomena (if non-graphic and contextually relevant), or a focus on a single, powerful symbolic color.*
  - **Relevance is Key:** While aiming for the dramatic and unique, the core visual must strongly resonate with and clearly illustrate the provided segment text. Avoid gratuitous or shocking elements that detract from the message or could be considered graphic. The goal is to be *remarkably relevant*.
  - Absolutely no nudity is permitted in any form. All human figures — male, female, or child — must be fully clothed in context-appropriate attire. Avoid low-cut clothing, exposed skin, lingerie-like styling, or ambiguous silhouettes. No suggestive or sensual postures. Backgrounds must also be free of any nude figures, statues, or artwork. This rule is critical.
  - Ensure the final prompt is well-structured and free from contradictions or surreal exaggerations unless clearly intentional and metaphorical.
  - If including humans, prefer neutral or emotionally expressive poses (e.g., resolve, awe, curiosity), with normal bodily focus.

- A **negative prompt** tailored for face adjustment and sharpening, excluding:
  - Unnatural skin tones
  - Facial distortions or warping
  - Plastic-like textures
  - Oversharpening artifacts
  - Harsh blurring effects
  - Nudity, exposed skin, lingerie, suggestive poses, sensual expression, erotic lighting, ambiguous silhouettes

- A list of **theme keywords** derived from the segment to optimize content creation.

Consider the duration of each segment when crafting the prompts, as this indicates how long the image will be displayed.

Full Transcript Context:
{full_transcript}

Segments to process:
{json.dumps(segment_batch, indent=2)}

Return the output as a JSON array, with each item corresponding to a segment:
[
  {{
    "duration": 12.18,
    "text": "Segment text...",
    "image_prompt": "A highly descriptive image prompt...",
    "negative_prompt": "No facial distortions, oversharpening artifacts, or plastic-like skin textures.",
    "theme_keywords": ["keyword1", "keyword2", "keyword3"]
  }},
  ...
]
"""

 
    
    # Set the Gemini API key for this block (overrides any .env or previous key)
    genai.configure(api_key="AIzaSyAmrfaRZ0EsoeTAb05K8sNPrTyBd7ncNag")
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    response_text = response.text
    
    json_match = re.search(r'``````', response_text, re.DOTALL)
    if json_match:
        cleaned_response = json_match.group(1)
    else:
        json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(1)
        else:
            cleaned_response = response_text
    
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print(colored("Error: Unable to parse JSON from API response", "red"))
        return []

    
    # model = genai.GenerativeModel("gemini-2.0-flash", api_key="AIzaSyAmrfaRZ0EsoeTAb05K8sNPrTyBd7ncNag")
    # response = model.generate_content(prompt)
    # response_text = response.text
    
    # json_match = re.search(r'``````', response_text, re.DOTALL)
    # if json_match:
    #     cleaned_response = json_match.group(1)
    # else:
    #     json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
    #     if json_match:
    #         cleaned_response = json_match.group(1)
    #     else:
    #         cleaned_response = response_text
    
    # try:
    #     return json.loads(cleaned_response)
    # except json.JSONDecodeError:
    #     print(colored("Error: Unable to parse JSON from API response", "red"))
    #     return []



    
    # client = Groq(api_key="api key here")
    # chat_completion = client.chat.completions.create(
    #     messages=[{"role": "user", "content": prompt}],
    #     model="llama-3.3-70b-versatile",
    # )
    # response_text = chat_completion.choices[0].message.content

    # json_match = re.search(r'``````', response_text, re.DOTALL)
    # if json_match:
    #     cleaned_response = json_match.group(1)
    # else:
    #     json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
    #     if json_match:
    #         cleaned_response = json_match.group(1)
    #     else:
    #         cleaned_response = response_text

    # try:
    #     return json.loads(cleaned_response)
    # except json.JSONDecodeError:
    #     print(colored("Error: Unable to parse JSON from API response", "red"))
    #     return []


VIDEO_TRANSCRIPT_LOG = "/home/ubuntu/crewgooglegemini/video_transcript_log.txt"
PROMPT_LOG_FILE = "/home/ubuntu/crewgooglegemini/prompt_log.txt"
CURRENT_PROMPT_FILE = "/home/ubuntu/crewgooglegemini/current_prompt.json"

def confirm_action(message):
    response = input(colored(f"{message} (y/n): ", "cyan"))
    return response.strip().lower() == 'y'

    print(colored("Step 1: Initialize the connection settings.", "cyan"))
    server_address = "54.80.78.81:8253"
    client_id = str(uuid.uuid4())
    
    print(colored(f"Server Address: {server_address}", "magenta"))
    print(colored(f"Generated Client ID: {client_id}", "magenta"))


# Renamed from get_image to be more generic for file types
def get_comfy_output_file(filename, subfolder, folder_type):
    """Fetches a file (image, video, etc.) from the ComfyUI server."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)

    # print(colored(f"Fetching file '{filename}' from ComfyUI server: http://{server_address}/view?{url_values}", "cyan"))
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    # print(colored(f"Fetching history for prompt ID: {prompt_id}.", "cyan"))
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def reconnect_websocket():
    ws = websocket.WebSocket()
    ws_url = f"ws://{server_address}/ws?clientId={client_id}"
    print(colored(f"Reconnecting WebSocket to {ws_url}", "cyan"))
    ws.connect(ws_url)
    return ws

# Renamed from get_output_images and adapted to be more generic
def get_comfy_output_files_data(prompt_id):
    """
    Retrieves file data (images, videos, etc.) for a completed ComfyUI prompt.
    It expects the output node to contain a list of files, e.g., 'gifs', 'videos', or 'images'.
    """
    files_data = {}
    history = get_history(prompt_id) # Get full history
    if prompt_id not in history:
        print(colored(f"Error: Prompt ID {prompt_id} not found in history.", "red"))
        return files_data

    prompt_history = history[prompt_id]

    for node_id in prompt_history.get('outputs', {}):
        node_output = prompt_history['outputs'][node_id]
        # Check for common output keys that might contain files
        # Common keys could be 'images', 'gifs', 'videos', etc.
        # For this task, we are expecting video files. Let's assume they might be under a 'videos' key
        # or a generic 'files' key. If it's images that are then turned to video by the workflow,
        # it might still be 'images'.

        output_key_to_check = None
        if 'videos' in node_output: # Ideal case for video output
            output_key_to_check = 'videos'
        elif 'gifs' in node_output: # Animated gifs might be an output
             output_key_to_check = 'gifs'
        elif 'images' in node_output: # Fallback if it's images to be animated later
            output_key_to_check = 'images'
        elif 'files' in node_output: # A generic file output
            output_key_to_check = 'files'

        if output_key_to_check and isinstance(node_output[output_key_to_check], list):
            collected_files = []
            for file_info in node_output[output_key_to_check]:
                if 'filename' in file_info and 'subfolder' in file_info and 'type' in file_info:
                    # print(colored(f"Found file '{file_info['filename']}' in node {node_id}. Type: {file_info['type']}", "yellow"))
                    try:
                        file_content = get_comfy_output_file(file_info['filename'], file_info['subfolder'], file_info['type'])
                        collected_files.append({
                            "filename": file_info['filename'],
                            "content": file_content,
                            "type": file_info['type']
                        })
                    except Exception as e:
                        print(colored(f"Error downloading file {file_info['filename']}: {e}", "red"))
                else:
                    print(colored(f"Warning: File info in node {node_id} is incomplete: {file_info}", "yellow"))
            if collected_files:
                files_data[node_id] = collected_files
    if not files_data:
        print(colored(f"No output files found in history for prompt_id {prompt_id}. Searched for 'videos', 'gifs', 'images', 'files' keys in output nodes.", "yellow"))
        # print(colored(f"Full history output for prompt {prompt_id}: {json.dumps(prompt_history.get('outputs',{}), indent=2)}", "magenta"))

    return files_data


# Renamed from generate_images and adapted
def await_comfy_job_completion(ws, prompt_id):
    """
    Monitors a ComfyUI job via WebSocket until completion.
    Returns True if execution completes successfully, False otherwise.
    """
    last_reported_percentage = 0
    timeout = 30

    ws.settimeout(timeout)
    while True:
        try:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'progress':
                    data = message['data']
                    current_progress = data['value']
                    max_progress = data['max']
                    percentage = int((current_progress / max_progress) * 100)
                    if percentage > last_reported_percentage:
                        #print(colored(f"Progress: {percentage}% in node {data['node']}", "yellow"))
                        last_reported_percentage = percentage
                elif message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id: # Execution complete for this prompt
                        # print(colored(f"Execution complete for prompt_id: {prompt_id}", "green"))
                        return True # Indicate successful completion
                elif message['type'] == 'status': # Other status messages
                    # print(colored(f"Status for prompt {prompt_id}: {message['data']}", "grey"))
                    pass
                elif message['type'] == 'execution_error':
                    print(colored(f"Execution error for prompt {prompt_id}: {message['data']}", "red"))
                    return False # Indicate error
                elif message['type'] == 'execution_interrupted':
                    print(colored(f"Execution interrupted for prompt {prompt_id}: {message['data']}", "red"))
                    return False # Indicate error


        except WebSocketTimeoutException:
            print(colored(f"Timeout waiting for ComfyUI response for prompt {prompt_id}. Reconnecting...", "yellow"))
            try:
                ws.ping() # Check if connection is still alive
            except: # If ping fails, WebSocket is likely closed
                print(colored("WebSocket connection lost. Attempting to reconnect...", "red"))
                try:
                    ws.close() # Ensure old ws is closed
                except:
                    pass
                ws = reconnect_websocket() # This global ws needs to be the one used by the caller
                                           # This is problematic if ws is passed by value.
                                           # For now, assume reconnect_websocket updates a global or class instance if used like that.
                                           # Or, this function should return the new ws.
                                           # For simplicity here, let's assume it might raise an error if reconnect fails.
                print(colored("Reconnected. Will retry receiving for current prompt if possible, or fail.", "yellow"))
                # Note: True state recovery after reconnect for a specific prompt is complex with current ComfyUI API.
                # Usually, you'd have to re-check history or re-queue.
                # For this function, a timeout might mean the job is stuck or server is unresponsive.
            # We might want to break or return False after a certain number of timeouts.
            # For now, it will keep trying indefinitely on timeout if ping succeeds.

        except WebSocketConnectionClosedException:
            print(colored(f"WebSocket connection closed unexpectedly for prompt {prompt_id}. Attempting to reconnect...", "red"))
            try:
                ws.close()
            except:
                pass
            ws = reconnect_websocket() # Similar issue as above with ws instance.
            print(colored("Reconnected. Will retry receiving for current prompt if possible, or fail.", "yellow"))

        except Exception as e:
            print(colored(f"Error receiving message from ComfyUI for prompt {prompt_id}: {e}", "red"))
            return False # Indicate error

    # Should not be reached if loop is truly infinite on recv, but as a fallback:
    return False


# Global server_address and client_id are assumed to be defined as they are used by helper functions.
# server_address = "54.80.78.81:8253"
# client_id = str(uuid.uuid4()) # This should be initialized once per script run ideally.

COMFYUI_WORKFLOW_FILE = "/home/ubuntu/crewgooglegemini/0001comfy2/720workflow003(API).json" # Assumed workflow for video scene generation
GENERATED_VIDEO_SCENES_DIR = "generated_video_scenes"


def run_video_generation_workflow(scene_prompts_data):
    """
    Generates video scenes using ComfyUI for each scene's video prompt.
    - Queues all prompts first.
    - Then, monitors and downloads each completed video.
    - Returns a list of paths to the generated video files.
    """
    if not scene_prompts_data:
        print(colored("No scene data provided to run_video_generation_workflow.", "yellow"))
        return []

    os.makedirs(GENERATED_VIDEO_SCENES_DIR, exist_ok=True)
    # print(colored(f"Ensured '{GENERATED_VIDEO_SCENES_DIR}' directory exists.", "green")) # Less verbose

    # Initialize WebSocket connection (client_id should be managed globally or passed)
    # For now, use the global client_id. This might need refinement if run in threads/processes.
    global client_id
    if not client_id: # Initialize if not already set (e.g. by process_and_generate_images)
        client_id = str(uuid.uuid4())
        print(colored(f"Initialized global client_id for ComfyUI: {client_id}", "magenta"))

    ws = websocket.WebSocket()
    ws_url = f"ws://{server_address}/ws?clientId={client_id}"
    try:
        print(colored(f"Connecting to ComfyUI WebSocket: {ws_url}", "cyan"))
        ws.connect(ws_url)
        print(colored("Successfully connected to ComfyUI WebSocket.", "green"))
    except Exception as e:
        print(colored(f"Failed to connect to ComfyUI WebSocket: {e}", "red"))
        return

    queued_jobs = [] # To store {"prompt_id": ..., "video_prompt_text": ..., "scene_index": ...}
    generated_video_paths = [] # To store paths of successfully generated videos

    print(colored(f"Queueing {len(scene_prompts_data)} video generation jobs with ComfyUI...", "cyan"))
    for i, scene_data_item in enumerate(scene_prompts_data):
        video_prompt_text = scene_data_item.get("video_prompt")
        if not video_prompt_text:
            print(colored(f"Skipping scene {i+1} due to missing 'video_prompt'. Data: {scene_data_item}", "yellow"))
            continue

        try:
            # The existing load_workflow takes positive and negative text prompts.
            # For this step, we only use the video_prompt.
            # The `load_workflow` in the main script is:
            #   def load_workflow(positive_prompt, negative_prompt):
            #       with open("/home/ubuntu/crewgooglegemini/0001comfy2/720workflow003(API).json", "r", encoding="utf-8") as f:
            #           workflow = json.load(f)
            #       workflow["4"]["inputs"]["text"] = positive_prompt
            #       workflow["5"]["inputs"]["text"] = negative_prompt
            #       return workflow
            # So, we pass the video_prompt_text as positive, and a generic negative.
            workflow = load_workflow(video_prompt_text, "text, watermark, signature, bad quality, low resolution")

            response = queue_prompt(workflow) # queue_prompt uses global client_id
            if response and 'prompt_id' in response:
                prompt_id = response['prompt_id']
                queued_jobs.append({"prompt_id": prompt_id, "video_prompt_text": video_prompt_text, "scene_index": i})
                # print(colored(f"Queued job for scene {i+1}/{len(scene_prompts_data)}: prompt_id {prompt_id}", "green")) # Less verbose
            else:
                print(colored(f"Failed to queue job for scene {i+1}. Response: {response}", "red"))
        except Exception as e:
            print(colored(f"Error queueing job for scene {i+1} ('{video_prompt_text[:50]}...'): {e}", "red"))

        time.sleep(0.2) # Shorter delay between queueing, as ComfyUI handles a queue.

    print(colored(f"\nAll {len(queued_jobs)} applicable jobs queued. Now awaiting completion and downloading...", "cyan"))

    for job_info in queued_jobs:
        prompt_id = job_info["prompt_id"]
        scene_index = job_info["scene_index"] # Original index from scene_prompts_data
        # print(colored(f"Awaiting completion of job for scene {scene_index + 1} (prompt_id: {prompt_id})...", "cyan")) # Less verbose

        completion_success = await_comfy_job_completion(ws, prompt_id) # Pass ws

        if completion_success:
            print(colored(f"Job {prompt_id} (scene {scene_index + 1}) completed. Fetching output files...", "green"))
            output_files_map = get_comfy_output_files_data(prompt_id) # Fetches content

            saved_file_this_scene = False
            for node_id, files_list in output_files_map.items():
                for file_data in files_list:
                    # We expect video files. Extension might be in filename or we guess.
                    # Example: save first video file found for the scene.
                    # filename could be like "ComfyUI_00001-.mp4"
                    file_extension = ".mp4" # Default, try to get from filename if possible
                    if '.' in file_data['filename']:
                        file_extension = "." + file_data['filename'].split('.')[-1]

                    # Prioritize .mp4 or other common video formats if type is generic like 'output'
                    if file_data['type'] == 'output' and file_extension not in ['.mp4', '.webm', '.mkv', '.gif']:
                        # If type is just 'output' and extension isn't video-like, it might be an image preview or other data.
                        # print(colored(f"Skipping non-video-like output file '{file_data['filename']}' for scene {scene_index + 1}", "yellow"))
                        # continue # Only save if it looks like a video
                        pass # For now, let's try to save it anyway and see.

                    output_filename = f"scene_{scene_index + 1:03d}{file_extension}"
                    output_filepath = os.path.join(GENERATED_VIDEO_SCENES_DIR, output_filename)
                    try:
                        with open(output_filepath, "wb") as f:
                            f.write(file_data["content"])
                        print(colored(f"Video for scene {scene_index + 1} saved: {output_filepath}", "blue"))
                        generated_video_paths.append(output_filepath) # Store path
                        saved_file_this_scene = True
                        break # Process only the first valid file for this scene's prompt_id
                    except Exception as e:
                        print(colored(f"Error saving file {output_filepath} for scene {scene_index + 1}: {e}", "red"))
                if saved_file_this_scene:
                    break

            if not saved_file_this_scene:
                 print(colored(f"No suitable video output file found or saved for prompt_id {prompt_id} (scene {scene_index + 1}).", "red"))
        else:
            print(colored(f"Job for prompt_id {prompt_id} (scene {scene_index + 1}) failed, was interrupted, or timed out.", "red"))

    try:
        ws.close()
        # print(colored("WebSocket connection closed.", "cyan")) # Less verbose
    except Exception as e:
        print(colored(f"Error closing WebSocket: {e}", "yellow"))

    if len(generated_video_paths) != len(queued_jobs):
        print(colored(f"Warning: Expected {len(queued_jobs)} videos, but only {len(generated_video_paths)} were successfully generated and saved.", "yellow"))

    print(colored(f"run_video_generation_workflow finished. {len(generated_video_paths)} videos saved to '{GENERATED_VIDEO_SCENES_DIR}'.", "green"))
    return generated_video_paths


GENERATED_VIDEO_SCENES_WITH_MMAUDIO_DIR = "generated_video_scenes_with_mmaudio"

def run_mmaudio_enhancement_workflow(video_scene_paths, scene_audio_data_list, mmaudio_workflow_path):
    """
    Enhances video scenes with audio using a specified MMAudio ComfyUI workflow.

    Args:
        video_scene_paths (list): List of file paths to the input video scenes.
        scene_audio_data_list (list): List of dicts, each with "audio_prompt",
                                      "negative_audio_prompt", "vocals_instruction".
        mmaudio_workflow_path (str): Path to the ComfyUI MMAudio workflow JSON file.

    Returns:
        list: A list of file paths to the audio-enhanced video scenes.
    """
    if not video_scene_paths:
        print(colored("No video scene paths provided for MMAudio enhancement.", "yellow"))
        return []
    if len(video_scene_paths) != len(scene_audio_data_list):
        print(colored(f"Mismatch between number of video paths ({len(video_scene_paths)}) and audio data entries ({len(scene_audio_data_list)}). Aborting MMAudio.", "red"))
        return []

    os.makedirs(GENERATED_VIDEO_SCENES_WITH_MMAUDIO_DIR, exist_ok=True)

    global client_id, server_address # Assuming these are set globally
    if not client_id: client_id = str(uuid.uuid4())

    ws = websocket.WebSocket()
    ws_url = f"ws://{server_address}/ws?clientId={client_id}"
    try:
        ws.connect(ws_url)
    except Exception as e:
        print(colored(f"Failed to connect to ComfyUI WebSocket for MMAudio: {e}", "red"))
        return []

    enhanced_video_paths = []

    print(colored(f"Starting MMAudio enhancement for {len(video_scene_paths)} video scenes...", "cyan"))

    for i, video_path in enumerate(video_scene_paths):
        if not os.path.exists(video_path):
            print(colored(f"Video file not found: {video_path}. Skipping MMAudio for scene {i+1}.", "red"))
            continue

        audio_data = scene_audio_data_list[i]
        positive_audio_prompt = audio_data.get("audio_prompt", "")
        negative_audio_prompt = audio_data.get("negative_audio_prompt", "")
        # vocals_instruction = audio_data.get("vocals_instruction", "") # Not directly used yet unless workflow has a node for it

        print(colored(f"\nProcessing scene {i+1}/{len(video_scene_paths)} for MMAudio: {os.path.basename(video_path)}", "blue"))
        print(colored(f"  Audio Prompt: {positive_audio_prompt}", "magenta"))
        print(colored(f"  Negative AP: {negative_audio_prompt}", "magenta"))

        # 1. Upload the video file for the current scene
        # Using subfolder to keep inputs organized on ComfyUI server side, if supported well by nodes
        upload_subfolder = "mmaudio_inputs"
        video_upload_resp = upload_file_to_comfyui(video_path, server_address, subfolder_name=upload_subfolder)
        if not video_upload_resp or 'name' not in video_upload_resp:
            print(colored(f"Failed to upload video {video_path} for MMAudio. Skipping scene {i+1}.", "red"))
            continue

        uploaded_video_name = video_upload_resp['name']
        if video_upload_resp.get('subfolder'):
            uploaded_video_name = f"{video_upload_resp['subfolder']}/{uploaded_video_name}"

        # 2. Load and Patch the MMAudio workflow
        try:
            with open(mmaudio_workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            print(colored(f"Error loading MMAudio workflow '{mmaudio_workflow_path}': {e}. Skipping scene {i+1}.", "red"))
            continue

        # --- PATCHING LOGIC ---
        # This is highly dependent on the structure of `vid_mmaudio.json`
        # User needs to verify these class_types and input field names.
        video_node_patched = False
        audio_prompt_node_patched = False
        neg_audio_prompt_node_patched = False

        for node_id, node_data in workflow.items():
            # Example: Patching video input node
            # Common class_types: LoadVideo, LoadVideoUpload, VideoFileInputNode, etc.
            # Common input fields: video, video_path, filename, file_path
            if node_data["class_type"] == "LoadVideo" or node_data["class_type"] == "VHS_LoadVideo": # Common custom node names
                node_data["inputs"]["video"] = uploaded_video_name # Assumes 'video' is the input field
                video_node_patched = True
                print(colored(f"  Patched MMAudio video input node '{node_id}' ({node_data['class_type']}) with: {uploaded_video_name}", "green"))

            # Example: Patching positive audio prompt node
            # Common class_types: PrimitiveNode, StringInputNode, MMAudioPromptNode, etc.
            # Common input fields: text, string, prompt, positive_prompt
            # Assuming a node for positive prompt, e.g., one that has a widget named 'text' or 'prompt'
            # This is a guess; a specific node class_type for prompts is better.
            if node_data["class_type"] == "CLIPTextEncode" and "text" in node_data["inputs"]: # A common pattern for text inputs
                 # Check if this node is likely for audio by looking at connected nodes or a conventional title in actual workflow
                 # For now, this is a placeholder. If MMAudio has specific prompt nodes, use their class_type.
                 # Let's assume there's a node specifically for the positive audio prompt.
                 # Placeholder: node_data["_meta"]["title"] == "Positive MMAudio Prompt"
                 # This requires the user to title their nodes in ComfyUI if we use _meta.title.
                 # A more robust way is to agree on a specific class_type or specific input name for audio prompts.
                 # For now, let's assume a specific node ID is known or a unique class_type for MMAudio prompts.
                 # If we assume node "10" is for positive audio prompt (hypothetical):
                 if node_id == "10": # HYPOTHETICAL NODE ID FOR POSITIVE AUDIO PROMPT
                     node_data["inputs"]["text"] = positive_audio_prompt
                     audio_prompt_node_patched = True
                     print(colored(f"  Patched positive audio prompt for node '{node_id}' with: {positive_audio_prompt[:50]}...", "green"))

            # Example: Patching negative audio prompt node
            # Similar assumptions as positive prompt. If node "11" is for negative (hypothetical):
            if node_id == "11": # HYPOTHETICAL NODE ID FOR NEGATIVE AUDIO PROMPT
                node_data["inputs"]["text"] = negative_audio_prompt
                neg_audio_prompt_node_patched = True
                print(colored(f"  Patched negative audio prompt for node '{node_id}' with: {negative_audio_prompt[:50]}...", "green"))

        if not video_node_patched:
            print(colored("  Warning: Could not find or patch the video input node in MMAudio workflow. Please check class_type/input field.", "yellow"))
        if not audio_prompt_node_patched:
            print(colored("  Warning: Could not find or patch the positive audio prompt node. Please check class_type/input field or node ID.", "yellow"))
        # Negative prompt is optional, so less critical if not patched.

        # 3. Queue the MMAudio job
        mmaudio_response = queue_prompt(workflow) # Uses global client_id
        if not mmaudio_response or 'prompt_id' not in mmaudio_response:
            print(colored(f"Failed to queue MMAudio job for scene {i+1}. Response: {mmaudio_response}", "red"))
            continue

        mmaudio_prompt_id = mmaudio_response['prompt_id']
        print(colored(f"  Queued MMAudio job for scene {i+1}: prompt_id {mmaudio_prompt_id}", "green"))

        # 4. Monitor and Download
        print(colored(f"  Awaiting MMAudio completion for prompt_id: {mmaudio_prompt_id}...", "cyan"))
        mmaudio_completion_success = await_comfy_job_completion(ws, mmaudio_prompt_id)

        if mmaudio_completion_success:
            print(colored(f"  MMAudio job {mmaudio_prompt_id} (scene {i+1}) completed. Fetching output...", "green"))
            output_files_map = get_comfy_output_files_data(mmaudio_prompt_id)

            saved_enhanced_video = False
            for node_id_out, files_list_out in output_files_map.items():
                for file_data_out in files_list_out:
                    file_extension_out = ".mp4"
                    if '.' in file_data_out['filename']:
                        file_extension_out = "." + file_data_out['filename'].split('.')[-1]

                    # Assume the output is a video file
                    output_filename_enhanced = f"scene_{i+1:03d}_mmaudio{file_extension_out}"
                    output_filepath_enhanced = os.path.join(GENERATED_VIDEO_SCENES_WITH_MMAUDIO_DIR, output_filename_enhanced)
                    try:
                        with open(output_filepath_enhanced, "wb") as f:
                            f.write(file_data_out["content"])
                        print(colored(f"  Successfully saved MMAudio enhanced video: {output_filepath_enhanced}", "blue"))
                        enhanced_video_paths.append(output_filepath_enhanced)
                        saved_enhanced_video = True
                        break
                    except Exception as e:
                        print(colored(f"  Error saving MMAudio enhanced file {output_filepath_enhanced}: {e}", "red"))
                if saved_enhanced_video:
                    break

            if not saved_enhanced_video:
                 print(colored(f"  No suitable output video found or saved from MMAudio job {mmaudio_prompt_id} (scene {i+1}).", "red"))
        else:
            print(colored(f"  MMAudio job for prompt_id {mmaudio_prompt_id} (scene {i+1}) failed or was interrupted.", "red"))

        time.sleep(1) # Small delay between processing each scene for MMAudio

    try:
        ws.close()
    except Exception as e:
        print(colored(f"Error closing WebSocket after MMAudio: {e}", "yellow"))

    print(colored(f"MMAudio enhancement workflow finished. {len(enhanced_video_paths)} videos processed and saved to '{GENERATED_VIDEO_SCENES_WITH_MMAUDIO_DIR}'.", "green"))
    return enhanced_video_paths


def process_and_generate_images():
    output_dir = "/home/ubuntu/crewgooglegemini/0001comfy2/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load segment IDs from image_video.json
    segment_ids = []
    with open("/home/ubuntu/crewgooglegemini/001videototxt/transcripts/image_video.json", "r", encoding="utf-8") as seg_file:
        segments_data = json.load(seg_file)
        segment_ids = [segment["segment_id"] for segment in segments_data]

    print(colored(f"Loaded {len(segment_ids)} segment IDs.", "cyan"))

    # Load current prompts
    with open(CURRENT_PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    ws = websocket.WebSocket()
    ws_url = f"ws://{server_address}/ws?clientId={client_id}"
    print(colored(f"Establishing WebSocket connection to {ws_url}", "cyan"))
    ws.connect(ws_url)

    queued_prompts = []
    for i, prompt in enumerate(prompt_data):
        positive_prompt = prompt["image_prompt"]
        negative_prompt = prompt["negative_prompt"]

        print(colored(f"\nQueuing prompt {i+1}/{len(prompt_data)}", "cyan"))

        workflow = load_workflow(positive_prompt, negative_prompt)
        response = queue_prompt(workflow)
        if response and 'prompt_id' in response:
            queued_prompts.append((response['prompt_id'], positive_prompt, negative_prompt))
        
        time.sleep(1)  # 1 second interval between queueing prompts

    print(colored("All prompts queued. Processing...", "green"))

    for i, (prompt_id, positive_prompt, negative_prompt) in enumerate(queued_prompts):
        print(colored(f"\nProcessing prompt {i+1}/{len(queued_prompts)}", "cyan"))
        images = generate_images(ws, prompt_id)

        print(colored("Saving the generated images locally.", "cyan"))

        # Get the corresponding segment_id for this image (based on index)
        try:
            segment_id = segment_ids[i]
        except IndexError:
            segment_id = f"segment_unknown_{i}"
            print(colored(f"Warning: No segment_id for prompt {i+1}, using {segment_id}", "red"))

        for node_id, image_list in images.items():
            for j, image_data in enumerate(image_list):
                try:
                    image = Image.open(io.BytesIO(image_data))
                    # Save image using segment_id only
                    filename = os.path.join(output_dir, f"{segment_id}.png")
                    image.save(filename)
                    print(colored(f"Image saved as {filename}", "blue"))
                except Exception as e:
                    print(colored(f"Error processing image: {e}", "red"))

        log_entry = {
            "segment_id": segment_id,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry, indent=4) + "\n\n")

        print(colored("Prompt and vid transcript logged successfully.", "green"))

    ws.close()
    print(colored("All prompts processed and images generated.", "green"))


def load_workflow(positive_prompt, negative_prompt):
    with open("/home/ubuntu/crewgooglegemini/0001comfy2/720workflow003(API).json", "r", encoding="utf-8") as f:
        workflow = json.load(f)
    workflow["4"]["inputs"]["text"] = positive_prompt
    workflow["5"]["inputs"]["text"] = negative_prompt
    return workflow

def queue_prompt(workflow):
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p, indent=4).encode("utf-8")
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)

    print(colored(f"Queuing the prompt for client ID {client_id}.", "cyan"))

    try:
        response = json.loads(urllib.request.urlopen(req).read())
        return response
    except Exception as e:
        print(colored(f"Error sending prompt: {e}", "red"))
        return {}

def upload_file_to_comfyui(local_file_path, server_address_param, subfolder_name=None, file_type="input"):
    """
    Uploads a file to the ComfyUI server via HTTP API.

    Args:
        local_file_path (str): The path to the local file to upload.
        server_address_param (str): The ComfyUI server address (e.g., "127.0.0.1:8188").
        subfolder_name (str, optional): The name of the subfolder on the ComfyUI server
                                        within the 'input' directory. Defaults to None.
        file_type (str, optional): The type of file for ComfyUI (e.g., 'input', 'temp').
                                   This often determines the root directory on the server.
                                   The API seems to use `type` for subfolder context too.

    Returns:
        dict: The JSON response from ComfyUI (e.g., {'name': 'filename.ext',
              'subfolder': 'subfolder_name_if_any', 'type': 'input'})
              or None if upload fails.
    """
    url = f"http://{server_address_param}/upload/image" # Endpoint seems generic
    filename = os.path.basename(local_file_path)

    files_payload = {'image': (filename, open(local_file_path, 'rb'))}

    data_payload = {'overwrite': 'true'} # Allow overwriting existing files
    if subfolder_name:
        data_payload['subfolder'] = subfolder_name
    # The 'type' (e.g. 'input', 'temp') is sometimes used by ComfyUI to determine
    # the root directory for the subfolder. The /upload/image endpoint itself
    # might implicitly use 'input' or allow 'type' to be specified here.
    # For file inputs to nodes, 'input' is typical.
    data_payload['type'] = file_type

    print(colored(f"Uploading '{local_file_path}' to ComfyUI ({url}) with data: {data_payload}", "cyan"))

    try:
        if HAS_REQUESTS:
            resp = requests.post(url, files=files_payload, data=data_payload, timeout=60) # Increased timeout for larger files
            resp.raise_for_status()
            response_json = resp.json()
        else:
            # Fallback to urllib (adapted from user's example)
            import mimetypes
            boundary = "----WebKitFormBoundary" + os.urandom(16).hex()

            body_parts = []
            # Add data fields (overwrite, subfolder, type)
            for key, value in data_payload.items():
                body_parts.append(f"--{boundary}".encode('utf-8'))
                body_parts.append(f'Content-Disposition: form-data; name="{key}"\r\n'.encode('utf-8'))
                body_parts.append(value.encode('utf-8'))

            # Add file field
            body_parts.append(f"--{boundary}".encode('utf-8'))
            content_type = mimetypes.guess_type(local_file_path)[0] or "application/octet-stream"
            file_header = f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n' \
                          f'Content-Type: {content_type}\r\n'
            body_parts.append(file_header.encode('utf-8'))

            with open(local_file_path, "rb") as f:
                filedata = f.read()
            body_parts.append(filedata)

            body_parts.append(f"--{boundary}--".encode("utf-8"))

            final_body = b"\r\n".join(body_parts)

            headers = {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(final_body)),
            }
            req = urllib.request.Request(url, data=final_body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as response: # Increased timeout
                response_json = json.loads(response.read().decode())

        print(colored(f"Successfully uploaded '{filename}'. Server response: {response_json}", "green"))
        return response_json

    except Exception as e:
        print(colored(f"Error uploading file '{local_file_path}' to ComfyUI: {e}", "red"))
        if HAS_REQUESTS and 'resp' in locals() and hasattr(resp, 'text'):
            print(colored(f"Server error response: {resp.text}", "red"))
        return None

import os
from moviepy.editor import ImageClip
from PIL import Image

def make_fullscreen_imageclip(image_path, output_size):
    w, h = output_size
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    # If already correct size, just load it directly, no resize/crop
    if (img_w, img_h) == (w, h):
        return ImageClip(image_path)

    # If not, do the usual process
    return (
        ImageClip(image_path)
        .resize(height=h)
        .resize(width=w)
        .crop(x_center=w/2, y_center=h/2, width=w, height=h)
    )

def fadein_effect(image_clip, duration):
    print(f"  [EFFECT] fadein ({duration}s)")
    return image_clip.fx(vfx.fadein, duration).set_duration(duration)


def color_shift_effect(image_clip, duration):
    #print(f"  [EFFECT] color_shift ({duration}s)")
    def brighten(get_frame, t):
        frame = get_frame(t).astype(float)
        factor = 1 + 0.5 * np.sin(2 * np.pi * t / duration)
        frame = np.clip(frame * factor, 0, 255)
        return frame.astype(np.uint8)
    return image_clip.fl(brighten).set_duration(duration)

def ken_burns_effect(image_clip, duration, zoom_start, zoom_end, pan_direction, image_size=(720, 1280)):
    #print(f"  [EFFECT] ken_burns {pan_direction} zoom {zoom_start}->{zoom_end} ({duration}s)")
    pan_options = {
        'top_left-to-bottom_right': ((0, 0), (1, 1)),
        'bottom_right-to_top_left': ((1, 1), (0, 0)),
        'left-to-right': ((0, 0.5), (1, 0.5)),
        'top-to-bottom': ((0.5, 0), (0.5, 1)),
        'center-in': ((0.5, 0.5), (0.5, 0.5)),
        'right-to-left': ((1, 0.5), (0, 0.5)),
        'bottom-to-top': ((0.5, 1), (0.5, 0)),
        'top_right-to-bottom_left': ((1, 0), (0, 1)),
        'bottom_left-to_top_right': ((0, 1), (1, 0)),
    }
    start_rel, end_rel = pan_options.get(pan_direction, ((0.5, 0.5), (0.5, 0.5)))
    w, h = image_size
    def crop_func(get_frame, t):
        progress = t / duration
        zoom = zoom_start + (zoom_end - zoom_start) * progress
        crop_w = w / zoom
        crop_h = h / zoom
        x_rel = start_rel[0] + (end_rel[0] - start_rel[0]) * progress
        y_rel = start_rel[1] + (end_rel[1] - start_rel[1]) * progress
        x_center = int(w * x_rel)
        y_center = int(h * y_rel)
        x1 = max(0, x_center - crop_w // 2)
        y1 = max(0, y_center - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        frame = get_frame(t)
        cropped = frame[int(y1):int(y2), int(x1):int(x2)]
        return np.array(ImageClip(cropped).resize((w, h)).get_frame(0))
    return VideoClip(lambda t: crop_func(image_clip.get_frame, t), duration=duration)

def shake_effect(image_clip, duration, intensity=1):
    #print(f"  [EFFECT] shake (intensity={intensity}, {duration}s)")
    w, h = image_clip.size
    def make_frame(t):
        dx = random.randint(-intensity, intensity)
        dy = random.randint(-intensity, intensity)
        frame = image_clip.get_frame(t)
        padded = np.pad(frame, ((intensity, intensity), (intensity, intensity), (0, 0)), mode='edge')
        y1 = intensity + dy
        x1 = intensity + dx
        return padded[y1:y1 + h, x1:x1 + w]
    return VideoClip(make_frame, duration=duration)

def animate_photo(image_path, image_size, ken_burns_variants, duration_kb=10.0):
    #print(f"[ANIMATION] {os.path.basename(image_path)}")
    base_clip = make_fullscreen_imageclip(image_path, image_size)
    clip = fadein_effect(base_clip, 0.5)
    kb_variant = random.choice(ken_burns_variants)
    clip = ken_burns_effect(clip, duration_kb, *kb_variant, image_size)
    if random.random() < 0.5:
        clip = color_shift_effect(clip, duration_kb)
    if random.random() < 0.5:
        clip = shake_effect(clip, duration_kb, intensity=1)
    # if random.random() < 0.2:
    #     clip = rotation_effect(clip, 0.2)
    # # Always enforce duration at the end
    clip = clip.set_duration(duration_kb)
    return clip

def step_5e_generate_video_clips():
    input_dir = "/home/ubuntu/crewgooglegemini/0001comfy2/outputs"
    output_dir = "/home/ubuntu/crewgooglegemini/SHORTCLIPSFACTS/WellnessGram"
    segment_json_path = "/home/ubuntu/crewgooglegemini/001videototxt/transcripts/image_video.json"

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist!")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(segment_json_path):
        print(f"Segment JSON {segment_json_path} does not exist!")
        return

    with open(segment_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    ken_burns_variants = [
        (1.0, 1.2, 'top_left-to-bottom_right'),
        (1.2, 1.0, 'bottom_right-to_top_left'),
        (1.0, 1.3, 'left-to-right'),
        (1.3, 1.0, 'top-to-bottom'),
        (1.0, 1.2, 'center-in'),
        (1.2, 1.0, 'right-to-left'),
        (1.0, 1.2, 'bottom_left-to_top_right'),
        (1.2, 1.0, 'top_right-to-bottom_left'),
    ]

    used_dir = os.path.join(input_dir, "used_images")
    os.makedirs(used_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')

    for segment in segments:
        image_id = segment['segment_id']
        duration = segment['duration']
        image_path = None
        for ext in image_extensions:
            candidate = os.path.join(input_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            print(f"Image for segment {image_id} not found.")
            continue

        #print(f"[SEGMENT] Animating {segment['segment_id']} ({os.path.basename(image_path)}) for {duration:.2f}s")
        try:
            clip = animate_photo(image_path, (720, 1280), ken_burns_variants, duration)
            clip = clip.set_duration(duration)  # Enforce duration
            out_path = os.path.join(output_dir, f"{image_id}.mp4")
            clip.write_videofile(
                out_path,
                fps=24,
                codec='libx264',
                audio=False,
                threads=4,
                preset='medium',
                ffmpeg_params=['-crf', '18']
            )
            print(f"Saved animated video for {image_id}: {out_path} (duration: {clip.duration:.2f}s)")
            dest_path = os.path.join(used_dir, os.path.basename(image_path))
            shutil.move(image_path, dest_path)
        except Exception as e:
            print(f"Error animating {image_id}: {e}")

    print("All images animated and saved as individual video clips.")





def log_transcript(transcript_text, new_video_title):
    log_entry = {
        "video_title": new_video_title,
        "transcript": transcript_text,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(VIDEO_TRANSCRIPT_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry, indent=4) + "\n\n")



from telegram import Bot, InputMediaPhoto

def step_5f_send_to_telegram():
    # Telegram Bot Configuration
    BOT_TOKEN = '6157935666:AAESXcHywVwdHZqurjz0kCcVTjzCv50gjlQ'
    CHAT_ID = '5034393732'

    FINAL_VIDEO_DIR = '/home/ubuntu/crewgooglegemini/FINALVIDEOS/WellnessGram'
    TRANSCRIPT_PATH = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json'
    THUMBNAIL_DIR = '/home/ubuntu/crewgooglegemini/CAPTACITY/assets/thumbnails'

    # Find the most recent video file in the FINAL_VIDEO_DIR
    if not os.path.exists(FINAL_VIDEO_DIR):
        print(f"Error: Directory {FINAL_VIDEO_DIR} does not exist.")
        return

    video_files = [f for f in os.listdir(FINAL_VIDEO_DIR) if f.endswith(".mp4")]
    if not video_files:
        print("Error: No video files found in the final video directory.")
        return

    latest_video_file = max(video_files, key=lambda f: os.path.getmtime(os.path.join(FINAL_VIDEO_DIR, f)))
    video_path = os.path.join(FINAL_VIDEO_DIR, latest_video_file)
    video_title = os.path.splitext(latest_video_file)[0]

    # Read the transcript from the JSON
    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"Error: Transcript log file {TRANSCRIPT_PATH} does not exist.")
        return

    with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        try:
            transcript_json = json.load(f)
        except json.JSONDecodeError:
            print(f"Transcript file {TRANSCRIPT_PATH} is not valid JSON.")
            return

    transcript_text = transcript_json.get('script', '')

    transcript_message = (
        f"Title: {video_title}\n\n"
        f"Transcript:\n{transcript_text}"
    )

    # --- Find and Rename Thumbnails to Match Final Video Base Name ---
    thumbnail_paths = []
    for size in ["720X1280"]:
        pattern = os.path.join(THUMBNAIL_DIR, f"*_{size}.png")
        candidates = glob.glob(pattern)
        if candidates:
            latest_thumb = max(candidates, key=os.path.getmtime)
            new_thumb_name = os.path.join(THUMBNAIL_DIR, f"{video_title}_thumbnail_{size}.png")
            if os.path.abspath(latest_thumb) != os.path.abspath(new_thumb_name):
                shutil.copy2(latest_thumb, new_thumb_name)
                print(f"Copied thumbnail {latest_thumb} -> {new_thumb_name}")
            else:
                print(f"Thumbnail already named: {new_thumb_name}")
            thumbnail_paths.append(new_thumb_name)
        else:
            print(f"No thumbnail found for size {size}. Skipping.")

    # Send to Telegram
    asyncio.run(send_telegram_message(
        BOT_TOKEN, CHAT_ID, video_path, video_title, transcript_message, thumbnail_paths=thumbnail_paths
    ))

async def send_telegram_message(bot_token, chat_id, video_path, video_title, transcript_message, thumbnail_paths=None):
    try:
        bot = Bot(token=bot_token)

        # Send the video
        with open(video_path, 'rb') as video:
            await bot.send_video(chat_id=chat_id, video=video, caption=f"Video Title: {video_title}")

        # Send the transcript as a text message
        await bot.send_message(chat_id=chat_id, text=transcript_message)

        # Send thumbnails as a media group if provided
        if thumbnail_paths:
            media = []
            file_handles = []
            for path in thumbnail_paths:
                f = open(path, "rb")
                file_handles.append(f)
                media.append(InputMediaPhoto(f))
            await bot.send_media_group(chat_id=chat_id, media=media)
            for f in file_handles:
                f.close()
            print("Thumbnails sent to Telegram.")

        print("Video, transcript, and thumbnails sent successfully to Telegram.")

    except Exception as e:
        print(f"An error occurred while sending to Telegram: {e}")


# You can call this function in your main script just like your other functions
# step_5f_send_to_telegram()


def get_next_link():
    new_links_file = "/home/ubuntu/crewgooglegemini/AUTO/shortsnewlinks.txt"
    
    if not os.path.exists(new_links_file):
        return None, None
    
    with open(new_links_file, 'r') as file:
        lines = file.readlines()
    
    if not lines:
        return None, None
    
    youtube_link = lines[0].strip()
    return youtube_link, new_links_file



def make_safe_filename(name):
    # Replace any character that is not alphanumeric, a hyphen, or underscore with an underscore
    return re.sub(r'[^A-Za-z0-9-_]', '_', name)

# Global variable to track the current background sound index
current_bg_sound_index = 0

def get_background_sound(audio_duration: float) -> AudioFileClip:
    """
    Get a background sound clip that matches the duration of the main audio.

    Parameters:
        audio_duration (float): The duration of the main audio in seconds.

    Returns:
        AudioFileClip: A background sound clip adjusted to match the main audio duration.
    """
    global current_bg_sound_index

    # Background sounds folder
    bg_sounds_folder = "/home/ubuntu/crewgooglegemini/CAPTACITY/assets/bgsound"

    # List all available background sound files in the folder
    bg_sound_files = [
        os.path.join(bg_sounds_folder, f)
        for f in os.listdir(bg_sounds_folder)
        if f.endswith(".mp3") or f.endswith(".wav")
    ]

    # Ensure there are background sounds available
    if not bg_sound_files:
        raise ValueError("No background sound files found in the specified folder.")

    # Select the current background sound file
    bg_sound_path = bg_sound_files[current_bg_sound_index]

    # Update the index for cyclic behavior
    current_bg_sound_index = (current_bg_sound_index + 1) % len(bg_sound_files)

    # Load the background sound and adjust its volume
    bg_sound = AudioFileClip(bg_sound_path).volumex(0.07)

    # Adjust the background sound duration to match the main audio duration
    if bg_sound.duration < audio_duration:
        # Repeat the background sound to cover the entire duration
        bg_sound_repeated = [bg_sound] * (int(audio_duration // bg_sound.duration) + 1)
        bg_sound = concatenate_audioclips(bg_sound_repeated).subclip(0, audio_duration)
        print("we repeated the audio to match video")
    else:
        # Trim the background sound to match the main audio duration
        bg_sound = bg_sound.subclip(0, audio_duration)
        print("we trimmed the audio to match video")

    return bg_sound



async def process_generated_content(new_video_title, script, base_output_dir_name, process_option_type):
    """
    Processes generated content to create a final video with voiceover, captions, and images/stock videos.
    """
    print(f"Starting process_generated_content for: {new_video_title} (Option: {process_option_type})")

    # 1. Path Setup
    VOICE_OVER_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/voice_over"
    TRANSCRIPT_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/transcripts"
    IMAGE_PROMPT_OUTPUT_FILE = '/home/ubuntu/crewgooglegemini/current_prompt.json'
    IMAGE_OUTPUT_DIR = "/home/ubuntu/crewgooglegemini/0001comfy2/outputs" # Base for Comfy outputs
    VIDEO_CLIP_OUTPUT_DIR = f"/home/ubuntu/crewgooglegemini/SHORTCLIPSFACTS/{base_output_dir_name}" # CUSTOM_FOLDER for image-generated clips
    FINAL_VIDEO_DIR = f"/home/ubuntu/crewgooglegemini/FINALVIDEOS/{base_output_dir_name}"
    STOCK_VIDEOS_DIR = "/home/ubuntu/crewgooglegemini/PIXABAY VIDEOS" # Fallback or primary for non-image based
    FULL_SCRIPT_JSON_PATH = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json' # To store the script

    ensure_folder_exists(VOICE_OVER_FOLDER)
    ensure_folder_exists(TRANSCRIPT_FOLDER)
    ensure_folder_exists(IMAGE_OUTPUT_DIR)
    ensure_folder_exists(VIDEO_CLIP_OUTPUT_DIR)
    ensure_folder_exists(FINAL_VIDEO_DIR)
    ensure_folder_exists(STOCK_VIDEOS_DIR)
    # Parent directory for IMAGE_PROMPT_OUTPUT_FILE and FULL_SCRIPT_JSON_PATH should exist or be ensured by calling functions.
    ensure_folder_exists(os.path.dirname(IMAGE_PROMPT_OUTPUT_FILE))
    ensure_folder_exists(os.path.dirname(FULL_SCRIPT_JSON_PATH))

    print("Paths configured and ensured.")

     # 2. Log Transcript (if provided)
    if transcript_text:
        log_transcript(transcript_text, new_video_title)
        print(colored("Transcript logged successfully.", "green"))
        
    # 2. Log Transcript (Assumed to be done *before* this function is called if it's the original video transcript)
    # If the `script` parameter itself needs to be logged as a "transcript", that would happen here.
    # For now, following the prompt, `log_transcript` is called outside with the original transcript.

    # # 3. Audio Generation
    # audio_path = os.path.join(VOICE_OVER_FOLDER, "voiceover.wav")
    # print(f"Generating audio for script: '{script[:50]}...'")
    # await generate_audio(script, audio_path)
    # print(f"Audio generated and saved to {audio_path}")

    # 4. Local Transcription for Captions
    print("Transcribing audio locally for captions...")
    transcription = transcribe_locally(audio_path)
    transcript_filename = os.path.join(TRANSCRIPT_FOLDER, f"{os.path.basename(audio_path).replace('.wav', '')}_transcription.txt")
    with open(transcript_filename, 'w') as f:
        if isinstance(transcription, list):
            for segment_data in transcription: # Ensure correct variable name if transcription format changes
                words_json = json.dumps(segment_data.get('words', []), default=lambda x: float(x) if isinstance(x, np.float64) else x)
                f.write(f"{segment_data.get('start', 0.0)} {segment_data.get('end', 0.0)} {segment_data.get('text', '')} {words_json}\n")
        else:
            f.write(str(transcription)) # Fallback if not list
    print(f"Local transcription saved to {transcript_filename}")
    video_segments_for_captions = extract_transcript_segments(transcription) # This returns a tuple (segments, durations)
    # If only segments are needed:
    # video_segments_for_captions, _ = extract_transcript_segments(transcription)
    # Based on current usage of add_captions, it needs the 'segments' part of the tuple.
    # The function `extract_transcript_segments` in the notebook returns `segments,` (a tuple with one element).
    # So, we might need to adjust how we get the actual list of segments.
    # Assuming `extract_transcript_segments` is adjusted to return just the segments list or we take the first element.
    if isinstance(video_segments_for_captions, tuple):
        video_segments_for_captions = video_segments_for_captions[0] 
    print(f"Extracted {len(video_segments_for_captions)} segments for captions.")


    # 5. Image-Video Segments & Prompts
    print("Parsing transcript for image-video segments...")
    parsed_transcript_for_images = parse_transcript_file(transcript_filename)
    segments_for_image_video = extract_transcript_segments_image_vid(parsed_transcript_for_images)
    image_video_json_path = os.path.join(TRANSCRIPT_FOLDER, 'image_video.json')
    with open(image_video_json_path, 'w') as f:
        json.dump(segments_for_image_video, f, indent=4)
    print(f"Image-video segments saved to {image_video_json_path}")

    print(f"Saving full script to {FULL_SCRIPT_JSON_PATH} for prompt generation context...")
    with open(FULL_SCRIPT_JSON_PATH, 'w') as f:
        json.dump({"script": script}, f) # Save the passed script here

    image_prompts = []
    batch_size = 5 # As used in the main loop
    full_script_for_prompts = script # Use the passed script as context
    print(f"Generating image prompts in batches (size {batch_size})...")
    for i in range(0, len(segments_for_image_video), batch_size):
        batch = segments_for_image_video[i:i+batch_size]
        # generate_image_prompts_batch expects `full_transcript` as the second argument.
        batch_prompts = generate_image_prompts_batch(batch, full_script_for_prompts)
        image_prompts.extend(batch_prompts)
    with open(IMAGE_PROMPT_OUTPUT_FILE, 'w') as f:
        json.dump(image_prompts, f, indent=4)
    print(f"Generated {len(image_prompts)} image prompts and saved to {IMAGE_PROMPT_OUTPUT_FILE}")

    # 6. Image Generation
    print("Starting image generation process...")
    process_and_generate_images() # Reads from IMAGE_PROMPT_OUTPUT_FILE, saves to IMAGE_OUTPUT_DIR
    print("Image generation process completed.")

    
#=============================================================================================================================
   

    # 10. Final Video Output & Captioning
    safe_title = make_safe_filename(new_video_title)
    intermediate_output_path = os.path.join(FINAL_VIDEO_DIR, f"{safe_title}_intermediate.mp4")
    
    print(f"Writing intermediate video to {intermediate_output_path}...")
    final_video_assembly.write_videofile(
        intermediate_output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23', "-bufsize", "16M", "-maxrate", "8M", '-profile:v', 'high'],
        threads=8,
        #verbose=False, # Keep verbose for now, or make it conditional
        logger="bar" # Progress bar
    )
    print("Intermediate video written.")

    final_output_path_with_captions = os.path.join(FINAL_VIDEO_DIR, f"{safe_title}.mp4")
    print(f"Adding captions. Outputting to {final_output_path_with_captions}...")
    add_captions(
        video_file=intermediate_output_path,
        output_file=final_output_path_with_captions,
        font="Bangers-Regular.ttf",
        font_size=80,
        font_color="yellow",
        stroke_width=3,
        stroke_color="black",
        highlight_current_word=True,
        word_highlight_color="red",
        line_count=2,
        padding=50, # Original padding from add_captions call
        shadow_strength=1.0,
        shadow_blur=0.1,
        print_info=True,
        segments=video_segments_for_captions # Pass the loaded video segments
    )
    print("Captions added.")

    if os.path.exists(intermediate_output_path):
        os.remove(intermediate_output_path)
        print(f"Removed intermediate file: {intermediate_output_path}")

    # Clean up .mp4 files in VIDEO_CLIP_OUTPUT_DIR (CUSTOM_FOLDER for image clips)
    print(f"Cleaning up clips from {VIDEO_CLIP_OUTPUT_DIR}...")
    if os.path.exists(VIDEO_CLIP_OUTPUT_DIR):
        for file_item in os.listdir(VIDEO_CLIP_OUTPUT_DIR):
            if file_item.endswith(".mp4"):
                file_path_to_delete = os.path.join(VIDEO_CLIP_OUTPUT_DIR, file_item)
                os.remove(file_path_to_delete)
                # print(f"Removed clip: {file_path_to_delete}")
        print(f"Cleaned up .mp4 files in {VIDEO_CLIP_OUTPUT_DIR}.")
    
    print(f"Final video with captions saved as {final_output_path_with_captions}")

    # 11. Distribution
    print("Starting distribution step (Telegram and Google Drive)...")
    # Note: These functions might need adjustment if they rely on global state not available here,
    # or if they need specific paths passed (currently they find latest files in fixed dirs).
   
    
    # Quick check for this specific case:
    if base_output_dir_name == "WellnessGram":
        step_5f_send_to_telegram() # Assumes this function finds the latest video in the correct FINAL_VIDEO_DIR
        step_5g_upload_to_google_drive() # Same assumption
    else:
        print(f"Skipping Telegram and Google Drive distribution for {base_output_dir_name} as it's not 'WellnessGram'. Distribution functions might need updates for dynamic paths.")

    print(f"Processing for {new_video_title} completed successfully.")



def step_5g_upload_to_google_drive():
    FINAL_VIDEO_DIR = '/home/ubuntu/crewgooglegemini/FINALVIDEOS/WellnessGram'
    TRANSCRIPT_PATH = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json'
    THUMBNAIL_DIR = '/home/ubuntu/crewgooglegemini/CAPTACITY/assets/thumbnails'

    # Find latest video
    video_files = [f for f in os.listdir(FINAL_VIDEO_DIR) if f.endswith(".mp4")]
    if not video_files:
        print("No video files found.")
        return
    latest_video_file = max(video_files, key=lambda f: os.path.getmtime(os.path.join(FINAL_VIDEO_DIR, f)))
    video_path = os.path.join(FINAL_VIDEO_DIR, latest_video_file)
    video_title = os.path.splitext(latest_video_file)[0]

    # Load script from output_transcript.json
    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"Transcript log file {TRANSCRIPT_PATH} does not exist.")
        return

    with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Transcript file {TRANSCRIPT_PATH} is not valid JSON.")
            return

    script_text = data.get('script', '')

    # Create the transcript file as a .txt
    transcript_filename = os.path.join(FINAL_VIDEO_DIR, f"{video_title}_transcript.txt")
    transcript_text = (
        f"Title: {video_title}\n\n"
        f"Transcript:\n{script_text}"
    )
    with open(transcript_filename, 'w', encoding='utf-8') as tf:
        tf.write(transcript_text)

    # Upload video to Google Drive
    print(f"Uploading video: {video_path}")
    video_upload_command = [
        'rclone', 'copy', video_path, 'mygdrive:/YouTubevids/', '--progress'
    ]
    subprocess.run(video_upload_command)
    print(f"Uploaded video: {latest_video_file}")

    # Upload transcript to Google Drive
    print(f"Uploading transcript: {transcript_filename}")
    transcript_upload_command = [
        'rclone', 'copy', transcript_filename, 'mygdrive:/YouTubevids/', '--progress'
    ]
    subprocess.run(transcript_upload_command)
    print(f"Uploaded transcript: {transcript_filename}")

    # --- Find and Rename Thumbnails to Match Final Video Base Name ---
    for size in ["720X1280"]:
        # Find any thumbnail ending with the correct size
        pattern = os.path.join(THUMBNAIL_DIR, f"*_{size}.png")
        candidates = glob.glob(pattern)
        # Pick the most recently modified one (assume it's for the latest video)
        if candidates:
            latest_thumb = max(candidates, key=os.path.getmtime)
            # Define the new name to match the video_title
            new_thumb_name = os.path.join(THUMBNAIL_DIR, f"{video_title}_thumbnail_{size}.png")
            # If it's not already named correctly, rename it
            if os.path.abspath(latest_thumb) != os.path.abspath(new_thumb_name):
                shutil.copy2(latest_thumb, new_thumb_name)
                print(f"Copied thumbnail {latest_thumb} -> {new_thumb_name}")
            else:
                print(f"Thumbnail already named: {new_thumb_name}")
            thumb_path = new_thumb_name
        else:
            print(f"No thumbnail found for size {size}. Skipping.")
            continue

        # Upload thumbnail to Google Drive
        if os.path.exists(thumb_path):
            print(f"Uploading thumbnail: {thumb_path}")
            thumb_upload_command = [
                'rclone', 'copy', thumb_path, 'mygdrive:/YouTubevids/', '--progress'
            ]
            subprocess.run(thumb_upload_command)
            print(f"Uploaded thumbnail: {thumb_path}")
        else:
            print(f"Thumbnail not found after renaming: {thumb_path}")

    print("All video assets (video, transcript, thumbnails) uploaded to Google Drive.")



def clean_json_string(json_str):
    # Find the script field and replace raw newlines inside it with literal \n
    # Assumes "script": "..." always comes after "keywords": [...]
    script_pattern = r'("script":\s*")([\s\S]*?)("(\s*,|}|\n))'
    def replacer(match):
        before = match.group(1)
        content = match.group(2).replace('\n', '\\n')
        after = match.group(3)
        return before + content + after
    cleaned = re.sub(script_pattern, replacer, json_str)
    return cleaned


# Updated schema
expected_schema = {
    "type": "object",
    "properties": {
        "new_video_title": {"type": "string"},
        "script": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["new_video_title", "script", "keywords"]
}

def normalize_text(text):
    return unidecode(text)

def generate_biblical_script(biblical_focus):
    prompt = f"""You are a distinguished biblical scholar and expert communicator, tasked with creating concise, compelling YouTube Shorts scripts (between 300 to 350 words long, or 2 minutes to 2 minutes 30 seconds spoken) that bring the Bible to life for modern audiences.

Your goal is to take the provided biblical topic, theme, or passage and create a short script that:
- Grabs attention immediately with a thought-provoking biblical insight
- Presents one powerful truth from Scripture, clearly and vividly
- Leaves viewers spiritually moved or challenged within the short time frame
- Feels complete — not like a teaser for something longer

The script must be biblically sound, spiritually insightful, and emotionally resonant — aimed at both mature believers and spiritually curious viewers.

Here’s how the script should flow:

- **Begin with a Revelation-Based Hook:**
    - Start with a statement or question that stops the scroll and challenges assumptions. Use lines like: “What if the most famous Bible verse doesn’t mean what you think?”, “This one verse shattered my understanding of God’s timing,” or “Most people skip this part of the Bible — but it holds a stunning truth.”

- **Deliver One Clear, Spirit-Filled Insight:**
    - Unpack one core idea or verse with emotional depth and clarity. Stick to a single biblical insight rather than broad teaching.
    - Quote Scripture in this format: “As it says in First Samuel 4 verse 4 to 6...”
    - If relevant, share the meaning of a key Hebrew, Greek, or Aramaic word — in a simple, non-academic way that adds clarity, not confusion.
    - Include historical or geographical context only if it strengthens the core idea — no long side journeys.
    - Keep language simple, reverent, and easily understood by non-native English speakers.
    - Inspire personal reflection and a sense of wonder or conviction.
    - If you give a quote thats in the bible, do not ascribe it to someone else, eg, Saint Augustine but only the bible if possible the real author if its verified.

- **Conclude with Purpose and Call to Thought:**
    - End with a compelling takeaway, either a final truth, a spiritual challenge, or a reflective question.
    - The ending should leave the viewer feeling emotionally seen, empowered, or quietly stunned — like they’ve just realized something personal or profound.
    - Examples: “So the next time you pray that verse, ask yourself — do you truly believe it?”, “That’s not just ancient history — it’s your invitation today.”
    - Leave the viewer with one line of emotional release or personal empowerment — as though the video itself just helped them break free of something.
    - Encourage engagement with a variation of: “If you love biblical truth like this, consider subscribing to stay connected,” or “More hidden gems from Scripture are coming — subscribe if you’re curious.”

**Stylistic and Formatting Requirements:**
- Keep each paragraph under 650 characters.
- Do not use asterisks, emojis, or parentheticals.
- Use proper punctuation for smooth TTS and voiceover pacing.
- Avoid the words “unlock,” “let’s dive,” and any visual instructions like “show image here.”
- Format numbered items with words (e.g., number one, number two).
- Preserve reverence, accuracy, and theological integrity. Clearly distinguish between interpretation, application, and historical context.

Additionally, generate:
- A **new video title** that is intriguing and YouTube-optimized (e.g., “Why God Let Them Fail — And Still Called Them Blessed”, “The Verse Everyone Quotes… Completely Wrong”).
- A list of **keywords** specific to the biblical topic, including book names, figures, and core theological themes for SEO.

Here is the provided biblical topic, theme, or passage:
{{{biblical_focus}}}

Return the output as a JSON object in the format:
{{
    "new_video_title": "Your catchy and biblically relevant video title",
    "keywords": ["keyword1_bible", "keyword2_theology", "keyword3_topic"],
    "script": "The generated script"
}}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    # Log the raw response for debugging
    # Log the raw response for debugging
    print("Raw API Response:", response.text)

    # Get the content from the response
    content = response.text

    # Extract JSON from the content using regex
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = content.strip()

    # Clean up control characters if needed
    json_str = clean_json_string(json_str)  # <--- ADD THIS LINE

    try:
        result = json.loads(json_str)

        # Validate JSON structure
        if validate_json_structure(result, expected_schema):
            new_video_title = result["new_video_title"]
            script = result["script"]

            # Normalize the script using your function
            script = normalize_text(script)
            # Remove asterisks (*) from the script if you wish
            script = script.replace("*", "")
            script = re.sub(r'\(([^)]*)\)', r'\1', script)
            # Optional: Log for debugging
            # print(f"Filtered Script:\n{script}")
            return new_video_title, script
        else:
            raise ValueError("JSON validation failed.")

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        # Retry logic or fallback approach can be added here
    return None, None  # In case of failure

def get_first_json_string_and_update_file(json_file_path):
    """
    Reads a JSON file containing a list of strings.
    Returns the first string, removes it from the list, and updates the file.
    If the file doesn't exist, is empty, not a list, or any error occurs, returns None.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {json_file_path}. File might be empty or malformed.")
                # To prevent repeated errors with a malformed file, overwrite with an empty list.
                # Alternatively, could move/rename the malformed file.
                with open(json_file_path, 'w', encoding='utf-8') as wf:
                    json.dump([], wf, indent=4)
                return None

        if not isinstance(data, list):
            print(f"Error: JSON content in {json_file_path} is not a list.")
            # Overwrite with an empty list to correct the structure for future runs.
            with open(json_file_path, 'w', encoding='utf-8') as wf:
                json.dump([], wf, indent=4)
            return None

        if not data: # Empty list
            # print(f"JSON list in {json_file_path} is empty. No items to process.") # This can be noisy if checked often
            return None

        item_to_process = data.pop(0) # Get and remove the first item

        # Write the rest of the list back (even if it's now empty)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4) 

        if isinstance(item_to_process, str):
            return item_to_process.strip() # Ensure leading/trailing whitespace is removed
        else:
            print(f"Warning: Item popped from {json_file_path} was not a string: {item_to_process}")
            return None # Or handle non-string items differently if needed

    except FileNotFoundError:
        # This case is handled by Option 4 logic by creating the file, 
        # but good to have for robustness if called elsewhere.
        print(f"JSON file {json_file_path} not found. No items to process.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_first_json_string_and_update_file for {json_file_path}: {e}")
        return None



# Kokoro model and voices file paths (no .env, hardcoded)
KOKORO_MODEL_FILE_PATH = "/home/ubuntu/crewgooglegemini/CAPTACITY/assets/kokoro_models/kokoro-v1.0.onnx"
KOKORO_VOICES_FILE_PATH = "/home/ubuntu/crewgooglegemini/CAPTACITY/assets/kokoro_models/voices-v1.0.bin"

# Curated list of major, high-quality Kokoro voices for faceless/narration videos
KOKORO_VOICE_POOL = [
    # American English - Female
    {"voice": "af_bella",   "lang": "en-us", "speed": 0.89, "label": "US Female, natural, warm"},
    # {"voice": "af_nicole",  "lang": "en-us", "speed": 0.92, "label": "US Female, clear, modern"},
    {"voice": "af_sarah",   "lang": "en-us", "speed": 0.90, "label": "US Female, expressive"},
    # American English - Male
    {"voice": "am_fenrir",  "lang": "en-us", "speed": 0.82, "label": "US Male, deep, calm"},
    {"voice": "am_michael", "lang": "en-us", "speed": 0.89, "label": "US Male, neutral, clear"},
    # British English - Female
    {"voice": "bf_emma",    "lang": "en-gb", "speed": 0.82, "label": "UK Female, natural, warm"},
    # British English - Male
    {"voice": "bm_george",  "lang": "en-gb", "speed": 0.87, "label": "UK Male, neutral, classic"},
    {"voice": "bm_fable",   "lang": "en-gb", "speed": 0.89, "label": "UK Male, modern, clear"},
]

try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

def generate_audio(
    text: str,
    output_filename: str,
    voice: str = None,
    speed: float = None,
    lang: str = None
):
    """
    Generate an audio file from text using Kokoro ONNX.
    Picks a random high-quality voice and settings for faceless/narration videos.
    """
    if not KOKORO_AVAILABLE:
        raise RuntimeError("Kokoro TTS is not available. Please install kokoro_onnx.")

    # Pick a random voice/settings from the curated pool if not specified
    if not voice or not speed or not lang:
        chosen = random.choice(KOKORO_VOICE_POOL)
        voice = chosen["voice"]
        speed = chosen["speed"]
        lang = chosen["lang"]
        print(f"[Kokoro] Using random voice: {voice} ({chosen['label']}), speed={speed}, lang={lang}")
    else:
        print(f"[Kokoro] Using specified voice: {voice}, speed={speed}, lang={lang}")

    # Initialize Kokoro with the specified model and voices files
    kokoro = Kokoro(KOKORO_MODEL_FILE_PATH, KOKORO_VOICES_FILE_PATH)

    # Synthesize audio
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=speed,
        lang=lang
    )

    # Save the audio file
    import soundfile as sf
    sf.write(output_filename, samples, sample_rate)
    print(f"Audio generated and saved to {output_filename}")



def add_voiceover_and_bgsound(video_path, voiceover_path, bgsound_folder, output_path):
    # Load video and voiceover
    video_clip = VideoFileClip(video_path)
    voiceover_audio = AudioFileClip(voiceover_path)

    video_duration = video_clip.duration
    voiceover_duration = voiceover_audio.duration

    # Load background sounds
    bgsound_files = [
        os.path.join(bgsound_folder, f)
        for f in os.listdir(bgsound_folder)
        if f.lower().endswith(('.mp3', '.wav'))
    ]
    if not bgsound_files:
        raise ValueError(f"No background sound files found in {bgsound_folder}")

    bgsound_path = random.choice(bgsound_files)
    bgsound_audio = AudioFileClip(bgsound_path)

    # Adjust video duration to match voiceover duration (never trim voiceover)
    if video_duration < voiceover_duration:
        # Extend video by freezing the last frame
        last_frame = video_clip.to_ImageClip(duration=voiceover_duration - video_duration)
        last_frame = last_frame.set_fps(video_clip.fps)
        video_clip = concatenate_videoclips([video_clip, last_frame])
        print(f"Extended video from {video_duration:.2f}s to {voiceover_duration:.2f}s")
    elif video_duration > voiceover_duration:
        # Trim video to voiceover duration
        video_clip = video_clip.subclip(0, voiceover_duration)
        print(f"Trimmed video from {video_duration:.2f}s to {voiceover_duration:.2f}s")

    # Adjust background sound duration to match voiceover duration
    if bgsound_audio.duration < voiceover_duration:
        loops = int(voiceover_duration // bgsound_audio.duration) + 1
        bgsound_audio = concatenate_audioclips([bgsound_audio] * loops).subclip(0, voiceover_duration)
        print(f"Looped background sound to {voiceover_duration:.2f}s")
    else:
        bgsound_audio = bgsound_audio.subclip(0, voiceover_duration)
        print(f"Trimmed background sound to {voiceover_duration:.2f}s")

    # Combine voiceover and background sound (voiceover always full volume, bg sound at 7%)
    combined_audio = CompositeAudioClip([voiceover_audio, bgsound_audio.volumex(0.07)])

    # Set combined audio to video
    final_video = video_clip.set_audio(combined_audio)

    # Apply 1 second fade out to video and audio
    final_video = final_video.fx(afx.fadeout, 1)
    final_audio = final_video.audio.fx(afx.audio_fadeout, 1)
    final_video = final_video.set_audio(final_audio)

    # Write output
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='medium',
        ffmpeg_params=['-crf', '18']
    )
    print(f"Final video with voiceover and background sound saved to {output_path}")

    return output_path





def get_random_endscreen(endscreen_folder_path):
    """Pick a random endscreen .mp4 from the given folder."""
    if not os.path.isdir(endscreen_folder_path):
        print(f"[ERROR] Endscreen folder not found: {endscreen_folder_path}")
        return None
    mp4_files = [f for f in os.listdir(endscreen_folder_path) if f.lower().endswith('.mp4')]
    if not mp4_files:
        print(f"[INFO] No .mp4 endscreen files found in {endscreen_folder_path}.")
        return None
    selected_file = random.choice(mp4_files)
    full_path = os.path.join(endscreen_folder_path, selected_file)
    print(f"[INFO] Randomly selected endscreen: {full_path}")
    return full_path

def append_endscreen_to_video(main_video_path, endscreen_video_path, final_output_path, target_width=720, target_height=1280):
    """
    Appends one endscreen video to the main video, re-encoding as needed to ensure compatibility.
    Both videos will be scaled and padded to target_width x target_height before concatenation.
    """
    if not os.path.exists(main_video_path):
        print(f"[ERROR] Main video not found: {main_video_path}")
        return False
    if not os.path.exists(endscreen_video_path):
        print(f"[ERROR] Endscreen video not found: {endscreen_video_path}")
        return False

    filter_complex = (
        f"[0:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p[v0];"
        f"[1:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p[v1];"
        f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a0];"
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a1];"
        f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
    )
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", main_video_path,
        "-i", endscreen_video_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", final_output_path
    ]
    print(f"[INFO] Running FFmpeg to append endscreen: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(final_output_path):
        print(f"[SUCCESS] Video with endscreen saved to: {final_output_path}")
        return True
    else:
        print(f"[ERROR] FFmpeg failed to append endscreen.")
        print(f"FFmpeg stdout:\n{result.stdout}")
        print(f"FFmpeg stderr:\n{result.stderr}")
        return False




PROMPT_LOG_FILE = "/home/ubuntu/crewgooglegemini/prompt_log.txt"

def generate_thumbnail_image(thumbnail_prompt, width, height, output_path):
    # Log the prompt and parameters
    try:
        with open(PROMPT_LOG_FILE, "a") as log_file:
            log_file.write(f"Thumbnail prompt: {thumbnail_prompt['thumbnail_prompt']}\n")
            log_file.write(f"Negative prompt: {thumbnail_prompt.get('negative_prompt', '')}\n")
            log_file.write(f"Width: {width}, Height: {height}, Output: {output_path}\n")
            log_file.write("-" * 40 + "\n")
    except Exception as e:
        print(f"Error logging prompt: {e}")

    # Load your workflow template
    workflow = load_workflow(thumbnail_prompt["thumbnail_prompt"], thumbnail_prompt["negative_prompt"])
    # Set the width and height for the thumbnail
    workflow["18"]["inputs"]["width"] = width
    workflow["18"]["inputs"]["height"] = height
    # You may also want to set a filename prefix or output path in your workflow if supported
    # For example, if your workflow supports it:
    # workflow["84"]["inputs"]["filename_prefix"] = output_path
    response = queue_prompt(workflow)
    if response and 'prompt_id' in response:
        ws = websocket.WebSocket()
        ws_url = f"ws://{server_address}/ws?clientId={client_id}"
        ws.connect(ws_url)
        images = generate_images(ws, response['prompt_id'])
        ws.close()
        # Save the image(s)
        for node_id, image_list in images.items():
            for i, image_data in enumerate(image_list):
                try:
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_data))
                    image.save(output_path)
                    print(f"Thumbnail saved as {output_path}")
                except Exception as e:
                    print(f"Error saving thumbnail: {e}")
    else:
        print("Error: Thumbnail prompt could not be queued.")




# # Directory setup (same as Kokoro)
# VOICE_OVER_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/voice_over"
# os.makedirs(VOICE_OVER_FOLDER, exist_ok=True)

# # Orpheus voice settings pool
# ORPHEUS_VOICE_SETTINGS = [
#     {"voice": "jess", "speed": 1.1},
#     {"voice": "jess", "speed": 1.15},
#     {"voice": "dan",  "speed": 1.0},
#     {"voice": "tara", "speed": 1.0},
#     {"voice": "leo",  "speed": 0.89},
#     {"voice": "leah", "speed": 0.95},
# ]

# ORPHEUS_API_URL = "http://54.80.78.81:5005/v1/audio/speech"

# def wait_for_file_stable(filepath, retries=40, interval=0.25):
#     """
#     Wait until the file size remains the same for two checks,
#     indicating the file is fully written and stable.
#     """
#     last_size = -1
#     for _ in range(retries):
#         try:
#             size = os.path.getsize(filepath)
#         except FileNotFoundError:
#             size = -1
#         if size == last_size and size > 0:
#             return True
#         last_size = size
#         time.sleep(interval)
#     return False

# def generate_audio(
#     text: str,
#     output_filename: str,
#     voice: str = None,
#     speed: float = None,
#     lang: str = None  # Not used by Orpheus, but kept for compatibility
# ):
#     """
#     Generate an audio file from text using Orpheus TTS API.
#     Randomly picks a voice/speed pair if not specified.
#     """
#     # Pick a random voice/speed pair if not specified
#     if voice is None or speed is None:
#         chosen = random.choice(ORPHEUS_VOICE_SETTINGS)
#         voice = chosen["voice"]
#         speed = chosen["speed"]

#     payload = {
#         "input": text,
#         "model": "orpheus",
#         "voice": voice,
#         "response_format": "wav",
#         "speed": speed
#     }

#     print(f"Sending text to Orpheus TTS: voice={voice}, speed={speed}")
#     try:
#         # Increased timeout for long texts (up to 5 minutes)
#         response = requests.post(ORPHEUS_API_URL, json=payload, timeout=500)
#         response.raise_for_status()
#     except Exception as e:
#         print(f"[ERROR] Orpheus TTS request failed: {e}")
#         raise

#     if not response.content:
#         raise RuntimeError("Orpheus TTS API returned empty audio content.")

#     # Save the audio file (same as Kokoro)
#     with open(output_filename, "wb") as f:
#         f.write(response.content)
#         f.flush()
#         os.fsync(f.fileno())
#     print(f"Audio generated and saved to {output_filename}")

#     # Wait for file stability
#     if not wait_for_file_stable(output_filename, retries=40, interval=0.99):
#         print(f"[ERROR] Audio file did not stabilize: {output_filename}")
#         raise RuntimeError("Audio file did not stabilize after writing.")


# get_latest_txt_file function removed as it's no longer needed.

# ---------------------------------------------------------------------------
# MAIN OPERATIONAL FUNCTION
# ---------------------------------------------------------------------------


async def main():
    first_run = True
    while True:
        if first_run:
            print("Choose an option:")
            print("1: Upload a video/audio file")
            print("2: Provide a YouTube link to download a video")
            print("3: Process next link from Newlinks directory FOR GENERAL TOPICS")
            print("4: Bible Stories")
            option = input("Enter 1, 2, 3, or 4: ").strip()
            first_run = False
        else:
            option = "3"

        if option == "1":
            print("File upload not implemented.")
            continue

        elif option == "2":
            youtube_link = input("Please paste the YouTube link: ").strip()
            filename = download_youtube_video(youtube_link, output_path="/home/ubuntu/downloads")
            if filename is None:
                print("Failed to download video from YouTube. Exiting.")
                continue
            else:
                print("Download successful. File saved as:", filename)

        elif option == "3":
            print("Processing Option 3: Channel Topics before Newlinks")
            topics_json = "/home/ubuntu/crewgooglegemini/AUTO/channeltopics.json"
            used_topics_json = "/home/ubuntu/crewgooglegemini/AUTO/Usedchanneltopics.json"
            failedtopics_json = "/home/ubuntu/crewgooglegemini/AUTO/failedtopics.json"
            server_address = "54.80.78.81:8253"
            client_id = str(uuid.uuid4())
            # 1. Process all channel topics first
            while True:
                try:
                    with open(topics_json, 'r', encoding='utf-8') as f:
                        topics = json.load(f)
                except Exception as e:
                    print(f"Error loading channel topics JSON: {e}")
                    topics = []
                if not topics:
                    print("No more topics in channeltopics.json. Switching to shortsnewlinks.txt.")
                    break
        
                topic = topics[0]
                print(f"Processing Channel Topic: {topic}")
                transcript_text = topic
        
                # Generate script
                try:
                    new_video_title, script = generate_script_from_video(transcript_text)
                except Exception as e:
                    print(f"Error generating script for topic: {e}")
                    # Move topic to failedtopics.json
                    try:
                        with open(failedtopics_json, 'r', encoding='utf-8') as f:
                            failed_topics = json.load(f)
                    except:
                        failed_topics = []
                    failed_topics.append(topic)
                    with open(failedtopics_json, 'w', encoding='utf-8') as f:
                        json.dump(failed_topics, f, indent=2)
                    # Remove topic from topics_json
                    topics.pop(0)
                    with open(topics_json, 'w', encoding='utf-8') as f:
                        json.dump(topics, f, indent=2)
                    continue

        
                if new_video_title and script:
                    # Run your media pipeline (use your shared function, e.g. process_generated_content or run_full_pipeline)
                    await run_full_pipeline(new_video_title, script, transcript_text=transcript_text)
                    print("Media pipeline for channel topic completed.")
                    clear_output(wait=True)
                    
                else:
                    print("Script generation failed for topic.")
        
                # Move topic to Usedchanneltopics.json and remove from topics_json
                try:
                    with open(used_topics_json, 'r', encoding='utf-8') as f:
                        used_topics = json.load(f)
                except:
                    used_topics = []
                used_topics.append(topic)
                with open(used_topics_json, 'w', encoding='utf-8') as f:
                    json.dump(used_topics, f, indent=2)
                topics.pop(0)
                with open(topics_json, 'w', encoding='utf-8') as f:
                    json.dump(topics, f, indent=2)
                print(f"Marked topic as used and removed from {topics_json}")
        
                time.sleep(2)
        
            # 2. When topics are finished, process links as usual
            print("Now processing links from shortsshortsnewlinks.txt as before.")
            while True:
                youtube_link, link_path = get_next_link()
                if youtube_link is None:
                    print("No more links found in Newlinks directory. Exiting Option 3 FOR GENERAL TOPICS.")
                    return
        
                print(f"Processing link from Newlinks: {youtube_link}")
                # (Existing Option 3 logic for processing YouTube link)
                # ... download, transcribe, generate_script_from_video, run_full_pipeline, move link to Usedlinks.txt, etc ...
                transcript_text, myfile, downloaded_video_path = None, None, None
                try:
                    downloaded_video_path = download_youtube_video(youtube_link, output_path="/home/ubuntu/downloads")
                    if downloaded_video_path:
                        myfile = genai.upload_file(downloaded_video_path)
                        start_time = time.time()
                        max_delay = 120
                        while True:
                            current_file_status = genai.get_file(myfile.name).state.name
                            print(f"Current file state for {myfile.name}: {current_file_status}. Waiting for file to become ACTIVE...")
                            if current_file_status == "ACTIVE":
                                break
                            if current_file_status == "FAILED":
                                myfile = None
                                break
                            if time.time() - start_time > max_delay:
                                myfile = None
                                break
                            time.sleep(10)
                        if myfile and genai.get_file(myfile.name).state.name == "ACTIVE":
                            model = genai.GenerativeModel("gemini-2.0-flash")
                            prompt_for_transcript = "Generate a transcript of the speech in this audio/video file. Only provide the transcript, without any additional commentary."
                            result = generate_transcript_with_retries(model, myfile, prompt_for_transcript)
                            transcript_text = result.text
                    else:
                        print(f"Failed to download video for link: {youtube_link}")
        
                except Exception as e:
                    print(f"An error occurred during download/transcript generation: {e}")
        
                finally:
                    if myfile:
                        try:
                            genai.delete_file(myfile.name)
                        except Exception:
                            pass
                    if downloaded_video_path and os.path.exists(downloaded_video_path):
                        os.remove(downloaded_video_path)
        
                if transcript_text and isinstance(transcript_text, str):
                    try:
                        new_video_title, script = generate_script_from_video(transcript_text)
                    except Exception as e:
                        print(f"Error generating script: {e}")
                        continue
        
                    if new_video_title and script:
                        await run_full_pipeline(new_video_title, script, transcript_text=transcript_text)
                        # Move processed link to Usedlinks.txt and remove from shortsnewlinks.txt
                        if os.path.exists(link_path):
                            with open(link_path, 'r') as file:
                                lines = file.readlines()
                            used_links_file = "/home/ubuntu/crewgooglegemini/AUTO/Usedlinks.txt"
                            with open(used_links_file, 'a') as used_file:
                                used_file.write(youtube_link + '\n')
                            with open(link_path, 'w') as new_file:
                                new_file.writelines(lines[1:])
                    else:
                        print(f"Failed to generate script for {youtube_link}. Skipping.")
        
                else:
                    print(f"Transcript generation or download failed for {youtube_link}. Skipping.")
        
                # print("Option 3 processing for current link finished. Ready for next iteration or option.")
                # # Move failed link to failed_bible_links_file.txt and remove from new_links_file.txt
                # with open(failed_bible_links_file, 'a') as ublf:
                #     ublf.write(youtube_link + '\n')
                # links.pop(0)
                # with open(new_links_file, 'w') as f:
                #     for l in links:
                #         f.write(l + '\n')
                # print(f"Moved processed link to {failed_bible_links_file} and removed from {new_links_file}")

                # time.sleep(2)
                time.sleep(3)

        elif option == "4":
            print("Processing Option 4: Bible Stories")
            bible_txt_json = "/home/ubuntu/crewgooglegemini/AUTO/BiblestoriesTxt.json"
            used_txtfiles_json = "/home/ubuntu/crewgooglegemini/AUTO/used_txtfiles.json"
            bible_links_file = "/home/ubuntu/crewgooglegemini/AUTO/Biblelinks.txt"
            used_bible_links_file = "/home/ubuntu/crewgooglegemini/AUTO/Usedbiblelinks.txt"
            failed_bible_links_file = "/home/ubuntu/crewgooglegemini/AUTO/checklink.txt"

            # Step 1: Process all stories in BiblestoriesTxt.json
            while True:
                try:
                    with open(bible_txt_json, 'r', encoding='utf-8') as f:
                        bible_stories = json.load(f)
                except Exception as e:
                    print(f"Error loading Bible stories JSON: {e}")
                    bible_stories = []
                if not bible_stories:
                    print("No more stories in BiblestoriesTxt.json. Switching to Biblelinks.txt.")
                    break

                next_story = bible_stories[0]
                print("Next Bible Story Selected:")
                print(next_story)

                transcript_text = next_story
                new_video_title, script = None, None
                try:
                    new_video_title, script = generate_biblical_script(transcript_text)
                except Exception as e:
                    print(f"Error generating script for story: {e}")

                if new_video_title and script:
                    await run_full_pipeline(new_video_title, script)
                    print("Media pipeline for text story completed.")
                else:
                    print("Script generation failed for story.")

                # Mark as used and remove from BiblestoriesTxt.json
                try:
                    with open(used_txtfiles_json, 'r', encoding='utf-8') as f:
                        used_stories = json.load(f)
                except:
                    used_stories = []
                used_stories.append(next_story)
                with open(used_txtfiles_json, 'w', encoding='utf-8') as f:
                    json.dump(used_stories, f, indent=2)
                bible_stories.pop(0)
                with open(bible_txt_json, 'w', encoding='utf-8') as f:
                    json.dump(bible_stories, f, indent=2)
                print(f"Marked story as used and removed from {bible_txt_json}")

                time.sleep(2)

            # Step 2: Now process links one by one from Biblelinks.txt
            while True:
                if not os.path.exists(bible_links_file):
                    print("No Biblelinks.txt file found! Option 4 complete.")
                    break
                with open(bible_links_file, 'r') as f:
                    links = [line.strip() for line in f if line.strip()]
                if not links:
                    print("No more links in Biblelinks.txt. Option 4 complete.")
                    break

                youtube_link = links[0]
                print(f"Processing Bible link: {youtube_link}")

                downloaded_video_path = download_youtube_video(youtube_link, output_path="/home/ubuntu/downloads")
                transcript_text = None
                myfile = None
                if downloaded_video_path:
                    try:
                        myfile = genai.upload_file(downloaded_video_path)
                        start_time = time.time()
                        max_delay = 120
                        while True:
                            current_state = genai.get_file(myfile.name).state.name
                            print(f"Current file state for {myfile.name}: {current_state}. Waiting for file to become ACTIVE...")
                            if current_state == "ACTIVE": break
                            if current_state == "FAILED":
                                myfile = None
                                break
                            if time.time() - start_time > max_delay:
                                myfile = None
                                break
                            time.sleep(10)
                        if myfile and genai.get_file(myfile.name).state.name == "ACTIVE":
                            model = genai.GenerativeModel("gemini-2.0-flash")
                            prompt_transcript = "Generate a transcript of the speech in this audio/video file. Only provide the transcript, without any additional commentary."
                            try:
                                result = generate_transcript_with_retries(model, myfile, prompt_transcript)
                                transcript_text = result.text
                            except Exception as e:
                                print(f"Error generating transcript for {youtube_link}: {e}")
                        else:
                            print(f"File upload processing for {downloaded_video_path} (link: {youtube_link}) did not result in an ACTIVE file.")
                    finally:
                        if myfile:
                            try:
                                genai.delete_file(myfile.name)
                            except Exception:
                                pass
                        if downloaded_video_path and os.path.exists(downloaded_video_path):
                            os.remove(downloaded_video_path)

                if transcript_text:
                    try:
                        new_video_title, script = generate_biblical_script(transcript_text)
                    except Exception as e:
                        print(f"Error generating script from transcript for {youtube_link}: {e}")
                    if new_video_title and script:
                        await run_full_pipeline(new_video_title, script)
                        print("Media pipeline for Bible link completed.")
                    else:
                        print("Script generation failed for Bible link.")
                else:
                    print("Transcript generation failed for Bible link.")

                # Move processed link to Usedbiblelinks.txt and remove from Biblelinks.txt
                with open(used_bible_links_file, 'a') as ublf:
                    ublf.write(youtube_link + '\n')
                links.pop(0)
                with open(bible_links_file, 'w') as f:
                    for l in links:
                        f.write(l + '\n')
                print(f"Moved processed link to {used_bible_links_file} and removed from {bible_links_file}")

                time.sleep(2)

        else:
            print("Invalid option selected. Exiting.")
            break

async def run_full_pipeline(new_video_title, script, transcript_text=None):
    if transcript_text: # This is the original transcript from YouTube video, if applicable
        log_transcript(transcript_text, new_video_title)

    # The 'script' variable here is assumed to be the script_json object (output of generate_script_from_video)
    # or a simple script text string (output of generate_biblical_script).
    script_input_arg = script # Keep original 'script' arg for clarity in this scope

    print(f"Running media pipeline for: {new_video_title}")
    print(colored("Step A: Generating video and audio prompts from script...", "blue"))

    # generate_video_prompts_from_script expects script_json (a dict with scene_sequence)
    # If script_input_arg is just text (e.g. from biblical_script), this step will be skipped by generate_video_prompts_from_script
    scene_prompts_data = generate_video_prompts_from_script(script_input_arg)

    video_scene_paths = []
    if scene_prompts_data:
        print(colored(f"Successfully generated {len(scene_prompts_data)} sets of video/audio prompts.", "green"))

        print(colored("Step B: Running video scene generation workflow...", "blue"))
        # run_video_generation_workflow expects a list of dicts, and uses item['video_prompt']
        video_scene_paths = run_video_generation_workflow(scene_prompts_data)
        if video_scene_paths:
            print(colored(f"Video scene generation completed. {len(video_scene_paths)} raw scenes in '{GENERATED_VIDEO_SCENES_DIR}'.", "green"))

            print(colored("Step C: Running MMAudio enhancement workflow...", "blue"))
            MMAUDIO_WORKFLOW_JSON_PATH = "/home/ubuntu/crewgooglegemini/0001comfy2/vid_mmaudio.json" # As per user info
            # Ensure scene_prompts_data (which contains audio_prompt etc.) is passed correctly
            enhanced_video_paths = run_mmaudio_enhancement_workflow(video_scene_paths, scene_prompts_data, MMAUDIO_WORKFLOW_JSON_PATH)
            if enhanced_video_paths:
                 print(colored(f"MMAudio enhancement completed. {len(enhanced_video_paths)} scenes in '{GENERATED_VIDEO_SCENES_WITH_MMAUDIO_DIR}'.", "green"))
            else:
                print(colored("MMAudio enhancement failed or produced no videos.", "red"))
        else:
            print(colored("Video scene generation failed or produced no videos. Skipping MMAudio.", "red"))
    else:
        print(colored("Failed to generate video/audio prompts. Skipping video scene generation and MMAudio.", "red"))

    # The 'script_input_arg' (which was the original 'script' parameter) is used to determine the text for TTS.
    # For Option 4 (generate_biblical_script), 'script' is just text.
    # The new video scene generation will effectively run if script_json is valid and contains scene_sequence.

    # Existing workflow continues below.
    # Note: The generated_video_scenes are NOT YET USED by the subsequent steps.
    # This will require further modification of how video_clips are sourced later.

    # Save script (text part) for image prompt context (original image generation) and other uses
    script_text_for_image_prompts_and_tts = ""
    if isinstance(script_json, dict) and "script" in script_json: # If script_json is from generate_biblical_script like format
        script_text_for_image_prompts_and_tts = script_json["script"]
    elif isinstance(script_json, dict) and "scene_sequence" in script_json: # If script_json is from generate_script_from_video
        # Concatenate dialogue or actions from scenes to form a cohesive script text if needed for TTS
        # For now, let's assume the main "script" for TTS is derived differently or handled by existing logic.
        # The user might want to generate TTS from the `dialogue` in `scene_sequence`.
        # For simplicity, if a top-level "script" field isn't in script_json, we might need user clarification
        # on what text to use for TTS and existing image prompt generation.
        # Let's check if the 'script' variable itself was the text (from biblical) or the dict.
        if isinstance(script, str): # This was likely from generate_biblical_script
             script_text_for_image_prompts_and_tts = script
        else: # It was a dict, try to extract a coherent text for TTS if possible.
            # This part is a bit ambiguous based on current plan.
            # The plan focuses on *video scene* generation from *video prompts* derived from *script scenes*.
            # It doesn't specify how the *audio narration/TTS script* is affected.
            # Let's assume for now that the 'script' variable passed to run_full_pipeline
            # is the text intended for TTS if it's a string.
            # If it's a dict from generate_script_from_video, the existing logic might use its 'synopsis' or join dialogues.
            # This ambiguity needs to be addressed if the TTS part is to be perfectly aligned.
            # For now, we prioritize getting *video scenes* generated.
            # The existing image prompt generation (generate_image_prompts_batch) uses a "full_transcript" argument.

            # Fallback: use synopsis if available, for the original image generation part.
            script_text_for_image_prompts_and_tts = script_json.get("synopsis", "No script text available for TTS/image prompts from scene_sequence.")
            print(colored(f"Note: Using synopsis for TTS/image prompts as script_json is a complex object. For better audio, ensure 'script' contains full narration.", "yellow"))


    output_script_path = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json'
    # Save the main script text (for TTS and original image pipeline)
    with open(output_script_path, 'w') as f:
        json.dump({"script": script}, f)

    # Step 4: Generate Audio
    VOICE_OVER_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/voice_over"
    ensure_folder_exists(VOICE_OVER_FOLDER)
    audio_path = os.path.join(VOICE_OVER_FOLDER, "voiceover.wav")
    print("Generating audio...")
    generate_audio(script, audio_path)
    print(f"Audio saved to {audio_path}")

    # Step 5 (Caption-related parts removed): Transcribe Audio (if needed for other purposes)
    # For now, assuming transcribe_locally and related segment extraction was SOLELY for captions.
    # If the transcription text or its segments were needed for something else (e.g. detailed logging, analytics)
    # then this part would need to be re-evaluated.
    # Current understanding: TTS is generated from LLM script, video scenes from LLM prompts. No other use for transcribing TTS.
    
    TRANSCRIPT_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/transcripts" # Path still used for image_video.json
    ensure_folder_exists(TRANSCRIPT_FOLDER)

    # The following block was for generating segments for image prompts (step 5b in old plan)
    # This is still needed for the *original* image generation pipeline if it's kept alongside the new video scene pipeline.
    # If the new video scenes (from run_video_generation_workflow + run_mmaudio_enhancement_workflow)
    # are intended to *replace* the old image-based pipeline, then this entire block for image prompts might also be removable.
    # For now, let's assume the old image pipeline might still be wanted as a fallback or for different content types.
    # However, it needs a source for `transcript_filename` if `transcribe_locally` on TTS output is removed.
    #
    # Let's re-evaluate: `parse_transcript_file` reads the _transcription.txt file.
    # `extract_transcript_segments_image_vid` then processes this.
    # This was for the ComfyUI *image* generation.
    # If we are not generating images this way anymore (because we generate video scenes directly),
    # then `parse_transcript_file`, `extract_transcript_segments_image_vid`, `generate_image_prompts_batch`,
    # and `process_and_generate_images` might also be removed or heavily refactored.
    #
    # The current request is to remove captions and assemble the MMAudio-enhanced videos.
    # I will proceed with removing caption-specific code. The image generation pipeline is a separate concern.
    # For now, the `image_video.json` and `current_prompt.json` (for images) will likely fail if their input
    # `transcript_filename` (from `transcribe_locally(audio_path)`) is removed.
    # This implies `generate_image_prompts_batch` and `process_and_generate_images` will also not run correctly.
    # This seems like an acceptable consequence if the new video scene pipeline is the primary focus.

    # Placeholder for where image prompt generation for the *original* image pipeline used to be.
    # This part will likely need to be removed or significantly rethought if the primary visual source
    # is now the MMAudio-enhanced video scenes.
    # For now, I will comment out the parts that directly depend on the removed transcription steps.
    
    # print("Processing transcript for image-video segments (original image pipeline)...")
    # # transcript_data = parse_transcript_file(transcript_filename) # transcript_filename is no longer generated here
    # # segments_image_vid = extract_transcript_segments_image_vid(transcript_data)
    # image_video_json_path = os.path.join(TRANSCRIPT_FOLDER, 'image_video.json') # Still referenced later
    # # with open(image_video_json_path, 'w') as f:
    # #     json.dump(segments_image_vid, f, indent=4)
    # # print(f"Image-video segments for original image pipeline might be outdated as their source is removed. Saved to {image_video_json_path}")
    
    # print("Generating image prompts for segments (original image pipeline)...")
    # # This section for generating prompts for ComfyUI *images* will likely not work correctly
    # # without its input 'segments' which came from 'output_file' (image_video.json)
    # # which in turn came from 'transcript_filename'.
    # # Keeping it commented out as a marker for now. The focus is on the new video scene pipeline.
    # # full_transcript_file = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json'
    # # with open(full_transcript_file, 'r') as f:
    # #     full_transcript_for_image_prompts = json.load(f) # This is the main script text
    # # if isinstance(full_transcript_for_image_prompts, dict):
    # #     full_transcript_for_image_prompts = full_transcript_for_image_prompts.get('script', '')
    # # elif not isinstance(full_transcript_for_image_prompts, str):
    # #     full_transcript_for_image_prompts = str(full_transcript_for_image_prompts)
    # #
    # # if os.path.exists(image_video_json_path):
    # #     with open(image_video_json_path, 'r') as f: # This file might be stale or empty now
    # #         segments_for_image_prompts = json.load(f)
    # #     batch_size = 5
    # #     image_prompts = []
    # #     if segments_for_image_prompts:
    # #         for i in range(0, len(segments_for_image_prompts), batch_size):
    # #             batch = segments_for_image_prompts[i:i+batch_size]
    # #             # print(f"Sending batch {i//batch_size + 1} for original image prompts: {len(batch)} segments")
    # #             try:
    # #                 batch_prompts = generate_image_prompts_batch(batch, full_transcript_for_image_prompts)
    # #                 image_prompts.extend(batch_prompts)
    # #                 prompt_file = '/home/ubuntu/crewgooglegemini/current_prompt.json'
    # #                 with open(prompt_file, 'w') as pf:
    # #                     json.dump(image_prompts, pf, indent=4)
    # #                 # print(f"Batch {i//batch_size + 1} for original image prompts processed and saved. Waiting 2 seconds...")
    # #                 time.sleep(2)
    # #             except Exception as e:
    # #                 print(f"An error occurred in original image prompt batch {i//batch_size + 1}: {e}")
    # #                 break
    # #     # print(f"Generated {len(image_prompts)} original image prompts and saved to {prompt_file}")
    # # else:
    # #     print(colored(f"Skipping original image prompt generation as {image_video_json_path} not found.", "yellow"))

    # # print("Step 5d: Generating images for each prompt (original image pipeline)...")
    # # if image_prompts: # Check if any image prompts were generated
    # #    process_and_generate_images() # This generates images from the above prompts
    # # else:
    # #    print(colored("Skipping original image generation as no image prompts were created.", "yellow"))


    # Thumbnail generation should still work as it uses the main script text (full_transcript)
    print("Step 5e: Generating thumbnail prompt...")
    # Need to ensure full_transcript is defined correctly here. It was previously based on output_script_path.
    # output_script_path = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json'
    # contains {"script": script_text_for_image_prompts_and_tts}
    # So, we need script_text_for_image_prompts_and_tts
    # This was defined near the top of run_full_pipeline.
    thumbnail_context_script = script_text_for_image_prompts_and_tts # Use the determined script text
    thumbnail_prompt = generate_thumbnail_prompt_purple_cow(thumbnail_context_script)


    THUMBNAIL_DIR = "/home/ubuntu/crewgooglegemini/CAPTACITY/assets/thumbnails"
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    base_name = new_video_title.replace(" ", "_") # Used for thumbnail name
    
    output_path_720x1280 = os.path.join(THUMBNAIL_DIR, f"{base_name}_thumbnail_720X1280.png")
    generate_thumbnail_image(thumbnail_prompt, 720, 1280, output_path_720x1280) # Thumbnail generated

    # --- Video Assembly Steps ---
    print(colored("Step D: Assembling final video from MMAudio-enhanced scenes...", "blue"))
    
    # Ensure main TTS audio path is available
    main_tts_audio_path = audio_path # From earlier in run_full_pipeline
    if not os.path.exists(main_tts_audio_path):
        print(colored(f"Main TTS audio file not found: {main_tts_audio_path}. Cannot proceed with final video assembly.", "red"))
        return # Or handle error appropriately

    # `enhanced_video_paths` should be available from the MMAudio step if it was successful.
    # If not, `video_scene_paths` (raw videos) could be a fallback, but they lack MMAudio sounds.
    # For now, proceed assuming `enhanced_video_paths` is the target.
    
    # If enhanced_video_paths is empty or None (because previous steps failed),
    # we cannot create the video from these scenes.
    if not enhanced_video_paths: # enhanced_video_paths is from run_mmaudio_enhancement_workflow
        print(colored("No MMAudio-enhanced video scenes available to assemble. Skipping final video creation from these scenes.", "yellow"))
        # Here, you might fall back to an older pipeline, or simply exit this part.
        # For now, we'll just not proceed with assembling *these* clips.
    else:
        print(colored(f"Found {len(enhanced_video_paths)} MMAudio-enhanced scenes for assembly.", "green"))

        # Load MMAudio-enhanced clips
        mmaudio_scene_clips = []
        for p in enhanced_video_paths:
            if os.path.exists(p):
                try:
                    mmaudio_scene_clips.append(VideoFileClip(p))
                except Exception as e:
                    print(colored(f"Error loading MMAudio scene clip {p}: {e}", "red"))
            else:
                print(colored(f"MMAudio scene file {p} not found. Skipping.", "yellow"))

        if not mmaudio_scene_clips:
            print(colored("No valid MMAudio scene clips could be loaded. Cannot assemble video.", "red"))
        else:
            # Concatenate MMAudio scenes - This is Step 3 of the new assembly plan
            # Audio transitions for mmaudio_scene_clips:
            # Simple concatenation's audio might be abrupt.
            # For smoother transitions, one might apply crossfades if concatenating audio separately
            # or ensure each clip has a slight fade in/out from MMAudio itself.
            # MoviePy's concatenate_videoclips with method="compose" does its best.
            print(colored("Concatenating MMAudio scenes...", "cyan"))
            assembled_scenes_clip = concatenate_videoclips(mmaudio_scene_clips, method="compose")

            # --- Duration Synchronization ---
            print(colored("Synchronizing duration of assembled scenes with main TTS audio...", "cyan"))
            main_tts_audio_clip = AudioFileClip(main_tts_audio_path)

            if assembled_scenes_clip.duration < main_tts_audio_clip.duration:
                duration_diff = main_tts_audio_clip.duration - assembled_scenes_clip.duration
                print(colored(f"Assembled scenes are shorter by {duration_diff:.2f}s. Extending last frame.", "yellow"))
                # Get the last frame of the assembled_scenes_clip
                # To avoid issues with getting frame from a CompositeVideoClip directly if it's complex,
                # it's safer to render it to a temporary path or ensure it has a simple structure.
                # However, for direct ImageClip from last frame:
                try:
                    last_frame_image = assembled_scenes_clip.get_frame(assembled_scenes_clip.duration - 1/assembled_scenes_clip.fps if assembled_scenes_clip.duration > 0 else 0)
                    freeze_frame_clip = ImageClip(last_frame_image).set_duration(duration_diff)
                    if assembled_scenes_clip.fps: # Ensure fps is not None or 0
                         freeze_frame_clip = freeze_frame_clip.set_fps(assembled_scenes_clip.fps)
                    else: # Default fps if original is problematic
                         freeze_frame_clip = freeze_frame_clip.set_fps(24) # A common default

                    assembled_scenes_clip = concatenate_videoclips([assembled_scenes_clip, freeze_frame_clip], method="compose")
                    print(colored(f"Extended assembled scenes to {assembled_scenes_clip.duration:.2f}s.", "green"))
                except Exception as e:
                    print(colored(f"Could not extend last frame: {e}. Duration mismatch may occur.", "red"))

            elif assembled_scenes_clip.duration > main_tts_audio_clip.duration:
                print(colored(f"Assembled scenes are longer. Trimming to {main_tts_audio_clip.duration:.2f}s.", "yellow"))
                assembled_scenes_clip = assembled_scenes_clip.subclip(0, main_tts_audio_clip.duration)

            print(colored(f"Assembled scenes duration: {assembled_scenes_clip.duration:.2f}s, TTS duration: {main_tts_audio_clip.duration:.2f}s", "magenta"))

            # Extract and adjust MMAudio track volume
            if assembled_scenes_clip.audio:
                print(colored("Extracting and adjusting volume of MMAudio track...", "cyan"))
                mmaudio_track = assembled_scenes_clip.audio.volumex(0.3) # Adjust volume as needed
            else:
                print(colored("Warning: Assembled scenes clip has no audio track. MMAudio sounds will be missing.", "yellow"))
                mmaudio_track = None # Explicitly None if no audio

            # Placeholder for next steps (audio mixing, overlay)
            # For now, `assembled_scenes_clip` (visuals) and `mmaudio_track` are prepared.

            # --- Audio Mixing ---
            print(colored("Starting audio mixing process...", "cyan"))
            audio_layers_to_composite = []

            # Main TTS Voiceover
            if main_tts_audio_clip and main_tts_audio_clip.duration > 0:
                audio_layers_to_composite.append(main_tts_audio_clip)
                print(colored(f"  Added main TTS audio (duration: {main_tts_audio_clip.duration:.2f}s).", "green"))
            else:
                print(colored("  Warning: Main TTS audio clip is missing or has zero duration.", "red"))
                # If no TTS, the video duration reference is lost. This should ideally not happen.
                # Fallback to assembled_scenes_clip duration if TTS is missing.
                if not main_tts_audio_clip and assembled_scenes_clip:
                     main_tts_audio_clip = MagicMock(duration=assembled_scenes_clip.duration) # Mock for duration reference
                     print(colored(f"  Using assembled scenes duration ({assembled_scenes_clip.duration:.2f}s) as fallback for main audio duration.", "yellow"))


            # MMAudio Track (Scene-specific sounds)
            if mmaudio_track and mmaudio_track.duration > 0:
                # Ensure mmaudio_track duration does not exceed main TTS audio (it should already be synced if video was)
                if main_tts_audio_clip: # Check if main_tts_audio_clip is not None
                    mmaudio_track = mmaudio_track.subclip(0, min(mmaudio_track.duration, main_tts_audio_clip.duration))
                audio_layers_to_composite.append(mmaudio_track)
                print(colored(f"  Added MMAudio track (volume adjusted, duration: {mmaudio_track.duration:.2f}s).", "green"))
            else:
                print(colored("  MMAudio track is missing or has zero duration. Skipping.", "yellow"))

            # Overall Background Sound
            if main_tts_audio_clip: # Check if main_tts_audio_clip is not None
                overall_bg_sound = get_background_sound(main_tts_audio_clip.duration) # Uses main TTS duration
                if overall_bg_sound and overall_bg_sound.duration > 0:
                    overall_bg_sound = overall_bg_sound.volumex(0.07) # Apply volume
                    audio_layers_to_composite.append(overall_bg_sound)
                    print(colored(f"  Added overall background sound (duration: {overall_bg_sound.duration:.2f}s).", "green"))
                else:
                    print(colored("  Overall background sound is missing or has zero duration. Skipping.", "yellow"))
            else: # Should not happen if fallback above works
                print(colored("  Cannot add overall background sound as main TTS audio reference is missing.", "red"))


            final_composite_audio = None
            if audio_layers_to_composite:
                final_composite_audio = CompositeAudioClip(audio_layers_to_composite)
                if main_tts_audio_clip: # Ensure final audio duration matches main TTS
                    final_composite_audio = final_composite_audio.set_duration(main_tts_audio_clip.duration)
                print(colored(f"  Successfully composited audio tracks. Final audio duration: {final_composite_audio.duration:.2f}s", "green"))
            else:
                print(colored("  No audio layers to composite. Video will have no audio.", "red"))

            # Set the composite audio to the assembled video clip
            # The variable `final_clip` will eventually hold the fully processed video.
            # To avoid NameError later if mmaudio_scene_clips was empty:
            if not mmaudio_scene_clips: # If the main video source was empty
                 print(colored("Cannot set audio as there are no video scenes.", "red"))
                 # Decide how to handle this - perhaps return or raise error earlier
                 final_clip = None # Indicates failure to produce video
            elif assembled_scenes_clip: # Check if assembled_scenes_clip is not None
                final_clip = assembled_scenes_clip.set_audio(final_composite_audio)
                print(colored("  Set final composite audio to video clip.", "green"))
            else: # Should not happen if mmaudio_scene_clips was not empty
                print(colored("Error: assembled_scenes_clip is None, cannot set audio.", "red"))
                final_clip = None


    # Old video assembly logic (commented out or to be removed)
    # voice_over_file = audio_path # This is now main_tts_audio_path

    # --- Apply Video Overlay ---
    # The `final_clip` here is the one with visuals and mixed audio.
    if final_clip: # Ensure final_clip exists (it might be None if earlier steps failed)
        print(colored("Applying video overlay...", "cyan"))
        overlays_dir = "/home/ubuntu/crewgooglegemini/CAPTACITY/assets/overlays"
        overlay_files = [f for f in os.listdir(overlays_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

        if overlay_files:
            overlay_path = os.path.join(overlays_dir, random.choice(overlay_files))
            print(colored(f"  Using overlay: {overlay_path}", "magenta"))
            try:
                overlay_clip_raw = VideoFileClip(overlay_path)
                # Resize overlay to match final_clip dimensions
                overlay_clip_resized = overlay_clip_raw.resize(final_clip.size)

                # Adjust overlay duration to match final_clip
                if overlay_clip_resized.duration < final_clip.duration:
                    overlay_clip_final = overlay_clip_resized.loop(duration=final_clip.duration)
                elif overlay_clip_resized.duration > final_clip.duration:
                    overlay_clip_final = overlay_clip_resized.subclip(0, final_clip.duration)
                else:
                    overlay_clip_final = overlay_clip_resized

                overlay_clip_final = overlay_clip_final.set_opacity(0.1).set_duration(final_clip.duration)

                # Composite with existing final_clip (which has audio)
                final_clip_with_overlay = CompositeVideoClip([final_clip, overlay_clip_final], use_bgclip=True)
                # Ensure the audio from final_clip is preserved if use_bgclip=True doesn't do it as expected
                if final_clip.audio:
                    final_clip_with_overlay = final_clip_with_overlay.set_audio(final_clip.audio)

                final_clip = final_clip_with_overlay # Update final_clip to be the one with overlay
                print(colored("  Overlay applied successfully.", "green"))
            except Exception as e:
                print(colored(f"  Error applying overlay: {e}. Proceeding without overlay.", "red"))
        else:
            print(colored("  No overlay videos found in assets. Skipping overlay.", "yellow"))

        # --- Final Fade Out ---
        if final_clip:
            print(colored("Applying final fade out...", "cyan"))
            fade_duration = 0.5 # Increased slightly
            final_clip = final_clip.fx(vfx.fadeout, duration=fade_duration)
            if final_clip.audio: # Ensure audio also fades out
                final_audio_faded = final_clip.audio.fx(audio_fadeout, duration=fade_duration)
                final_clip = final_clip.set_audio(final_audio_faded)
            print(colored("  Final fade out applied.", "green"))

        # --- Write Final Video ---
        output_dir = "/home/ubuntu/crewgooglegemini/FINALVIDEOS/WellnessGram"
        safe_title = make_safe_filename(new_video_title) # From earlier in the function
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{safe_title}.mp4")

        print(colored(f"Writing final video to: {output_file_path}", "blue"))
        try:
            final_clip.write_videofile(
                output_file_path,
                codec="libx264",
                audio_codec="aac",
                ffmpeg_params=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '22', "-bufsize", "16M", "-maxrate", "8M", '-profile:v', 'high'], # Slightly better CRF
                threads=8, # Adjust based on server capabilities
                logger="bar" # Progress bar
            )
            print(colored(f"Successfully wrote final video: {output_file_path}", "green"))

            # --- End Screen Logic ---
            print(colored("Attempting to append end screen...", "cyan"))
            ENDSCREEN_ANIMATIONS_FOLDER = "/home/ubuntu/crewgooglegemini/PodcastProd/Endanim"
            endscreen_video_file = get_random_endscreen(ENDSCREEN_ANIMATIONS_FOLDER)
            if endscreen_video_file:
                # Create a new path for the video that includes the endscreen
                base_name_for_final_output, ext = os.path.splitext(output_file_path)
                final_video_with_endscreen_path = f"{base_name_for_final_output}_endscreen{ext}"

                print(colored(f"  Appending endscreen '{os.path.basename(endscreen_video_file)}' to '{os.path.basename(output_file_path)}' -> '{os.path.basename(final_video_with_endscreen_path)}'", "magenta"))

                append_success = append_endscreen_to_video(
                    main_video_path=output_file_path,
                    endscreen_video_path=endscreen_video_file,
                    final_output_path=final_video_with_endscreen_path
                )
                if append_success and os.path.exists(final_video_with_endscreen_path):
                    print(colored(f"  Endscreen append successful. New final video: {final_video_with_endscreen_path}", "green"))
                    # Optionally, remove the version without endscreen if desired, and update output_file_path
                    try:
                        os.remove(output_file_path)
                        print(colored(f"  Removed original file: {output_file_path}", "magenta"))
                        output_file_path = final_video_with_endscreen_path # Update to the true final path
                    except Exception as e:
                        print(colored(f"  Could not remove original file after appending endscreen: {e}", "yellow"))
                else:
                    print(colored(f"  [ERROR] Failed to append endscreen or output file missing. Using video before endscreen: {output_file_path}", "red"))
            else:
                print(colored("  No endscreen video found/selected. Skipping append step.", "yellow"))

        except Exception as e:
            print(colored(f"Error writing final video {output_file_path}: {e}", "red"))

    else: # Case where final_clip was not successfully created (e.g. no MMAudio scenes)
        print(colored("Final video clip was not created due to earlier errors. Skipping final write and overlay.", "red"))


    # --- Cleanup ---
    # Cleanup of CUSTOM_FOLDER (which was output for old animated images pipeline)
    CUSTOM_FOLDER = "/home/ubuntu/crewgooglegemini/SHORTCLIPSFACTS/WellnessGram"
    if os.path.exists(CUSTOM_FOLDER):
        print(colored(f"Cleaning up old animated clips from: {CUSTOM_FOLDER}", "cyan"))
        for file_item in os.listdir(CUSTOM_FOLDER):
            if file_item.endswith(".mp4"):
                file_path_to_delete = os.path.join(CUSTOM_FOLDER, file_item)
                try:
                    os.remove(file_path_to_delete)
                except Exception as e:
                    print(colored(f"  Error deleting file {file_path_to_delete}: {e}", "yellow"))
        print(colored(f"  Cleanup of {CUSTOM_FOLDER} complete.", "green"))

    # Note: Cleanup of GENERATED_VIDEO_SCENES_DIR and GENERATED_VIDEO_SCENES_WITH_MMAUDIO_DIR
    # is not done here automatically, allowing for inspection. Could be added if desired.

    print(colored(f"--- Pipeline for '{new_video_title}' finished. ---", "blue"))
    time.sleep(2)
    # Optionally clear Jupyter output if running in notebook
    # clear_output(wait=True)
    time.sleep(3)

    step_5f_send_to_telegram()
    time.sleep(3)
    step_5g_upload_to_google_drive()
    time.sleep(3)


if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        print(f"An error occurred: {e}")
