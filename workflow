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
import segment_parser
import transcriber
import soundfile as sf
from text_drawer import (
    get_text_size_ex,
    create_text_ex,
    blur_text_clip,
    Word,
)

shadow_cache = {}
lines_cache = {}
#from ._init_ import add_captions, fits_frame, calculate_lines, create_shadow, get_font_path, detect_local_whisper


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



def fits_frame(line_count, font, font_size, stroke_width, frame_width):
    def fit_function(text):
        lines = calculate_lines(
            text,
            font,
            font_size,
            stroke_width,
            frame_width
        )
        return len(lines["lines"]) <= line_count
    return fit_function

def calculate_lines(text, font, font_size, stroke_width, frame_width):
    global lines_cache

    arg_hash = hash((text, font, font_size, stroke_width, frame_width))

    if arg_hash in lines_cache:
        return lines_cache[arg_hash]

    lines = []

    line_to_draw = None
    line = ""
    words = text.split()
    word_index = 0
    total_height = 0
    while word_index < len(words):
        word = words[word_index]
        line += word + " "
        text_size = get_text_size_ex(line.strip(), font, font_size, stroke_width)
        text_width = text_size[0]
        line_height = text_size[1]

        if text_width < frame_width:
            line_to_draw = {
                "text": line.strip(),
                "height": line_height,
            }
            word_index += 1
        else:
            if not line_to_draw:
                print(f"NOTICE: Word '{line.strip()}' is too long for the frame!")
                line_to_draw = {
                    "text": line.strip(),
                    "height": line_height,
                }
                word_index += 1

            lines.append(line_to_draw)
            total_height += line_height
            line_to_draw = None
            line = ""

    if line_to_draw:
        lines.append(line_to_draw)
        total_height += line_height

    data = {
        "lines": lines,
        "height": total_height,
    }

    lines_cache[arg_hash] = data

    return data

def ffmpeg(command):
    return subprocess.run(command, capture_output=True)

def create_shadow(text: str, font_size: int, font: str, blur_radius: float, opacity: float=1.0):
    global shadow_cache

    arg_hash = hash((text, font_size, font, blur_radius, opacity))

    if arg_hash in shadow_cache:
        return shadow_cache[arg_hash].copy()

    shadow = create_text_ex(text, font_size, "black", font, opacity=opacity)
    shadow = blur_text_clip(shadow, int(font_size*blur_radius))

    shadow_cache[arg_hash] = shadow.copy()

    return shadow



def get_font_path(font):
    # Check if Font Exists Directly:
    if os.path.exists(font):
        return font

    # Get the current working directory
    dirname = os.path.abspath('')

    # Search in Assets Folder:
    font = os.path.join(dirname, "assets", "fonts", font)

    if not os.path.exists(font):
        raise FileNotFoundError(f"Font '{font}' not found")

    return font

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


def add_captions(
    video_file,
    output_file = "with_transcript.mp4",

    font = "Bangers-Regular.ttf",
    font_size = 80,
    font_color = "yellow",

    stroke_width = 3,
    stroke_color = "black",

    highlight_current_word = True,
    word_highlight_color = "red",

    line_count = 2,
    fit_function = None,

    padding = 30,
    position = ("center", "center"), # TODO: Implement this

    shadow_strength = 1.0,
    shadow_blur = 0.1,

    print_info = False,

    initial_prompt = None,
    segments = None,

):
    _start_time = time.time()

    font = get_font_path(font)


    if print_info:
        print("Generating video elements...")

    # Open the video file
    video = VideoFileClip(video_file)
    text_bbox_width = video.w-padding*2
    clips = [video]

    captions = segment_parser.parse(
        segments=segments,
        fit_function=fit_function if fit_function else fits_frame(
            line_count,
            font,
            font_size,
            stroke_width,
            text_bbox_width,
        ),
    )

    for caption in captions:
        captions_to_draw = []
        if highlight_current_word:
            for i, word in enumerate(caption["words"]):
                if i+1 < len(caption["words"]):
                    end = caption["words"][i+1]["start"]
                else:
                    end = word["end"]

                captions_to_draw.append({
                    "text": caption["text"],
                    "start": word["start"],
                    "end": end,
                })
        else:
            captions_to_draw.append(caption)

        for current_index, caption in enumerate(captions_to_draw):
            line_data = calculate_lines(caption["text"], font, font_size, stroke_width, text_bbox_width)

            #text_y_offset = video.h // 2 - line_data["height"] // 2
            #original above


            #    # Base Y position from bottom with padding
            base_y_position = video.h - padding
            ## Calculate total height for all lines
            total_height = line_data["height"]
            # Adjust Y offset to pull it up a little if needed (e.g., by 20 pixels)
            text_y_offset = base_y_position - total_height - 370  # Adjust '20' as needed for spacing



            index = 0
            for line in line_data["lines"]:
                pos = ("center", text_y_offset)
                #pos = ("center", video.h - padding - line_data["height"])


                words = line["text"].split()
                word_list = []
                for w in words:
                    word_obj = Word(w)
                    if highlight_current_word and index == current_index:
                        word_obj.set_color(word_highlight_color)
                    index += 1
                    word_list.append(word_obj)

                # Create shadow
                shadow_left = shadow_strength
                while shadow_left >= 1:
                    shadow_left -= 1
                    shadow = create_shadow(line["text"], font_size, font, shadow_blur, opacity=1)
                    shadow = shadow.set_start(caption["start"])
                    shadow = shadow.set_duration(caption["end"] - caption["start"])
                    shadow = shadow.set_position(pos)
                    clips.append(shadow)

                if shadow_left > 0:
                    shadow = create_shadow(line["text"], font_size, font, shadow_blur, opacity=shadow_left)
                    shadow = shadow.set_start(caption["start"])
                    shadow = shadow.set_duration(caption["end"] - caption["start"])
                    shadow = shadow.set_position(pos)
                    clips.append(shadow)

                # Create text
                text = create_text_ex(word_list, font_size, font_color, font, stroke_color=stroke_color, stroke_width=stroke_width)
                text = text.set_start(caption["start"])
                text = text.set_duration(caption["end"] - caption["start"])
                text = text.set_position(pos)
                clips.append(text)

                text_y_offset += line["height"]

    end_time = time.time()
    generation_time = end_time - _start_time

    if print_info:
        print(f"Generated in {generation_time//60:02.0f}:{generation_time%60:02.0f} ({len(clips)} clips)")

    if print_info:
        print("Rendering video...")

    video_with_text = CompositeVideoClip(clips)

    # video_with_text.write_videofile(
    #     filename=output_file,
    #     codec="h264_nvenc",  # Use NVIDIA NVENC for H.264 encoding
    #     fps=30,
    #     threads=2,  # Let FFmpeg decide the number of threads
    #     logger="bar" if print_info else None,
    #     ffmpeg_params=[
    #         "-preset", "slow",      # Use 'medium' for a balance between speed and quality
    #         "-b:v", "3500k",          # Set target bitrate to 2500 kbps (2.5 Mbps) for good quality
    #         "-maxrate:v", "4000k",    # Maximum bitrate set to 2500 kbps
    #         "-bufsize:v", "8000k",
    #         "-crf", "23",
    #         "-pix_fmt", "yuv420p"                        # Buffer size (2x maxrate is a good rule of thumb)
    #     ]
    # )


    video_with_text.write_videofile(
    filename=output_file,
    codec="libx264",
    fps=30,
    threads=6,
    #logger="bar" if print_info else None,
    logger=None,
    ffmpeg_params=[
        "-preset", "slow",
        "-crf", "23",
        "-bufsize", "8000k",
        "-maxrate", "4000k",
        "-pix_fmt", "yuv420p"
    ]
)





    end_time = time.time()
    total_time = end_time - _start_time
    render_time = total_time - generation_time

    if print_info:
        print(f"Generated in {generation_time//60:02.0f}:{generation_time%60:02.0f}")
        print(f"Rendered in {render_time//60:02.0f}:{render_time%60:02.0f}")
        print(f"Done in {total_time//60:02.0f}:{total_time%60:02.0f}")



def load_video_segments(transcript_folder):
    transcript_files = [f for f in os.listdir(transcript_folder) if f.endswith('_transcription.txt')]

    if not transcript_files:
        raise FileNotFoundError("No transcript files found in the specified folder.")

    transcript_filename = os.path.join(transcript_folder, sorted(transcript_files)[-1])

    segments = []

    with open(transcript_filename, 'r') as f:
        for line in f:
            #print("Reading line:", line)  # This will show you each line being read

            match = re.match(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(.*?)\s+(\[.*\])$', line.strip())
            if match:
                start_time, end_time, text, words_str = match.groups()

                try:
                    # Safely evaluate words_str using ast.literal_eval
                    words = ast.literal_eval(words_str)

                    # Ensure words are properly formatted (if necessary)
                    if not isinstance(words, list):
                        raise ValueError(f"Expected a list but got {type(words)}")

                except Exception as e:
                    print(f"Error parsing words for segment: {text}. Exception: {e}")
                    words = []  # Fallback to an empty list if parsing fails

                segments.append({
                    'start': float(start_time),
                    'end': float(end_time),
                    'text': text,
                    'words': words,
                })

    print(f"Loaded {len(segments)} segments from {transcript_filename}.")
    return segments



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




def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)

    print(colored(f"Fetching image from the server: {server_address}/view", "cyan"))
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    print(colored(f"Fetching history for prompt ID: {prompt_id}.", "cyan"))
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def reconnect_websocket():
    ws = websocket.WebSocket()
    ws_url = f"ws://{server_address}/ws?clientId={client_id}"
    print(colored(f"Reconnecting WebSocket to {ws_url}", "cyan"))
    ws.connect(ws_url)
    return ws

def get_output_images(prompt_id):
    output_images = {}
    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            images_output = []
            for image in node_output['images']:
                print(colored(f"Downloading image: {image['filename']} from the server.", "yellow"))
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
            output_images[node_id] = images_output
    return output_images
#==================================================================
#REVISED OF THE BELOW


def generate_images(ws, prompt_id):
    output_images = {}
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
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print(colored("Execution complete.", "green"))
                        return get_output_images(prompt_id)
        except WebSocketTimeoutException:
            print(colored(f"No message received for {timeout} seconds. Checking connection...", "yellow"))
            try:
                ws.ping()
            except:
                ws = reconnect_websocket()

    return output_images


server_address = "54.80.78.81:8253"
client_id = str(uuid.uuid4())

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
    if transcript_text:
        log_transcript(transcript_text, new_video_title)
    print(f"Running media pipeline for: {new_video_title}")

    # Save script for image prompt context
    output_script_path = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json'
    with open(output_script_path, 'w') as f:
        json.dump({"script": script}, f)

    # Step 4: Generate Audio
    VOICE_OVER_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/voice_over"
    ensure_folder_exists(VOICE_OVER_FOLDER)
    audio_path = os.path.join(VOICE_OVER_FOLDER, "voiceover.wav")
    print("Generating audio...")
    generate_audio(script, audio_path)
    print(f"Audio saved to {audio_path}")

    # Step 5: Transcribe Audio
    print("Transcribing audio...")
    transcription = transcribe_locally(audio_path)
    print("Transcription completed.")
    TRANSCRIPT_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/transcripts"
    ensure_folder_exists(TRANSCRIPT_FOLDER)
    transcript_filename = os.path.join(TRANSCRIPT_FOLDER, f"{os.path.basename(audio_path).replace('.wav', '')}_transcription.txt")
    with open(transcript_filename, 'w') as f:
        if isinstance(transcription, list):
            for segment in transcription:
                words_json = json.dumps(segment['words'], default=lambda x: float(x) if isinstance(x, np.float64) else x)
                f.write(f"{segment['start']} {segment['end']} {segment['text']} {words_json}\n")
        else:
            f.write(transcription)
    print(f"Transcription saved to {transcript_filename}")
    
    segments = extract_transcript_segments(transcription)
    if isinstance(segments, tuple):
        segments = segments[0]
    print(f"Extracted {len(segments)} raw transcription segments from the transcript.")
    
    # Step 5b: Process transcript for image-video segments
    print("Processing transcript for image-video segments...")
    transcript_data = parse_transcript_file(transcript_filename)
    segments_image_vid = extract_transcript_segments_image_vid(transcript_data)
    output_file = os.path.join(TRANSCRIPT_FOLDER, 'image_video.json')
    with open(output_file, 'w') as f:
        json.dump(segments_image_vid, f, indent=4)
    print(f"Extracted {len(segments_image_vid)} image-video segments from the transcript.")
    print(f"Image-video segments have been saved to {output_file}")
    


    # Step 5c: Generate image prompts for segments, sending 1 batch at a time and waiting 2 seconds between batches
    print("Generating image prompts for segments...")
    full_transcript_file = '/home/ubuntu/crewgooglegemini/001videototxt/output_transcript.json'
    with open(full_transcript_file, 'r') as f:
        full_transcript = json.load(f)
    if isinstance(full_transcript, dict):
        full_transcript = full_transcript.get('script', '')
    elif not isinstance(full_transcript, str):
        full_transcript = str(full_transcript)
    with open(output_file, 'r') as f:
        segments = json.load(f)
    batch_size = 5
    image_prompts = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        print(f"Sending batch {i//batch_size + 1}: {len(batch)} segments")
        try:
            batch_prompts = generate_image_prompts_batch(batch, full_transcript)
            image_prompts.extend(batch_prompts)
            # Save after each batch if needed
            prompt_file = '/home/ubuntu/crewgooglegemini/current_prompt.json'
            with open(prompt_file, 'w') as f:
                json.dump(image_prompts, f, indent=4)
            print(f"Batch {i//batch_size + 1} processed and saved. Waiting 2 seconds before next batch...")
            time.sleep(2)
        except Exception as e:
            print(f"An error occurred in batch {i//batch_size + 1}: {e}")
            break  # Stop processing on error (optional: implement retry/backoff here)
    print(f"Generated {len(image_prompts)} image prompts and saved to {prompt_file}")



    # Step 5d: Generate images for each prompt
    print("Step 5d: Generating images for each prompt...")
    process_and_generate_images()

    print("Step 5e: Generating thumbnail prompt...")
    thumbnail_prompt = generate_thumbnail_prompt_purple_cow(full_transcript)


    THUMBNAIL_DIR = "/home/ubuntu/crewgooglegemini/CAPTACITY/assets/thumbnails"
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    base_name = new_video_title.replace(" ", "_")
    
    # 720x1280 (vertical)
    output_path_720x1280 = os.path.join(THUMBNAIL_DIR, f"{base_name}_thumbnail_720X1280.png")
    generate_thumbnail_image(thumbnail_prompt, 720, 1280, output_path_720x1280)
    
    # 1280x720 (horizontal)
    # output_path_1280x720 = os.path.join(THUMBNAIL_DIR, f"{base_name}_thumbnail_1280X720.png")
    # generate_thumbnail_image(thumbnail_prompt, 1280, 720, output_path_1280x720)
    


    # Step 5e: Animate images to video clips (one per segment, no overlay/audio yet)
    step_5e_generate_video_clips()

    # Simulate video search and download
    PIXABAY_FOLDER = "/home/ubuntu/crewgooglegemini/PIXABAY VIDEOS"
    CUSTOM_FOLDER = "/home/ubuntu/crewgooglegemini/SHORTCLIPSFACTS/WellnessGram"
    TRANSCRIPT_FOLDER = "/home/ubuntu/crewgooglegemini/001videototxt/transcripts"
    downloaded_videos, video_titles, durations = search_and_download_best_videos(PIXABAY_FOLDER, CUSTOM_FOLDER)
    print(f"Downloaded {len(downloaded_videos)} videos.")
    for i, (video_path, title, duration) in enumerate(zip(downloaded_videos, video_titles, durations), start=1):
        print(f"Video {i}: {title} ({duration:.2f} seconds) at path: {video_path}")

    # Step 6: Concatenate Videos and Add Voice Over
    video_segments = load_video_segments(TRANSCRIPT_FOLDER)
    voice_over_file = audio_path

    print("Concatenating videos and adding voice over...")
    video_clips = []
    for i, (video_path, duration) in enumerate(zip(downloaded_videos, durations)):
        clip = VideoFileClip(video_path)
        if clip.duration > 0:
            if clip.duration < duration:
                loop_count = int(duration // clip.duration) + 1
                extended_clip = concatenate_videoclips([clip] * loop_count, method="compose")
                trimmed_clip = extended_clip.subclip(0, duration)
            else:
                trimmed_clip = clip.subclip(0, duration)
            video_clips.append(trimmed_clip)
        else:
            print(f"Warning: Skipping empty clip {video_path}")
    
    print(f"Number of clips to concatenate: {len(video_clips)}")
    if len(video_clips) > 0:
        final_clip = concatenate_videoclips(video_clips, method="compose")
    else:
        raise ValueError("No valid video clips available for concatenation.")

    # Add overlay after final_clip is created
    overlays_dir = "/home/ubuntu/crewgooglegemini/CAPTACITY/assets/overlays"
    overlay_files = [f for f in os.listdir(overlays_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    
    if overlay_files:
        overlay_path = os.path.join(overlays_dir, random.choice(overlay_files))
        print(f"Using overlay: {overlay_path}")
        overlay_clip = VideoFileClip(overlay_path).resize(final_clip.size)
        # Adjust overlay duration to match final_clip
        if overlay_clip.duration < final_clip.duration:
            overlay_clip = overlay_clip.loop(duration=final_clip.duration)
        elif overlay_clip.duration > final_clip.duration:
            overlay_clip = overlay_clip.subclip(0, final_clip.duration)
        overlay_clip = overlay_clip.set_opacity(0.1).set_duration(final_clip.duration)
        final_clip = CompositeVideoClip([final_clip, overlay_clip]).set_duration(final_clip.duration)
        print("Overlay applied to final video.")
    else:
        print("No overlay videos found. Skipping overlay step.")


    audio = AudioFileClip(audio_path)
    if final_clip.duration < audio.duration:
        difference = audio.duration - final_clip.duration
        print(f"Video is {difference:.2f} seconds shorter than the audio. Extending video.")
        last_clip = video_clips[-1]
        loops_needed = int(difference // last_clip.duration) + 1
        extension_clip = concatenate_videoclips([last_clip] * loops_needed)
        extension_clip = extension_clip.subclip(0, difference)
        final_clip = concatenate_videoclips([final_clip, extension_clip], method="compose")
    final_clip = final_clip.subclip(0, audio.duration)

    # Step 7: Add background sound
    bg_sound = get_background_sound(audio.duration)
    combined_audio = CompositeAudioClip([audio, bg_sound])
    final_clip = final_clip.set_audio(combined_audio)
    print(f"Final video duration: {final_clip.duration:.2f} seconds")
    print(f"Audio duration: {audio.duration:.2f} seconds")
    fade_duration = 0.4  # seconds
    final_clip = final_clip.fx(fadeout, duration=fade_duration)
    final_audio = final_clip.audio.fx(audio_fadeout, duration=fade_duration)
    final_clip = final_clip.set_audio(final_audio)

    # Step 8: Write Final Video and Add Captions
    output_dir = "/home/ubuntu/crewgooglegemini/FINALVIDEOS/WellnessGram"
    safe_title = make_safe_filename(new_video_title)
    os.makedirs(output_dir, exist_ok=True)
    intermediate_output = os.path.join(output_dir, f"{safe_title}_intermediate_video_with_audio.mp4")
    final_clip.write_videofile(
        intermediate_output,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            "-bufsize", "16M",
            "-maxrate", "8M",
            '-profile:v', 'high'
        ],
        threads=8,
        verbose=False
    )

    output_file_path = os.path.join(output_dir, f"{safe_title}.mp4")
    add_captions(
        video_file=intermediate_output,
        output_file=output_file_path,
        font="Bangers-Regular.ttf",
        font_size=80,
        font_color="yellow",
        stroke_width=3,
        stroke_color="black",
        highlight_current_word=True,
        word_highlight_color="red",
        line_count=2,
        padding=50,
        shadow_strength=1.0,
        shadow_blur=0.1,
        print_info=True,
        segments=video_segments
    )
#=============================================================================================================================
    # --- INSERT ENDSCREEN LOGIC HERE ---crewgooglegemini/PodcastProd/Endanim
    ENDSCREEN_ANIMATIONS_FOLDER = "/home/ubuntu/crewgooglegemini/PodcastProd/Endanim"
    endscreen_video_file = get_random_endscreen(ENDSCREEN_ANIMATIONS_FOLDER)
    if endscreen_video_file:
        base_name_for_final_output = os.path.splitext(os.path.basename(output_file_path))[0]
        final_video_with_endscreen_path = os.path.join(
            output_dir, f"{base_name_for_final_output}_mmp4.mp4"
        )
        print(f"Appending endscreen '{endscreen_video_file}' to '{output_file_path}' -> '{final_video_with_endscreen_path}'")
        append_success = append_endscreen_to_video(
            main_video_path=output_file_path,
            endscreen_video_path=endscreen_video_file,
            final_output_path=final_video_with_endscreen_path
        )
        if append_success and os.path.exists(final_video_with_endscreen_path):
            print(f"Endscreen append successful. New final video: {final_video_with_endscreen_path}")
            # Optionally remove the video without endscreen
            # os.remove(output_file_path)
            output_file_path = final_video_with_endscreen_path
        else:
            print(f"[ERROR] Failed to append endscreen or output file missing. Using previous video: {output_file_path}")
    else:
        print(f"No endscreen video found/selected in {ENDSCREEN_ANIMATIONS_FOLDER}. Skipping append step.")

    # Continue with cleanup and distribution
    if os.path.exists(intermediate_output):
        os.remove(intermediate_output)
    if os.path.exists(CUSTOM_FOLDER):
        for file in os.listdir(CUSTOM_FOLDER):
            if file.endswith(".mp4"):
                file_path = os.path.join(CUSTOM_FOLDER, file)
                os.remove(file_path)
    print(f"Final video with captions saved as {output_file_path}")
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
