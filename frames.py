import os
import subprocess
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import base64
import requests
import glob
import json

from dotenv import load_dotenv
load_dotenv()

VIDEO_PATH = os.getenv("VIDEO_PATH")
PY_SCENE_DETECT_THRESHOLD = float(os.getenv("PY_SCENE_DETECT_THRESHOLD", 50.0))
MIN_SCENE_LENGTH = int(os.getenv("MIN_SCENE_LENGTH", 60))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

"""
    Extracts mid frames from detected scenes in a video using ffmpeg.
"""
def extract_frame_with_ffmpeg(video_path, time_sec, output_path):
    time_str = f"{time_sec:.3f}"   # Format time to seconds with milliseconds
    cmd = [
        "ffmpeg", "-y", # overwrite output file if it exists
        "-ss", time_str, # seek to the specified time
        "-i", video_path, # input video file
        "-frames:v", "1", # extract one frame
        "-q:v", "2",   # quality level (lower is better, 2 is good)
        output_path 
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

"""
    Extracts mid frames from detected scenes in a video and saves them to the specified output directory.
    Uses ffmpeg for frame extraction.
    Scenes are detected using the ContentDetector from scenedetect with a specified threshold.
    The mid frame of each scene is saved as a JPEG image.
    Scenes are detected based on content changes in the video.
"""
def extract_mid_frames(video_path, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name from video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open video
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=PY_SCENE_DETECT_THRESHOLD, min_scene_len=MIN_SCENE_LENGTH))

    # Detect scenes
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    
    for i, (start, end) in enumerate(scenes, 1):
        mid_time = (start.get_seconds() + end.get_seconds()) / 2
        filename = f"{video_name}_scene_{i:03}.jpg"
        filepath = os.path.join(output_dir, filename)
        extract_frame_with_ffmpeg(video_path, mid_time, filepath)
        print(f"Scene {i:03}: {start} → {end} | Midpoint: {mid_time:.2f}s | Saved: {filepath}")
    
    return scenes

def caption_image_with_chatgpt(image_path, OPENAI_API_KEY, prompt="Describe this image in one sentence."):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
    data = response.json()
    
    try:
        return data["choices"][0]["message"]["content"]
    except KeyError:
        print("Error:", data)
        return None
    
def run_captioning_on_frames(video_path, scenes, output_dir="captions"):
    print("\nStarting image captioning...\n")
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(output_dir, f"{video_name}_captions.json")

    captions = {}

    for i, (start, end) in enumerate(scenes, 1):
        frame_filename = f"{video_name}_scene_{i:03}.jpg"
        frame_path = os.path.join("frames", frame_filename)
        mid_time = (start.get_seconds() + end.get_seconds()) / 2

        caption = caption_image_with_chatgpt(frame_path, OPENAI_API_KEY)
        if caption:
            captions[frame_filename] = {
                "start_time": str(start),
                "end_time": str(end),
                "midpoint_seconds": round(mid_time, 2),
                "caption": caption.strip()
            }
            print(f"{frame_filename} → {caption}")
        else:
            print(f"{frame_filename} → Captioning failed")

    with open(json_path, "w") as f:
        json.dump(captions, f, indent=2)
        print(f"\n Captions saved to {json_path}")

scenes = extract_mid_frames(VIDEO_PATH)
run_captioning_on_frames(VIDEO_PATH, scenes)
