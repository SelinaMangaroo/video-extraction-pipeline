import os
import time
from datetime import timedelta
import json
import logging
import cv2
from dotenv import load_dotenv
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from utils.logging_utils import setup_logger
from utils.chat_gpt_utils import caption_scene_with_images
from utils.extract_utils import extract_frame_with_ffmpeg
from utils.whisper_utils import transcribe_video_with_whisper

# Load environment variables
load_dotenv()
VIDEO_PATH = os.getenv("VIDEO_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PY_SCENE_DETECT_THRESHOLD = float(os.getenv("PY_SCENE_DETECT_THRESHOLD", 50.0))
MIN_SCENE_LENGTH = int(os.getenv("MIN_SCENE_LENGTH", 60))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE")
TRANSCRIPT_CACHE_DIR = os.getenv("TRANSCRIPT_CACHE_DIR")
TRANSCRIPT_REPEAT_THRESHOLD = int(os.getenv("TRANSCRIPT_REPEAT_THRESHOLD", 8))

def get_scene_frames(video_path, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Get total duration to avoid overshooting on last frame
    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=PY_SCENE_DETECT_THRESHOLD, min_scene_len=MIN_SCENE_LENGTH))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    logging.info(f"Detected {len(scenes)} scenes in {VIDEO_PATH}.")

    scene_data = []

    for i, (start, end) in enumerate(scenes, 1):
        start_sec = start.get_seconds()
        end_sec = min(end.get_seconds(), total_duration - 0.1)  # clamp to avoid ffmpeg failure

        times = {
            "start": start_sec,
            "end": end_sec
        }

        # Add 4 evenly spaced points
        num_frames = 4
        step = (end_sec - start_sec) / (num_frames + 1)
        for j in range(num_frames):
            times[f"t{j+1}"] = start_sec + (j + 1) * step

        frame_paths = {}
        for tag, t in times.items():
            if tag == "end":
                t = max(0, min(t, total_duration - 0.1)) 
            filename = f"{video_name}_scene_{i:03}_{tag}.jpg"
            filepath = os.path.join(output_dir, filename)
            extract_frame_with_ffmpeg(video_path, t, filepath)
            frame_paths[tag] = filepath
            logging.info(f"Scene {i:03} ({tag}): {t:.2f}s → {filepath}")

        scene_data.append({
            "scene_number": i,
            "start_time": start_sec,
            "end_time": end_sec,
            "frames": frame_paths
        })

    return scene_data

def has_excessive_repeats(text):
    """
    Check if any word or phrase repeats consecutively more than `threshold` times.
    """
    words = text.split()

    # Check repeated words
    count = 1
    last_word = None
    for word in words:
        if word == last_word:
            count += 1
            if count >= TRANSCRIPT_REPEAT_THRESHOLD:
                return True
        else:
            count = 1
            last_word = word

    # Check repeated phrases (2 to 20 words)
    for size in range(2, min(20, len(words) // 2)):
        for i in range(len(words) - size * TRANSCRIPT_REPEAT_THRESHOLD + 1):
            phrase = words[i:i + size]
            repeated = True
            for j in range(1, TRANSCRIPT_REPEAT_THRESHOLD):
                if words[i + j * size:i + (j + 1) * size] != phrase:
                    repeated = False
                    break
            if repeated:
                return True
    return False

def run_captioning(video_path, scenes, transcript_segments, api_key, output_dir="captions"):
    logging.info("\nStarting captioning...\n")
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_captions.json")
    captions = {}

    # extract transcript for the given scene range
    def get_scene_transcript(scene_start, scene_end, segments):
        return " ".join(
            seg["text"].strip()
            for seg in segments
            if seg.get("end", 0) > scene_start and seg.get("start", 0) < scene_end and seg.get("text", "").strip()
        ).strip()
    
    for scene in scenes:
        scene_id = f"scene_{scene['scene_number']:03}"
        scene_text = get_scene_transcript(scene["start_time"], scene["end_time"], transcript_segments)
        
        # Skip transcript if it has excessive repeats
        if scene_text and has_excessive_repeats(scene_text):
            logging.warning(f"Scene {scene_id}: transcript text skipped due to excessive repeats.")
            scene_text = ""
        
        caption, keywords = caption_scene_with_images(scene["frames"], api_key, scene_text)
        
        # normalize return just in case
        if not caption:
            logging.error(f"{scene_id} → Captioning failed\n")
            continue
        if keywords is None:
            keywords = []

        if caption:
            captions[scene_id] = {
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "frame_files": list(scene["frames"].values()),
                "transcript_text": scene_text,
                "caption": caption.strip(),
                "keywords": keywords
            }
            # logging.info(f"{scene_id} → {caption}\nKeywords: {keywords}\n")
            logging.info(f"{scene_id} → Captions and keywords generated successfully\n")
        else:
            logging.error(f"{scene_id} → Captioning failed\n")

    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)
        logging.info(f"Captions saved to {output_path}")
    
if __name__ == "__main__":
    log_path = setup_logger(VIDEO_PATH)
    start_time = time.time()
    logging.info("Starting video processing pipeline...")
    
    logging.info("==== Starting Scene Detection ====")
    scenes = get_scene_frames(VIDEO_PATH)
    logging.info(f"Extracted {len(scenes)} scenes from video.")

    logging.info("==== Starting Whisper Transcription ====")
    transcript_segments = transcribe_video_with_whisper(VIDEO_PATH, TRANSCRIPT_CACHE_DIR, WHISPER_MODEL_SIZE)
    logging.info(f"Transcription complete with {len(transcript_segments)} segments.")

    logging.info("==== Starting GPT Captioning ====")
    run_captioning(VIDEO_PATH, scenes, transcript_segments, OPENAI_API_KEY)
    logging.info("Captioning process completed successfully.")
    
    logging.info(f"Total processing time: {timedelta(seconds=int(time.time() - start_time))}")
