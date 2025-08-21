import os
import time
import json
import subprocess
import logging
import argparse
from datetime import timedelta
from dotenv import load_dotenv
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from utils.logging_utils import setup_logger
from utils.chat_gpt_utils import caption_scene_with_images, build_prompt_and_images
import utils.chat_gpt_utils as chat_gpt_utils
from utils.batch_utils import submit_and_parse_batch 
from utils.extract_utils import extract_frame_with_ffmpeg
from utils.whisper_utils import transcribe_video_with_whisper

# CLI argument parsing
parser = argparse.ArgumentParser(description="Video captioning pipeline")
parser.add_argument("--input", required=False, help="Path to video file or directory")
parser.add_argument("--output", required=False, default="captions", help="Path to output directory")
parser.add_argument("--log", required=False, default="logs", help="Path to log directory")
args = parser.parse_args()

# Load environment variables
load_dotenv()
VIDEO_PATH = args.input or os.getenv("VIDEO_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PY_SCENE_DETECT_THRESHOLD = float(os.getenv("PY_SCENE_DETECT_THRESHOLD", 50.0))
MIN_SCENE_LENGTH = int(os.getenv("MIN_SCENE_LENGTH", 60))
NUM_FRAMES_PER_SCENE = int(os.getenv("NUM_FRAMES_PER_SCENE", 4))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", 'tiny')
TRANSCRIPT_CACHE_DIR = os.getenv("TRANSCRIPT_CACHE_DIR")
TRANSCRIPT_REPEAT_THRESHOLD = int(os.getenv("TRANSCRIPT_REPEAT_THRESHOLD", 8))
INCLUDE_TRANSCRIPT_IN_GPT = os.getenv("INCLUDE_TRANSCRIPT_IN_GPT", "true").lower() in ("true", "1", "yes")
CHAT_GPT_MODEL = os.getenv("CHAT_GPT_MODEL", "gpt-4o-mini")
CHAT_GPT_RETRIES = int(os.getenv("CHAT_GPT_RETRIES", 10))
CAPTION_DIR = args.output or os.getenv("CAPTIONS_DIR", "captions")
LOGS_DIR = args.log or os.getenv("LOGS_DIR", "logs")
USE_BATCH_API = os.getenv("USE_BATCH_API", "true").lower() in ("true", "1", "yes")

def get_scene_frames(video_path, output_dir="frames"):
    '''Extract frames from video scenes using SceneDetect and save them to output directory.'''
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = os.path.join(output_dir, f"{video_name}_frames")
    os.makedirs(output_dir, exist_ok=True)

    # Get accurate duration using ffprobe
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ], text=True)
    total_duration = float(json.loads(out)["format"]["duration"])
    SAFETY = 0.25  # margin to avoid end of frame issues

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=PY_SCENE_DETECT_THRESHOLD, min_scene_len=MIN_SCENE_LENGTH))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    logging.info(f"Detected {len(scenes)} scenes in {VIDEO_PATH}.")

    scene_data = []

    for i, (start, end) in enumerate(scenes, 1):
        start_sec = start.get_seconds()
        end_sec = min(end.get_seconds(), total_duration - SAFETY)

        if end_sec <= start_sec:
            logging.info(f"Skipping scene {i:03}: start={start_sec:.3f} end={end_sec:.3f}")
            continue

        times = {"start": start_sec, "end": end_sec}
        step = (end_sec - start_sec) / (NUM_FRAMES_PER_SCENE + 1)
        for j in range(NUM_FRAMES_PER_SCENE):
            times[f"t{j+1}"] = start_sec + (j + 1) * step

        frame_paths = {}
        for tag, t in times.items():
            t = max(0.0, min(t, total_duration - SAFETY))
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
    """Check if any word or phrase repeats consecutively more than 'threshold' times."""
    
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
    """Run the captioning process for a video using GPT and save results to output directory."""
    
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_captions.json")
    captions = {}

    batch_requests = []
    scene_id_map = {}

    for scene in scenes:
        scene_id = f"scene_{scene['scene_number']:03}"
        
        scene_text = " ".join(
            seg["text"].strip()
            for seg in transcript_segments
            if seg.get("end", 0) > scene["start_time"] and seg.get("start", 0) < scene["end_time"] and seg.get("text", "").strip()
        ).strip()

        if scene_text and has_excessive_repeats(scene_text):
            logging.warning(f"Scene {scene_id}: transcript text skipped due to excessive repeats.")
            scene_text = ""

        transcript_for_prompt = scene_text if INCLUDE_TRANSCRIPT_IN_GPT else ""

        if USE_BATCH_API:
            full_prompt, image_inputs = build_prompt_and_images(transcript_for_prompt, scene["frames"])

            custom_id = f"{video_name}_{scene_id}"
            messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}] + image_inputs}]
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CHAT_GPT_MODEL,
                    "messages": messages,
                    "max_tokens": 400
                }
            })
            scene_id_map[custom_id] = {
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "frame_files": list(scene["frames"].values()),
                "transcript_text": scene_text
            }
            logging.info(f"{custom_id} → Queued for batch captioning.")
        else:
            caption, keywords = caption_scene_with_images(
                scene["frames"], api_key, transcript_for_prompt, CHAT_GPT_MODEL, CHAT_GPT_RETRIES
            )

            if not caption:
                logging.error(f"{scene_id} → Captioning failed\n")
                continue
            keywords = keywords or []

            captions[scene_id] = {
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "frame_files": list(scene["frames"].values()),
                "transcript_text": scene_text,
                "caption": caption.strip(),
                "keywords": keywords
            }
            logging.info(f"{scene_id} → Captions and keywords generated successfully\n")

    if USE_BATCH_API and batch_requests:
        captions_from_batch = submit_and_parse_batch(batch_requests, scene_id_map, api_key)
        captions.update(captions_from_batch)

    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)
        logging.info(f"Captions saved to {output_path}")
    
if __name__ == "__main__":
    start_time_all = time.time()

    if os.path.isdir(VIDEO_PATH):
        exts = {".mp4", ".mov", ".mpg", ".mpeg"}
        video_files = [
            os.path.join(VIDEO_PATH, f)
            for f in sorted(os.listdir(VIDEO_PATH))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not video_files:
            raise SystemExit(f"No supported video files found in directory: {VIDEO_PATH}")
    else:
        video_files = [VIDEO_PATH]

    for vp in video_files:
        # New log per video
        log_path = setup_logger(vp, output_dir=LOGS_DIR)
        start_time = time.time()
        logging.info(f"=== Starting video processing pipeline for: {vp} ===")

        logging.info("==== Starting Scene Detection ====")
        scenes = get_scene_frames(vp)
        logging.info(f"Extracted {len(scenes)} scenes from video.")

        logging.info("==== Starting Whisper Transcription ====")
        transcript_segments = transcribe_video_with_whisper(vp, TRANSCRIPT_CACHE_DIR, WHISPER_MODEL_SIZE)
        logging.info(f"Transcription complete with {len(transcript_segments)} segments.")

        logging.info("==== Starting GPT Captioning ====")
        run_captioning(vp, scenes, transcript_segments, OPENAI_API_KEY, output_dir=CAPTION_DIR)

        logging.info("Captioning process completed successfully.")

        logging.info(f"Total processing time for {vp}: {timedelta(seconds=int(time.time() - start_time))}")

    logging.info(f"Total processing time overall: {timedelta(seconds=int(time.time() - start_time_all))}")
    logging.info(f"Total time lost to rate limit backoffs: {chat_gpt_utils.TOTAL_RATE_LIMIT_BACKOFF:.1f} seconds")
    
    logging.info("=== GPT Token Usage Summary ===")
    logging.info(f"Prompt tokens:     {chat_gpt_utils.TOTAL_TOKENS_USED['prompt_tokens']}")
    logging.info(f"Completion tokens: {chat_gpt_utils.TOTAL_TOKENS_USED['completion_tokens']}")
    logging.info(f"Total tokens:      {chat_gpt_utils.TOTAL_TOKENS_USED['total_tokens']}")
    logging.info(f"Estimated total cost for all videos: ${chat_gpt_utils.TOTAL_COST['value']:.6f}")