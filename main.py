import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

from utils.logging_utils import setup_logger
from utils.audio_utils import extract_audio
from utils.whisper_utils import run_whisper
from utils.diarization_utils import run_diarization
from utils.speaker_utils import load_speaker_map, match_speakers

# === Load environment variables ===
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
VIDEO_PATH = os.getenv("VIDEO_PATH")

if not VIDEO_PATH:
    raise ValueError("VIDEO_PATH is not set in your .env file")

AUDIO_PATH = "tmp/audio.wav"
os.makedirs("tmp", exist_ok=True)

def main():
    start_time = time.time()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    log_path = setup_logger(VIDEO_PATH)
    print(f"Log file saved to: {log_path}")
    logging.info("Started processing")

    speaker_map = load_speaker_map()

    extract_audio(VIDEO_PATH, AUDIO_PATH)
    transcript = run_whisper(AUDIO_PATH)
    diarization = run_diarization(AUDIO_PATH, HUGGINGFACE_TOKEN)
    results = match_speakers(transcript, diarization, speaker_map)

    for r in results:
        print(f"[{r['speaker']}] {r['start']:.2f}–{r['end']:.2f}: {r['text']}")

    # Save transcript
    video_stem = Path(VIDEO_PATH).stem
    output_path = os.path.join(output_dir, f"{video_stem}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Transcript saved to: {output_path}")

    # Clean up
    try:
        os.remove(AUDIO_PATH)
        logging.info(f"Temporary audio file deleted: {AUDIO_PATH}")
    except Exception as e:
        logging.warning(f"Could not delete temp audio file: {e}")

    duration = time.time() - start_time
    logging.info(f"Total processing time: {duration:.2f} seconds")
    print(f"Total processing time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()
