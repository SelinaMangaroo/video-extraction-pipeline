import whisper
import logging
import json
import os

# Run Whisper transcription on the audio file, used in diarization.py
def run_whisper(audio_path):
    logging.info("Transcribing with Whisper...")
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path, word_timestamps=True)
    return result["segments"]

# Transcribe video using Whisper and cache the result, used in captioning.py
def transcribe_video_with_whisper(video_path, TRANSCRIPT_CACHE_DIR, WHISPER_MODEL_SIZE="tiny"):
    os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{video_name}.json")
    
    # Load cached transcript if it exists
    if os.path.exists(cache_path):
        try:
            logging.info(f"Loading cached transcript from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Cache file {cache_path} is corrupted. Re-running Whisper...")

    logging.info("Transcribing video with Whisper...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    result = model.transcribe(video_path, verbose=False)

    segments = result.get("segments") or []
    if not segments:
        logging.warning("No speech segments detected in video.")

    try:
        with open(cache_path, "w") as f:
            json.dump(segments, f, indent=2)
            logging.info(f"Transcript saved to {cache_path}")
    except Exception as e:
        logging.error(f"Failed to save transcript cache: {e}")

    return segments