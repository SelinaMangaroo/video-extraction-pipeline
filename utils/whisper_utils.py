import whisper
import logging

def run_whisper(audio_path):
    logging.info("Transcribing with Whisper...")
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path, word_timestamps=True)
    return result["segments"]
