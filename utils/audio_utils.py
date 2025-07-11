import subprocess
import logging

def extract_audio(video_path, audio_path):
    logging.info("Extracting audio...")
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, check=True)
