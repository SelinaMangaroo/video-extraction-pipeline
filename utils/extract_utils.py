import subprocess
import logging
import os

# Extract audio from video using ffmpeg, used in diarization.py
def extract_audio(video_path, audio_path):
    logging.info("Extracting audio...")
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, check=True)

# Extract single frame with ffmpeg, used in captioning.py
def extract_frame_with_ffmpeg(video_path, time_sec, output_path):
    time_str = f"{time_sec:.3f}"
    cmd = [
        "ffmpeg", "-y",
        "-ss", time_str,
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if os.path.exists(output_path):
        logging.info(f"Frame successfully extracted: {output_path}")
    else:
        logging.info(f"Failed to extract frame at {time_sec}s → {output_path}")
        logging.info(result.stderr.decode().strip())