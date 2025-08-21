import json
import os
import logging

def load_speaker_map(path="speaker_map.json"):
    '''Load the speaker map from a JSON file.'''
    
    if os.path.exists(path):
        with open(path) as f:
            logging.info("Loaded speaker map.")
            return json.load(f)
    logging.info("No speaker_map.json found. Defaulting to raw speaker labels.")
    return {}

def match_speakers(transcript_segments, diarization_segments, speaker_map):
    '''Match speakers to transcript segments based on diarization data.'''
    
    logging.info("Matching speakers to transcript...")
    matched = []
    for seg in transcript_segments:
        speaker = next(
            (d["speaker"] for d in diarization_segments if d["start"] <= (seg["start"] + seg["end"]) / 2 <= d["end"]),
            "Unknown"
        )
        matched.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": speaker_map.get(speaker, speaker)
        })
    return matched
