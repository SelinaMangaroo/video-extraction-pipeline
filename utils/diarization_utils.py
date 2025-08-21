from pyannote.audio import Pipeline
import logging

def run_diarization(audio_path, token):
    '''Run speaker diarization on the audio file using PyAnnote and return the segments.'''
    
    logging.info("Running speaker diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments
