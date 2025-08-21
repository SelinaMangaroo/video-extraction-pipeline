# Video Captioning Pipeline

This project is a Python pipeline for automatically captioning video scenes using AI. It performs the following steps:

1. Detects scenes using PySceneDetect  
2. Extracts key frames using FFmpeg  
3. Transcribes audio using Whisper  
4. Uses OpenAI GPT to generate captions and searchable keywords per scene  

Supports the Batch APIs for GPT, with token usage tracking and cost estimation included.

---

## Features

- Automatic scene detection
- Frame extraction per scene
- Whisper transcription
- GPT-4o-based scene captioning
- Keyword tagging for search
- Logs usage stats + estimated cost
- Works on individual videos or full folders
- Skips scenes with excessive repeated transcript text

---

## Project Structure

```
project_root/
├── captioning.py                # Main pipeline script
├── .env                         # Environment config
├── requirements.txt             # Dependencies
├── utils/
│   ├── chat_gpt_utils.py        # GPT captioning logic
│   ├── batch_utils.py           # OpenAI batch API handling
│   ├── whisper_utils.py         # Whisper transcription
│   ├── extract_utils.py         # Frame extraction (FFmpeg)
│   └── logging_utils.py         # Log setup
├── logs/                        # Per-video logs
├── captions/                    # Final output JSONs
├── frames/                      # Extracted scene frames
└── transcripts/                 # Transcription cache
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SelinaMangaroo/video-captioning-pipeline.git
cd video-captioning-pipeline
```

### 2. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up `.env`

## Environment Variables

The project uses a `.env` file for configuration. Create a `.env` file. Below is a description of each variable:

| Variable                     | Example Value          | Description                                                                  |
|------------------------------|------------------------|------------------------------------------------------------------------------|
| `HUGGINGFACE_TOKEN`          | `hf_....`              | Token for accessing Hugging Face models (used by Whisper or other models).   |
| `VIDEO_PATH`                 | `video/media.mp4`      | Path to a single video file or directory of videos to process.               |
| `PY_SCENE_DETECT_THRESHOLD`  | `40.0`                 | Sensitivity threshold for PySceneDetect content detection. Lower = more cuts.|
| `MIN_SCENE_LENGTH`           | `300`                  | Minimum scene length in frames (prevents over-segmentation).                 |
| `NUM_FRAMES_PER_SCENE`       | `4`                    | Number of frames extracted per scene (not including start & end).            |
| `OPENAI_API_KEY`             | `sk-...`               | API key for OpenAI models.                                                   |
| `CHAT_GPT_MODEL`             | `gpt-4o-mini`          | ChatGPT model to use for captioning.                                         |
| `CHAT_GPT_RETRIES`           | `10`                   | Number of retries on rate limit or transient errors.                         |
| `USE_BATCH_API`              | `true`                 | Whether to use the OpenAI Batch API (cheaper & asynchronous).                |
| `INCLUDE_TRANSCRIPT_IN_GPT`  | `true`                 | If `true`, transcripts are included in GPT prompts for more context.         |
| `TRANSCRIPT_REPEAT_THRESHOLD`| `8`                    | Filter threshold for repeated transcript words/phrases (to avoid noise).     |
| `TRANSCRIPT_CACHE_DIR`       | `transcripts`          | Directory for cached Whisper transcripts.                                    |
| `WHISPER_MODEL_SIZE`         | `base`                 | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`).             |
| `LOGS_DIR`                   | `logs`                 | Directory for logs.                                                          |
| `CAPTIONS_DIR`               | `captions`             | Directory where caption JSON files will be saved.                            |

---

## Usage

### Process a single video

```bash
python3 captioning.py --input ./videos/my_clip.mp4
```

### Process all videos in a folder

```bash
python3 captioning.py --input ./videos
```

Optional CLI args:

- `--output` → captions output folder (default: `captions/`)
- `--log` → log output folder (default: `logs/`)

---

## Output

Each processed video creates:

- A JSON file in `captions/`:
Example:
```json
"scene_001": {
  "start_time": 0.0,
  "end_time": 12.53,
  "frame_files": [...],
  "transcript_text": "She walks into the room and smiles.",
  "caption": "A woman enters with a warm smile.",
  "keywords": ["woman", "entrance", "smile", "scene intro"]
}
```

- A `.log` file in `logs/` with:
  - Scene count and timing
  - Token usage breakdown
  - Total runtime
  - Total cost

- A directory of frames collected from the video in `frames/`

---

## Batch API Mode

If `USE_BATCH_API=true`, all scenes are submitted in one OpenAI batch job (cheaper, async).

> **Batch mode costs ~50% less** than real-time GPT API calls

You can toggle between batch and sync modes with the `.env` setting.

---

## Token Usage & Cost Tracking

The system logs:

- Prompt + completion token count
- Total tokens used
- Estimated total OpenAI cost

---

## License

MIT License. © 2025 Selina Mangaroo