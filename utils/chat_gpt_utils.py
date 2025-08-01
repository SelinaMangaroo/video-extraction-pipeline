import os
import re
import time
import base64
import requests
import logging

CHAT_GPT_MODEL = os.getenv("CHAT_GPT_MODEL", "gpt-4o-mini")
CHAT_GPT_RETRIES = int(os.getenv("CHAT_GPT_RETRIES", 10))

# Convert image to base64 for OpenAI API
def encode_image_to_base64(path):
    if not os.path.exists(path):
        logging.info(f"Skipping missing image: {path}")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def caption_scene_with_images(frame_paths, api_key, transcript):
    # Updated prompt: ask for caption and keywords
    full_prompt = (
        "Analyze the following transcript and images. "
        "1) Write a single-sentence caption for this scene (do not infer beyond what is visible). "
        "2) Provide a list of concise, comma-separated keywords for searching purposes describing this scene.\n\n"
        f"Transcript:\n{transcript}"
    )

    image_inputs = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for p in frame_paths.values()
        if (b64 := encode_image_to_base64(p)) is not None
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    body = {
        "model": CHAT_GPT_MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": full_prompt}] + image_inputs}
        ],
        "max_tokens": 300
    }

    for attempt in range(CHAT_GPT_RETRIES):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        data = response.json()

        if "choices" in data:
            output = data["choices"][0]["message"]["content"].strip()
            # Split output into caption and keywords
            parts = re.split(r'(?i)keywords?:', output, maxsplit=1)
            caption = parts[0].strip()
            keywords = parts[1].strip() if len(parts) > 1 else ""
            return caption, keywords
        
        if "error" in data and data["error"].get("code") == "rate_limit_exceeded":
            wait_time = 2 ** (attempt + 1)
            logging.warning(f"Rate limit hit. Waiting {wait_time:.1f}s (attempt {attempt+1}/{CHAT_GPT_RETRIES})...")
            time.sleep(wait_time)
            continue

        logging.error(f"Unexpected error: {data}")
        break

    logging.error("Final attempt failed.")
    return None, ""