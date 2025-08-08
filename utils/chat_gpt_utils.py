import os
import re
import time
import base64
import requests
import logging
import json

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
    full_prompt = (
        "Analyze the following transcript and images. "
        "Return a single-sentence caption for this scene and do not infer beyond what is visible and within the transcript text. "
        "Then provide a list of keywords for searching purposes describing this scene. "
        "Provide this in JSON format with two keys: 'caption' (string) and 'keywords' (list of strings). "
        f"\n\nTranscript:\n{transcript}"
    )

    image_inputs = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for p in frame_paths.values()
        if (b64 := encode_image_to_base64(p)) is not None
    ]

    headers = {"Content-Type": "application/json","Authorization": f"Bearer {api_key}"}
    body = {
        "model": CHAT_GPT_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt}] + image_inputs}],
        "max_tokens": 400,
    }

    for attempt in range(CHAT_GPT_RETRIES):
        
        try:
            logging.debug(f"OpenAI model={CHAT_GPT_MODEL}, frames={len(image_inputs)}, transcript_chars={len(transcript or '')}")      
            try:
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            except Exception as e:
                logging.error(f"OpenAI request failed before body was received: {e}")
                return None, []
            
            # resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            
            data = resp.json()
            
            logging.debug(f"OpenAI response status={resp.status_code} x-request-id={resp.headers.get('x-request-id', 'n/a')}")
            logging.debug(f"Raw API response text: {resp.text}")

            # API-side errors
            if "error" in data:
                code = data["error"].get("code")
                if code == "rate_limit_exceeded":
                    wait_time = 2 ** (attempt + 1)
                    logging.warning(f"Rate limit hit. Waiting {wait_time:.1f}s (attempt {attempt+1}/{CHAT_GPT_RETRIES})...")
                    time.sleep(wait_time)
                    continue
                logging.error(f"Unexpected error: {data}")
                break

            output = data["choices"][0]["message"]["content"].strip()

            # Strip improper json if present
            output = re.sub(r'^\s*```(?:json)?\s*', '', output, flags=re.IGNORECASE)
            output = re.sub(r'\s*```\s*$', '', output)

            # Grab first {...} block if there’s any extra text
            m = re.search(r'\{.*\}', output, flags=re.DOTALL)
            json_text = m.group(0) if m else output

            obj = json.loads(json_text)
            caption = (obj.get("caption") or "").strip()

            keywords = obj.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in re.split(r'[,\n;]+', keywords) if kw.strip()]
            elif isinstance(keywords, list):
                keywords = [str(kw).strip() for kw in keywords if str(kw).strip()]
            else:
                keywords = []

            return caption, keywords

        except Exception as e:
            logging.error(f"Request/parse failed: {e}")
            break

    logging.error("Final attempt failed.")
    return None, []
