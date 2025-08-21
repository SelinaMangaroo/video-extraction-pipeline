import os
import re
import time
import base64
import requests
import logging
import json

TOTAL_RATE_LIMIT_BACKOFF = 0.0
TOTAL_TOKENS_USED = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}
TOTAL_COST = {"value": 0.0}

def build_prompt_and_images(transcript_text, frames):
    '''Build the prompt and image inputs for the OpenAI API request.'''
    
    prompt = (
        "Analyze the following transcript and images. "
        "Return a single-sentence caption for this scene and do not infer beyond what is visible and within the transcript text. "
        "Then provide a list of keywords for searching purposes describing this scene. "
        "Provide this in JSON format with two keys: 'caption' (string) and 'keywords' (list of strings). "
        f"\n\nTranscript:\n{transcript_text}"
    )

    image_inputs = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for p in frames.values()
        if (b64 := encode_image_to_base64(p)) is not None
    ]

    return prompt, image_inputs

# Convert image to base64 for OpenAI API
def encode_image_to_base64(path):
    '''Encode an image file to base64 string.'''
    
    if not os.path.exists(path):
        logging.info(f"Skipping missing image: {path}")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def calculate_token_cost(model_name, prompt_tokens, completion_tokens, batch_mode=False):
    '''Calculate the estimated cost of tokens used in a request.'''
    
    prompt_rate = 0.15 / 1_000_000
    completion_rate = 0.60 / 1_000_000

    if batch_mode:
        prompt_rate /= 2
        completion_rate /= 2

    return round(prompt_tokens * prompt_rate + completion_tokens * completion_rate, 6)

def safe_json_extract(output):
    '''Safely extract JSON from the OpenAI response, handling common formatting issues.'''
    
    output = re.sub(r'^\s*```(?:json)?\s*', '', output, flags=re.IGNORECASE)
    output = re.sub(r'\s*```\s*$', '', output)
    m = re.search(r'\{.*\}', output, flags=re.DOTALL)
    return m.group(0) if m else output

def caption_scene_with_images(frame_paths, api_key, transcript, model, retries):
    '''Caption a scene using OpenAI's chat completion API with images.'''
    
    prompt, images = build_prompt_and_images(transcript, frame_paths)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}] + images}],
        "max_tokens": 400,
    }

    for attempt in range(retries):
        try:
            logging.debug(f"OpenAI model={model}, frames={len(images)}, transcript_chars={len(transcript or '')}")
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            data = resp.json()

            if "usage" in data:
                usage = data["usage"]
                cost = calculate_token_cost(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                for key in TOTAL_TOKENS_USED:
                    TOTAL_TOKENS_USED[key] += usage.get(key, 0)
                TOTAL_COST["value"] += cost

                logging.info(
                    f"Token usage — Prompt: {usage['prompt_tokens']}, Completion: {usage['completion_tokens']}, "
                    f"Total: {usage['total_tokens']}, Estimated scene cost: ${cost:.6f}"
                )

            logging.debug(f"OpenAI response status={resp.status_code} x-request-id={resp.headers.get('x-request-id', 'n/a')}")
            logging.debug(f"Raw API response text: {resp.text}")

            if "error" in data:
                if data["error"].get("code") == "rate_limit_exceeded":
                    wait = 2 ** (attempt + 1)
                    logging.warning(f"Rate limit hit. Waiting {wait:.1f}s (attempt {attempt+1}/{retries})...")
                    global TOTAL_RATE_LIMIT_BACKOFF
                    TOTAL_RATE_LIMIT_BACKOFF += wait
                    time.sleep(wait)
                    continue
                logging.error(f"Unexpected API error: {data}")
                break

            content = data["choices"][0]["message"]["content"].strip()
            obj = json.loads(safe_json_extract(content))

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

    logging.error("Final captioning attempt failed.")
    return None, []