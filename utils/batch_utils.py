import os
import json
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from utils.chat_gpt_utils import TOTAL_TOKENS_USED, TOTAL_COST, calculate_token_cost

load_dotenv()
BATCH_INPUT_FILE = "batch_input.jsonl"

def submit_batch_and_get_output_file(batch_requests, api_key):
    '''Submit a batch of requests to OpenAI and return the output file ID.'''
    
    client = OpenAI(api_key=api_key)
    
    with open(BATCH_INPUT_FILE, "w") as f:
        for req in batch_requests:
            json.dump(req, f)
            f.write("\n")
            
    upload = client.files.create(file=open(BATCH_INPUT_FILE, "rb"), purpose="batch")
    
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    logging.info(f"Submitted batch: {batch.id}")
    
    while True:
        batch = client.batches.retrieve(batch.id)
        logging.info(f"Status: {batch.status}")
        if batch.status == "completed":
            return batch.output_file_id
        elif batch.status in ["failed", "expired", "cancelled"]:
            raise RuntimeError(f"Batch failed: {batch.status}")
        time.sleep(10)

def parse_batch_output(output_file_id, captions_by_scene, model_name="gpt-4o-mini"):
    '''Parse the output file from a batch request and update captions_by_scene.'''
    
    client = OpenAI()
    output = client.files.content(output_file_id).text
    for line in output.splitlines():
        obj = json.loads(line)
        logging.info(f"OBJ: {obj}")
        try:
            scene_id = obj["custom_id"]
            content = obj["response"]["body"]["choices"][0]["message"]["content"]
            usage = obj["response"]["body"].get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = calculate_token_cost(model_name, prompt_tokens, completion_tokens, batch_mode=True)

            TOTAL_TOKENS_USED["prompt_tokens"] += prompt_tokens
            TOTAL_TOKENS_USED["completion_tokens"] += completion_tokens
            TOTAL_TOKENS_USED["total_tokens"] += usage.get("total_tokens", 0)
            TOTAL_COST["value"] += cost
            
            logging.debug(f"Raw content for {scene_id}:\n{content}")

            logging.info(
                f"{scene_id} → Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                f"Total: {usage.get('total_tokens', 0)}, Estimated cost: ${cost:.6f}"
            )

            try:
                cleaned = content.strip("```json").strip("```").strip()
                data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON for {scene_id}: {e}")
                continue

            if scene_id in captions_by_scene:
                captions_by_scene[scene_id]["caption"] = data.get("caption", "").strip()
                captions_by_scene[scene_id]["keywords"] = data.get("keywords", [])
            else:
                logging.error(f"{scene_id} not found in scene_id_map. Skipping.")

        except Exception as e:
            logging.error(f"Failed to parse scene: {e}")

def submit_and_parse_batch(batch_requests, scene_id_map, api_key):
    '''Submit a batch of requests and parse the output.'''
    
    output_file_id = submit_batch_and_get_output_file(batch_requests, api_key)
    parse_batch_output(output_file_id, scene_id_map)
    return scene_id_map