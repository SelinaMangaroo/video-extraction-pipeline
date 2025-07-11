import os
import logging
from datetime import datetime
from pathlib import Path

def setup_logger(video_path: str, output_dir: str = "output") -> str:
    """Initializes and configures logging. Returns path to log file."""
    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(video_path).stem
    timestamp = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{timestamp}_{video_stem}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Logging initialized.")
    return log_path
