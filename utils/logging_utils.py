import os
import logging
from datetime import datetime
from pathlib import Path

def setup_logger(video_path: str, output_dir: str = "output") -> str:
    """Initializes and configures logging. Returns path to log file."""
    os.makedirs(output_dir, exist_ok=True)
        
    path = Path(video_path).resolve()
    
    if path.suffix:
        video_stem = path.stem
    else:
        video_stem = path.name
    
    timestamp = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    log_filename = f"{timestamp}_{video_stem}.log"
    log_path = os.path.join(output_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

    logging.info("Logging initialized.")
    return log_path