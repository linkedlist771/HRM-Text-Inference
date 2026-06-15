from pathlib import Path
from loguru import logger

ROOT = Path(__file__).parent

MODEL_PATH = ROOT / "checkpoints/HRM-Text-1B"


if __name__ == "__main__":
    logger.debug(f"ROOT: {ROOT}")
    logger.debug(f"MODEL_PATH: {MODEL_PATH}")
