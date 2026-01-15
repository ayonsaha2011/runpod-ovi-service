#!/usr/bin/env python3
"""
Runtime Model Download for Ovi 1.1
Downloads models to RunPod Network Volume on first cold start
"""
import os
import sys
import logging
import time
import fcntl
from pathlib import Path
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Model file mappings
NAME_TO_MODELS_MAP = {
    "720x720_5s": "model.safetensors",
    "960x960_5s": "model_960x960.safetensors",
    "960x960_10s": "model_960x960_10s.safetensors"
}

# Required files for verification
REQUIRED_BASE_FILES = [
    ("Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth"),
    ("Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"),
    ("MMAudio", "ext_weights/best_netG.pt"),
    ("MMAudio", "ext_weights/v1-16.pth"),
]


def check_models_exist(ckpt_dir: str, model_name: str = "960x960_10s") -> bool:
    """Check if all required model files exist"""
    if not os.path.exists(ckpt_dir):
        return False
    
    # Check base files
    for subdir, filename in REQUIRED_BASE_FILES:
        file_path = os.path.join(ckpt_dir, subdir, filename)
        if not os.path.exists(file_path):
            logger.debug(f"Missing file: {file_path}")
            return False
    
    # Check model-specific file
    if model_name in NAME_TO_MODELS_MAP:
        model_file = os.path.join(ckpt_dir, "Ovi", NAME_TO_MODELS_MAP[model_name])
        if not os.path.exists(model_file):
            logger.debug(f"Missing model file: {model_file}")
            return False
    
    return True


def download_models_if_needed(ckpt_dir: str, model_name: str = "960x960_10s") -> bool:
    """
    Download models if they don't exist.
    Uses file locking to prevent multiple workers from downloading simultaneously.
    
    Returns True if models are ready, False if download failed.
    """
    # Check if models already exist
    if check_models_exist(ckpt_dir, model_name):
        logger.info(f"Models already present at {ckpt_dir}")
        return True
    
    # Create lock file path
    lock_file = os.path.join(os.path.dirname(ckpt_dir), ".model_download.lock")
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    
    logger.info(f"Models not found at {ckpt_dir}. Starting download...")
    
    try:
        # Acquire lock (blocking)
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            # Double-check after acquiring lock (another worker may have completed download)
            if check_models_exist(ckpt_dir, model_name):
                logger.info("Models downloaded by another worker")
                return True
            
            # Perform downloads
            start_time = time.time()
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Download Wan components
            logger.info("Downloading Wan2.2 components...")
            snapshot_download(
                repo_id="Wan-AI/Wan2.2-TI2V-5B",
                local_dir=os.path.join(ckpt_dir, "Wan2.2-TI2V-5B"),
                allow_patterns=["google/*", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"]
            )
            
            # Download MMAudio components
            logger.info("Downloading MMAudio components...")
            snapshot_download(
                repo_id="hkchengrex/MMAudio",
                local_dir=os.path.join(ckpt_dir, "MMAudio"),
                allow_patterns=["ext_weights/best_netG.pt", "ext_weights/v1-16.pth"]
            )
            
            # Download Ovi models (all variants for flexibility)
            logger.info("Downloading Ovi model weights...")
            snapshot_download(
                repo_id="chetwinlow1/Ovi",
                local_dir=os.path.join(ckpt_dir, "Ovi"),
                allow_patterns=list(NAME_TO_MODELS_MAP.values())
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Model download complete in {elapsed:.1f} seconds")
            
            # Verify
            if check_models_exist(ckpt_dir, model_name):
                return True
            else:
                logger.error("❌ Model verification failed after download")
                return False
                
    except Exception as e:
        logger.exception(f"❌ Model download failed: {e}")
        return False


def ensure_models_ready(ckpt_dir: str = None, model_name: str = None) -> bool:
    """
    Main entry point - ensures models are downloaded and ready.
    
    Args:
        ckpt_dir: Directory to store models (default: OVI_CKPT_DIR env or /runpod-volume/models)
        model_name: Model variant name (default: OVI_MODEL_NAME env or 960x960_10s)
    
    Returns:
        True if models are ready, False otherwise
    """
    if ckpt_dir is None:
        ckpt_dir = os.environ.get("OVI_CKPT_DIR", "/runpod-volume/models")
    
    if model_name is None:
        model_name = os.environ.get("OVI_MODEL_NAME", "960x960_10s")
    
    return download_models_if_needed(ckpt_dir, model_name)


if __name__ == "__main__":
    # CLI for manual testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OVI_CKPT_DIR", "/runpod-volume/models")
    model_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OVI_MODEL_NAME", "960x960_10s")
    
    success = ensure_models_ready(ckpt_dir, model_name)
    sys.exit(0 if success else 1)
