#!/usr/bin/env python3
"""
Model Download Script for Ovi 1.1
Downloads all required model weights from HuggingFace
"""
import os
import argparse
import logging
import time
from huggingface_hub import snapshot_download, hf_hub_download

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Model file mappings
NAME_TO_MODELS_MAP = {
    "720x720_5s": "model.safetensors",
    "960x960_5s": "model_960x960.safetensors",
    "960x960_10s": "model_960x960_10s.safetensors"
}


def timed_download(repo_id: str, local_dir: str, allow_patterns: list = None, ignore_patterns: list = None):
    """Download files from HF repo and log time + destination."""
    logger.info(f"Starting download from {repo_id} into {local_dir}")
    start_time = time.time()

    os.makedirs(local_dir, exist_ok=True)

    kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "local_dir_use_symlinks": False,
    }
    
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    snapshot_download(**kwargs)

    elapsed = time.time() - start_time
    logger.info(
        f"✅ Finished downloading {repo_id} "
        f"in {elapsed:.2f} seconds. Files saved at: {local_dir}"
    )


def download_wan_components(output_dir: str):
    """Download Wan2.2 VAE and T5 text encoder"""
    wan_dir = os.path.join(output_dir, "Wan2.2-TI2V-5B")
    
    timed_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        local_dir=wan_dir,
        allow_patterns=[
            "google/*",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.2_VAE.pth"
        ]
    )
    
    return wan_dir


def download_mmaudio_components(output_dir: str):
    """Download MMAudio VAE components"""
    mm_audio_dir = os.path.join(output_dir, "MMAudio")
    
    timed_download(
        repo_id="hkchengrex/MMAudio",
        local_dir=mm_audio_dir,
        allow_patterns=[
            "ext_weights/best_netG.pt",
            "ext_weights/v1-16.pth"
        ]
    )
    
    return mm_audio_dir


def download_ovi_models(output_dir: str, models: list = None):
    """Download Ovi model weights"""
    if models is None:
        models = list(NAME_TO_MODELS_MAP.keys())
    
    # Validate models
    for model in models:
        if model not in NAME_TO_MODELS_MAP:
            raise ValueError(f"Invalid model name {model}. Valid options: {list(NAME_TO_MODELS_MAP.keys())}")
    
    ovi_dir = os.path.join(output_dir, "Ovi")
    model_files = [NAME_TO_MODELS_MAP[m] for m in models]
    
    timed_download(
        repo_id="chetwinlow1/Ovi",
        local_dir=ovi_dir,
        allow_patterns=model_files
    )
    
    return ovi_dir


def download_fp8_model(output_dir: str):
    """Download optional FP8 quantized model for 24GB GPUs"""
    ovi_dir = os.path.join(output_dir, "Ovi")
    os.makedirs(ovi_dir, exist_ok=True)
    
    logger.info("Downloading FP8 quantized model...")
    start_time = time.time()
    
    hf_hub_download(
        repo_id="rkfg/Ovi-fp8_quantized",
        filename="model_fp8_e4m3fn.safetensors",
        local_dir=ovi_dir
    )
    
    elapsed = time.time() - start_time
    logger.info(f"✅ FP8 model downloaded in {elapsed:.2f} seconds")


def verify_downloads(output_dir: str, models: list = None):
    """Verify all required files are present"""
    if models is None:
        models = list(NAME_TO_MODELS_MAP.keys())
    
    required_files = [
        # Wan components
        os.path.join(output_dir, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(output_dir, "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"),
        # MMAudio components
        os.path.join(output_dir, "MMAudio", "ext_weights", "best_netG.pt"),
        os.path.join(output_dir, "MMAudio", "ext_weights", "v1-16.pth"),
    ]
    
    # Add Ovi model files
    for model in models:
        required_files.append(
            os.path.join(output_dir, "Ovi", NAME_TO_MODELS_MAP[model])
        )
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        logger.error(f"❌ Missing {len(missing)} required files:")
        for f in missing:
            logger.error(f"  - {f}")
        return False
    
    logger.info(f"✅ All {len(required_files)} required files present")
    return True


def main(output_dir: str, models: list = None, include_fp8: bool = False):
    """Download all required models"""
    logger.info(f"=== Ovi 1.1 Model Downloader ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to download: {models or 'all'}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Download components
    download_wan_components(output_dir)
    download_mmaudio_components(output_dir)
    download_ovi_models(output_dir, models)
    
    if include_fp8:
        download_fp8_model(output_dir)
    
    total_time = time.time() - start_time
    
    # Verify
    success = verify_downloads(output_dir, models)
    
    if success:
        logger.info(f"=== Download complete in {total_time:.2f} seconds ===")
    else:
        logger.error("=== Download verification failed ===")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Ovi 1.1 models from HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/models",
        help="Base directory to save downloaded models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["720x720_5s", "960x960_5s", "960x960_10s"],
        choices=["720x720_5s", "960x960_5s", "960x960_10s"],
        help="Model variants to download"
    )
    parser.add_argument(
        "--include-fp8",
        action="store_true",
        help="Also download FP8 quantized model for 24GB GPUs"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (optional)"
    )
    
    args = parser.parse_args()
    
    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    main(args.output_dir, args.models, args.include_fp8)
