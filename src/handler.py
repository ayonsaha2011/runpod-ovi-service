"""
RunPod Serverless Handler for Ovi 1.1 Video Generation
OpenAI-compatible API with Cloudflare R2 storage
"""
import os
import sys

# =============================================================================
# CRITICAL: Initialize CUDA BEFORE any other imports that trigger Ovi modules
# The Ovi T5 module uses torch.cuda.current_device() as a default argument,
# which is evaluated at class definition (import) time, not at instantiation.
# =============================================================================
import torch

# Force CUDA initialization before importing any Ovi modules
if torch.cuda.is_available():
    # Initialize CUDA context
    torch.cuda.init()
    _ = torch.cuda.current_device()
    print(f"CUDA initialized: {torch.cuda.get_device_name(0)}", file=sys.stdout, flush=True)
else:
    print("WARNING: CUDA not available, running on CPU", file=sys.stdout, flush=True)

import logging
import time
import uuid
import base64
import tempfile
from typing import Optional, Dict, Any

import runpod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import local modules
from api_models import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoData,
    UsageInfo,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    GenerationMode,
    ModelVariant
)
from ovi_engine_wrapper import get_engine, OviEngineWrapper
from r2_storage import get_storage_handler, R2StorageHandler


# ============================================================================
# Global Model Initialization (Outside Handler for Cold Start Optimization)
# ============================================================================

# Load model at worker startup
DEFAULT_MODEL = os.environ.get("OVI_MODEL_NAME", "960x960_10s")
CKPT_DIR = os.environ.get("OVI_CKPT_DIR", "/runpod-volume/models")

logger.info(f"Initializing Ovi engine with model={DEFAULT_MODEL}, ckpt_dir={CKPT_DIR}")

# Download models if not present (first cold start on new network volume)
try:
    from download_on_startup import ensure_models_ready
    
    if not ensure_models_ready(CKPT_DIR, DEFAULT_MODEL):
        logger.error("Failed to download models - engine will not be initialized")
        engine = None
    else:
        # Initialize engine (lazy loading - will load model on first request)
        engine = get_engine(
            model_name=DEFAULT_MODEL,
            ckpt_dir=CKPT_DIR,
            cpu_offload=os.environ.get("OVI_CPU_OFFLOAD", "false").lower() == "true",
            fp8=os.environ.get("OVI_FP8", "false").lower() == "true",
            qint8=os.environ.get("OVI_QINT8", "false").lower() == "true"
        )
        logger.info("Ovi engine wrapper initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Ovi engine: {e}")
    engine = None

# Initialize R2 storage handler
try:
    storage = get_storage_handler()
    logger.info("R2 storage handler initialized successfully")
except Exception as e:
    logger.warning(f"R2 storage not configured (will fail on requests): {e}")
    storage = None


# ============================================================================
# Helper Functions
# ============================================================================

def save_base64_image_to_temp(base64_image: str) -> str:
    """Save base64 image to temporary file and return path"""
    # Remove data URL prefix if present
    if "," in base64_image:
        base64_image = base64_image.split(",")[1]
    
    image_data = base64.b64decode(base64_image)
    
    # Detect image format from magic bytes
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        ext = ".png"
    elif image_data[:2] == b'\xff\xd8':
        ext = ".jpg"
    elif image_data[:4] == b'RIFF':
        ext = ".webp"
    else:
        ext = ".png"  # Default to PNG
    
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(image_data)
    temp_file.close()
    
    return temp_file.name


def create_error_response(message: str, error_type: str = "invalid_request_error", code: str = None) -> Dict:
    """Create OpenAI-compatible error response"""
    error = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            code=code
        )
    )
    return error.model_dump()


def get_model_variant_from_request(model_str: str) -> str:
    """Convert model string to internal variant name"""
    model_map = {
        "720x720_5s": "720x720_5s",
        "960x960_5s": "960x960_5s", 
        "960x960_10s": "960x960_10s",
        # Aliases
        "ovi-720-5s": "720x720_5s",
        "ovi-960-5s": "960x960_5s",
        "ovi-960-10s": "960x960_10s",
        "ovi-1.1": "960x960_10s",
        "ovi-1.1-10s": "960x960_10s",
        "ovi-1.1-5s": "960x960_5s"
    }
    return model_map.get(model_str, model_str)


# ============================================================================
# Main Handler Function
# ============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for video generation
    
    Input format:
    {
        "input": {
            "prompt": "A cat playing piano. Audio: upbeat jazz piano music",
            "mode": "t2v",
            "model": "960x960_10s",
            "seed": 42,
            ...
        }
    }
    
    Returns OpenAI-compatible response with Cloudflare R2 URL
    """
    global engine, storage
    
    job_id = job.get("id", str(uuid.uuid4()))
    job_input = job.get("input", {})
    
    logger.info(f"Job {job_id}: Processing request")
    
    # Validate engine and storage are available
    if engine is None:
        return create_error_response(
            "Model not initialized. Check server logs.",
            error_type="server_error",
            code="model_not_loaded"
        )
    
    if storage is None:
        return create_error_response(
            "Storage not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY environment variables.",
            error_type="server_error",
            code="storage_not_configured"
        )
    
    try:
        # Parse and validate request
        request = VideoGenerationRequest(**job_input)
        
        # Get model variant
        model_name = get_model_variant_from_request(request.model.value)
        
        # Handle image input for i2v mode
        image_path = None
        temp_image_path = None
        
        if request.mode == GenerationMode.I2V:
            if not request.image:
                return create_error_response(
                    "Image input required for i2v mode",
                    code="missing_image"
                )
            temp_image_path = save_base64_image_to_temp(request.image)
            image_path = temp_image_path
            logger.info(f"Job {job_id}: Saved input image to {image_path}")
        
        # Ensure model is loaded
        if not engine.is_loaded():
            runpod.serverless.progress_update(job, "Loading model...")
            mode_str = request.mode.value
            engine.load_model(mode=mode_str)
        
        # Check if requested model matches loaded model
        if model_name != engine.model_name:
            logger.warning(f"Requested model {model_name} differs from loaded model {engine.model_name}. Using loaded model.")
            model_name = engine.model_name
        
        # Report progress
        runpod.serverless.progress_update(job, "Generating video...")
        
        # Generate video
        result = engine.generate(
            prompt=request.prompt,
            image_path=image_path,
            height=request.height,
            width=request.width,
            seed=request.seed,
            sample_steps=request.sample_steps,
            video_guidance_scale=request.video_guidance_scale,
            audio_guidance_scale=request.audio_guidance_scale,
            video_negative_prompt=request.video_negative_prompt,
            audio_negative_prompt=request.audio_negative_prompt
        )
        
        logger.info(f"Job {job_id}: Generation complete in {result.generation_time_seconds:.2f}s")
        
        # Report progress
        runpod.serverless.progress_update(job, "Uploading to storage...")
        
        # Upload to R2
        metadata = {
            "job_id": job_id,
            "prompt": request.prompt[:200],  # Truncate for metadata
            "model": model_name,
            "seed": str(request.seed) if request.seed else "random"
        }
        
        video_url = storage.upload_video_file(
            file_path=result.video_path,
            job_id=job_id,
            model=model_name,
            metadata=metadata
        )
        
        logger.info(f"Job {job_id}: Video uploaded to {video_url}")
        
        # Cleanup temporary files
        engine.cleanup_file(result.video_path)
        if result.image_path:
            engine.cleanup_file(result.image_path)
        if temp_image_path:
            try:
                os.remove(temp_image_path)
            except:
                pass
        
        # Build response
        response = VideoGenerationResponse(
            id=f"video-{job_id}",
            created=int(time.time()),
            model=f"ovi-1.1-{model_name}",
            data=[
                VideoData(
                    url=video_url,
                    revised_prompt=result.formatted_prompt
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(request.prompt.split()),
                total_frames=result.total_frames,
                duration_seconds=result.duration_seconds,
                generation_time_seconds=result.generation_time_seconds
            )
        )
        
        logger.info(f"Job {job_id}: Complete")
        return response.model_dump()
        
    except Exception as e:
        logger.exception(f"Job {job_id}: Error during generation")
        return create_error_response(
            str(e),
            error_type="generation_error",
            code="generation_failed"
        )


# ============================================================================
# Health Check Handler
# ============================================================================

def health_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint"""
    global engine
    
    memory = {"allocated_gb": 0, "reserved_gb": 0}
    if engine:
        memory = engine.get_memory_usage()
    
    response = HealthResponse(
        status="healthy" if engine and engine.is_loaded() else "unhealthy",
        model_loaded=engine.is_loaded() if engine else False,
        available_models=engine.get_available_models() if engine else [],
        gpu_memory_allocated_gb=memory.get("allocated_gb"),
        gpu_memory_reserved_gb=memory.get("reserved_gb")
    )
    
    return response.model_dump()


# ============================================================================
# RunPod Serverless Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Ovi 1.1 RunPod Serverless Worker")
    
    # Pre-load model if configured
    if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
        if engine:
            logger.info("Pre-loading model at startup...")
            try:
                engine.load_model(mode="i2v")
                logger.info("Model pre-loaded successfully")
            except Exception as e:
                logger.error(f"Failed to pre-load model: {e}")
    
    # Start serverless worker
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })
