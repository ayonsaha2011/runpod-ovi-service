"""
Pydantic models for OpenAI-compatible Video Generation API
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from enum import Enum


class GenerationMode(str, Enum):
    """Video generation mode"""
    T2V = "t2v"  # Text to Video
    I2V = "i2v"  # Image to Video
    T2I2V = "t2i2v"  # Text to Image to Video


class ModelVariant(str, Enum):
    """Available Ovi model variants"""
    MODEL_720_5S = "720x720_5s"
    MODEL_960_5S = "960x960_5s"
    MODEL_960_10S = "960x960_10s"


class VideoGenerationRequest(BaseModel):
    """
    OpenAI-compatible video generation request
    
    Example:
    {
        "prompt": "A cat playing piano. Audio: upbeat jazz piano music",
        "mode": "t2v",
        "model": "960x960_10s",
        "seed": 42
    }
    """
    # Required fields
    prompt: str = Field(
        ...,
        description="Text prompt for video generation. Include 'Audio: description' for audio guidance.",
        min_length=1,
        max_length=4096
    )
    
    # Model configuration
    mode: GenerationMode = Field(
        default=GenerationMode.T2V,
        description="Generation mode: t2v (text-to-video), i2v (image-to-video), t2i2v (text-to-image-to-video)"
    )
    model: ModelVariant = Field(
        default=ModelVariant.MODEL_960_10S,
        description="Model variant determining resolution and duration"
    )
    
    # Optional image input for i2v mode
    image: Optional[str] = Field(
        default=None,
        description="Base64-encoded input image for i2v mode"
    )
    
    # Generation parameters
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducible results"
    )
    sample_steps: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Number of denoising steps (higher = better quality, slower)"
    )
    video_guidance_scale: float = Field(
        default=4.0,
        ge=1.0,
        le=20.0,
        description="Video conditioning strength"
    )
    audio_guidance_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="Audio conditioning strength"
    )
    
    # Frame dimensions for t2v/t2i2v modes
    height: Optional[int] = Field(
        default=None,
        ge=256,
        le=1280,
        description="Video height in pixels (for t2v/t2i2v modes)"
    )
    width: Optional[int] = Field(
        default=None,
        ge=256,
        le=1280,
        description="Video width in pixels (for t2v/t2i2v modes)"
    )
    
    # Negative prompts
    video_negative_prompt: str = Field(
        default="jitter, bad hands, blur, distortion",
        description="Artifacts to avoid in video"
    )
    audio_negative_prompt: str = Field(
        default="robotic, muffled, echo, distorted",
        description="Artifacts to avoid in audio"
    )
    
    # Advanced options
    cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offloading to reduce GPU memory (slower)"
    )
    fp8: bool = Field(
        default=False,
        description="Use FP8 quantization (720x720_5s only, reduces quality)"
    )
    qint8: bool = Field(
        default=False,
        description="Use INT8 quantization (reduces quality slightly)"
    )


class VideoData(BaseModel):
    """Individual video result data"""
    url: str = Field(
        description="Cloudflare R2 URL to the generated video"
    )
    revised_prompt: Optional[str] = Field(
        default=None,
        description="Formatted prompt used for generation"
    )


class UsageInfo(BaseModel):
    """Token and generation usage information"""
    prompt_tokens: int = Field(description="Number of tokens in prompt")
    total_frames: int = Field(description="Total video frames generated")
    duration_seconds: float = Field(description="Video duration in seconds")
    generation_time_seconds: float = Field(description="Time taken for generation")


class VideoGenerationResponse(BaseModel):
    """
    OpenAI-compatible video generation response
    """
    id: str = Field(description="Unique generation ID")
    object: Literal["video.generation"] = "video.generation"
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used for generation")
    data: List[VideoData] = Field(description="List of generated videos")
    usage: UsageInfo = Field(description="Usage statistics")


class ErrorDetail(BaseModel):
    """Error detail structure"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response"""
    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    available_models: List[str]
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None
