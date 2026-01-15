"""
Ovi Fusion Engine Wrapper for RunPod Serverless
Provides simplified interface to Ovi 1.1 video generation
"""
import os
import sys
import logging
import tempfile
import torch
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Add Ovi source to path
OVI_PATH = os.environ.get("OVI_PATH", "/app/Ovi")
if OVI_PATH not in sys.path:
    sys.path.insert(0, OVI_PATH)

from omegaconf import OmegaConf
from ovi.ovi_fusion_engine import OviFusionEngine, NAME_TO_MODEL_SPECS_MAP
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import format_prompt_for_filename

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of video generation"""
    video_path: str
    image_path: Optional[str]
    prompt: str
    formatted_prompt: str
    model_name: str
    duration_seconds: float
    total_frames: int
    generation_time_seconds: float


class OviEngineWrapper:
    """
    Wrapper around Ovi's OviFusionEngine for RunPod serverless deployment
    
    Handles:
    - Model initialization with configurable variants
    - Memory optimization (cpu_offload, fp8, qint8)
    - Generation with automatic resource cleanup
    - Video file output management
    """
    
    # Model specs from Ovi
    MODEL_SPECS = {
        "720x720_5s": {
            "duration": 5.0,
            "fps": 24,
            "frames": 121,
            "default_height": 720,
            "default_width": 720
        },
        "960x960_5s": {
            "duration": 5.0,
            "fps": 24,
            "frames": 121,
            "default_height": 960,
            "default_width": 960
        },
        "960x960_10s": {
            "duration": 10.0,
            "fps": 24,
            "frames": 241,
            "default_height": 960,
            "default_width": 960
        }
    }
    
    def __init__(
        self,
        model_name: str = "960x960_10s",
        ckpt_dir: str = "/models",
        device: int = 0,
        cpu_offload: bool = False,
        fp8: bool = False,
        qint8: bool = False
    ):
        """
        Initialize Ovi engine with specified configuration
        
        Args:
            model_name: Model variant to load
            ckpt_dir: Directory containing model checkpoints
            device: CUDA device index
            cpu_offload: Enable CPU offloading for memory savings
            fp8: Use FP8 quantization (720x720_5s only)
            qint8: Use INT8 quantization
        """
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.cpu_offload = cpu_offload
        self.fp8 = fp8
        self.qint8 = qint8
        
        if model_name not in self.MODEL_SPECS:
            raise ValueError(f"Invalid model_name: {model_name}. Must be one of {list(self.MODEL_SPECS.keys())}")
        
        self.specs = self.MODEL_SPECS[model_name]
        self._engine: Optional[OviFusionEngine] = None
        self._output_dir = tempfile.mkdtemp(prefix="ovi_output_")
        
        logger.info(f"OviEngineWrapper initialized for model: {model_name}")
    
    def load_model(self, mode: str = "i2v") -> None:
        """
        Load the Ovi model into memory
        
        Args:
            mode: Generation mode (t2v, i2v, t2i2v)
        """
        if self._engine is not None:
            logger.info("Model already loaded, skipping initialization")
            return
        
        logger.info(f"Loading Ovi model: {self.model_name} with mode={mode}")
        
        # Build configuration
        config = OmegaConf.create({
            "ckpt_dir": self.ckpt_dir,
            "output_dir": self._output_dir,
            "model_name": self.model_name,
            "mode": mode,
            "cpu_offload": self.cpu_offload,
            "fp8": self.fp8,
            "qint8": self.qint8,
            "sp_size": 1,
            "sample_steps": 50,
            "solver_name": "unipc",
            "shift": 5.0,
            "video_guidance_scale": 4.0,
            "audio_guidance_scale": 3.0,
            "slg_layer": 11,
            "video_negative_prompt": "jitter, bad hands, blur, distortion",
            "audio_negative_prompt": "robotic, muffled, echo, distorted"
        })
        
        # Initialize engine
        self._engine = OviFusionEngine(
            config=config,
            device=self.device,
            target_dtype=torch.bfloat16
        )
        
        logger.info(f"Ovi model loaded successfully. GPU memory: {self.get_memory_usage()}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._engine is not None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9
            }
        return {"allocated_gb": 0, "reserved_gb": 0}
    
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        sample_steps: int = 50,
        video_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 3.0,
        video_negative_prompt: str = "jitter, bad hands, blur, distortion",
        audio_negative_prompt: str = "robotic, muffled, echo, distorted",
        slg_layer: int = 11,
        shift: float = 5.0,
        solver_name: str = "unipc"
    ) -> GenerationResult:
        """
        Generate video from prompt and optional input image
        
        Args:
            prompt: Text prompt (include "Audio:" for audio guidance)
            image_path: Path to input image for i2v mode
            height: Video height (for t2v mode)
            width: Video width (for t2v mode)
            seed: Random seed
            sample_steps: Denoising steps
            video_guidance_scale: Video conditioning strength
            audio_guidance_scale: Audio conditioning strength
            video_negative_prompt: Video negative prompt
            audio_negative_prompt: Audio negative prompt
            slg_layer: SLG layer index
            shift: Timestep shift factor
            solver_name: Sampling algorithm
            
        Returns:
            GenerationResult with paths to generated files
        """
        import time
        
        if not self.is_loaded():
            mode = "i2v" if image_path else "t2v"
            self.load_model(mode=mode)
        
        # Use default dimensions from model specs
        if height is None:
            height = self.specs["default_height"]
        if width is None:
            width = self.specs["default_width"]
        
        # Use random seed if not provided
        if seed is None:
            seed = torch.randint(0, 2147483647, (1,)).item()
        
        video_frame_height_width = [height, width]
        
        logger.info(f"Starting generation: seed={seed}, steps={sample_steps}, size={height}x{width}")
        
        start_time = time.time()
        
        # Generate video and audio
        generated_video, generated_audio, generated_image = self._engine.generate(
            text_prompt=prompt,
            image_path=image_path,
            video_frame_height_width=video_frame_height_width,
            seed=seed,
            solver_name=solver_name,
            sample_steps=sample_steps,
            shift=shift,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            slg_layer=slg_layer,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt
        )
        
        generation_time = time.time() - start_time
        
        # Save output files
        formatted_prompt = format_prompt_for_filename(prompt)
        output_filename = f"{formatted_prompt}_{height}x{width}_{seed}.mp4"
        output_path = os.path.join(self._output_dir, output_filename)
        
        save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
        
        # Save generated image if available (t2i2v mode)
        image_output_path = None
        if generated_image is not None:
            image_output_path = output_path.replace(".mp4", ".png")
            generated_image.save(image_output_path)
        
        logger.info(f"Generation complete in {generation_time:.2f}s: {output_path}")
        
        return GenerationResult(
            video_path=output_path,
            image_path=image_output_path,
            prompt=prompt,
            formatted_prompt=formatted_prompt,
            model_name=self.model_name,
            duration_seconds=self.specs["duration"],
            total_frames=self.specs["frames"],
            generation_time_seconds=generation_time
        )
    
    def cleanup_file(self, file_path: str) -> None:
        """Remove generated file after upload"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    def get_available_models(self) -> list:
        """Get list of available model variants"""
        return list(self.MODEL_SPECS.keys())
    
    def unload_model(self) -> None:
        """Unload model from GPU memory"""
        if self._engine is not None:
            del self._engine
            self._engine = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded from GPU memory")


# Global engine instance for RunPod serverless
_engine_instance: Optional[OviEngineWrapper] = None


def get_engine(
    model_name: str = None,
    force_reload: bool = False,
    **kwargs
) -> OviEngineWrapper:
    """
    Get or create global engine instance
    
    Args:
        model_name: Model variant to load
        force_reload: Force reload even if already loaded
        **kwargs: Additional arguments for OviEngineWrapper
        
    Returns:
        OviEngineWrapper instance
    """
    global _engine_instance
    
    # Use environment variable or default
    if model_name is None:
        model_name = os.environ.get("OVI_MODEL_NAME", "960x960_10s")
    
    ckpt_dir = kwargs.pop("ckpt_dir", os.environ.get("OVI_CKPT_DIR", "/models"))
    
    if _engine_instance is None or force_reload:
        if _engine_instance is not None:
            _engine_instance.unload_model()
        
        _engine_instance = OviEngineWrapper(
            model_name=model_name,
            ckpt_dir=ckpt_dir,
            **kwargs
        )
    
    return _engine_instance
