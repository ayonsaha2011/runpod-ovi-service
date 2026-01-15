# Ovi 1.1 RunPod Serverless Service

Production-ready video+audio generation service using Character AI's Ovi 1.1 model, deployed on RunPod Serverless with OpenAI-compatible API and Cloudflare R2 storage.

## Features

- **3 Model Variants**: 720x720 5s, 960x960 5s, 960x960 10s
- **3 Generation Modes**: Text-to-Video, Image-to-Video, Text-to-Image-to-Video
- **Audio Generation**: Synchronized audio with video
- **OpenAI-Compatible API**: Familiar request/response format
- **Cloudflare R2 Storage**: Videos uploaded and URLs returned
- **24GB GPU Support**: FP8/INT8 quantization options

---

## Quick Start

### 1. Build Docker Image

```bash
# Standard build
docker build -t your-registry/ovi-runpod:latest .

# With HuggingFace token (if you need gated models)
# First set the token in your environment:
export HF_TOKEN=your_huggingface_token
docker build --build-arg HF_TOKEN=$HF_TOKEN -t your-registry/ovi-runpod:latest .

# Alternative: pass token directly (less secure, visible in shell history)
docker build --build-arg HF_TOKEN=hf_xxxxx -t your-registry/ovi-runpod:latest .
```

### 2. Push to Registry

```bash
docker push your-registry/ovi-runpod:latest
```

### 3. Deploy on RunPod

1. Go to [RunPod Console](https://console.runpod.io)
2. Create new Serverless Endpoint
3. Enter your Docker image URL
4. Configure GPU: **A100 40GB/80GB** or **H100** recommended
5. Set environment variables (see below)
6. Deploy

### 4. Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `R2_ACCOUNT_ID` | Cloudflare account ID | ✅ |
| `R2_ACCESS_KEY_ID` | R2 access key | ✅ |
| `R2_SECRET_ACCESS_KEY` | R2 secret key | ✅ |
| `R2_BUCKET_NAME` | R2 bucket name | ✅ |
| `R2_PUBLIC_URL` | Custom domain URL (optional) | ❌ |
| `OVI_MODEL_NAME` | Default model (960x960_10s) | ❌ |
| `OVI_CPU_OFFLOAD` | CPU offload for memory (false) | ❌ |
| `PRELOAD_MODEL` | Pre-load model at startup (false) | ❌ |

---

## API Reference

### Generate Video

**Endpoint:** `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

**Headers:**
```
Authorization: Bearer {RUNPOD_API_KEY}
Content-Type: application/json
```

**Request:**
```json
{
  "input": {
    "prompt": "A cat playing piano. Audio: upbeat jazz piano music",
    "mode": "t2v",
    "model": "960x960_10s",
    "seed": 42,
    "sample_steps": 50,
    "video_guidance_scale": 4.0,
    "audio_guidance_scale": 3.0
  }
}
```

**Response:**
```json
{
  "id": "video-abc123",
  "object": "video.generation",
  "created": 1705334567,
  "model": "ovi-1.1-960x960_10s",
  "data": [{
    "url": "https://your-bucket.r2.cloudflarestorage.com/videos/2024/01/15/abc123.mp4",
    "revised_prompt": "a_cat_playing_piano"
  }],
  "usage": {
    "prompt_tokens": 12,
    "total_frames": 241,
    "duration_seconds": 10.0,
    "generation_time_seconds": 45.2
  }
}
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text prompt with optional "Audio:" section |
| `mode` | string | "t2v" | Generation mode: t2v, i2v, t2i2v |
| `model` | string | "960x960_10s" | Model variant |
| `image` | string | null | Base64 image for i2v mode |
| `seed` | int | random | Random seed |
| `sample_steps` | int | 50 | Denoising steps (10-100) |
| `height` | int | auto | Video height (t2v/t2i2v) |
| `width` | int | auto | Video width (t2v/t2i2v) |
| `video_guidance_scale` | float | 4.0 | Video conditioning (1-20) |
| `audio_guidance_scale` | float | 3.0 | Audio conditioning (1-20) |

---

## Model Variants

| Variant | Resolution | Duration | Frames | VRAM |
|---------|------------|----------|--------|------|
| `720x720_5s` | 720×720 | 5 sec | 121 | ~32GB |
| `960x960_5s` | 960×960 | 5 sec | 121 | ~40GB |
| `960x960_10s` | 960×960 | 10 sec | 241 | ~48GB |

**For 24GB GPUs:** Enable `OVI_CPU_OFFLOAD=true` and optionally `OVI_QINT8=true`

---

## Prompt Format

Include audio description using "Audio:" prefix:

```
A majestic waterfall in a tropical forest. Audio: rushing water sounds with birds chirping
```

For 720x720_5s model, use special tags:
```
A majestic waterfall. <AUDCAP>rushing water sounds</ENDAUDCAP>
```

---

## Cloudflare R2 Setup

1. Create R2 bucket in Cloudflare dashboard
2. Create API token with R2 read/write permissions
3. (Optional) Set up custom domain for public access
4. Configure environment variables

---

## Local Testing

```bash
# Test handler locally
cd runpod-ovi-service
python src/handler.py --test_input test_input.json
```

---

## Project Structure

```
runpod-ovi-service/
├── Dockerfile
├── requirements.txt
├── README.md
├── test_input.json
├── configs/
│   └── inference_config.yaml
├── scripts/
│   └── download_models.py
└── src/
    ├── handler.py           # RunPod handler
    ├── api_models.py        # Pydantic models
    ├── ovi_engine_wrapper.py # Ovi engine wrapper
    └── r2_storage.py        # R2 upload handler
```

---

## License

MIT License - See Ovi repository for model license terms.
