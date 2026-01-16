# Ovi 1.1 RunPod Serverless Service

Production-ready video+audio generation service using Character AI's Ovi 1.1 model, deployed on RunPod Serverless with OpenAI-compatible API and Cloudflare R2 storage.

## Features

- **3 Model Variants**: 720x720 5s, 960x960 5s, 960x960 10s
- **3 Generation Modes**: Text-to-Video, Image-to-Video, Text-to-Image-to-Video
- **Audio Generation**: Synchronized audio with video
- **OpenAI-Compatible API**: Familiar request/response format
- **Cloudflare R2 Storage**: Videos uploaded and URLs returned
- **24GB GPU Support**: FP8/INT8 quantization options
- **Lightweight Image**: ~5GB image, models downloaded to network volume on first run

---

## RunPod Serverless Deployment Guide

### Step 1: Build and Push Docker Image

```bash
# Clone the repository
git clone https://github.com/your-org/runpod-ovi-service.git
cd runpod-ovi-service

# Build the Docker image (~5GB, takes 5-10 minutes)
docker build -t your-dockerhub-username/ovi-runpod:latest .

# Login to Docker Hub (or your registry)
docker login

# Push to registry
docker push your-dockerhub-username/ovi-runpod:latest
```

> **Note**: The image is lightweight (~5GB). Models download automatically to Network Volume on first run.

---

### Step 2: Create Cloudflare R2 Bucket

1. Login to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Go to **R2 Object Storage** → **Create bucket**
3. Name your bucket (e.g., `ovi-videos`)
4. Go to **R2 Overview** → **Manage R2 API Tokens** → **Create API token**
5. Select permissions: **Object Read & Write**
6. Copy and save:
   - **Account ID** (shown in URL or sidebar)
   - **Access Key ID**
   - **Secret Access Key**
7. (Optional) Set up custom domain for public access under bucket settings

---

### Step 3: Create RunPod Network Volume

1. Login to [RunPod Console](https://console.runpod.io)
2. Go to **Storage** → **Network Volumes**
3. Click **+ New Network Volume**
4. Configure:
   - **Name**: `ovi-models` (or any name)
   - **Region**: Choose same region as your endpoint (e.g., `EU-RO-1`)
   - **Size**: `50 GB` (minimum, 100GB recommended for all variants + future models)
5. Click **Create**
6. Note the **Volume ID** for the next step

---

### Step 4: Create Serverless Endpoint

1. Go to **Serverless** → **+ New Endpoint**
2. Configure the endpoint:

   **Basic Settings:**
   | Setting | Value |
   |---------|-------|
   | Endpoint Name | `ovi-video-generator` |
   | Docker Image | `your-dockerhub-username/ovi-runpod:latest` |
   | GPU Type | `48GB` or `80GB` (A100 recommended) |

   **Worker Configuration:**
   | Setting | Recommended Value |
   |---------|-------------------|
   | Active Workers | `0` (scale to zero when idle) |
   | Max Workers | `1-3` (based on your usage) |
   | GPU per Worker | `1` |
   | Idle Timeout | `60` seconds |
   | Execution Timeout | `600` seconds (10 min for long videos) |

3. Click **Advanced** to expand advanced options

4. **Attach Network Volume:**
   - Select your `ovi-models` volume (created in Step 3)
   - RunPod automatically mounts it at `/runpod-volume`

5. **Environment Variables** (click **+ Add Environment Variable** for each):

   | Key | Value |
   |-----|-------|
   | `R2_ACCOUNT_ID` | Your Cloudflare Account ID |
   | `R2_ACCESS_KEY_ID` | Your R2 Access Key |
   | `R2_SECRET_ACCESS_KEY` | Your R2 Secret Key |
   | `R2_BUCKET_NAME` | `ovi-videos` (your bucket name) |
   | `R2_PUBLIC_URL` | `https://your-custom-domain.com` (optional) |
   | `OVI_MODEL_NAME` | `960x960_10s` (default model) |
   | `OVI_CKPT_DIR` | `/runpod-volume/models` |
   | `PRELOAD_MODEL` | `false` |

6. Click **Create Endpoint**

---

### Step 5: Test Your Endpoint

1. Copy your **Endpoint ID** from the endpoint details page
2. Get your **RunPod API Key** from [Settings → API Keys](https://www.runpod.io/console/user/settings)
3. Send a test request:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cat playing piano. Audio: upbeat jazz piano music",
      "mode": "t2v",
      "model": "960x960_10s"
    }
  }'
```

> **First Request**: Will take 3-5 minutes (downloads models + generates video). Subsequent requests: ~45-90 seconds.

---

### Step 6: Monitor and Scale

**View Logs:**
- Go to your endpoint → **Logs** tab
- Monitor model downloads on first run
- Check for any errors

**Scale Workers:**
- Increase **Max Workers** for higher throughput
- Each worker needs its own GPU
- Network volume is shared across workers

**Cost Optimization:**
- Set **Active Workers** to `0` to scale to zero when idle
- Use `720x720_5s` model for cheaper 32GB GPUs
- Enable `OVI_CPU_OFFLOAD=true` for 24GB GPUs

---

## First Run Behavior

On the **first cold start** with a new network volume:
1. Handler checks if models exist in `/runpod-volume/models`
2. If not, downloads ~30GB of model weights from HuggingFace
3. Download takes ~2-3 minutes (one-time only)
4. Models are cached on network volume for all future cold starts

Subsequent cold starts skip the download and load models directly.

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

**Request 2:**
```json
{
  "input": {
    "prompt": "A young man wearing a light blue hoodie, dark pants, and a white baseball cap is performing a dynamic street dance on a stone terrace. In the background, there's a stunning panoramic view of a city sprawling along a large body of water, with a long bridge and a distant statue visible under a bright, sunny sky. He begins on one knee, leaning back with one hand raised, then fluidly rises, bringing his hands to his head before dropping them down. He executes a quick succession of intricate footwork, shifting his weight rapidly and performing small hops and shuffles. His movements are sharp and precise, with a strong rhythmic quality. He bends his knees and extends his arms, then continues with more fast-paced footwork, incorporating body isolations and flowing arm movements. The dance is energetic and expressive, with the dancer's shadow stretching out behind him on the sunlit pavement. <S>Ovi diez segundos, mira cómo baila este algoritmo.<E> Audio: Upbeat, electronic dance music with a strong beat and synthesised elements, a processed, slightly robotic-sounding male voice speaking a Spanish phrase.",
    "mode": "t2v",
    "model": "960x960_10s"
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

## Project Structure

```
runpod-ovi-service/
├── Dockerfile
├── requirements.txt
├── README.md
├── configs/
│   └── inference_config.yaml
├── scripts/
│   └── download_models.py       # Manual model download script
└── src/
    ├── handler.py               # RunPod handler
    ├── download_on_startup.py   # Runtime model download
    ├── api_models.py            # Pydantic models
    ├── ovi_engine_wrapper.py    # Ovi engine wrapper
    └── r2_storage.py            # R2 upload handler
```

---

## Local Development

```bash
# Manual model download (for local testing)
python scripts/download_models.py --output-dir ./models --models 960x960_10s

# Test handler locally
OVI_CKPT_DIR=./models python src/handler.py --test_input test_input.json
```

---

## License

MIT License - See Ovi repository for model license terms.
