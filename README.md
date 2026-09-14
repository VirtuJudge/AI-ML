# VirtuJudge AI-ML Engine

[![AI Engine Status](https://img.shields.io/badge/AI_Engine-Live_on_HuggingFace-blue?style=flat&logo=huggingface)](https://moadel01-virtujudge-ai-engine.hf.space)
[![Tests](https://img.shields.io/badge/Tests-329%20passed-success)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey.svg)]()

Production AI worker for multimodal startup pitch analysis, interactive Q&A assessment, and rubric-grounded evaluation report generation.

---

## 🚀 Live Cloud Deployment

The VirtuJudge AI Engine runs 24/7 as an asynchronous worker on Hugging Face Spaces:

- **Live Status Dashboard**: [https://moadel01-virtujudge-ai-engine.hf.space](https://moadel01-virtujudge-ai-engine.hf.space)
- **Deployment Tier**: Hugging Face Space (ZeroGPU / 2 vCPU · 16 GB RAM)
- **Queue Consumer**: Redis async long-polling on `virtujudge:jobs` and `virtujudge:local:jobs`
- **Health Check API**: Direct SSE status endpoint verifying model caches (`ffmpeg`, `mediapipe`, `pyannote`) and worker thread liveliness.

---

## 🧠 Multimodal AI Stack

| Capability | Provider / Engine | Role |
| :--- | :--- | :--- |
| **Speech Transcription** | Groq Whisper Large V3 Turbo | High-accuracy speech-to-text with word-level timestamps |
| **Speaker Diarization** | PyAnnote Audio 3.1 | Identifying individual presenters, clean turn segmentation |
| **Computer Vision** | MediaPipe (Face & Pose Landmarkers) | Eye gaze direction, head pose, posture openness, body movement |
| **Acoustic Analysis** | Librosa | Speaking tempo (WPM), pitch stability, pause & filler tracking |
| **Document Retrieval** | PyMuPDF + Supabase `pgvector` | Slide extraction & vector similarity search for grounded Q&A |
| **LLM Judge Panel** | Groq LPU Cloud (OpenAI GPT-OSS / Qwen) | Multi-judge question generation, answer scoring & report synthesis |
| **Object Storage** | Cloudflare R2 (`virtujudge-prod`) | Analysis artifacts (`analysis.json`), evaluations (`evaluation.json`, `report.md`) |
| **Queue & Messaging** | Redis KeyValue | Asynchronous job dispatching and heartbeat monitoring |

---

## 📋 Pipeline Stages

1. **`analyze_session`**: Processes pitch video/audio and pitch deck PDF. Performs speaker diarization, audio/visual analysis, slide extraction, and generates grounded primary questions with citations.
2. **`analyze_answer`**: Transcribes presenter answers during interactive Q&A rounds, evaluates answer quality against slide evidence, and produces dynamic follow-up questions.
3. **`generate_report`**: Executes the 7-category startup rubric scoring engine (with exactly 20% Q&A weight and skipped answer handling), synthesizes team and individual member feedback with diarization timestamps, and generates structured `evaluation.json` and human-readable `report.md`.
4. **`erase_ai_data`**: Handles secure deletion of session document chunks and artifacts.

---

## 🧪 Testing & Verification

All pipeline stages are verified with unit and integration tests:

```bash
# Run full test suite
pytest -q

# Run with coverage report
pytest --cov=app tests/
```

**Test Status:** 329 passed, 1 skipped, 0 failures.
