# VirtuJudge AI-ML Engine

[![Latest Release](https://img.shields.io/badge/Release-v1.2.1-purple?style=flat)](https://github.com/VirtuJudge/AI-ML/releases/tag/v1.2.1)
[![AI Engine Status](https://img.shields.io/badge/AI_Engine-Live_on_HuggingFace-blue?style=flat&logo=huggingface)](https://moadel01-virtujudge-ai-engine.hf.space)
[![Tests](https://img.shields.io/badge/Tests-pytest-success)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen)](pyproject.toml)
[![Compliance](https://img.shields.io/badge/ADR_0007-Compliant_Erasure-success)](tests/test_erasure.py)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey.svg)]()

Asynchronous AI worker for multimodal startup pitch evaluation, interactive Q&A assessment, and rubric-grounded report generation.

---

## 🚀 Live Cloud Deployment

The AI Engine runs 24/7 as an autonomous worker on Hugging Face Spaces:

- **Live Status Dashboard**: [https://moadel01-virtujudge-ai-engine.hf.space](https://moadel01-virtujudge-ai-engine.hf.space)
- **Deployment Tier**: Hugging Face Space (ZeroGPU / CPU Basic 2 vCPU · 16 GB RAM)
- **Queue Consumer**: Celery worker consuming the `ai_jobs` queue through a Redis broker.
- **Production checks**: Readiness verifies broker connectivity, task registration, backend reachability, callback credentials, and object-storage configuration.

---

## 🏗️ Architecture Overview

Jobs are consumed from Redis and processed through modular extraction and evaluation stages:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Queue & Dispatch"]
        Redis[("Redis / Celery Broker")] -->|"ai_jobs"| Celery["Celery task: app.worker.process_job"]
        Celery --> Worker["Persistent-loop worker (app/worker.py)"]
        Worker -->|"POST /updates (started)"| Backend["Backend Orchestrator"]
    end

    subgraph DualProcessing ["2. Parallel Preprocessing & Extraction"]
        Worker -->|"Video / Audio Media"| MediaSplit["FFmpeg Normalizer (app/stages/media.py)"]
        MediaSplit -->|"16kHz Mono WAV"| Whisper["Groq Whisper Large V3 Turbo"]
        MediaSplit -->|"16kHz Mono WAV"| Diarize["PyAnnote Audio 3.1 (VAD & Clustering)"]
        MediaSplit -->|"Stripped Video"| Vision["Google MediaPipe (Face 468 + Pose 33)"]
        MediaSplit -->|"Acoustic Stream"| Audio["Librosa Acoustic Engine (Pace & Pitch)"]
        
        Worker -->|"Slide Deck (.pdf / .pptx)"| DocExtract["PyMuPDF / python-pptx (app/stages/documents.py)"]
        DocExtract -->|"Vector Chunks"| PgVector[("Supabase PostgreSQL pgvector")]
    end

    subgraph FusionCheckpoints ["3. Fusion & Intermediate Checkpointing"]
        Whisper --> Aggregation["Speaker Pre-Aggregation & Lip-MAR Sync (app/stages/aggregation.py)"]
        Diarize --> Aggregation
        Vision --> Aggregation
        Audio --> Aggregation
        Aggregation --> Bundle["EvidenceBundle Registry"]
        
        Whisper -.->|"SHA-256 Checkpoint"| Storage["Cloudflare R2 Storage"]
        Vision -.->|"SHA-256 Checkpoint"| Storage
        Audio -.->|"SHA-256 Checkpoint"| Storage
    end

    subgraph Intelligence ["4. Multi-Judge Persona Panel & Synthesis"]
        Bundle --> KeyPool["GroqKeyPool (4-Key Rotation & Failover)"]
        KeyPool --> J1["Business Strategist (gpt-oss-120b · Key 1)"]
        KeyPool --> J2["Technical Evaluator (gpt-oss-120b · Key 2)"]
        KeyPool --> J3["Product Analyst (qwen3.8-27b · Key 3)"]
        
        J1 --> OutputRouter{"Lifecycle Router"}
        J2 --> OutputRouter
        J3 --> OutputRouter
    end

    subgraph Delivery ["5. Artifact Delivery & Retention"]
        OutputRouter -->|"analyze_session"| AnalysisJSON["ai/session/{id}/analysis.json"]
        OutputRouter -->|"analyze_answer"| AnswerJSON["ai/session/{id}/answers/{aid}/assessment.json"]
        OutputRouter -->|"generate_report"| ReportMD["ai/session/{id}/report.md & evaluation.json"]
        OutputRouter -->|"erase_ai_data"| RetentionPurge["ADR 0007 Physical Erasure Engine"]
        
        AnalysisJSON --> FinalUpdate["POST /updates (completed)"]
        AnswerJSON --> FinalUpdate
        ReportMD --> FinalUpdate
        RetentionPurge --> FinalUpdate
        FinalUpdate --> Backend
    end
```

---

## ✅ Reliable Job Execution

Release `v1.2.1` documents the production execution model introduced after the
previous README revision:

- **Celery transport**: JSON-only messages are consumed from `ai_jobs`; late acknowledgements and worker-loss rejection protect in-flight work.
- **Persistent async runtime**: the Celery worker uses one reusable event loop, preventing HTTP and provider clients from being reused on a closed event loop.
- **Contract validation**: malformed envelopes, missing required fields, unknown job types, invalid checksums, and stale attempts are rejected safely before pipeline execution.
- **Callback lifecycle**: the worker emits ordered `started`, progress, terminal, and cancellation updates. Backend `401`, `404`, and `409` responses stop the affected task rather than triggering unsafe retries.
- **Transient retry policy**: retryable provider failures use bounded Celery retries; invalid input and superseded jobs are acknowledged without retrying.

---

## 📋 The 4 Pipeline Lifecycle Phases

Defined in `app/pipeline.py` and `app/contracts.py`:

### 1. `analyze_session` (Pitch Analysis & Questions)
- **Media Demuxing**: FFmpeg normalizes media into 16kHz mono WAV and stripped video.
- **Multimodal Extraction**:
  - **Speech**: Groq Whisper Large V3 Turbo transcription with word-level timestamps.
  - **Diarization**: PyAnnote Audio 3.1 speaker attribution (`SPEAKER_00`, `SPEAKER_01`).
  - **Vision**: MediaPipe Face & Pose landmarkers extract gaze, head pose, and posture openness.
  - **Acoustics**: Librosa measures speaking pace (WPM), pitch variation, and vocal pauses.
  - **Slide Parsing**: PyMuPDF / python-pptx extracts slide text into Supabase pgvector.
- **Evidence Aggregation**: Correlates lip activity with diarization to build speaker profiles.
- **Judge Panel**: 3-judge panel generates 3 grounded questions.
- **Output**: Uploads `analysis.json` and stage checkpoints to Cloudflare R2.

### 2. `analyze_answer` (Q&A Assessment)
- **Audio Ingestion**: Transcribes answer audio using Groq Whisper (handles skips gracefully).
- **Substantive Evaluation**: Assesses answer depth against the question and pitch materials.
- **Follow-Up Generation**: Generates targeted follow-up questions when quota remains (`remaining_follow_ups > 0`).
- **Output**: Saves `transcript.json` and `assessment.json`.

### 3. `generate_report` (Report & Feedback Synthesis)
- **Evidence Ingestion**: Aggregates `analysis.json`, Q&A transcripts, and answer assessments.
- **Calibrated Scoring**: Evaluates deterministic scores across the 6 rubric dimensions.
- **Presenter Scorecards**: Computes individual speaking time, pace, gaze, and posture metrics.
- **Qualitative Synthesis**: Master Judge synthesizes executive summaries, strengths, and recommendations.
- **Output**: Generates `evaluation.json` and Markdown `report.md`.

### 4. `erase_ai_data` (Physical Erasure & Compliance)
- **`practice_session` Scope**: Purges intermediate observations, checkpoints, scratch media, and vector chunks while preserving final reports.
- **`project` / `team` Scope**: Complete purge across all associated sessions.
- **`asset` Scope**: Deletes uploaded asset files from object storage.

---

## ⚖️ Multi-Judge Persona Panel

Primary questions and answer assessments are evaluated by a specialized 3-judge panel:

- **Business Strategist (`gpt-oss-120b`)**: Market sizing, unit economics, CAC/LTV, monetization.
- **Technical Evaluator (`gpt-oss-120b`)**: Architecture, defensibility, technical feasibility.
- **Product Analyst (`qwen3.8-27b`)**: Product-market fit, user friction, milestones, roadmap.
- **GroqKeyPool**: Resilient key manager rotating across configured API keys with transparent 429/503/529 failover.
- **Guardrails**: Regex filtering suppresses subjective or emotional claims to ensure grounded, objective evaluations.

---

## 📊 Calibrated Startup Rubric Scoring Engine

Evaluates pitches against a calibrated 6-dimension rubric (`STARTUP_PITCH_RUBRIC_V1`):

| Dimension | Configured Weight | Focus Area |
| :--- | :---: | :--- |
| **`pitch_content_and_evidence`** | **25%** | Problem clarity, market sizing (TAM/SAM), and evidence-backed claims. |
| **`business_and_problem_solution_reasoning`** | **20%** | Unit economics, monetization model, and competitive differentiation. |
| **`technical_feasibility`** | **15%** | Proprietary technology, technical defensibility, and architecture. |
| **`delivery_and_body_language`** | **15%** | Gaze alignment, posture openness, and body movement stability. |
| **`timing_and_speech_mechanics`** | **5%** | Speaking pace (130–160 WPM ideal), pause discipline, and verbal fillers. |
| **`qa_quality`** | **20%** | Answer completeness, grounded reasoning, and responsiveness. |

- **Score Scaling**: Dimension scores in `[0.0, 1.0]` scale to display scores `[0, 100]`.
- **Rating Bands**: `0–39` Needs Work · `40–59` Developing · `60–79` Good · `80–100` Strong.
- **Skipped Questions**: Count as `0.0` towards Q&A score without altering rubric weights.
- **Dynamic Rebalancing**: Automatically renormalizes weights when optional media streams (e.g. video) are omitted.

---

## 📂 Repository Structure

```text
AI-ML/
├── app/
│   ├── contracts.py              # Pydantic v2 schemas for backend contracts & updates
│   ├── pipeline.py               # PitchAnalysisPipeline protocol & FakePipeline implementation
│   ├── celery_app.py             # Celery broker, queue, serializer, and acknowledgement settings
│   ├── worker.py                 # Celery task, persistent async runtime, cancellation & retry handling
│   ├── backend_client.py         # Async HTTP client for status callbacks & cancellation checks
│   ├── readiness.py              # Broker, task, backend, credential, and storage readiness checks
│   ├── document_store.py         # Supabase PostgreSQL pgvector store & in-memory fake
│   ├── compat.py                 # Python 3.10 and PyTorch compatibility shims
│   ├── prompts/                  # Judge and report generation prompt templates
│   ├── providers/                # Model adapters (Groq, PyAnnote, MediaPipe, Librosa, PyMuPDF)
│   ├── stages/                   # Pipeline stages (media, speech, vision, audio, scoring, reporting)
│   └── storage/                  # Cloudflare R2 / AWS S3 and local storage clients
├── scripts/                      # Worker runners and end-to-end verification scripts
├── tests/                        # Unit, contract, integration, and reliability coverage
├── pyproject.toml                # Build configuration & dependency definitions
└── README.md                     # Project documentation
```

---

## 🧪 Testing & Verification

```bash
# Install the development dependencies, then run the test suite.
pip install -e ".[dev]"
pytest -q
```

For the Hugging Face production image, dependencies are pinned in
[`requirements.txt`](requirements.txt) and required OS packages are listed in
[`packages.txt`](packages.txt).

---

## ⚙️ Configuration

Copy [`.env.example`](.env.example) to `.env` for local development. Never
commit real credentials.

- `CELERY_BROKER_URL` or `REDIS_URL`: Redis broker used by Celery.
- `AI_QUEUE_NAME`: queue name; defaults to `ai_jobs`.
- `BACKEND_INTERNAL_URL` and `AI_WORKER_SHARED_SECRET`: authenticated backend callback configuration. Both are required in production.
- `OBJECT_STORAGE_*`: S3-compatible object storage configuration, including Cloudflare R2.
- `AI_PROVIDER_MODE`: use `fake` locally; use `live` only when the required model, storage, and API credentials are configured.
- `GROQ_API_KEY` through `GROQ_API_KEY_4`: optional Groq key pool used for transcription and judge calls.

Start the local worker with:

```bash
python -m app
```
