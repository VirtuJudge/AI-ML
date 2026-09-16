# Working in the AI-ML repository

The AI-ML repository runs one Celery worker behind one `PitchAnalysisPipeline` interface. It processes backend-created AI Jobs and returns versioned progress, artifact references, limitations, and results. The backend remains the source of truth for product state.

The repository currently contains only its initial README. Scaffold it from `../Docs/Architecture/AI-ML-Architecture.md` and the active setup issue.

## Read before changing code

- Always use the domain language in `../Docs/CONTEXT.md`.
- For worker shape, pipeline stages, provider ports, and checkpoints, read `../Docs/Architecture/AI-ML-Architecture.md`.
- For job inputs, callbacks, ordering, retries, and result rules, read `../Docs/Contracts/Backend-AI-Contract.md`.
- For Evidence, Evaluation, Report, limitation, and provenance fields, read `../Docs/Contracts/Data-Contracts.md`.
- For consent, private media, human-impact rules, retention, or erasure, read `../Docs/Architecture/Security-Privacy-and-Retention.md`.
- For deterministic tests and model evaluation gates, read `../Docs/Planning/Testing-and-Quality.md`.

## Architecture

Keep one worker and one deep pipeline interface with four operations:

- `analyze_session` returns an Analysis Artifact, exactly three Primary Questions, speaker labels, and limitations;
- `analyze_answer` returns a transcript, assessment, and an optional grounded Follow-up Question;
- `generate_report` returns the final Evaluation and Report with team and individual feedback;
- `erase_data` removes AI-owned data and reports safe counts.

Internal stages may cover media normalization, speech, diarization, vision, audio, documents, retrieval, questions, answers, and reporting. They remain internal to the pipeline. Do not expose separate stage services or make the backend orchestrate them.

Put model and provider SDKs behind small ports. Provider request objects and raw responses stay inside adapters.

## Initial adapters

- Whisper Large V3 Turbo for transcription and word timestamps.
- pyannote Community-1 for anonymous speaker labels.
- MediaPipe Pose and Face Landmarker for timed posture, movement, and gaze observations.
- librosa for speaking rate, pauses, fillers, and pitch measurements.
- LLM, embedding model, hosting, exact versions, and sampling remain explicit implementation choices behind provider ports.

Record every model, prompt, embedding, sampling, stage, and rubric version needed to reproduce a result.

## Evidence and data rules

- Every question, material finding, and score links to typed Evidence or carries an explicit limitation.
- Missing evidence stays missing. Do not turn unavailable signals into zero scores or invented observations.
- Report observable measurements only. Never infer emotion, confidence, honesty, anxiety, personality, or mental state.
- Q&A contributes exactly 20 percent to the final Evaluation.
- A Report contains whole-team feedback and one separate feedback section for every mapped presenter.
- Raw files and large artifacts stay in object storage. AI-owned chunks, embeddings, checkpoints, and derived metadata use `ai_*` tables only.
- Never write backend product tables.

## Jobs, retries, and cancellation

Treat repeated delivery of the same `job_id` as the same work. Reuse a checkpoint only when input checksums and every relevant version match. Check cancellation between expensive stages and before publishing a result. A cancelled or superseded job must not publish usable output.

Post only versioned, authenticated worker updates. Keep provider bodies, private content, stack traces, signed URLs, and secrets out of failures and logs.

## Verification and delivery

Use deterministic fakes for normal tests. Add focused unit, pipeline, shared-schema, idempotency, checkpoint, retry, cancellation, and erasure tests with each change. Run provider-dependent benchmarks before changing a model, prompt, embedding, chunking, sampling, aggregation, rubric, or feature calculation, and record quality, grounding, latency, cost, licence, and data-handling impact.

Update the owning executable schema and the matching Docs contract together. Review benchmark evidence and the full local diff before any push or PR. Follow `../Docs/Planning/Review-Workflow.md`.
