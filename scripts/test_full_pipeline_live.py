"""End-to-end test runner for the full VirtuJudge pipeline in LIVE mode.

Flow:
1. Stage 1 (analyze_session):
   - Ingests test.mkv and sample_2page.pdf.
   - Splits media via FFmpeg into 16kHz mono WAV + video.
   - Transcribes audio via Groq Whisper Large V3 Turbo.
   - Diarizes speakers via PyAnnote (or graceful fallback).
   - Analyzes vision via MediaPipe (gaze, posture, lip activity).
   - Analyzes acoustics via Librosa (pitch, tempo, pauses).
   - Combines timed evidence and pre-aggregates 10s discrete windows by_speaker.
   - Parses pitch deck via PyMuPDF.
   - Generates 3 grounded primary questions via Groq 3-Judge LLM panel.
   - Stores ai/session/{session_id}/analysis.json.

2. Stage 2 (analyze_answer):
   - Slices audio clips from the recording to simulate founder answers to each of the 3 questions.
   - Transcribes answers via Groq Whisper.
   - Evaluates answers and generates follow-up questions via Groq Judge panel.
   - Stores ai/answer/{answer_id}/transcript.json and assessment.json.

3. Stage 3 (generate_report):
   - Computes rubric math scores (whole-team pitch, Q&A quality, individual delivery).
   - Generates qualitative feedback narrative via Groq Judge model.
   - Generates formatted Markdown report (report.md).
   - Stores ai/session/{report_id}/evaluation.json and report.md.
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure Conda Library\bin (where ffmpeg.exe lives on Windows) is in PATH
_conda_bin = Path(sys.prefix) / "Library" / "bin"
if _conda_bin.is_dir() and str(_conda_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_conda_bin) + os.pathsep + os.environ.get("PATH", "")

# Load .env variables
_env_path = Path(".env")
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Keep third-party loggers less verbose during CLI demo
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pydub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from app.contracts import (  # noqa: E402
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    ArtifactRef,
    AssetInput,
    AudioAssetInput,
    GenerateReportPayload,
    RubricRef,
    SpeakerMapping,
)
from app.document_store import create_document_store, PgVectorDocumentStore  # noqa: E402
from app.pipeline import FakePipeline, generate_deterministic_ulid  # noqa: E402
from app.providers.fake_audio import FakeAudioMetricsProvider  # noqa: E402
from app.providers.fake_documents import FakeDocumentProvider  # noqa: E402
from app.providers.fake_speech import FakeDiarizationProvider  # noqa: E402
from app.providers.fake_vision import FakeVisionProvider  # noqa: E402
from app.providers.groq_judge import GroqJudgeModelProvider  # noqa: E402
from app.providers.groq_speech import GroqSpeechProvider  # noqa: E402
from app.storage import S3ObjectStorage, create_object_storage  # noqa: E402
from app.storage.local import LocalDiskObjectStorage  # noqa: E402


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def slice_audio_clip(source_audio: Path, output_path: Path, start_s: float, duration_s: float) -> None:
    """Extract a short audio clip from a WAV file using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(duration_s),
        "-i",
        str(source_audio),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


async def main() -> None:
    print("=" * 80)
    print("      VIRTUJUDGE FULL END-TO-END PIPELINE (LIVE EVALUATION)")
    print("=" * 80)

    video_file = Path("test.mkv")
    deck_file = Path("tests/fixtures/documents/sample_2page.pdf")

    if not video_file.is_file():
        print(f"Error: Video file '{video_file}' not found.")
        sys.exit(1)
    if not deck_file.is_file():
        print(f"Error: Supporting deck '{deck_file}' not found.")
        sys.exit(1)

    print(f"\n[INPUTS]")
    print(f"  Presentation Video: {video_file.resolve()} ({video_file.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"  Supporting Deck:    {deck_file.resolve()} ({deck_file.stat().st_size / 1024:.1f} KB)")

    # 1. Initialize live providers
    print("\n[INIT] Initializing Live Pipeline Providers...")
    speech_provider = GroqSpeechProvider()

    try:
        from app.providers.pyannote_diarization import PyannoteDiarizationProvider
        diar_provider = PyannoteDiarizationProvider()
        print("  [OK] PyAnnote Diarization Provider active")
    except Exception as exc:
        print(f"  [INFO] PyAnnote fallback to FakeDiarizationProvider ({exc})")
        diar_provider = FakeDiarizationProvider()

    try:
        from app.providers.mediapipe_vision import MediaPipeVisionProvider
        vision_provider = MediaPipeVisionProvider()
        print("  [OK] MediaPipe Vision Provider active (Face + Pose landmarkers)")
    except Exception as exc:
        print(f"  [INFO] MediaPipe fallback to FakeVisionProvider ({exc})")
        vision_provider = FakeVisionProvider()

    try:
        from app.providers.librosa_audio import LibrosaAudioProvider
        audio_provider = LibrosaAudioProvider()
        print("  [OK] Librosa Acoustic Audio Provider active (prosody, tempo, pauses)")
    except Exception as exc:
        print(f"  [INFO] Librosa fallback to FakeAudioMetricsProvider ({exc})")
        audio_provider = FakeAudioMetricsProvider()

    try:
        from app.providers.pymupdf_documents import PyMuPDFDocumentProvider
        doc_provider = PyMuPDFDocumentProvider()
        print("  [OK] PyMuPDF Document Provider active (deck text extraction)")
    except Exception as exc:
        print(f"  [INFO] PyMuPDF fallback to FakeDocumentProvider ({exc})")
        doc_provider = FakeDocumentProvider()

    judge_provider = GroqJudgeModelProvider()
    print("  [OK] Groq 3-Judge LLM Model Provider active (Key Pool multi-key retry)")

    storage = create_object_storage()
    if isinstance(storage, S3ObjectStorage):
        print(f"  [OK] Object Storage initialized (Cloudflare R2 S3: bucket={storage.bucket} @ {storage.endpoint_url})")
    else:
        print(f"  [OK] Object Storage initialized ({type(storage).__name__}: target={getattr(storage, 'base_dir', '.storage')})")

    doc_store = await create_document_store()
    print(f"  [OK] Document Store initialized ({type(doc_store).__name__} -> Supabase PostgreSQL / pgvector)")

    pipeline = FakePipeline(
        speech_provider=speech_provider,
        diarization_provider=diar_provider,
        vision_provider=vision_provider,
        audio_provider=audio_provider,
        document_provider=doc_provider,
        judge_provider=judge_provider,
        document_store=doc_store,
        object_storage=storage,
    )

    # -------------------------------------------------------------------------
    # STAGE 1: Analyze Session
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 1: Running analyze_session (Pitch Video + Deck Analysis)...")
    print("-" * 80)

    session_id = "01JSESSION_LIVE_DEMO_01"
    video_checksum = compute_sha256(video_file)
    deck_checksum = compute_sha256(deck_file)

    session_job = AnalyzeSessionPayload(
        practice_session_id=session_id,
        presentation=AssetInput(
            artifact_id="01J_PRES_VIDEO_001",
            object_key=str(video_file),
            checksum=video_checksum,
            media_type="video/x-matroska",
        ),
        supporting_documents=[
            AssetInput(
                artifact_id="01J_DOC_DECK_001",
                object_key=str(deck_file),
                checksum=deck_checksum,
                media_type="application/pdf",
            )
        ],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=[
            "speech",
            "diarization",
            "vision",
            "audio",
            "documents",
            "questions",
        ],
    )

    t0 = asyncio.get_event_loop().time()
    session_res = await pipeline.analyze_session(session_job)
    stage1_duration = asyncio.get_event_loop().time() - t0

    print(f"\n  [STAGE 1 COMPLETED in {stage1_duration:.1f}s]")
    print(f"  Detected Speakers ({len(session_res.speaker_labels)}):   {session_res.speaker_labels}")
    print(f"  Primary Questions Generated: {len(session_res.primary_questions)}")
    if isinstance(doc_store, PgVectorDocumentStore):
        async with doc_store.db_pool.acquire() as conn:
            cnt = await conn.fetchval(
                "SELECT COUNT(*) FROM ai_document_chunks WHERE session_id = $1;",
                session_id,
            )
            print(f"  Document Chunks in Supabase: {cnt} rows stored in 'ai_document_chunks' table")

    print("\n--- 3 GROUNDED PRIMARY QUESTIONS GENERATED BY GROQ JUDGES ---")
    for idx, q in enumerate(session_res.primary_questions, start=1):
        print(f"\n  [Judge {idx}] Rubric Dimension: {q.rubric_dimension}")
        print(f"    Question: {q.text}")
        print(f"    Reason:   {q.reason}")
        print(f"    Evidence: {q.evidence_ids}")

    # -------------------------------------------------------------------------
    # STAGE 2: Analyze Answers (Q&A Simulation)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 2: Running analyze_answer for each generated question...")
    print("-" * 80)

    # We will slice 3 realistic 15s snippets from the speech audio to represent founder answers
    temp_dir = Path(".storage/temp_answers")
    temp_dir.mkdir(parents=True, exist_ok=True)

    qa_assessments: list[dict] = []
    round_offsets = [10.0, 45.0, 90.0]

    for idx, q in enumerate(session_res.primary_questions):
        print(f"\n  [Q&A Round {idx + 1}/3] Evaluating Founder Answer to Judge {idx + 1}...")
        answer_wav = temp_dir / f"answer_{idx + 1}.wav"
        
        # Split a 15-second snippet from the test video
        slice_audio_clip(video_file, answer_wav, start_s=round_offsets[idx], duration_s=15.0)
        ans_checksum = compute_sha256(answer_wav)

        answer_job = AnalyzeAnswerPayload(
            qa_round_id=f"01J_ROUND_{idx + 1}",
            question_id=q.candidate_id,
            answer_id=f"01J_ANSWER_{idx + 1}",
            answered_by="user_founder_01",
            audio=AudioAssetInput(
                artifact_id=f"01J_AUDIO_ANS_{idx + 1}",
                object_key=str(answer_wav),
                checksum=ans_checksum,
                media_type="audio/wav",
                duration_ms=15000,
            ),
            remaining_follow_ups=1 if idx == 0 else 0,
        )

        t_ans = asyncio.get_event_loop().time()
        ans_res = await pipeline.analyze_answer(answer_job)
        ans_duration = asyncio.get_event_loop().time() - t_ans

        assessment_data = await storage.read_json(f"ai/answer/{answer_job.answer_id}/assessment.json")
        qa_assessments.append(assessment_data)

        print(f"    Assessment Duration: {ans_duration:.1f}s")
        print(f"    Answer Score:        {assessment_data.get('score', 'N/A')}/100")
        print(f"    Confidence:          {assessment_data.get('confidence', 'N/A')}")
        print(f"    Follow-up Generated: {ans_res.follow_up.text if ans_res.follow_up else 'None (Final round)'}")

    # Build Q&A aggregate artifact for report stage
    qa_artifact_id = generate_deterministic_ulid(f"{session_id}:qa_aggregate")
    qa_artifact_key = f"ai/session/{session_id}/qa_aggregate.json"
    qa_aggregate_data = {
        "artifact_id": qa_artifact_id,
        "session_id": session_id,
        "qa_rounds": qa_assessments,
        "created_at": datetime.now(UTC).isoformat(),
    }
    qa_ref = await storage.upload_json(qa_artifact_key, qa_aggregate_data, artifact_id=qa_artifact_id)

    # -------------------------------------------------------------------------
    # STAGE 3: Generate Final Report & Evaluation (AI-07)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 3: Running generate_report (Final Evaluation + report.md)...")
    print("-" * 80)

    speaker_labels = session_res.speaker_labels or ["SPEAKER_00"]
    speaker_mappings = [
        SpeakerMapping(
            speaker_label=speaker_labels[0],
            user_id="user_alice_ceo",
            display_name="Alice (Founder & CEO)",
        )
    ]
    if len(speaker_labels) > 1:
        speaker_mappings.append(
            SpeakerMapping(
                speaker_label=speaker_labels[1],
                user_id="user_bob_cto",
                display_name="Bob (Co-founder & CTO)",
            )
        )

    report_id = "01JREPORT_LIVE_DEMO_01"
    report_job = GenerateReportPayload(
        report_id=report_id,
        analysis_artifact=session_res.analysis_artifact,
        qa_artifact=qa_ref,
        speaker_mappings=speaker_mappings,
    )

    t_rep = asyncio.get_event_loop().time()
    report_res = await pipeline.generate_report(report_job)
    rep_duration = asyncio.get_event_loop().time() - t_rep

    print(f"\n  [STAGE 3 COMPLETED in {rep_duration:.1f}s]")
    print(f"  Evaluation Artifact: {report_res.evaluation_artifact.object_key}")
    print(f"  Report Markdown:     {report_res.report_artifact.object_key}")
    print(f"  Evaluated Members:   {report_res.member_feedback_user_ids}")

    eval_json = await storage.read_json(report_res.evaluation_artifact.object_key)
    overall_score = eval_json.get("overall_score")
    score_str = f"{overall_score:.1f}/100" if overall_score is not None else "N/A"
    print("\n" + "=" * 80)
    print("                     EVALUATION SCORECARD")
    print("=" * 80)
    print(f"  Overall Score: {score_str}")
    print("\n  [Component Scores]")
    for comp in eval_json.get("components", []):
        d_score = f"{comp.get('display_score')}/100" if comp.get("display_score") is not None else "N/A"
        print(f"    - {comp.get('dimension'):<25}: {d_score:<10} (Weight: {comp.get('configured_weight', 0):.2f})")

    # Read and print the report markdown preview
    temp_report_file = Path(".storage/temp_report_preview.md")
    temp_report_file.parent.mkdir(parents=True, exist_ok=True)
    await storage.download_file(report_res.report_artifact.object_key, temp_report_file)
    report_md_content = temp_report_file.read_text(encoding="utf-8")
    temp_report_file.unlink(missing_ok=True)

    print("\n" + "=" * 80)
    print("               FINAL GENERATED REPORT.MD (PREVIEW)")
    print("=" * 80)
    # Print the first 60 lines of the report markdown
    for line in report_md_content.splitlines()[:60]:
        print(line)
    if len(report_md_content.splitlines()) > 60:
        print(f"\n... [{len(report_md_content.splitlines()) - 60} more lines in object storage ({report_res.report_artifact.object_key})]")

    print("\n" + "=" * 80)
    print("  >>> FULL PIPELINE END-TO-END EXECUTION SUCCEEDED! <<<")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
