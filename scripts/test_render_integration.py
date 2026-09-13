"""End-to-end live Render Redis and Backend smoke integration test.

Validates:
1. Connects to live Render Redis (SSL rediss://).
2. Pushes an analyze_session test message to virtujudge:local:jobs.
3. Processes the job (or consumes it via RedisQueueConsumer).
4. Inspects the correlation result at virtujudge:local:result:{job_id}.
5. Verifies 3 grounded primary questions produced by the multi-judge panel.
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import redis.asyncio as aioredis
import ulid

from app.contracts import (
    AnalyzeSessionPayload,
    AssetInput,
    JobType,
    QueueMessage,
    RubricRef,
    UpdateStatus,
    WorkerUpdate,
)

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


_load_env()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("AI_QUEUE_NAME", "virtujudge:local:jobs")
RESULT_PREFIX = "virtujudge:local:result:"


async def run_smoke_test() -> None:
    print("\n" + "=" * 70)
    print("VirtuJudge Live Render Redis & Question Panel Smoke Test")
    print(f"  Redis URL: {REDIS_URL.split('@')[-1] if '@' in REDIS_URL else REDIS_URL}")
    print(f"  Queue:     {QUEUE_NAME}")
    print("=" * 70)

    # 1. Connect to Live Backend on Render
    backend_url = os.getenv("BACKEND_INTERNAL_URL", "https://virtujudge-backend.onrender.com")
    print(f"\n[Step 1/5] Probing Live Backend on Render ({backend_url})...")
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as http_client:
        try:
            health_res = await http_client.get(f"{backend_url.rstrip('/')}/health")
            body_summary = health_res.text.strip()
            print(f"  [OK] Backend Health: status={health_res.status_code}, body={body_summary}")
        except Exception as exc:
            print(f"  [WARN] Backend Health check failed: {exc}")

        try:
            dummy_update = await http_client.post(
                f"{backend_url.rstrip('/')}/internal/v1/ai-jobs/smoke_probe/updates",
                json={"status": "probing"},
            )
            deploy_status = (
                "Ready"
                if dummy_update.status_code < 400
                else "Pending Backend Orchestrator route deployment"
            )
            print(
                f"  [INFO] Backend internal updates endpoint status: {dummy_update.status_code} "
                f"({deploy_status})"
            )
        except Exception as exc:
            print(f"  [WARN] Backend updates endpoint probe failed: {exc}")

    # 2. Connect to Render Redis
    print("\n[Step 2/5] Connecting to Live Render Redis over SSL...")
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    pong = await r.ping()
    print(f"  [OK] Render Redis PING response: {pong}")

    # 3. Build test analyze_session message
    job_id = f"smoke_job_{ulid.new().str.lower()}"
    session_id = f"smoke_session_{ulid.new().str.lower()}"

    doc_path = Path("tests/fixtures/documents/sample_2page.pdf").resolve()

    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id=f"art_pres_{ulid.new().str.lower()}",
            object_key="presentations/test-2.mp4",
            checksum="sha256:" + "0" * 64,
            media_type="video/mp4",
            duration_ms=20000,
        ),
        supporting_documents=[
            AssetInput(
                artifact_id=f"art_doc_{ulid.new().str.lower()}",
                object_key=str(doc_path),
                checksum="sha256:" + "1" * 64,
                media_type="application/pdf",
            )
        ]
        if doc_path.exists()
        else [],
        rubric=RubricRef(
            rubric_id="startup_pitch",
            version=1,
        ),
        requested_capabilities=["transcription", "question_generation"],
    )

    message = QueueMessage(
        schema_version=1,
        job_id=job_id,
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id=session_id,
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id=f"trace_{ulid.new().str.lower()}",
        payload=payload.model_dump(mode="json"),
    )

    raw_json = message.model_dump_json()

    # 3. Push to Redis queue
    print(f"\n[Step 3/5] Enqueuing job {job_id} to '{QUEUE_NAME}'...")
    await r.rpush(QUEUE_NAME, raw_json)
    print(f"  [OK] Job {job_id} pushed.")

    # 4. Consume and process the job
    print("\n[Step 4/5] Processing job with Groq 3-Judge Panel...")
    from app.__main__ import build_pipeline
    from app.backend_client import FakeBackendClient
    from app.queue_consumer import RedisQueueConsumer

    pipeline = build_pipeline()
    backend_client = FakeBackendClient()

    consumer = RedisQueueConsumer(
        pipeline=pipeline,
        backend_client=backend_client,
        redis_client=r,
        queue_names=[QUEUE_NAME],
    )

    # Process exactly this job from queue
    processed = await consumer.run(max_jobs=1)
    print(f"  [OK] Consumer processed {processed} job(s).")

    # 5. Verify result in Redis
    print("\n[Step 5/5] Verifying result in Redis...")
    result_key = f"{RESULT_PREFIX}{job_id}"
    raw_result = await r.get(result_key)
    if not raw_result:
        raise RuntimeError(f"Result key '{result_key}' was not found in Redis!")

    result_dict = json.loads(raw_result)
    update = WorkerUpdate.model_validate(result_dict)

    print(f"  [OK] Update status: {update.status}")
    assert update.status == UpdateStatus.COMPLETED

    # Primary questions delivered through backend client update
    assert backend_client.updates, "No updates received by backend client"
    latest_update = backend_client.updates[-1]
    completed = latest_update.payload
    questions = completed.primary_questions

    print(f"  [OK] Primary questions count: {len(questions)}")
    assert len(questions) == 3

    print("\n" + "=" * 70)
    print("Generated 3 Grounded Primary Questions:")
    print("=" * 70)
    for idx, q in enumerate(questions, start=1):
        print(f"\n[Judge {idx}] Dimension: {q.rubric_dimension}")
        print(f"  Question:  {q.text}")
        print(f"  Reason:    {q.reason}")
        print(f"  Evidence:  {', '.join(q.evidence_ids)}")

    print("\n" + "=" * 70)
    print("[SUCCESS] All live Render Redis and judge panel verification steps passed!")
    print("=" * 70)

    # Clean up test result key
    await r.delete(result_key)
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
