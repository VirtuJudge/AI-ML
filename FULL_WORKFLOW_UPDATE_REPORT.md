# AI Team Full Workflow Update Report

## Purpose

This report defines the AI worker changes required to consume backend-created AI Jobs and complete the VirtuJudge workflow from presentation analysis through Primary Questions, answer analysis, and final report generation.

## Executive Summary

The backend and deployed AI service currently use incompatible queue transports:

- The backend publishes Celery tasks to the `ai_jobs` queue using task name `app.worker.process_job`.
- The deployed AI service uses a custom Redis `BLPOP` consumer listening on `virtujudge:local:jobs` and `virtujudge:jobs`.
- The backend payload is wrapped in the Celery task protocol.
- The AI consumer expects the Redis value itself to be a raw `QueueMessage` JSON object.

Changing only the queue name will not fix the integration because the payload formats are also different.

The architecture decision and repository guidance describe the AI runtime as a Celery worker. The recommended fix is therefore to replace the production `BLPOP` consumer with an actual Celery worker that registers the backend’s task name and consumes `ai_jobs`.

There is also callback configuration risk: the deployed callback client defaults to `http://localhost:3000`. In a Hugging Face Space, that does not identify the Render backend. Production must explicitly configure the backend URL and the same worker secret used by the backend.

## Confirmed Integration Mismatch

### Backend producer

The backend uses:

```text
task name: app.worker.process_job
queue: ai_jobs
serializer: json
broker: REDIS_URL
```

It publishes the AI job envelope as the first Celery positional argument.

### Deployed AI consumer

The deployed worker reports:

```json
{
  "status": "healthy",
  "worker_thread_alive": true,
  "provider_mode": "live",
  "queue_name": "virtujudge:local:jobs"
}
```

Its consumer runs approximately:

```python
item = await redis.blpop(
    ["virtujudge:local:jobs", "virtujudge:jobs"],
    timeout=1,
)
message = QueueMessage.model_validate_json(item[1])
```

It never observes messages placed on `ai_jobs`. If pointed at `ai_jobs`, it would receive a Celery protocol envelope instead of a raw `QueueMessage`, and deserialization would still fail.

## Required Updates

### 1. Adopt one queue protocol: Celery

Implement the production AI runtime as a Celery worker.

Required behavior:

- Use the same `REDIS_URL` broker as the backend.
- Consume queue `ai_jobs`.
- Register task name `app.worker.process_job`.
- Accept one serialized AI job envelope as the first task argument.
- Validate that argument with `QueueMessage.model_validate`.
- Execute the asynchronous pipeline safely from the Celery task context.
- Send monotonic worker updates to the backend.
- Acknowledge jobs according to the agreed retry policy.

Do not keep both the raw Redis and Celery production paths active unless there is a documented migration window. Two consumers with different semantics make delivery and retry behavior ambiguous.

### 2. Create a Celery application owned by the AI repository

Add an AI-side Celery application with explicit configuration:

```python
task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"
timezone = "UTC"
enable_utc = True
task_default_queue = "ai_jobs"
task_ignore_result = True
broker_connection_retry_on_startup = True
```

The task declaration should preserve the backend-owned task name:

```python
@celery_app.task(name="app.worker.process_job", bind=True)
def process_job_task(self, envelope: dict[str, object]) -> None:
    ...
```

Keep the existing asynchronous `process_job` function as the application-level dispatcher. The Celery task should remain a thin transport adapter.

### 3. Define delivery, acknowledgement, and retry semantics

The worker must distinguish three failure classes:

1. Invalid or non-retryable input: report a safe failed update and acknowledge.
2. Transient provider, storage, or backend failure: retry with bounded exponential backoff.
3. Cancellation or supersession: publish a cancelled update and acknowledge.

Recommended Celery settings:

- `task_acks_late = True`
- `task_reject_on_worker_lost = True`
- finite retry count
- exponential retry delay with jitter
- task time limits appropriate for multimodal analysis
- worker prefetch of 1 for expensive jobs

The same `job_id` must remain stable across broker retries. Do not create a new backend AI Job for a delivery retry.

### 4. Configure the production callback destination explicitly

The deployed code currently falls back to `http://localhost:3000`. That is not a valid backend address inside a Hugging Face Space.

Production must set values equivalent to:

```text
BACKEND_INTERNAL_URL=https://virtujudge-backend.onrender.com
AI_WORKER_SHARED_SECRET=<matching backend secret>
```

The exact values must be environment-owned and must not be committed. The worker should fail fast during production startup when either setting is missing.

Do not use a permissive development-secret fallback in production. A healthy UI must not be reported if callbacks cannot authenticate.

### 5. Add a callback readiness check

The current health endpoint proves that the worker thread exists, but not that the full integration works.

Expose safe readiness fields such as:

```json
{
  "status": "ready",
  "broker_connected": true,
  "queue": "ai_jobs",
  "task_registered": true,
  "backend_reachable": true,
  "callback_credentials_configured": true,
  "object_storage_configured": true,
  "provider_mode": "live"
}
```

Do not expose URLs containing credentials, tokens, Redis DSNs, object keys, prompts, or provider responses.

`healthy` should describe the web process. `ready` should describe whether the worker can accept and finish a job.

### 6. Preserve the backend callback contract

For each job, the worker must use:

```text
GET  /internal/v1/ai-jobs/{job_id}
POST /internal/v1/ai-jobs/{job_id}/updates
```

Authentication headers must include the configured worker credential expected by the backend.

Update rules:

- `sequence` starts at 1.
- Sequence numbers increase monotonically.
- `trace_id` matches the queue message.
- The first applied update is `started`.
- Progress updates contain only safe stage metadata.
- Exactly one terminal `completed`, `failed`, or `cancelled` outcome is produced.
- A stale or superseded job must not publish a usable result.

Handle backend responses intentionally:

| Status | Meaning and action |
|---|---|
| `200` | Update applied or safely ignored. |
| `401` | Configuration error; stop retry storms and mark the worker unready. |
| `404` | Unknown or stale job; stop processing and do not publish artifacts. |
| `409` | Invalid state transition; inspect cancellation or supersession. |
| `5xx` | Retry the callback with bounded backoff. |

### 7. Verify the shared queue schema

Add producer/consumer fixture tests for all job types:

- `analyze_session`
- `analyze_answer`
- `generate_report`
- `erase_ai_data`

The tests must use backend-generated fixtures or a published shared schema. They must not construct an AI-only approximation of the message.

For `analyze_session`, validate:

- stable `job_id`
- `practice_session_id`
- `analysis_attempt`
- `created_at`
- `trace_id`
- presentation asset reference
- zero to five supporting document references
- rubric reference
- requested capabilities

### 8. Validate asset access before expensive model work

At the beginning of `analyze_session`:

1. Check cancellation.
2. Validate every input checksum and media type.
3. Confirm object storage access.
4. Download or stream the presentation safely.
5. Download supporting documents through immutable references.
6. Fail with a safe typed code if a required asset is missing.

Do not log signed URLs, object-storage credentials, private document text, transcripts, or raw provider bodies.

### 9. Keep document analysis observable

Emit safe progress at these stage boundaries:

```text
ingestion
speech
diarization
vision
audio_features
documents
aggregation
grounding
questions
```

Each update should include a normalized progress value and a user-safe message. Long document processing should emit bounded progress or heartbeats without repeating confidential text.

Missing or unsupported document signals should produce a limitation when the presentation can still be analyzed. They should fail the job only when the contract makes the document mandatory or the input itself is invalid.

### 10. Enforce question completion invariants

An `analyze_session` completion must contain:

- one valid analysis artifact reference
- exactly three unique Primary Questions
- non-empty grounded Evidence IDs for each question
- allowed rubric dimensions
- detected speaker labels
- safe limitations

Fallback questions must still be grounded in available Evidence. Do not invent citations when speech, visual, or document evidence is unavailable.

The backend validates these invariants and will reject an invalid completed callback.

### 11. Verify answer and report jobs under the same transport

The queue repair must cover more than initial question generation.

`analyze_answer` must:

- verify the submitted answer audio reference
- produce transcript and assessment artifact IDs
- optionally produce a grounded Follow-up Question
- respect the two-follow-up maximum enforced by the backend

`generate_report` must:

- load the correct Practice Session analysis and Q&A artifacts
- preserve the Practice Session ID in artifact object paths
- return Evaluation and Report references
- include team feedback and one separate member section per mapped presenter
- preserve the 20% Q&A weighting

The local uncommitted report-path changes should be reviewed and converted into tests before deployment rather than copied blindly.

### 12. Make deployment topology explicit

Hugging Face Spaces are primarily web-app runtimes. Confirm that the selected runtime keeps the worker process alive and allows a long-running broker connection.

The Space should start:

- the health/readiness web UI
- one supervised Celery worker process or thread

If the platform suspends the Space, marks it idle, or cannot reliably supervise a background worker, move the worker to a dedicated always-on worker platform. A green Gradio page alone is not evidence that queue processing is available.

## Testing Requirements

### Queue integration test

Add a test that:

1. Starts a real Redis test broker.
2. Uses the backend Celery producer configuration to publish a job to `ai_jobs`.
3. Runs the AI Celery consumer.
4. Confirms the exact `QueueMessage` reaches `process_job`.
5. Confirms the job is acknowledged after the expected outcome.

The test must fail when the worker listens only to `virtujudge:local:jobs`.

### Callback integration tests

Run the worker against a backend test server and assert:

- started update accepted
- progress sequence accepted
- completed update accepted
- duplicate sequence ignored safely
- stale attempt rejected or ignored safely
- incorrect shared secret returns 401 without an infinite retry loop
- cancellation observed between expensive stages

### Full pipeline test

For an `analyze_session` job containing a short video and one PDF:

1. Publish through Celery.
2. Fetch inputs from the configured object store.
3. Execute deterministic fake providers.
4. Send callbacks to the real backend test app.
5. Assert the backend creates one Q&A Round with exactly three Primary Questions.
6. Assert the Practice Session becomes `questions_ready`.

Run provider-dependent live tests separately from deterministic CI.

### Deployment smoke test

After deployment, create one disposable Practice Session through the public backend workflow and verify:

- an AI Job changes from `pending` or `queued` to `running`
- the Space logs the same job ID and trace ID
- the backend receives sequence 1
- document analysis completes or records an explicit limitation
- three questions are persisted
- the job reaches `completed`
- no secrets or private content appear in logs

## Acceptance Criteria

- Backend and AI worker use the same Redis broker, queue name, task name, and message protocol.
- A backend-published Celery task is consumed without manual Redis insertion.
- Production startup fails clearly when callback URL or worker secret is missing.
- Worker readiness distinguishes a live UI from a functioning queue consumer.
- A valid session job reaches `started`, progress, and `completed` callbacks.
- The backend persists exactly three Primary Questions and transitions the session to `questions_ready`.
- Document analysis emits safe progress and limitations.
- Answer analysis and report generation use the same repaired queue path.
- Cancellation, retry, duplicate delivery, and stale-attempt behavior are covered by tests.
- Credentials, signed URLs, prompts, transcripts, private document content, and provider bodies do not appear in logs.

## Recommended Implementation Order

1. Add the AI Celery application and registered task.
2. Add a cross-repository broker integration test.
3. Remove the production `BLPOP` path.
4. Enforce production callback configuration.
5. Add backend reachability and broker readiness checks.
6. Verify `analyze_session` callbacks and three-question persistence.
7. Verify document ingestion and object-storage access.
8. Verify `analyze_answer` and `generate_report` through Celery.
9. Deploy and run the complete smoke test with a new Practice Session.
10. Monitor queue depth, job age, callback failures, and worker readiness.

## Cross-Team Dependencies

The AI worker will not receive jobs from the currently inspected sessions because the frontend never creates an Analysis Attempt. The frontend team must first perform the `draft -> ready -> analyzing` lifecycle.

The backend remains the owner of:

- public workflow state
- AI Job persistence
- queue-message schemas
- worker callback schemas
- Q&A persistence
- Evaluation and Report persistence

Any boundary change must update the executable schema, shared fixtures, and documentation together.

## Definition of Done

The AI work is complete only when a job created by the deployed backend is consumed automatically by the deployed worker, processes the real presentation and supporting documents, sends authenticated monotonic updates, causes exactly three grounded Primary Questions to be persisted, and later processes answers and produces the final report without manual queue or database intervention.
