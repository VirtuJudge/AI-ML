"""app.py: Gradio entry point & keep-alive server for Hugging Face Space.

Runs 24/7 on Hugging Face Gradio Space (CPU Basic 2 vCPU · 16 GB RAM).
Serves a public status dashboard and health-check endpoint on port 7860,
while running the RedisQueueConsumer in a background daemon thread.
"""

# ZeroGPU requires `import spaces` to be imported BEFORE torch or any library that imports torch
try:
    import spaces

    @spaces.GPU(duration=10)
    def _zero_gpu_ready() -> bool:
        """Satisfies ZeroGPU startup check if hosted on Hugging Face Zero-GPU hardware."""
        return True

    _zero_gpu_ready()
except Exception:
    pass

import asyncio
import datetime
import logging
import os
import sys
import threading
from typing import Any

import gradio as gr

import app.compat  # noqa: F401
from app.__main__ import _load_env
from consumer import run_redis_consumer
from init_models import warm_up_all

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("virtujudge.space")

# 1. Load environment variables
_load_env()

# 2. Run model warmup and system verification on container boot
init_status = warm_up_all()

worker_error: str | None = None


# 3. Start Redis queue consumer in a background daemon thread
def _start_worker() -> None:
    global worker_error
    logger.info("Starting background Redis queue consumer thread...")
    while True:
        try:
            worker_error = None
            asyncio.run(run_redis_consumer())
        except Exception as exc:
            worker_error = f"{type(exc).__name__}: {exc}"
            logger.error("Background worker thread exited with error: %s. Retrying in 15s...", exc)
            import time

            time.sleep(15)


worker_thread = threading.Thread(target=_start_worker, daemon=True, name="RedisQueueWorker")
worker_thread.start()


# 4. Status and Health Check Helper
def get_system_health() -> dict[str, Any]:
    """Return live system telemetry and health status for monitoring."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "status": "healthy",
        "timestamp": now_iso,
        "worker_thread_alive": worker_thread.is_alive(),
        "worker_error": worker_error,
        "provider_mode": os.getenv("AI_PROVIDER_MODE", "fake"),
        "models_cached": init_status,
        "queue_name": os.getenv("AI_QUEUE_NAME", "virtujudge:jobs"),
        "object_storage_bucket": os.getenv("OBJECT_STORAGE_BUCKET", "virtujudge-prod"),
        "python_version": sys.version.split()[0],
    }


# 5. Build Gradio UI
with gr.Blocks(
    title="VirtuJudge AI Multimodal Engine",
) as demo:
    gr.Markdown("# 🟢 VirtuJudge AI Multimodal Engine")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🛠️ Container Initialization Status")
            gr.JSON(value=init_status, label="Model Cache & Dependencies")

        with gr.Column():
            gr.Markdown("### 🩺 Live Engine Health Check")
            health_output = gr.JSON(value=get_system_health, label="Current System State")
            refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
            refresh_btn.click(fn=get_system_health, inputs=[], outputs=[health_output])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    logger.info("Launching Gradio Space server on 0.0.0.0:%d...", port)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
