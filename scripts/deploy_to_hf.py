"""deploy_to_hf.py: Deployment helper script for Hugging Face Gradio Space.

Uploads the VirtuJudge AI-ML worker to Hugging Face Spaces using huggingface_hub,
automatically excluding large media files, local storage, and private keys.

Usage:
  python scripts/deploy_to_hf.py --space-id <username>/<space-name> [--token <hf_token>]
  python scripts/deploy_to_hf.py --dry-run
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure app/__main__ can load .env
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from app.__main__ import _load_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("virtujudge.deploy")

SPACE_README_CONTENT = """---
title: VirtuJudge AI Engine
emoji: 🟢
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: mit
short_description: Multimodal AI Worker for Pitch Evaluation & Q&A
---

# 🟢 VirtuJudge Multimodal AI Engine

Automated Startup Pitch Evaluation Worker running 24/7 on Hugging Face Spaces.

- **Compute:** 2 vCPU · 16 GB RAM (CPU Basic Zero-GPU)
- **Queue:** Upstash Redis Cloud (`virtujudge:jobs`)
- **Speech & LLM:** Groq LPU (Whisper Large V3 Turbo + OpenAI GPT-OSS / Qwen)
- **Vision:** MediaPipe FaceLandmarker + PoseLandmarker
- **Acoustics:** Librosa prosody & tempo analysis
- **Database:** Supabase PostgreSQL with `pgvector`
- **Object Storage:** Cloudflare R2 (`virtujudge-prod`)
"""


def prepare_deployment_package(target_dir: Path) -> list[str]:
    """Copy only required deployment files into the staging directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []

    # 1. Root configuration files
    root_files = [
        "packages.txt",
        "requirements.txt",
        "app.py",
        "init_models.py",
        "consumer.py",
    ]
    for rf in root_files:
        src = ROOT_DIR / rf
        if not src.is_file():
            raise FileNotFoundError(f"Missing required deployment file: {src}")
        dest = target_dir / rf
        shutil.copy2(src, dest)
        copied_files.append(rf)

    # 2. Write Space README.md with YAML metadata
    readme_path = target_dir / "README.md"
    readme_path.write_text(SPACE_README_CONTENT, encoding="utf-8")
    copied_files.append("README.md (Hugging Face Space Card)")

    # 3. Copy production app package
    app_src = ROOT_DIR / "app"
    app_dest = target_dir / "app"
    if app_dest.exists():
        shutil.rmtree(app_dest)

    shutil.copytree(
        app_src,
        app_dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )
    copied_files.append("app/ (Production Codebase)")

    return copied_files


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Deploy VirtuJudge AI-ML Worker to Hugging Face Spaces.")
    parser.add_argument(
        "--space-id",
        type=str,
        default=os.getenv("HF_SPACE_ID"),
        help="Target Hugging Face Space repository ID (e.g. 'username/virtujudge-ai-engine').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        help="Hugging Face User Access Token (with Write permissions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage and validate the deployment package locally without uploading.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional local output directory to export the staged deployment files.",
    )

    args = parser.parse_args()

    stage_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="hf_space_stage_"))
    logger.info("Staging deployment files in: %s", stage_dir)

    copied = prepare_deployment_package(stage_dir)
    logger.info("Deployment bundle prepared successfully with %d components:", len(copied))
    for item in copied:
        logger.info("  ✓ %s", item)

    if args.dry_run:
        logger.info("\n[DRY RUN COMPLETE] Deployment bundle verified. No files were uploaded.")
        print(f"\nStaged files are located at: {stage_dir}")
        print("\nTo deploy via Git manually:")
        print("  1. git clone https://huggingface.co/spaces/<username>/<space-name>")
        print(f"  2. Copy staged contents from '{stage_dir}' into your space clone.")
        print("  3. git add . && git commit -m 'Deploy VirtuJudge AI Worker' && git push")
        return

    if not args.space_id:
        logger.error(
            "Missing --space-id! Specify --space-id <username>/<space-name> or set HF_SPACE_ID environment variable."
        )
        sys.exit(1)

    if not args.token:
        logger.error(
            "Missing Hugging Face token! Specify --token <hf_token> or set HF_TOKEN environment variable."
        )
        sys.exit(1)

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=args.token)
        logger.info("Uploading deployment bundle to Hugging Face Space '%s'...", args.space_id)

        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=args.space_id,
            repo_type="space",
            commit_message="Deploy VirtuJudge AI-ML 24/7 Engine",
        )

        logger.info("\n🎉 Deployment upload succeeded!")
        logger.info("Space URL: https://huggingface.co/spaces/%s", args.space_id)
        logger.info("Gradio Keep-Alive Endpoint: https://%s.hf.space", args.space_id.replace("/", "-").lower())
    except Exception as exc:
        logger.error("Failed to upload to Hugging Face Space: %s", exc)
        sys.exit(1)
    finally:
        if not args.output_dir and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
