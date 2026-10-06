"""Check that the live-provider image contains every required adapter dependency."""

import sys
from importlib import import_module

REQUIRED_MODULES = (
    "pyannote.audio",
    "mediapipe",
    "cv2",
    "librosa",
    "soundfile",
    "pymupdf",
    "pptx",
    "asyncpg",
    "pgvector",
)


def main() -> int:
    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name} ({type(exc).__name__})")

    if missing:
        print("Live provider dependencies are unavailable: " + ", ".join(missing), file=sys.stderr)
        return 1

    print("Live provider dependencies are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
