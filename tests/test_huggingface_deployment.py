"""Deployment contract checks for the Hugging Face Space runtime."""

from pathlib import Path


def test_huggingface_installs_mediapipe_gles_runtime() -> None:
    """MediaPipe's native landmarker library requires libGLESv2.so.2."""
    packages_file = Path(__file__).resolve().parents[1] / "packages.txt"

    assert packages_file.is_file(), "Hugging Face requires a packages.txt manifest"
    package_names = {
        line.strip()
        for line in packages_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "libgles2" in package_names
