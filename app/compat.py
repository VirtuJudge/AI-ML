"""compat.py: Compatibility shims for Python 3.10 and PyTorch / TorchAudio / HuggingFace."""

import types

# 1. torchaudio legacy compatibility shims for pyannote.audio 3.x & speechbrain
try:
    import torchaudio

    # AudioMetaData shim
    if not hasattr(torchaudio, "AudioMetaData"):
        try:
            from torchaudio.backend.common import AudioMetaData

            torchaudio.AudioMetaData = AudioMetaData
        except Exception:
            try:
                from torchaudio._backend.common import AudioMetaData

                torchaudio.AudioMetaData = AudioMetaData
            except Exception:

                class AudioMetaData:  # type: ignore[no-redef]
                    """Fallback AudioMetaData shim for type annotations."""

                    def __init__(
                        self,
                        sample_rate: int = 0,
                        num_frames: int = 0,
                        num_channels: int = 0,
                        bits_per_sample: int = 0,
                        encoding: str = "",
                    ) -> None:
                        self.sample_rate = sample_rate
                        self.num_frames = num_frames
                        self.num_channels = num_channels
                        self.bits_per_sample = bits_per_sample
                        self.encoding = encoding

                torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    # Shims for list_audio_backends, get_audio_backend, set_audio_backend
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile", "sox_io", "ffmpeg"]

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None

    # Also shim torchaudio.backend namespace if present or missing
    if not hasattr(torchaudio, "backend"):
        backend_mod = types.ModuleType("torchaudio.backend")
        backend_mod.list_audio_backends = lambda: ["soundfile", "sox_io", "ffmpeg"]  # type: ignore[attr-defined]
        backend_mod.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
        backend_mod.set_audio_backend = lambda backend: None  # type: ignore[attr-defined]
        torchaudio.backend = backend_mod  # type: ignore[attr-defined]
    else:
        if not hasattr(torchaudio.backend, "list_audio_backends"):
            torchaudio.backend.list_audio_backends = lambda: ["soundfile", "sox_io", "ffmpeg"]  # type: ignore[attr-defined]
        if not hasattr(torchaudio.backend, "get_audio_backend"):
            torchaudio.backend.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
        if not hasattr(torchaudio.backend, "set_audio_backend"):
            torchaudio.backend.set_audio_backend = lambda backend: None  # type: ignore[attr-defined]
except Exception:
    pass


# 2. huggingface_hub use_auth_token -> token compatibility shim for pyannote.audio 3.x
try:
    import huggingface_hub
    import huggingface_hub.file_download

    _orig_hf_hub_download = huggingface_hub.file_download.hf_hub_download

    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            if "token" not in kwargs and tok is not None:
                kwargs["token"] = tok
        return _orig_hf_hub_download(*args, **kwargs)

    huggingface_hub.file_download.hf_hub_download = _patched_hf_hub_download
    huggingface_hub.hf_hub_download = _patched_hf_hub_download

    if hasattr(huggingface_hub, "snapshot_download"):
        _orig_snapshot = huggingface_hub.snapshot_download

        def _patched_snapshot(*args, **kwargs):
            if "use_auth_token" in kwargs:
                tok = kwargs.pop("use_auth_token")
                if "token" not in kwargs and tok is not None:
                    kwargs["token"] = tok
            return _orig_snapshot(*args, **kwargs)

        huggingface_hub.snapshot_download = _patched_snapshot
except Exception:
    pass


# 3. PyTorch 2.6 weights_only compatibility shim for model checkpoints
try:
    import torch
    import torch.serialization

    try:
        import torch.torch_version

        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass

    # Allowlist pyannote classes that torch.load needs to unpickle.
    # This is the proper fix: it works even when pyannote captures its own
    # reference to torch.load before our monkey-patch runs.
    try:
        from pyannote.audio.core.task import Specifications

        torch.serialization.add_safe_globals([Specifications])
    except Exception:
        pass

    # Also allowlist any Enum/dataclass members that Specifications may reference
    try:
        from pyannote.audio.core.task import Problem, Resolution

        torch.serialization.add_safe_globals([Problem, Resolution])
    except Exception:
        pass

    # Fallback: monkey-patch torch.load to default weights_only=False
    # for any remaining globals that aren't allowlisted.
    _orig_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
except Exception:
    pass


# 4. Prevent SpeechBrain lazy-module imports from crashing Gradio hot-reload
#
# Root cause: Gradio's reload server (jurigged) iterates over sys.modules and
# calls `getattr(module, "__file__", None)` on each.  SpeechBrain registers
# LazyModule objects for optional integrations (k2, transformers, deepspeed…).
# The __getattr__ on LazyModule triggers a real import, which raises ImportError
# when the optional dependency isn't installed — crashing the Gradio thread.
#
# Fix: patch LazyModule.__getattr__ so introspection attributes (__file__,
# __path__, __spec__, __loader__) return None without triggering the import.
try:
    from speechbrain.utils.importutils import LazyModule

    _orig_lazy_getattr = LazyModule.__getattr__

    # Attributes that Gradio / jurigged / importlib introspection access
    _INTROSPECTION_ATTRS = frozenset({
        "__file__", "__path__", "__spec__", "__loader__",
        "__name__", "__package__", "__all__",
    })

    def _safe_lazy_getattr(self, attr):  # type: ignore[override]
        if attr in _INTROSPECTION_ATTRS:
            return None
        return _orig_lazy_getattr(self, attr)

    LazyModule.__getattr__ = _safe_lazy_getattr  # type: ignore[assignment]
except Exception:
    pass

