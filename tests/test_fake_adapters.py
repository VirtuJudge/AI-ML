"""Tests verifying fake stage adapters return valid types and deterministic results."""

from pathlib import Path

import pytest

from app.contracts import PrimaryQuestion
from app.providers import (
    AudioObservation,
    DiarizationResult,
    DocumentChunk,
    FakeAudioMetricsProvider,
    FakeDiarizationProvider,
    FakeDocumentProvider,
    FakeEmbeddingProvider,
    FakeJudgeModelProvider,
    FakeSpeechProvider,
    FakeVisionProvider,
    TranscriptionResult,
    VisualObservation,
)

DUMMY_PATH = Path("dummy_file.ext")


@pytest.mark.asyncio
async def test_fake_speech_provider() -> None:
    """Test FakeSpeechProvider returns valid TranscriptionResult and is deterministic."""
    provider = FakeSpeechProvider()
    res1 = await provider.transcribe(DUMMY_PATH)
    res2 = await provider.transcribe(DUMMY_PATH)

    assert isinstance(res1, TranscriptionResult)
    assert len(res1.segments) == 3
    assert res1.segments[0].confidence > 0.9
    assert res1 == res2


@pytest.mark.asyncio
async def test_fake_diarization_provider() -> None:
    """Test FakeDiarizationProvider returns valid DiarizationResult and is deterministic."""
    provider = FakeDiarizationProvider()
    res1 = await provider.diarize(DUMMY_PATH)
    res2 = await provider.diarize(DUMMY_PATH)

    assert isinstance(res1, DiarizationResult)
    assert len(res1.segments) == 2
    assert res1.speaker_labels == ["SPEAKER_00", "SPEAKER_01"]
    assert res1 == res2


@pytest.mark.asyncio
async def test_fake_vision_provider() -> None:
    """Test FakeVisionProvider returns valid VisualObservation list and is deterministic."""
    provider = FakeVisionProvider()
    res1 = await provider.analyze_video(DUMMY_PATH)
    res2 = await provider.analyze_video(DUMMY_PATH)

    assert isinstance(res1, list)
    assert all(isinstance(obs, VisualObservation) for obs in res1)
    assert len(res1) == 3
    assert res1 == res2


@pytest.mark.asyncio
async def test_fake_audio_metrics_provider() -> None:
    """Test FakeAudioMetricsProvider returns valid AudioObservation list and is deterministic."""
    provider = FakeAudioMetricsProvider()
    res1 = await provider.extract_metrics(DUMMY_PATH)
    res2 = await provider.extract_metrics(DUMMY_PATH)

    assert isinstance(res1, list)
    assert all(isinstance(obs, AudioObservation) for obs in res1)
    assert len(res1) == 3
    assert res1[0].pitch_hz > 0
    assert res1 == res2


@pytest.mark.asyncio
async def test_fake_document_and_embedding_providers() -> None:
    """Test FakeDocumentProvider and FakeEmbeddingProvider."""
    doc_provider = FakeDocumentProvider()
    embed_provider = FakeEmbeddingProvider(dimension=64)

    doc_res1 = await doc_provider.extract_and_embed(DUMMY_PATH)
    doc_res2 = await doc_provider.extract_and_embed(DUMMY_PATH)

    assert isinstance(doc_res1, list)
    assert all(isinstance(chunk, DocumentChunk) for chunk in doc_res1)
    assert len(doc_res1) == 2
    assert doc_res1 == doc_res2

    embeddings = await embed_provider.embed(["text a", "text b"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 64
    assert len(embeddings[1]) == 64


@pytest.mark.asyncio
async def test_fake_judge_model_provider() -> None:
    """Test FakeJudgeModelProvider returns 3 PrimaryQuestions and is deterministic."""
    provider = FakeJudgeModelProvider()
    questions1 = await provider.generate_questions("dummy transcript", "startup_pitch")
    questions2 = await provider.generate_questions("dummy transcript", "startup_pitch")

    assert len(questions1) == 3
    assert all(isinstance(q, PrimaryQuestion) for q in questions1)
    assert questions1 == questions2
