"""Unit tests for app.stages.evidence (EvidenceBundle & build_evidence_bundle)."""

from app.contracts import Limitation
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    DocumentChunk,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    VisualObservation,
)
from app.stages.audio import AudioStageResult
from app.stages.evidence import (
    EvidenceBundle,
    EvidenceItem,
    build_evidence_bundle,
)
from app.stages.speech import SpeechStageResult
from app.stages.vision import VisionStageResult


def _make_speech_result() -> SpeechStageResult:
    return SpeechStageResult(
        transcription=TranscriptionResult(
            full_text="Our platform reduces customer acquisition cost by 40%.",
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=3000,
                    text="Our platform reduces customer acquisition cost by 40%.",
                    confidence=0.98,
                ),
                TranscriptionSegment(
                    start_ms=3500,
                    end_ms=6000,
                    text="We achieved this through automated algorithmic matching.",
                    confidence=0.95,
                ),
            ],
        ),
        diarization=DiarizationResult(
            speaker_labels=["SPEAKER_00"],
            segments=[
                SpeakerSegment(start_ms=0, end_ms=6000, speaker_label="SPEAKER_00"),
            ],
        ),
        speaker_labels=["SPEAKER_00"],
        limitations=[],
    )


def _make_vision_result() -> VisionStageResult:
    return VisionStageResult(
        observations=[
            VisualObservation(
                start_ms=500,
                end_ms=1500,
                metric="posture_openness",
                value=0.85,
                unit="score",
                speaker_label="SPEAKER_00",
            ),
        ],
        limitations=[
            Limitation(
                code="camera_distance_warning",
                scope="vision",
                message="Presenter stood far from camera.",
                affected_dimensions=["visual_delivery"],
            )
        ],
    )


def _make_audio_result() -> AudioStageResult:
    return AudioStageResult(
        observations=[
            AudioObservation(
                start_ms=0,
                end_ms=3000,
                metric="speaking_rate_wpm",
                value=145.0,
                unit="wpm",
                speaker_label="SPEAKER_00",
            ),
        ],
        limitations=[],
    )


def _make_document_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_01",
            page_or_slide=1,
            text="VirtuJudge Executive Summary: Next-gen AI pitch feedback platform.",
            extraction_method="pymupdf_text",
            chunking_version="1.0",
        ),
        DocumentChunk(
            chunk_id="chunk_02",
            page_or_slide=3,
            text="Financial Projections: Projected ARR of $2M by Year 2.",
            extraction_method="pymupdf_text",
            chunking_version="1.0",
        ),
    ]


def test_build_evidence_bundle_all_modalities() -> None:
    """Verify bundle contains all 4 modalities and flags are properly set."""
    speech = _make_speech_result()
    vision = _make_vision_result()
    audio = _make_audio_result()
    docs = _make_document_chunks()

    bundle = build_evidence_bundle(
        speech_result=speech,
        vision_result=vision,
        audio_result=audio,
        document_chunks=docs,
    )

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.has_documents is True
    assert bundle.has_vision is True
    assert bundle.has_audio is True
    assert bundle.speaker_labels == ["SPEAKER_00"]
    assert len(bundle.items) == 2 + 2 + 1 + 1  # 2 speech, 2 docs, 1 vision, 1 audio = 6

    # Verify ID format and prefixes
    speech_items = bundle.get_items_for_source("speech")
    assert len(speech_items) == 2
    assert speech_items[0].evidence_id == "ev_speech_001"
    assert speech_items[1].evidence_id == "ev_speech_002"
    assert speech_items[0].speaker_label == "SPEAKER_00"

    doc_items = bundle.get_items_for_source("documents")
    assert len(doc_items) == 2
    assert doc_items[0].evidence_id == "ev_doc_slide_01"
    assert doc_items[1].evidence_id == "ev_doc_slide_02"
    assert doc_items[1].page_or_slide == 3

    vision_items = bundle.get_items_for_source("vision")
    assert len(vision_items) == 1
    assert vision_items[0].evidence_id == "ev_vision_001"
    assert vision_items[0].metric == "posture_openness"

    audio_items = bundle.get_items_for_source("audio")
    assert len(audio_items) == 1
    assert audio_items[0].evidence_id == "ev_audio_001"
    assert audio_items[0].metric == "speaking_rate_wpm"

    # Limitations from vision merged
    assert any(lim.code == "camera_distance_warning" for lim in bundle.limitations)


def test_build_evidence_bundle_presentation_only() -> None:
    """Verify presentation-only session (no documents) produces valid speech+vision+audio."""
    speech = _make_speech_result()
    vision = _make_vision_result()
    audio = _make_audio_result()

    bundle = build_evidence_bundle(
        speech_result=speech,
        vision_result=vision,
        audio_result=audio,
        document_chunks=None,
    )

    assert bundle.has_documents is False
    assert bundle.has_vision is True
    assert bundle.has_audio is True
    assert len(bundle.get_items_for_source("documents")) == 0
    assert len(bundle.get_items_for_source("speech")) == 2
    assert bundle.has_evidence_id("ev_speech_001")
    assert not bundle.has_evidence_id("ev_doc_slide_01")


def test_build_evidence_bundle_missing_vision() -> None:
    """Verify audio-only session (missing vision) builds without vision items."""
    speech = _make_speech_result()
    audio = _make_audio_result()
    docs = _make_document_chunks()

    bundle = build_evidence_bundle(
        speech_result=speech,
        vision_result=None,
        audio_result=audio,
        document_chunks=docs,
    )

    assert bundle.has_vision is False
    assert bundle.has_audio is True
    assert bundle.has_documents is True
    assert len(bundle.get_items_for_source("vision")) == 0


def test_evidence_id_uniqueness_and_query_helpers() -> None:
    """Verify all evidence IDs are unique across all 4 prefixes and query helpers work."""
    speech = _make_speech_result()
    vision = _make_vision_result()
    audio = _make_audio_result()
    docs = _make_document_chunks()

    bundle = build_evidence_bundle(
        speech_result=speech,
        vision_result=vision,
        audio_result=audio,
        document_chunks=docs,
    )

    all_ids = [item.evidence_id for item in bundle.items]
    assert len(all_ids) == len(set(all_ids))

    # Test lookup
    item = bundle.get_item("ev_doc_slide_01")
    assert item is not None
    assert isinstance(item, EvidenceItem)
    assert item.page_or_slide == 1
    assert bundle.get_item("non_existent_id") is None

    # Test dimension query
    dim_items = bundle.get_items_for_dimension("market_and_business_model")
    assert len(dim_items) > 0


def test_format_summary_for_prompt() -> None:
    """Verify prompt formatting outputs expected headings and items."""
    speech = _make_speech_result()
    docs = _make_document_chunks()

    bundle = build_evidence_bundle(
        speech_result=speech,
        document_chunks=docs,
    )

    summary = bundle.format_summary_for_prompt()
    assert "### Spoken Presentation Evidence:" in summary
    assert "ev_speech_001" in summary
    assert "### Supporting Document & Slide Evidence:" in summary
    assert "ev_doc_slide_01" in summary
    assert "Slide 1" in summary


def test_evidence_bundle_add_limitations() -> None:
    """Verify EvidenceBundle encapsulation methods for limitations."""
    bundle = EvidenceBundle(items=[])
    assert bundle.limitations == []

    lim1 = Limitation(
        code="lim_1",
        scope="question_generation",
        message="Message 1",
        affected_dimensions=["market_and_business_model"],
    )
    bundle.add_limitation(lim1)
    assert len(bundle.limitations) == 1
    assert bundle.limitations[0].code == "lim_1"

    lim2 = Limitation(
        code="lim_2",
        scope="question_generation",
        message="Message 2",
        affected_dimensions=["technology_and_moat"],
    )
    bundle.add_limitations([lim2])
    assert len(bundle.limitations) == 2
    assert [lim.code for lim in bundle.limitations] == ["lim_1", "lim_2"]

