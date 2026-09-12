"""Evidence bundle model and builder for VirtuJudge AI-ML pipeline.

Aggregates all 4 stage outputs (speech transcript, slide chunks, visual observations,
and acoustic metrics) into a structured EvidenceBundle with stable evidence IDs:
- ev_speech_*: Diarized speech transcript statements.
- ev_doc_slide_*: Document / pitch deck chunks with slide provenance.
- ev_vision_*: Correlated visual observations (gaze, posture openness, movement).
- ev_audio_*: Correlated acoustic observations (speaking rate, pauses, pitch variation).

Provides `get_rag_context()` to isolate speech + document evidence for the primary
question judges, while preserving the complete multimodal registry for AI-07 (Report).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.contracts import Limitation
from app.providers.types import (
    AudioObservation,
    DocumentChunk,
    VisualObservation,
)
from app.stages.audio import AudioStageResult
from app.stages.speech import SpeechStageResult
from app.stages.vision import VisionStageResult

EvidenceSource = Literal["speech", "documents", "vision", "audio"]

DEFAULT_SPEECH_DIMENSIONS: list[str] = [
    "market_and_business_model",
    "technology_and_moat",
    "execution_and_milestones",
    "business_reasoning",
    "technical_feasibility",
    "pitch_content_and_evidence",
]

DEFAULT_DOCUMENT_DIMENSIONS: list[str] = [
    "market_and_business_model",
    "technology_and_moat",
    "execution_and_milestones",
    "pitch_content_and_evidence",
]

DEFAULT_VISION_DIMENSIONS: list[str] = [
    "visual_delivery",
    "individual_feedback",
]

DEFAULT_AUDIO_DIMENSIONS: list[str] = [
    "acoustic_prosody",
    "individual_feedback",
]


class EvidenceItem(BaseModel):
    """A discrete unit of evidence from pitch speech, slides, vision, or audio."""

    evidence_id: str
    source: EvidenceSource
    speaker_label: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    page_or_slide: int | None = None
    metric: str | None = None
    value: float | None = None
    unit: str | None = None
    title: str
    text: str | None = None
    rubric_dimensions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _populate_text_from_excerpt(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("text") and data.get("excerpt"):
            data["text"] = data.get("excerpt")
        return data

    @property
    def excerpt(self) -> str:
        """Convenience accessor for textual representation of the evidence."""
        if self.text is not None:
            return self.text
        if self.metric is not None and self.value is not None:
            unit_str = f" {self.unit}" if self.unit else ""
            return f"{self.metric}: {self.value:.2f}{unit_str}"
        return self.title


class EvidenceBundle(BaseModel):
    """Complete multimodal evidence registry and RAG context provider."""

    items: list[EvidenceItem] = Field(default_factory=list)
    transcript_full_text: str = ""
    speaker_labels: list[str] = Field(default_factory=list)
    rubric_id: str = "startup_pitch"
    has_documents: bool = False
    has_vision: bool = False
    has_audio: bool = False
    limitations: list[Limitation] = Field(default_factory=list)

    def get_items_for_source(self, source: str) -> list[EvidenceItem]:
        """Return all evidence items originating from the given modality."""
        return [item for item in self.items if item.source == source]

    def get_items_for_dimension(self, dimension: str) -> list[EvidenceItem]:
        """Return all evidence items linked to the given rubric dimension."""
        return [item for item in self.items if dimension in item.rubric_dimensions]

    def get_rag_context(self) -> tuple[str, list[EvidenceItem]]:
        """Extract spoken transcript and slide chunks specifically for question generation."""
        rag_items = [
            item for item in self.items if item.source in ("speech", "documents")
        ]
        return self.transcript_full_text, rag_items

    def has_evidence_id(self, evidence_id: str) -> bool:
        """Check whether an evidence ID exists in the bundle."""
        return any(item.evidence_id == evidence_id for item in self.items)

    def get_item(self, evidence_id: str) -> EvidenceItem | None:
        """Find an evidence item by its unique ID."""
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def format_summary_for_prompt(self, max_items_per_source: int = 25) -> str:
        """Format speech and slide RAG evidence into prompt context for Question Judges."""
        lines: list[str] = []

        # Spoken Presentation Evidence
        speech_items = self.get_items_for_source("speech")[:max_items_per_source]
        if speech_items:
            lines.append("### Spoken Presentation Evidence:")
            for item in speech_items:
                spk = f"[{item.speaker_label}] " if item.speaker_label else ""
                t_range = (
                    f" ({item.start_ms}ms-{item.end_ms}ms)"
                    if item.start_ms is not None and item.end_ms is not None
                    else ""
                )
                lines.append(f"- [{item.evidence_id}] {spk}{t_range}: \"{item.excerpt}\"")

        # Supporting Document Evidence
        doc_items = self.get_items_for_source("documents")[:max_items_per_source]
        if doc_items:
            lines.append("\n### Supporting Document & Slide Evidence:")
            for item in doc_items:
                page_info = f" (Slide {item.page_or_slide})" if item.page_or_slide else ""
                lines.append(f"- [{item.evidence_id}]{page_info}: \"{item.excerpt}\"")

        return "\n".join(lines)


def build_evidence_bundle(
    *,
    speech_result: SpeechStageResult | None = None,
    transcript: str | None = None,
    speaker_labels: list[str] | None = None,
    vision_result: VisionStageResult | None = None,
    audio_result: AudioStageResult | None = None,
    document_chunks: list[DocumentChunk] | None = None,
    visual_observations: list[VisualObservation] | None = None,
    audio_observations: list[AudioObservation] | None = None,
    rubric_id: str = "startup_pitch",
    extra_limitations: list[Limitation] | None = None,
) -> EvidenceBundle:
    """Build an EvidenceBundle combining all available presentation evidence.

    Guarantees:
    - Stable, sequential IDs:
      - ev_speech_001...
      - ev_doc_slide_01...
      - ev_vision_001...
      - ev_audio_001...
    - Clear flags: has_documents, has_vision, has_audio.
    - Complete multi-modal registry for downstream evaluation report (AI-07).
    - `get_rag_context()` extracts strictly speech + doc evidence for AI-05 question judges.
    - Merged limitations from all participating stages.
    """
    items: list[EvidenceItem] = []
    all_limitations: list[Limitation] = []
    transcript_full_text = ""
    speakers: list[str] = []

    # 1. Spoken Transcript Evidence
    if speech_result is not None:
        transcript_full_text = speech_result.transcription.full_text
        speakers = list(speech_result.speaker_labels)
        all_limitations.extend(speech_result.limitations)

        segments = speech_result.transcription.segments
        if segments:
            for idx, seg in enumerate(segments, start=1):
                text_clean = seg.text.strip()
                if not text_clean:
                    continue

                spk = None
                if speech_result.diarization and speech_result.diarization.segments:
                    best_overlap = 0
                    for d_seg in speech_result.diarization.segments:
                        overlap = max(
                            0,
                            min(seg.end_ms, d_seg.end_ms) - max(seg.start_ms, d_seg.start_ms),
                        )
                        if overlap > best_overlap:
                            best_overlap = overlap
                            spk = d_seg.speaker_label

                items.append(
                    EvidenceItem(
                        evidence_id=f"ev_speech_{idx:03d}",
                        source="speech",
                        speaker_label=spk,
                        start_ms=seg.start_ms,
                        end_ms=seg.end_ms,
                        title=f"Spoken Statement ({spk or 'Unknown'})",
                        text=text_clean,
                        rubric_dimensions=list(DEFAULT_SPEECH_DIMENSIONS),
                        metadata={"confidence": seg.confidence},
                    )
                )
        elif transcript_full_text:
            items.append(
                EvidenceItem(
                    evidence_id="ev_speech_001",
                    source="speech",
                    speaker_label=speakers[0] if speakers else None,
                    start_ms=0,
                    end_ms=None,
                    title="Spoken Presentation Transcript",
                    text=transcript_full_text,
                    rubric_dimensions=list(DEFAULT_SPEECH_DIMENSIONS),
                )
            )
    elif transcript:
        transcript_full_text = transcript.strip()
        speakers = list(speaker_labels or [])
        if transcript_full_text:
            items.append(
                EvidenceItem(
                    evidence_id="ev_speech_001",
                    source="speech",
                    speaker_label=speakers[0] if speakers else None,
                    start_ms=0,
                    end_ms=None,
                    title="Spoken Presentation Transcript",
                    text=transcript_full_text,
                    rubric_dimensions=list(DEFAULT_SPEECH_DIMENSIONS),
                )
            )

    # 2. Document & Slide Evidence
    has_documents = False
    if document_chunks:
        has_documents = True
        for idx, chunk in enumerate(document_chunks, start=1):
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            display_text = chunk_text if len(chunk_text) <= 500 else f"{chunk_text[:497]}..."

            items.append(
                EvidenceItem(
                    evidence_id=f"ev_doc_slide_{idx:02d}",
                    source="documents",
                    page_or_slide=chunk.page_or_slide,
                    title=f"Slide {chunk.page_or_slide}",
                    text=display_text,
                    rubric_dimensions=list(DEFAULT_DOCUMENT_DIMENSIONS),
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "extraction_method": chunk.extraction_method,
                    },
                )
            )

    # 3. Visual Observations Registry (preserved for AI-07)
    v_obs_list = visual_observations
    if v_obs_list is None and vision_result is not None:
        v_obs_list = vision_result.observations

    if vision_result is not None:
        all_limitations.extend(vision_result.limitations)

    has_vision = False
    if v_obs_list:
        has_vision = True
        for idx, obs in enumerate(v_obs_list, start=1):
            items.append(
                EvidenceItem(
                    evidence_id=f"ev_vision_{idx:03d}",
                    source="vision",
                    speaker_label=obs.speaker_label,
                    start_ms=obs.start_ms,
                    end_ms=obs.end_ms,
                    metric=obs.metric,
                    value=obs.value,
                    unit=obs.unit,
                    title=f"Visual Observation: {obs.metric}",
                    text=f"{obs.metric}: {obs.value:.2f} {obs.unit}",
                    rubric_dimensions=list(DEFAULT_VISION_DIMENSIONS),
                    metadata={"confidence": obs.confidence},
                )
            )

    # 4. Acoustic Observations Registry (preserved for AI-07)
    a_obs_list = audio_observations
    if a_obs_list is None and audio_result is not None:
        a_obs_list = audio_result.observations

    if audio_result is not None:
        all_limitations.extend(audio_result.limitations)

    has_audio = False
    if a_obs_list:
        has_audio = True
        for idx, obs in enumerate(a_obs_list, start=1):
            items.append(
                EvidenceItem(
                    evidence_id=f"ev_audio_{idx:03d}",
                    source="audio",
                    speaker_label=obs.speaker_label,
                    start_ms=obs.start_ms,
                    end_ms=obs.end_ms,
                    metric=obs.metric,
                    value=obs.value,
                    unit=obs.unit,
                    title=f"Acoustic Observation: {obs.metric}",
                    text=f"{obs.metric}: {obs.value:.2f} {obs.unit}",
                    rubric_dimensions=list(DEFAULT_AUDIO_DIMENSIONS),
                    metadata={"confidence": obs.confidence},
                )
            )

    # 5. Extra limitations
    if extra_limitations:
        all_limitations.extend(extra_limitations)

    return EvidenceBundle(
        items=items,
        transcript_full_text=transcript_full_text,
        speaker_labels=speakers,
        rubric_id=rubric_id,
        has_documents=has_documents,
        has_vision=has_vision,
        has_audio=has_audio,
        limitations=all_limitations,
    )


__all__ = [
    "DEFAULT_AUDIO_DIMENSIONS",
    "DEFAULT_DOCUMENT_DIMENSIONS",
    "DEFAULT_SPEECH_DIMENSIONS",
    "DEFAULT_VISION_DIMENSIONS",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceSource",
    "build_evidence_bundle",
]
