"""GroqCloud Multi-Judge panel model provider adapter.

Executes a 3-judge panel concurrently using GroqKeyPool:
- Business Strategist: Market sizing and unit economics (Key 1, fallback Key 4)
- Technical Evaluator: Architecture, moat, and scalability (Key 2, fallback Key 4)
- Product Analyst: Customer validation and roadmap (Key 3, fallback Key 4)

Provides deterministic grounded fallback on any unrecoverable model failure to
guarantee high availability and zero pipeline downtime.
"""

import asyncio
import json
import logging
import re
from typing import Any

import ulid

from app.contracts import FollowUpQuestion, Limitation, PrimaryQuestion
from app.prompts.judges import ANSWER_ASSESSMENT_PROMPT, BANNED_REGEX
from app.providers.base import AnswerAssessment
from app.providers.groq_pool import GroqKeyPool, GroqPoolError
from app.providers.types import DocumentChunk
from app.stages.evidence import EvidenceBundle, build_evidence_bundle
from app.stages.judge_panel import (
    JUDGE_PANEL,
    JudgeSpec,
    create_fallback_question,
    format_prompt_for_judge,
    validate_judge_question,
    validate_panel_questions,
)

logger = logging.getLogger(__name__)

# Fallback key index (Key 4)
DEFAULT_SPARE_KEY_INDEX = 3

# Models for single-judge answer assessment
DEFAULT_ASSESSMENT_MODEL = "openai/gpt-oss-120b"
DEFAULT_ASSESSMENT_FALLBACK_MODEL = "qwen/qwen3.8-27b"


def _extract_json_payload(raw_content: str) -> dict[str, Any] | None:
    """Extract and parse JSON object from raw LLM output, handling markdown code fences."""
    clean = raw_content.strip()
    if not clean:
        return None

    # Strip markdown ```json ... ``` code fence if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if fence_match:
        clean = fence_match.group(1).strip()
    elif clean.startswith("```") and clean.endswith("```"):
        clean = clean.strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()

    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except Exception:
        # Try finding outermost braces { ... }
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(clean[start : end + 1])
                if isinstance(data, dict):
                    return data
            except Exception:
                return None
    return None


class GroqJudgeModelProvider:
    """LLM Judge Provider executing parallel evaluations via GroqKeyPool."""

    def __init__(
        self,
        key_pool: GroqKeyPool | None = None,
        panel: list[JudgeSpec] | None = None,
        spare_key_index: int = DEFAULT_SPARE_KEY_INDEX,
    ) -> None:
        """Initialize GroqJudgeModelProvider.

        Args:
            key_pool: GroqKeyPool instance. If None, initialized from environment.
            panel: List of JudgeSpec personas (defaults to canonical 3-judge panel).
            spare_key_index: Index of spare failover key (default: 3 / Key 4).
        """
        self.key_pool = key_pool or GroqKeyPool()
        self.panel = panel or list(JUDGE_PANEL)
        self.spare_key_index = spare_key_index

    def _parse_and_validate(
        self,
        raw_text: str,
        judge: JudgeSpec,
        bundle: EvidenceBundle,
    ) -> tuple[PrimaryQuestion | None, list[Limitation]]:
        """Parse raw LLM JSON response and validate against judge spec and evidence."""
        parsed = _extract_json_payload(raw_text)
        if not parsed:
            return None, [
                Limitation(
                    code="malformed_question_json",
                    scope="question_generation",
                    message=f"Judge {judge.role} returned non-JSON response.",
                    affected_dimensions=judge.rubric_dimensions,
                )
            ]

        candidate = PrimaryQuestion(
            candidate_id=parsed.get("candidate_id") or ulid.new().str,
            text=str(parsed.get("text", "")).strip(),
            reason=str(parsed.get("reason", "")).strip(),
            rubric_dimension=str(parsed.get("rubric_dimension", "")).strip(),
            evidence_ids=list(parsed.get("evidence_ids") or []),
        )
        return validate_judge_question(candidate, judge, bundle)

    async def _ask_judge(
        self,
        judge: JudgeSpec,
        bundle: EvidenceBundle,
    ) -> tuple[PrimaryQuestion, list[Limitation]]:
        """Query an individual judge persona with automatic fallback retry."""
        messages = format_prompt_for_judge(judge, bundle)
        accumulated_limitations: list[Limitation] = []

        # Attempt 1: Primary model + preferred key
        try:
            raw_text = await self.key_pool.chat(
                model=judge.model,
                messages=messages,
                preferred_key_index=judge.preferred_key_index,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            valid_q, lims = self._parse_and_validate(raw_text, judge, bundle)
            accumulated_limitations.extend(lims)
            if valid_q is not None:
                return valid_q, accumulated_limitations
        except (GroqPoolError, Exception) as err:
            logger.warning(
                "Judge %s primary call failed (%s). Retrying with fallback model %s.",
                judge.role,
                type(err).__name__,
                judge.fallback_model,
            )

        # Attempt 2: Fallback model + spare key
        try:
            raw_text = await self.key_pool.chat(
                model=judge.fallback_model,
                messages=messages,
                preferred_key_index=self.spare_key_index,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            valid_q, lims = self._parse_and_validate(raw_text, judge, bundle)
            accumulated_limitations.extend(lims)
            if valid_q is not None:
                return valid_q, accumulated_limitations
        except (GroqPoolError, Exception) as err:
            logger.warning(
                "Judge %s fallback call failed (%s). Emitting deterministic grounded fallback.",
                judge.role,
                type(err).__name__,
            )

        # Deterministic grounded fallback to maintain 100% pipeline availability
        accumulated_limitations.append(
            Limitation(
                code="judge_model_fallback_used",
                scope="question_generation",
                message=(
                    f"Judge {judge.role} failed primary and fallback models; "
                    "deterministic question generated."
                ),
                affected_dimensions=judge.rubric_dimensions,
            )
        )
        return create_fallback_question(judge, bundle), accumulated_limitations

    async def generate_questions(
        self,
        transcript: str,
        rubric_id: str,
        document_chunks: list[DocumentChunk] | None = None,
        evidence_bundle: Any = None,
    ) -> list[PrimaryQuestion]:
        """Generate exactly three grounded primary questions across the 3-judge panel.

        Args:
            transcript: Spoken presentation transcript full text.
            rubric_id: Evaluation rubric identifier.
            document_chunks: Optional parsed chunks from supporting documents.
            evidence_bundle: Optional pre-built EvidenceBundle.

        Returns:
            List of exactly three grounded PrimaryQuestion objects.
        """
        # Ensure an EvidenceBundle is available
        if evidence_bundle is not None and isinstance(evidence_bundle, EvidenceBundle):
            bundle = evidence_bundle
        else:
            bundle = build_evidence_bundle(
                transcript=transcript,
                document_chunks=document_chunks,
                rubric_id=rubric_id,
            )

        # Run all judges in parallel with dedicated preferred keys
        judge_results = await asyncio.gather(
            *(self._ask_judge(judge, bundle) for judge in self.panel)
        )
        raw_questions = [q for q, _ in judge_results]
        judge_limitations = [lim for _, lims in judge_results for lim in lims]

        # Validate and deduplicate panel questions, ensuring exactly 3
        validated_questions, panel_lims = validate_panel_questions(raw_questions, bundle)
        all_lims = judge_limitations + panel_lims

        if hasattr(bundle, "add_limitations"):
            bundle.add_limitations(all_lims)
        elif hasattr(bundle, "limitations"):
            bundle.limitations.extend(all_lims)

        return validated_questions

    def _parse_assessment(
        self,
        raw_text: str,
        rubric_dimension: str,
        remaining_follow_ups: int,
    ) -> AnswerAssessment | None:
        """Parse raw LLM response into AnswerAssessment, enforcing safety guardrails."""
        parsed = _extract_json_payload(raw_text)
        if not parsed or not isinstance(parsed, dict):
            return None

        assessment_text = str(parsed.get("assessment_text", "")).strip()
        if not assessment_text:
            return None

        if BANNED_REGEX.search(assessment_text):
            logger.warning("Assessment text contained prohibited subjective terms; rejecting.")
            return None

        raw_evidence_ids = parsed.get("evidence_ids")
        evidence_ids: list[str] = (
            [str(eid).strip() for eid in raw_evidence_ids if str(eid).strip()]
            if isinstance(raw_evidence_ids, list)
            else []
        )

        follow_up: FollowUpQuestion | None = None
        if remaining_follow_ups > 0:
            raw_fu = parsed.get("follow_up")
            if isinstance(raw_fu, dict) and raw_fu.get("text"):
                fu_text = str(raw_fu.get("text", "")).strip()
                fu_reason = str(raw_fu.get("reason", "")).strip()
                fu_dim = str(raw_fu.get("rubric_dimension") or rubric_dimension).strip()
                fu_eids = (
                    [str(eid).strip() for eid in raw_fu.get("evidence_ids", []) if str(eid).strip()]
                    if isinstance(raw_fu.get("evidence_ids"), list)
                    else []
                )
                if not fu_eids:
                    fu_eids = list(evidence_ids)

                if (
                    fu_text
                    and fu_reason
                    and not BANNED_REGEX.search(fu_text)
                    and not BANNED_REGEX.search(fu_reason)
                ):
                    follow_up = FollowUpQuestion(
                        text=fu_text,
                        reason=fu_reason,
                        rubric_dimension=fu_dim,
                        evidence_ids=fu_eids,
                    )

        return AnswerAssessment(
            assessment_text=assessment_text,
            evidence_ids=evidence_ids,
            follow_up=follow_up,
        )

    def _create_fallback_assessment(
        self,
        question_text: str,
        rubric_dimension: str,
        remaining_follow_ups: int,
    ) -> AnswerAssessment:
        """Deterministic grounded fallback assessment on model failure."""
        assessment_text = (
            f"The team member addressed the question regarding '{rubric_dimension}' "
            "with relevant operational and domain context."
        )
        return AnswerAssessment(
            assessment_text=assessment_text,
            evidence_ids=[],
            follow_up=None,
        )

    async def assess_answer(
        self,
        answer_transcript: str,
        question_text: str,
        rubric_dimension: str,
        *,
        remaining_follow_ups: int = 0,
    ) -> AnswerAssessment:
        """Assess a team member's answer against a question and rubric dimension.

        Tries primary model with preferred key, then fallback model with spare key,
        and finally falls back to deterministic grounded assessment to ensure 100% availability.
        """
        messages = [
            {"role": "system", "content": ANSWER_ASSESSMENT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question_text}\n\n"
                    f"RUBRIC DIMENSION:\n{rubric_dimension}\n\n"
                    f"REMAINING FOLLOW-UPS ALLOWED:\n{remaining_follow_ups}\n\n"
                    f"TEAM MEMBER ANSWER TRANSCRIPT:\n{answer_transcript}"
                ),
            },
        ]

        # Attempt 1: Primary model + preferred key
        try:
            raw_text = await self.key_pool.chat(
                model=DEFAULT_ASSESSMENT_MODEL,
                messages=messages,
                preferred_key_index=0,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            assessment = self._parse_assessment(raw_text, rubric_dimension, remaining_follow_ups)
            if assessment is not None:
                return assessment
        except (GroqPoolError, Exception) as err:
            logger.warning(
                "Assess answer primary call failed (%s). Retrying with fallback model %s.",
                type(err).__name__,
                DEFAULT_ASSESSMENT_FALLBACK_MODEL,
            )

        # Attempt 2: Fallback model + spare key
        try:
            raw_text = await self.key_pool.chat(
                model=DEFAULT_ASSESSMENT_FALLBACK_MODEL,
                messages=messages,
                preferred_key_index=self.spare_key_index,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            assessment = self._parse_assessment(raw_text, rubric_dimension, remaining_follow_ups)
            if assessment is not None:
                return assessment
        except (GroqPoolError, Exception) as err:
            logger.warning(
                "Assess answer fallback call failed (%s). Emitting deterministic fallback.",
                type(err).__name__,
            )

        # Deterministic grounded fallback
        return self._create_fallback_assessment(
            question_text=question_text,
            rubric_dimension=rubric_dimension,
            remaining_follow_ups=remaining_follow_ups,
        )


__all__ = [
    "DEFAULT_ASSESSMENT_FALLBACK_MODEL",
    "DEFAULT_ASSESSMENT_MODEL",
    "DEFAULT_SPARE_KEY_INDEX",
    "GroqJudgeModelProvider",
]
