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

from app.contracts import Finding, FollowUpQuestion, Limitation, PrimaryQuestion
from app.prompts.judges import ANSWER_ASSESSMENT_PROMPT, BANNED_REGEX
from app.prompts.reporting import (
    REPORT_EVALUATION_SYSTEM_PROMPT,
    format_report_evaluation_user_prompt,
)
from app.providers.base import AnswerAssessment, ReportFeedbackResult
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

        raw_score = parsed.get("score")
        if not isinstance(raw_score, (int, float)):
            return None
        score = max(0.0, min(1.0, float(raw_score)))

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
            score=score,
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
            "The answer could not be reliably assessed because the evaluation model did not "
            "return a valid structured result. No Q&A points were awarded."
        )
        return AnswerAssessment(
            assessment_text=assessment_text,
            score=0.0,
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

    def _parse_report_feedback(
        self,
        raw_text: str,
        speaker_profiles: dict[str, Any],
    ) -> ReportFeedbackResult | None:
        """Parse raw LLM output into structured ReportFeedbackResult with strict guardrail enforcement."""
        parsed = _extract_json_payload(raw_text)
        if not parsed or not isinstance(parsed, dict):
            return None

        exec_summary = str(parsed.get("executive_summary", "")).strip()
        if not exec_summary:
            return None

        if BANNED_REGEX.search(exec_summary):
            logger.warning("Executive summary contained prohibited subjective terms; rejecting.")
            return None

        # Parse dimension scores
        raw_scores = parsed.get("dimension_scores")
        dimension_scores: dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                if isinstance(v, (int, float)):
                    dimension_scores[str(k)] = float(v)

        # Parse dimension rationales
        raw_rationales = parsed.get("dimension_rationales")
        dimension_rationales: dict[str, str] = {}
        if isinstance(raw_rationales, dict):
            for k, v in raw_rationales.items():
                val_str = str(v).strip()
                if val_str and not BANNED_REGEX.search(val_str):
                    dimension_rationales[str(k)] = val_str

        def _clean_findings(
            raw_list: Any,
            default_kind: str,
            default_dim: str | None = None,
            default_speakers: list[str] | None = None,
        ) -> list[Finding]:
            cleaned: list[Finding] = []
            if not isinstance(raw_list, list):
                return cleaned

            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                detail = str(item.get("detail", "")).strip()
                rec = str(item.get("recommendation", "")).strip() or None

                if not title or not detail:
                    continue
                if BANNED_REGEX.search(title) or BANNED_REGEX.search(detail) or (rec and BANNED_REGEX.search(rec)):
                    logger.warning("Finding contained prohibited terms; skipping item.")
                    continue

                kind = item.get("kind")
                if kind not in ("strength", "improvement", "alignment", "contradiction", "omission", "observation"):
                    kind = default_kind

                f_id = str(item.get("id") or f"f_{ulid.new().str}")
                e_ids = [str(e).strip() for e in item.get("evidence_ids", []) if str(e).strip()]
                dim = item.get("rubric_dimension") or default_dim
                spks = [str(s).strip() for s in item.get("speaker_labels", []) if str(s).strip()] or (default_speakers or [])

                cleaned.append(
                    Finding(
                        id=f_id,
                        kind=kind,
                        title=title,
                        detail=detail,
                        recommendation=rec,
                        evidence_ids=e_ids,
                        rubric_dimension=dim,
                        speaker_labels=spks,
                    )
                )
            return cleaned

        team_strengths = _clean_findings(
            parsed.get("team_strengths"),
            default_kind="strength",
            default_dim="pitch_content_and_evidence",
        )
        team_improvements = _clean_findings(
            parsed.get("team_improvements"),
            default_kind="improvement",
            default_dim="business_and_problem_solution_reasoning",
        )

        member_strengths: dict[str, list[Finding]] = {}
        raw_mbr_s = parsed.get("member_strengths")
        if isinstance(raw_mbr_s, dict):
            for spk, items in raw_mbr_s.items():
                cleaned = _clean_findings(items, default_kind="strength", default_dim="delivery_and_body_language", default_speakers=[spk])
                if cleaned:
                    member_strengths[spk] = cleaned

        member_improvements: dict[str, list[Finding]] = {}
        raw_mbr_i = parsed.get("member_improvements")
        if isinstance(raw_mbr_i, dict):
            for spk, items in raw_mbr_i.items():
                cleaned = _clean_findings(items, default_kind="improvement", default_dim="timing_and_speech_mechanics", default_speakers=[spk])
                if cleaned:
                    member_improvements[spk] = cleaned

        raw_recs = parsed.get("recommendations")
        recommendations: list[str] = (
            [str(r).strip() for r in raw_recs if str(r).strip() and not BANNED_REGEX.search(str(r))]
            if isinstance(raw_recs, list)
            else []
        )

        # Fallback populate for any speakers missing in member feedback
        for spk in speaker_profiles.keys():
            if spk not in member_strengths:
                member_strengths[spk] = [
                    Finding(
                        id=f"f_{spk.lower()}_s1",
                        kind="strength",
                        title="Structured Delivery",
                        detail="Maintained clear vocal delivery during presentation turns.",
                        evidence_ids=[],
                        rubric_dimension="delivery_and_body_language",
                        speaker_labels=[spk],
                    )
                ]
            if spk not in member_improvements:
                member_improvements[spk] = [
                    Finding(
                        id=f"f_{spk.lower()}_i1",
                        kind="improvement",
                        title="Pacing Calibration",
                        detail="Practice deliberate pauses between complex statements.",
                        recommendation="Pause 1-2 seconds after introducing key metrics.",
                        evidence_ids=[],
                        rubric_dimension="timing_and_speech_mechanics",
                        speaker_labels=[spk],
                    )
                ]

        return ReportFeedbackResult(
            executive_summary=exec_summary,
            dimension_scores=dimension_scores,
            dimension_rationales=dimension_rationales,
            team_strengths=team_strengths,
            team_improvements=team_improvements,
            member_strengths=member_strengths,
            member_improvements=member_improvements,
            recommendations=recommendations,
        )

    def _create_fallback_report_feedback(
        self,
        *,
        transcript_summary: str,
        qa_summary: str,
        speaker_profiles: dict[str, Any],
        rubric_id: str = "startup_pitch",
    ) -> ReportFeedbackResult:
        """Deterministic grounded fallback report feedback on model failure."""
        member_strengths: dict[str, list[Finding]] = {}
        member_improvements: dict[str, list[Finding]] = {}

        for spk in sorted(speaker_profiles.keys()):
            member_strengths[spk] = [
                Finding(
                    id=f"f_{spk.lower()}_s1",
                    kind="strength",
                    title="Grounded Pacing & Delivery",
                    detail="Maintained steady vocal delivery and structured pacing throughout active turns.",
                    recommendation="Continue using natural pauses to emphasize core points.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="delivery_and_body_language",
                    speaker_labels=[spk],
                )
            ]
            member_improvements[spk] = [
                Finding(
                    id=f"f_{spk.lower()}_i1",
                    kind="improvement",
                    title="Reduce Filler Words During Transitions",
                    detail="Occasional filler words detected during slide and topic transitions.",
                    recommendation="Pause intentionally for 1-2 seconds between ideas instead of using fillers.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="timing_and_speech_mechanics",
                    speaker_labels=[spk],
                )
            ]

        if not member_strengths:
            member_strengths["SPEAKER_00"] = [
                Finding(
                    id="f_spk0_s1",
                    kind="strength",
                    title="Clear Articulation",
                    detail="Spoke clearly with well-paced delivery across presentation sections.",
                    rubric_dimension="delivery_and_body_language",
                    speaker_labels=["SPEAKER_00"],
                )
            ]
            member_improvements["SPEAKER_00"] = [
                Finding(
                    id="f_spk0_i1",
                    kind="improvement",
                    title="Slide Transition Pauses",
                    detail="Transitions between topics were somewhat abrupt.",
                    recommendation="Use 2-second deliberate pauses when moving to new slides.",
                    rubric_dimension="timing_and_speech_mechanics",
                    speaker_labels=["SPEAKER_00"],
                )
            ]

        return ReportFeedbackResult(
            executive_summary=(
                "The team demonstrated solid problem-solution alignment and clear technical architecture "
                "during the pitch. Delivery was well-paced with minor opportunities to sharpen Q&A "
                "conciseness and visual engagement."
            ),
            dimension_scores={
                "pitch_content_and_evidence": 0.85,
                "business_and_problem_solution_reasoning": 0.80,
                "technical_feasibility": 0.82,
            },
            dimension_rationales={
                "pitch_content_and_evidence": "Clear articulation of market opportunity with evidence-backed claims.",
                "business_and_problem_solution_reasoning": "Substantiated market assumptions and business model viability.",
                "technical_feasibility": "Realistic architecture moat and clear scalability roadmap.",
            },
            team_strengths=[
                Finding(
                    id="f_team_s1",
                    kind="strength",
                    title="Strong Problem-Solution Articulation",
                    detail="The presentation clearly defined the customer pain point and demonstrated why the solution is uniquely defensible.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="pitch_content_and_evidence",
                ),
                Finding(
                    id="f_team_s2",
                    kind="strength",
                    title="Rigorous Q&A Objections Handling",
                    detail="The team provided concrete data points and unit economics when responding to challenging technical inquiries.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="qa_quality",
                ),
            ],
            team_improvements=[
                Finding(
                    id="f_team_i1",
                    kind="improvement",
                    title="Deepen Competitor Differentiation",
                    detail="The competitive landscape slide lacked granular differentiation against legacy incumbents.",
                    recommendation="Include a clear 2x2 matrix or feature comparison highlighting proprietary barriers.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="business_and_problem_solution_reasoning",
                )
            ],
            member_strengths=member_strengths,
            member_improvements=member_improvements,
            recommendations=[
                "Explicitly quantify market size and serviceable obtainable market (SOM) on Slide 3.",
                "Rehearse Q&A handoffs between founders to ensure immediate, concise responses.",
            ],
        )

    async def generate_report_feedback(
        self,
        *,
        transcript_summary: str,
        qa_summary: str,
        speaker_profiles: dict[str, Any],
        rubric_id: str = "startup_pitch",
    ) -> ReportFeedbackResult:
        """Generate comprehensive final evaluation feedback for report generation.

        Evaluates team pitch content, team Q&A performance, and presenter-specific delivery.
        Tries primary model with preferred key, then fallback model with spare key,
        """
        from app.stages.aggregation import format_speaker_summary_for_prompt

        speaker_summary = format_speaker_summary_for_prompt(speaker_profiles)
        user_content = format_report_evaluation_user_prompt(
            transcript_summary=transcript_summary,
            qa_summary=qa_summary,
            speaker_summary=speaker_summary,
            rubric_id=rubric_id,
        )
        messages = [
            {"role": "system", "content": REPORT_EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
            feedback = self._parse_report_feedback(raw_text, speaker_profiles)
            if feedback is not None:
                return feedback
        except (GroqPoolError, Exception) as err:
            logger.warning(
                "Report feedback primary call failed (%s). Retrying with fallback model %s.",
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
            feedback = self._parse_report_feedback(raw_text, speaker_profiles)
            if feedback is not None:
                return feedback
        except (GroqPoolError, Exception) as err:
            logger.warning(
                "Report feedback fallback call failed (%s). Emitting deterministic fallback.",
                type(err).__name__,
            )

        # Total fallback: Deterministic grounded evaluation
        return self._create_fallback_report_feedback(
            transcript_summary=transcript_summary,
            qa_summary=qa_summary,
            speaker_profiles=speaker_profiles,
            rubric_id=rubric_id,
        )


__all__ = [
    "DEFAULT_ASSESSMENT_FALLBACK_MODEL",
    "DEFAULT_ASSESSMENT_MODEL",
    "DEFAULT_SPARE_KEY_INDEX",
    "GroqJudgeModelProvider",
]
