"""Judge panel specifications, prompt templates, and strict validation guardrails.

Configures a 3-judge panel with specialized evaluation personas:
- Business Strategist: market sizing, unit economics, monetization, competitive dynamics.
- Technical Evaluator: architecture, proprietary moat, scalability, feasibility.
- Product Analyst: customer validation, milestones, go-to-market evidence, roadmaps.

Enforces zero-emotional-claims safety guardrails, evidence grounding, and strict
conformance to PrimaryQuestion data contracts.
"""

import re
from collections.abc import Mapping
from typing import Any

import ulid
from pydantic import BaseModel

from app.contracts import MAX_QUESTION_TEXT_LENGTH, Limitation, PrimaryQuestion
from app.prompts.judges import (
    BANNED_REGEX,
    BANNED_SUBJECTIVE_TERMS,
    BUSINESS_STRATEGIST_PROMPT,
    COMMON_GUARDRAILS,
    PRODUCT_ANALYST_PROMPT,
    TECHNICAL_EVALUATOR_PROMPT,
)
from app.stages.evidence import EvidenceBundle


EVIDENCE_ID_IN_QUESTION_TEXT_REGEX = re.compile(
    r"(?i)\s*\(?\s*(?:evidence\s*(?:id)?\s*[:#-]?\s*)?ev_[a-z0-9_-]+\s*\)?\s*[,;:]?"
)


def strip_evidence_ids_from_question_text(text: str) -> str:
    """Keep machine evidence identifiers out of the candidate-facing question."""
    cleaned = EVIDENCE_ID_IN_QUESTION_TEXT_REGEX.sub(" ", text)
    return " ".join(cleaned.split())


class JudgeSpec(BaseModel):
    """Specification for an AI judge persona in the 3-judge panel."""

    role: str
    title: str
    model: str
    preferred_key_index: int
    fallback_model: str
    rubric_dimensions: list[str]
    system_prompt: str
    default_rubric_dimension: str
    fallback_text: str
    fallback_reason: str


JUDGE_PANEL: list[JudgeSpec] = [
    JudgeSpec(
        role="business_strategist",
        title="Business Strategist",
        model="openai/gpt-oss-120b",
        preferred_key_index=0,
        fallback_model="qwen/qwen3.8-27b",
        rubric_dimensions=[
            "market_and_business_model",
            "business_reasoning",
            "customer_retention",
        ],
        default_rubric_dimension="market_and_business_model",
        fallback_text=(
            "What specific unit economics assumptions drive your projected customer acquisition "
            "cost at scale, and what evidence supports them?"
        ),
        fallback_reason="Validates financial feasibility and go-to-market model assumptions.",
        system_prompt=BUSINESS_STRATEGIST_PROMPT,
    ),
    JudgeSpec(
        role="technical_evaluator",
        title="Technical Evaluator",
        model="openai/gpt-oss-120b",
        preferred_key_index=1,
        fallback_model="qwen/qwen3.8-27b",
        rubric_dimensions=[
            "technology_and_moat",
            "technical_feasibility",
        ],
        default_rubric_dimension="technology_and_moat",
        fallback_text=(
            "How does your core technical architecture maintain defensibility against incumbent "
            "competitors and resolve anticipated scaling bottlenecks?"
        ),
        fallback_reason="Assesses technical defensibility and implementation feasibility.",
        system_prompt=TECHNICAL_EVALUATOR_PROMPT,
    ),
    JudgeSpec(
        role="product_analyst",
        title="Product Analyst",
        model="qwen/qwen3.8-27b",
        preferred_key_index=2,
        fallback_model="qwen/qwen3.6-27b",
        rubric_dimensions=[
            "execution_and_milestones",
            "pitch_content_and_evidence",
        ],
        default_rubric_dimension="execution_and_milestones",
        fallback_text=(
            "What specific milestones and customer feedback metrics must be achieved during "
            "the initial deployment to confirm product-market fit?"
        ),
        fallback_reason="Evaluates execution roadmap against early user validation.",
        system_prompt=PRODUCT_ANALYST_PROMPT,
    ),
]


def format_prompt_for_judge(judge: JudgeSpec, bundle: EvidenceBundle) -> list[dict[str, str]]:
    """Construct system and user messages for a judge LLM query."""
    evidence_text = bundle.format_summary_for_prompt()
    if not evidence_text.strip():
        evidence_text = (
            "(No detailed evidence items available. Ground on full transcript if available.)"
        )

    user_content = f"""Presentation Evidence Available:
{evidence_text}

Full Transcript:
{bundle.transcript_full_text or "(None provided)"}

Based on this evidence, produce exactly ONE primary grounded question in JSON format.
Ensure you cite at least one valid evidence_id from the evidence list above.
Put evidence IDs only in the evidence_ids JSON field. Never include an evidence ID in the
candidate-facing text field.
"""
    return [
        {"role": "system", "content": judge.system_prompt},
        {"role": "user", "content": user_content},
    ]


def check_for_subjective_emotional_claims(text: str) -> str | None:
    """Return matching banned term if text contains prohibited emotional/subjective language."""
    match = BANNED_REGEX.search(text)
    if match:
        return match.group(0)
    return None


def create_fallback_question(judge: JudgeSpec, bundle: EvidenceBundle) -> PrimaryQuestion:
    """Generate a deterministic grounded question for a judge if LLM fails."""
    # Prioritize speech or document items for question grounding
    ev_ids: list[str] = []
    for item in bundle.items:
        if item.source in ("speech", "documents") and any(
            d in judge.rubric_dimensions for d in item.rubric_dimensions
        ):
            ev_ids.append(item.evidence_id)
            break

    if not ev_ids:
        for item in bundle.items:
            if item.source in ("speech", "documents"):
                ev_ids.append(item.evidence_id)
                break

    if not ev_ids and bundle.items:
        ev_ids.append(bundle.items[0].evidence_id)

    return PrimaryQuestion(
        candidate_id=ulid.new().str,
        text=judge.fallback_text,
        reason=judge.fallback_reason,
        rubric_dimension=judge.default_rubric_dimension,
        evidence_ids=ev_ids,
    )


def _extract_question_fields(
    question: PrimaryQuestion | Mapping[str, Any],
) -> tuple[str, str, str, str, list[str]]:
    """Extract normalized candidate fields from a PrimaryQuestion or raw judge dict."""
    if isinstance(question, Mapping):
        raw_cand = question.get("candidate_id")
        cand_id = str(raw_cand).strip() if raw_cand is not None else ""
        text = str(question.get("text") or "").strip()
        reason = str(question.get("reason") or "").strip()
        rubric_dim = str(question.get("rubric_dimension") or "").strip()
        raw_eids = question.get("evidence_ids")
        evidence_ids = (
            [str(ev_id).strip() for ev_id in raw_eids if str(ev_id).strip()]
            if isinstance(raw_eids, list)
            else []
        )
        return cand_id, text, reason, rubric_dim, evidence_ids

    cand_id = question.candidate_id.strip() if question.candidate_id else ""
    return (
        cand_id,
        (question.text or "").strip(),
        (question.reason or "").strip(),
        (question.rubric_dimension or "").strip(),
        [str(ev_id).strip() for ev_id in (question.evidence_ids or []) if str(ev_id).strip()],
    )


def validate_judge_question(
    question: PrimaryQuestion | Mapping[str, Any],
    judge: JudgeSpec,
    bundle: EvidenceBundle,
) -> tuple[PrimaryQuestion | None, list[Limitation]]:
    """Validate a judge question against grounding, rubric, and guardrails rules.

    Returns:
        (validated_question, limitations)
        If validation fails, validated_question is None with explanatory limitation.
    """
    limitations: list[Limitation] = []
    cand_id, raw_text, raw_reason, raw_rubric_dim, raw_evidence_ids = _extract_question_fields(
        question
    )

    # 1. Text non-empty and within contract length. Evidence identifiers are metadata,
    # not words for the candidate to see or hear.
    cleaned_question_text = strip_evidence_ids_from_question_text(raw_text)
    if not cleaned_question_text:
        limitations.append(
            Limitation(
                code="empty_question_text",
                scope="question_generation",
                message=f"Judge {judge.role} generated an empty question text.",
                affected_dimensions=[judge.default_rubric_dimension],
            )
        )
        return None, limitations

    if len(cleaned_question_text) > MAX_QUESTION_TEXT_LENGTH:
        limitations.append(
            Limitation(
                code="over_length_question_rejected",
                scope="question_generation",
                message=(
                    f"Judge {judge.role} generated a question exceeding "
                    f"{MAX_QUESTION_TEXT_LENGTH} characters."
                ),
                affected_dimensions=[raw_rubric_dim or judge.default_rubric_dimension],
            )
        )
        return None, limitations

    if not raw_reason:
        limitations.append(
            Limitation(
                code="empty_question_reason",
                scope="question_generation",
                message=f"Judge {judge.role} generated a question without a reason.",
                affected_dimensions=[judge.default_rubric_dimension],
            )
        )
        return None, limitations

    # 2. Check for subjective/emotional claims
    combined_text = f"{raw_text} {raw_reason}"
    banned_match = check_for_subjective_emotional_claims(combined_text)
    if banned_match:
        limitations.append(
            Limitation(
                code="subjective_emotional_claim_rejected",
                scope="safety_guardrails",
                message=(
                    f"Question from judge {judge.role} rejected for prohibited "
                    f"subjective/emotional claim: '{banned_match}'."
                ),
                affected_dimensions=[raw_rubric_dim or judge.default_rubric_dimension],
            )
        )
        return None, limitations

    # 3. Rubric dimension validation
    if raw_rubric_dim not in judge.rubric_dimensions:
        # Try normalizing or defaulting if closely aligned
        if raw_rubric_dim:
            # Check if valid dimension in another persona
            all_valid_dimensions = {d for j in JUDGE_PANEL for d in j.rubric_dimensions}
            if raw_rubric_dim in all_valid_dimensions:
                dim_to_use = raw_rubric_dim
            else:
                dim_to_use = judge.default_rubric_dimension
        else:
            dim_to_use = judge.default_rubric_dimension
    else:
        dim_to_use = raw_rubric_dim

    # 4. Evidence IDs validation
    valid_evidence_ids: list[str] = []
    for ev_id in raw_evidence_ids:
        if bundle.has_evidence_id(ev_id):
            valid_evidence_ids.append(ev_id)

    if not valid_evidence_ids:
        limitations.append(
            Limitation(
                code="unsupported_question_rejected",
                scope="question_grounding",
                message=(
                    f"Judge {judge.role} question cited unknown or unsupported "
                    f"evidence IDs {raw_evidence_ids}."
                ),
                affected_dimensions=[dim_to_use],
            )
        )
        return None, limitations

    # 5. Ensure non-empty candidate_id
    final_cand_id = cand_id if cand_id else ulid.new().str

    validated = PrimaryQuestion(
        candidate_id=final_cand_id,
        text=cleaned_question_text,
        reason=raw_reason,
        rubric_dimension=dim_to_use,
        evidence_ids=valid_evidence_ids,
    )
    return validated, limitations


def validate_panel_questions(
    questions: list[PrimaryQuestion],
    bundle: EvidenceBundle,
    judges: list[JudgeSpec] | None = None,
) -> tuple[list[PrimaryQuestion], list[Limitation]]:
    """Validate full panel output: exactly 3 non-duplicate grounded questions.

    Deduplicates questions and enforces candidate count invariants.
    """
    actual_bundle = bundle
    actual_judges = judges if judges is not None else list(JUDGE_PANEL)

    limitations: list[Limitation] = []
    seen_texts: set[str] = set()
    deduped: list[PrimaryQuestion] = []

    for q in questions:
        cleaned_text = strip_evidence_ids_from_question_text(q.text)
        if not cleaned_text or len(cleaned_text) > MAX_QUESTION_TEXT_LENGTH:
            is_over_length = len(cleaned_text) > MAX_QUESTION_TEXT_LENGTH
            limitations.append(
                Limitation(
                    code=(
                        "over_length_question_rejected" if is_over_length else "empty_question_text"
                    ),
                    scope="question_panel",
                    message=(
                        f"Question rejected because text exceeded "
                        f"{MAX_QUESTION_TEXT_LENGTH} characters."
                        if is_over_length
                        else "Question rejected because text was empty."
                    ),
                    affected_dimensions=[q.rubric_dimension] if q.rubric_dimension else [],
                )
            )
            continue
        normalized_text = " ".join(cleaned_text.lower().split())
        if normalized_text in seen_texts:
            limitations.append(
                Limitation(
                    code="duplicate_question_filtered",
                    scope="question_panel",
                    message=f"Duplicate question filtered: '{cleaned_text[:60]}...'",
                    affected_dimensions=[q.rubric_dimension],
                )
            )
            continue
        seen_texts.add(normalized_text)
        deduped.append(q if cleaned_text == q.text else q.model_copy(update={"text": cleaned_text}))

    # Ensure exactly 3 questions
    if len(deduped) < 3:
        # Fill missing questions from judge defaults
        for judge in actual_judges:
            if len(deduped) >= 3:
                break
            existing_dims = {q.rubric_dimension for q in deduped}
            if judge.default_rubric_dimension not in existing_dims:
                fallback_q = create_fallback_question(judge, actual_bundle)
                deduped.append(fallback_q)
                limitations.append(
                    Limitation(
                        code="missing_judge_question_filled",
                        scope="question_panel",
                        message=f"Generated fallback question for missing judge {judge.role}.",
                        affected_dimensions=[judge.default_rubric_dimension],
                    )
                )

        # If still < 3, add additional fallbacks
        while len(deduped) < 3:
            idx = len(deduped)
            judge = actual_judges[idx % len(actual_judges)]
            deduped.append(create_fallback_question(judge, actual_bundle))

    return deduped[:3], limitations


__all__ = [
    "BANNED_REGEX",
    "BANNED_SUBJECTIVE_TERMS",
    "BUSINESS_STRATEGIST_PROMPT",
    "COMMON_GUARDRAILS",
    "JUDGE_PANEL",
    "PRODUCT_ANALYST_PROMPT",
    "TECHNICAL_EVALUATOR_PROMPT",
    "JudgeSpec",
    "strip_evidence_ids_from_question_text",
    "check_for_subjective_emotional_claims",
    "create_fallback_question",
    "format_prompt_for_judge",
    "validate_judge_question",
    "validate_panel_questions",
]
