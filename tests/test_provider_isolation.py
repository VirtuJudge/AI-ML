"""Tests verifying strict isolation between provider-internal types and boundary contracts."""

from typing import get_args, get_origin

from pydantic import BaseModel

import app.providers.types as provider_types
from app.contracts import (
    AnswerAnalysisCompleted,
    ErasureCompleted,
    ReportCompleted,
    SessionAnalysisCompleted,
)


def _extract_all_types(annotation: type) -> set[type]:
    """Recursively extract all underlying types from a type annotation."""
    origin = get_origin(annotation)
    if origin is None:
        return {annotation} if isinstance(annotation, type) else set()

    types = set()
    for arg in get_args(annotation):
        types.update(_extract_all_types(arg))
    return types


def _collect_nested_model_types(
    model_cls: type[BaseModel], visited: set[type[BaseModel]] | None = None
) -> set[type]:
    """Collect all types present across all fields of a Pydantic model recursively."""
    if visited is None:
        visited = set()
    if model_cls in visited:
        return set()
    visited.add(model_cls)

    collected: set[type] = {model_cls}
    for field_info in model_cls.model_fields.values():
        if field_info.annotation is not None:
            field_types = _extract_all_types(field_info.annotation)
            collected.update(field_types)
            for t in field_types:
                if isinstance(t, type) and issubclass(t, BaseModel) and t not in visited:
                    collected.update(_collect_nested_model_types(t, visited))
    return collected


def test_no_provider_types_in_boundary_contracts() -> None:
    """Verify boundary models do not leak any provider-internal domain types."""
    provider_type_classes = {
        cls
        for name, cls in vars(provider_types).items()
        if isinstance(cls, type) and issubclass(cls, BaseModel) and cls is not BaseModel
    }
    assert len(provider_type_classes) > 0, "Provider type classes should not be empty"

    boundary_models = [
        SessionAnalysisCompleted,
        AnswerAnalysisCompleted,
        ReportCompleted,
        ErasureCompleted,
    ]

    for boundary_model in boundary_models:
        all_referenced_types = _collect_nested_model_types(boundary_model)
        leaked = provider_type_classes.intersection(all_referenced_types)
        assert not leaked, (
            f"Boundary model {boundary_model.__name__} leaks internal provider types: "
            f"{[t.__name__ for t in leaked]}"
        )
