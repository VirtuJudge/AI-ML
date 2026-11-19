"""Typed document extraction errors, importable without the optional parser libraries."""


class DocumentExtractionError(Exception):
    """Exception raised when document extraction fails."""


class DocumentExtractionTimeout(DocumentExtractionError):
    """The isolated parser exceeded its wall-clock deadline and was killed."""


class DocumentParserCrashed(DocumentExtractionError):
    """The isolated parser process exited without returning a result."""


class DocumentResourceLimitExceeded(DocumentExtractionError):
    """The isolated parser ran out of its memory budget."""


__all__ = [
    "DocumentExtractionError",
    "DocumentExtractionTimeout",
    "DocumentParserCrashed",
    "DocumentResourceLimitExceeded",
]
