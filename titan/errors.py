"""Domain-specific exceptions for the Titan pipeline.

Wrap external/infrastructure failures in these so callers can pattern-match
on intent rather than on third-party SDK exception classes (which leak the
provider into business logic).
"""

from __future__ import annotations


class TitanError(Exception):
    """Base for all Titan-raised errors."""


class OCRFailedError(TitanError):
    """Raised when every parser in the OCR router fails for a document."""


class LowConfidenceError(TitanError):
    """Raised when extraction quality falls below an acceptable threshold."""


class ExtractionError(TitanError):
    """Raised when BAML/structured extraction cannot produce a TitleDocument."""


class RetrievalError(TitanError):
    """Raised when hybrid retrieval cannot return any candidate chunks."""


class GenerationError(TitanError):
    """Raised when LLM-driven section drafting fails after retries."""


class RuleDistillationError(TitanError):
    """Raised when LLM-as-judge rule distillation fails after retries."""


class ConfigurationError(TitanError):
    """Raised when required configuration (e.g. an API key) is missing."""


__all__ = [
    "TitanError",
    "OCRFailedError",
    "LowConfidenceError",
    "ExtractionError",
    "RetrievalError",
    "GenerationError",
    "RuleDistillationError",
    "ConfigurationError",
]
