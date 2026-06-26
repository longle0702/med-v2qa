"""
guardrail/__init__.py
----------------------
Public surface of the ``guardrail`` package.

Importing from ``guardrail`` directly gives access to the pipeline and
result types without needing to know the internal module layout.

Example
-------
::

    from guardrail import GuardrailPipeline, GuardrailResult

    pipeline = GuardrailPipeline()
    result: GuardrailResult = pipeline.run(
        image_path="xray.jpg",
        question="What abnormalities are visible?",
    )
"""

from guardrail.pipeline import GuardrailPipeline
from guardrail.result_types import (
    ConfidenceResult,
    GuardrailResult,
    IntentResult,
)

__all__ = [
    "GuardrailPipeline",
    "GuardrailResult",
    "IntentResult",
    "ConfidenceResult",
]
