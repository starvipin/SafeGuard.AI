"""Core domain types shared across application layers."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    status: str
    reason: str
    alert_class: str
    source: str
    confidence: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
