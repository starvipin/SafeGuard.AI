"""Core domain types shared across application layers."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    status: str
    reason: str
    alert_class: str
    source: str
    confidence: float | None = None

    def as_tuple(self) -> tuple[str, str, str]:
        return self.status, self.reason, self.alert_class

    def to_dict(self) -> dict:
        return asdict(self)
