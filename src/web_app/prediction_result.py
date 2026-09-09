# Har prediction ka ek common format, taaki detector, API aur UI same fields samjhein.
"""Core domain types shared across application layers."""

from dataclasses import asdict, dataclass


# dataclass constructor khud banata hai; frozen fields ko assign karne se rokta hai, slots fixed attributes rakhta hai.
@dataclass(frozen=True, slots=True)
class Prediction:
    # Verdict: FRAUD, WARNING ya LEGIT; reason mein us verdict ka explanation hota hai.
    status: str
    reason: str
    # UI color ke liye danger/warning/success; source model/keywords/hybrid batata hai.
    alert_class: str
    source: str
    # Model ka selected-class score 0 se 1 tak; keyword-only result mein None hota hai.
    confidence: float | None = None

    # Fields ko dictionary bana do, jise Flask JSON response mein bhej sake.
    def to_dict(self) -> dict:
        return asdict(self)
