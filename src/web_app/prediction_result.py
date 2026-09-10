# A common prediction format keeps the detector, API, and interface consistent.
"""Core domain types shared across application layers."""

from dataclasses import asdict, dataclass


# The dataclass generates its constructor; frozen prevents field assignment, and slots restricts attributes.
@dataclass(frozen=True, slots=True)
class Prediction:
    # The verdict is FRAUD, WARNING, or LEGIT; reason explains the decision.
    status: str
    reason: str
    # The interface uses danger/warning/success for colors; source identifies model, keywords, or hybrid analysis.
    alert_class: str
    source: str
    # The selected model class score ranges from 0 to 1; keyword-only results use None.
    confidence: float | None = None

    # Convert the fields into a dictionary for a Flask JSON response.
    def to_dict(self) -> dict:
        return asdict(self)
