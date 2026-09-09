"""Public contracts for the post-DNN reliability policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Mapping, Sequence


FEATURE_NAMES = (
    "mean_hr",
    "std_hr",
    "tinn",
    "hrv_index",
    "nn50",
    "pnn50",
    "mean_hrv",
    "std_hrv",
    "rmssd",
    "fft_mean",
    "fft_std",
    "sum_psd",
)


class ReliabilityState(str, Enum):
    """User-facing state emitted by the policy."""

    DATA_INVALID = "DATA_INVALID"
    OOD_OR_UNCERTAIN = "OOD_OR_UNCERTAIN"
    MONITOR = "MONITOR"
    RELIABLE_LOW = "RELIABLE_LOW"
    RELIABLE_ELEVATED = "RELIABLE_ELEVATED"


class Action(str, Enum):
    """Actions that downstream orchestration is allowed to expose."""

    MONITOR = "monitor"
    RETRY_SENSOR = "retry_sensor"
    CONTINUE_BY_TEXT = "continue_by_text"
    ASK_USER = "ask_user"
    SUPPORT = "support"
    PROPOSE_DRY_RUN = "propose_dry_run"


class ReasonCode(str, Enum):
    INPUT_OK = "INPUT_OK"
    FEATURE_COUNT_MISMATCH = "FEATURE_COUNT_MISMATCH"
    FEATURE_ORDER_MISMATCH = "FEATURE_ORDER_MISMATCH"
    NONFINITE_FEATURE = "NONFINITE_FEATURE"
    PROBABILITY_INVALID = "PROBABILITY_INVALID"
    TIMESTAMP_INVALID = "TIMESTAMP_INVALID"
    TIMESTAMP_OUT_OF_ORDER = "TIMESTAMP_OUT_OF_ORDER"
    MODEL_ID_MISMATCH = "MODEL_ID_MISMATCH"
    BASELINE_VERSION_REQUIRED = "BASELINE_VERSION_REQUIRED"
    BASELINE_VERSION_MISMATCH = "BASELINE_VERSION_MISMATCH"
    SIGNAL_WINDOW_TOO_SHORT = "SIGNAL_WINDOW_TOO_SHORT"
    SAMPLING_RATE_INVALID = "SAMPLING_RATE_INVALID"
    RPEAK_QUALITY_LOW = "RPEAK_QUALITY_LOW"
    ABNORMAL_RR_HIGH = "ABNORMAL_RR_HIGH"
    VALID_SEGMENT_LOW = "VALID_SEGMENT_LOW"
    FLATLINE_HIGH = "FLATLINE_HIGH"
    SATURATION_HIGH = "SATURATION_HIGH"
    SIGNAL_QUALITY_INVALID = "SIGNAL_QUALITY_INVALID"
    EXTERNAL_QUALITY_LOW = "EXTERNAL_QUALITY_LOW"
    FEATURE_OUTLIER_EXTREME = "FEATURE_OUTLIER_EXTREME"
    QUALITY_OK = "QUALITY_OK"
    OOD_HIGH = "OOD_HIGH"
    IN_DOMAIN = "IN_DOMAIN"
    CONFORMAL_AMBIGUOUS = "CONFORMAL_AMBIGUOUS"
    CONFIDENCE_LOW = "CONFIDENCE_LOW"
    SELECTIVE_SINGLETON = "SELECTIVE_SINGLETON"
    HYSTERESIS_PENDING = "HYSTERESIS_PENDING"
    HYSTERESIS_ACTIVE = "HYSTERESIS_ACTIVE"
    HYSTERESIS_RELEASED = "HYSTERESIS_RELEASED"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    CALIBRATED_LOW = "CALIBRATED_LOW"
    CALIBRATED_HIGH = "CALIBRATED_HIGH"


def _parse_timestamp(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class InferenceInput:
    """One frozen-DNN result plus the model-ready feature vector."""

    features: Sequence[float]
    feature_names: Sequence[str] = FEATURE_NAMES
    raw_probability: float | None = None
    raw_logit: float | None = None
    timestamp: str | datetime | None = None
    model_id: str = "wesad_attention_v1"
    session_id: str = "default"
    external_quality_score: float | None = None
    input_source: str | None = None
    baseline_version: str | None = None
    signal_quality: Mapping[str, float] = field(default_factory=dict)
    top_drivers: Sequence[Mapping[str, Any] | str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def probability(self) -> float:
        if (self.raw_probability is None) == (self.raw_logit is None):
            raise ValueError("Provide exactly one of raw_probability or raw_logit")
        if self.raw_probability is not None:
            probability = float(self.raw_probability)
        else:
            logit = float(self.raw_logit)
            if not math.isfinite(logit):
                raise ValueError("raw_logit must be finite")
            if logit >= 0:
                probability = 1.0 / (1.0 + math.exp(-logit))
            else:
                exp_value = math.exp(logit)
                probability = exp_value / (1.0 + exp_value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("raw_probability must be finite and within [0, 1]")
        return probability

    def utc_timestamp(self) -> datetime:
        return _parse_timestamp(self.timestamp)


@dataclass(frozen=True)
class QualityResult:
    valid: bool
    score: float
    max_robust_z: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OODResult:
    is_ood: bool
    score: float | None
    threshold: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectiveResult:
    prediction_set: tuple[str, ...]
    confidence: float
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TemporalResult:
    state: ReliabilityState
    active: bool
    proactive_notification: bool
    elevated_count: int
    window_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReliabilityDecision:
    """Versioned, JSON-safe result consumed by the orchestrator and UI."""

    raw_probability: float | None
    calibrated_probability: float | None
    data_quality_score: float
    ood_score: float | None
    ood_threshold: float | None
    prediction_set: tuple[str, ...]
    reliability_state: ReliabilityState
    allowed_actions: tuple[Action, ...]
    reason_codes: tuple[str, ...]
    policy_version: str
    model_id: str
    timestamp: str
    session_id: str
    proactive_notification: bool = False
    input_source: str | None = None
    baseline_version: str | None = None
    top_drivers: tuple[Mapping[str, Any] | str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reliability_state"] = self.reliability_state.value
        payload["allowed_actions"] = [action.value for action in self.allowed_actions]
        payload["prediction_set"] = list(self.prediction_set)
        payload["reason_codes"] = list(self.reason_codes)
        payload["top_drivers"] = list(self.top_drivers)
        payload["diagnostics"] = dict(self.diagnostics)
        return payload

    def audit_event(self) -> dict[str, Any]:
        """Return one append-only audit event without duplicating raw features."""
        return {
            "timestamp": self.timestamp,
            "node": "physio_reliability",
            "summary": f"Physio reliability decision: {self.reliability_state.value}",
            "details": {
                "session_id": self.session_id,
                "model_id": self.model_id,
                "policy_version": self.policy_version,
                "reliability_state": self.reliability_state.value,
                "reason_codes": list(self.reason_codes),
                "allowed_actions": [action.value for action in self.allowed_actions],
                "data_quality_score": self.data_quality_score,
                "ood_score": self.ood_score,
                "prediction_set": list(self.prediction_set),
                "proactive_notification": self.proactive_notification,
                "input_source": self.input_source,
                "baseline_version": self.baseline_version,
            },
        }
