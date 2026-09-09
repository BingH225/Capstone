"""End-to-end post-DNN reliability policy and checksummed manifest persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .calibration import (
    BinaryCalibrator,
    calibrator_from_dict,
    fit_calibrator,
    probabilities_to_logits,
)
from .contracts import (
    Action,
    FEATURE_NAMES,
    InferenceInput,
    OODResult,
    QualityResult,
    ReasonCode,
    ReliabilityDecision,
    ReliabilityState,
    SelectiveResult,
)
from .ood import RobustDiagonalOODDetector
from .quality import FeatureQualityProfile
from .selective import BinarySelectivePredictor
from .temporal import TemporalPolicy, TemporalPolicyConfig


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ReliabilityPolicyConfig:
    policy_version: str = "physio-rel-v1"
    expected_model_id: str = "wesad_attention_v1"
    calibrator: str = "temperature"
    conformal_alpha: float = 0.1
    minimum_confidence: float = 0.6
    ood_threshold_quantile: float = 0.99
    minimum_external_quality: float = 0.5
    quality_soft_z_limit: float = 6.0
    quality_hard_z_limit: float = 12.0
    require_baseline_version: bool = False
    temporal: TemporalPolicyConfig = field(default_factory=TemporalPolicyConfig)

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("policy_version cannot be empty")
        if not self.expected_model_id.strip():
            raise ValueError("expected_model_id cannot be empty")
        if self.calibrator not in {"identity", "temperature", "platt"}:
            raise ValueError("unsupported calibrator")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must be between 0 and 1")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "expected_model_id": self.expected_model_id,
            "calibrator": self.calibrator,
            "conformal_alpha": self.conformal_alpha,
            "minimum_confidence": self.minimum_confidence,
            "ood_threshold_quantile": self.ood_threshold_quantile,
            "minimum_external_quality": self.minimum_external_quality,
            "quality_soft_z_limit": self.quality_soft_z_limit,
            "quality_hard_z_limit": self.quality_hard_z_limit,
            "require_baseline_version": self.require_baseline_version,
            "temporal": self.temporal.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReliabilityPolicyConfig":
        return cls(
            policy_version=str(payload["policy_version"]),
            expected_model_id=str(payload["expected_model_id"]),
            calibrator=str(payload["calibrator"]),
            conformal_alpha=float(payload["conformal_alpha"]),
            minimum_confidence=float(payload["minimum_confidence"]),
            ood_threshold_quantile=float(payload["ood_threshold_quantile"]),
            minimum_external_quality=float(payload["minimum_external_quality"]),
            quality_soft_z_limit=float(payload["quality_soft_z_limit"]),
            quality_hard_z_limit=float(payload["quality_hard_z_limit"]),
            require_baseline_version=bool(payload.get("require_baseline_version", False)),
            temporal=TemporalPolicyConfig.from_dict(payload["temporal"]),
        )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_timestamp(value: str | datetime | None) -> datetime:
    try:
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ReliabilityPolicy:
    config: ReliabilityPolicyConfig
    calibrator: BinaryCalibrator
    quality_profile: FeatureQualityProfile
    ood_detector: RobustDiagonalOODDetector
    selective_predictor: BinarySelectivePredictor
    fitted_at: str
    data_provenance: Mapping[str, Any] = field(default_factory=dict)
    temporal_policy: TemporalPolicy = field(init=False)
    _baseline_versions: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        feature_orders = {
            tuple(self.quality_profile.feature_names),
            tuple(self.ood_detector.feature_names),
        }
        if feature_orders != {tuple(FEATURE_NAMES)}:
            raise ValueError("all policy components must use the WESAD 12-feature contract")
        self.temporal_policy = TemporalPolicy(self.config.temporal)

    @classmethod
    def fit(
        cls,
        *,
        reference_features: Iterable[Sequence[float]],
        calibration_labels: Iterable[int],
        calibration_features: Iterable[Sequence[float]],
        calibration_probabilities: Iterable[float] | None = None,
        calibration_logits: Iterable[float] | None = None,
        config: ReliabilityPolicyConfig | None = None,
        data_provenance: Mapping[str, Any] | None = None,
    ) -> "ReliabilityPolicy":
        cfg = config or ReliabilityPolicyConfig()
        reference = np.asarray(list(reference_features), dtype=np.float64)
        calibration_matrix = np.asarray(list(calibration_features), dtype=np.float64)
        labels = np.asarray(list(calibration_labels), dtype=np.int64)
        if calibration_matrix.ndim != 2 or calibration_matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError("calibration_features do not match the 12-feature contract")
        if labels.shape != (calibration_matrix.shape[0],):
            raise ValueError("calibration labels and features must be aligned")
        if (calibration_probabilities is None) == (calibration_logits is None):
            raise ValueError("provide exactly one calibration probability/logit sequence")
        if calibration_probabilities is not None:
            raw_probabilities = np.asarray(list(calibration_probabilities), dtype=np.float64)
            if not np.isfinite(raw_probabilities).all() or np.any(
                (raw_probabilities < 0.0) | (raw_probabilities > 1.0)
            ):
                raise ValueError("calibration probabilities must be finite within [0, 1]")
            logits = probabilities_to_logits(raw_probabilities)
        else:
            assert calibration_logits is not None
            logits = np.asarray(list(calibration_logits), dtype=np.float64)
        if logits.shape != labels.shape or not np.isfinite(logits).all():
            raise ValueError("calibration predictions must be finite and aligned")

        quality = FeatureQualityProfile.fit(
            reference,
            soft_z_limit=cfg.quality_soft_z_limit,
            hard_z_limit=cfg.quality_hard_z_limit,
            minimum_external_quality=cfg.minimum_external_quality,
        )
        ood = RobustDiagonalOODDetector.fit(
            reference, threshold_quantile=cfg.ood_threshold_quantile
        )
        calibrator = fit_calibrator(cfg.calibrator, logits, labels)
        calibrated = calibrator.predict_from_logits(logits)
        selective = BinarySelectivePredictor.fit(
            calibrated,
            labels,
            alpha=cfg.conformal_alpha,
            minimum_confidence=cfg.minimum_confidence,
        )
        return cls(
            config=cfg,
            calibrator=calibrator,
            quality_profile=quality,
            ood_detector=ood,
            selective_predictor=selective,
            fitted_at=_isoformat(datetime.now(timezone.utc)),
            data_provenance=dict(data_provenance or {}),
        )

    def reset_temporal_state(self, session_id: str | None = None) -> None:
        self.temporal_policy.reset(session_id)
        if session_id is None:
            self._baseline_versions.clear()
        else:
            self._baseline_versions.pop(session_id, None)

    def _invalid_decision(
        self,
        inference: InferenceInput,
        *,
        timestamp: datetime,
        reasons: Sequence[str],
        quality_score: float = 0.0,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> ReliabilityDecision:
        try:
            raw_probability = inference.probability()
        except (TypeError, ValueError, OverflowError):
            raw_probability = None
        return ReliabilityDecision(
            raw_probability=raw_probability,
            calibrated_probability=None,
            data_quality_score=quality_score,
            ood_score=None,
            ood_threshold=self.ood_detector.threshold,
            prediction_set=(),
            reliability_state=ReliabilityState.DATA_INVALID,
            allowed_actions=(Action.RETRY_SENSOR, Action.CONTINUE_BY_TEXT),
            reason_codes=tuple(dict.fromkeys(reasons)),
            policy_version=self.config.policy_version,
            model_id=inference.model_id,
            timestamp=_isoformat(timestamp),
            session_id=inference.session_id,
            input_source=inference.input_source,
            baseline_version=inference.baseline_version,
            top_drivers=tuple(inference.top_drivers),
            diagnostics=dict(diagnostics or {}),
        )

    def evaluate(self, inference: InferenceInput) -> ReliabilityDecision:
        try:
            timestamp = inference.utc_timestamp()
        except (AttributeError, TypeError, ValueError):
            timestamp = _safe_timestamp(None)
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=(ReasonCode.TIMESTAMP_INVALID.value,),
            )
        if inference.model_id != self.config.expected_model_id:
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=(ReasonCode.MODEL_ID_MISMATCH.value,),
                diagnostics={
                    "expected_model_id": self.config.expected_model_id,
                    "received_model_id": inference.model_id,
                },
            )
        try:
            feature_order_matches = tuple(inference.feature_names) == FEATURE_NAMES
        except TypeError:
            feature_order_matches = False
        if not feature_order_matches:
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=(ReasonCode.FEATURE_ORDER_MISMATCH.value,),
                diagnostics={"expected_feature_order": list(FEATURE_NAMES)},
            )
        previous_baseline = self._baseline_versions.get(inference.session_id)
        if self.config.require_baseline_version and not inference.baseline_version:
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=(ReasonCode.BASELINE_VERSION_REQUIRED.value,),
            )
        if previous_baseline is not None and inference.baseline_version != previous_baseline:
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=(ReasonCode.BASELINE_VERSION_MISMATCH.value,),
                diagnostics={"expected_baseline_version": previous_baseline},
            )

        quality: QualityResult = self.quality_profile.evaluate(
            inference.features,
            external_quality_score=inference.external_quality_score,
            signal_quality=inference.signal_quality,
        )
        if not quality.valid:
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=quality.reasons,
                quality_score=quality.score,
                diagnostics={"max_robust_z": quality.max_robust_z},
            )
        try:
            raw_probability = inference.probability()
        except (TypeError, ValueError, OverflowError) as exc:
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=(ReasonCode.PROBABILITY_INVALID.value,),
                quality_score=quality.score,
                diagnostics={"error": str(exc), "max_robust_z": quality.max_robust_z},
            )

        if inference.baseline_version:
            self._baseline_versions[inference.session_id] = inference.baseline_version
        ood: OODResult = self.ood_detector.evaluate(inference.features)
        reasons = [*quality.reasons, *ood.reasons]
        if ood.is_ood:
            return ReliabilityDecision(
                raw_probability=raw_probability,
                calibrated_probability=None,
                data_quality_score=quality.score,
                ood_score=ood.score,
                ood_threshold=ood.threshold,
                prediction_set=(),
                reliability_state=ReliabilityState.OOD_OR_UNCERTAIN,
                allowed_actions=(Action.MONITOR, Action.ASK_USER, Action.CONTINUE_BY_TEXT),
                reason_codes=tuple(dict.fromkeys(reasons)),
                policy_version=self.config.policy_version,
                model_id=inference.model_id,
                timestamp=_isoformat(timestamp),
                session_id=inference.session_id,
                input_source=inference.input_source,
                baseline_version=inference.baseline_version,
                top_drivers=tuple(inference.top_drivers),
                diagnostics={"max_robust_z": quality.max_robust_z},
            )

        calibrated_probability = float(
            self.calibrator.predict_from_probabilities([raw_probability])[0]
        )
        selective: SelectiveResult = self.selective_predictor.evaluate(calibrated_probability)
        reasons.extend(selective.reasons)
        if not selective.accepted:
            return ReliabilityDecision(
                raw_probability=raw_probability,
                calibrated_probability=calibrated_probability,
                data_quality_score=quality.score,
                ood_score=ood.score,
                ood_threshold=ood.threshold,
                prediction_set=selective.prediction_set,
                reliability_state=ReliabilityState.OOD_OR_UNCERTAIN,
                allowed_actions=(Action.MONITOR, Action.ASK_USER, Action.CONTINUE_BY_TEXT),
                reason_codes=tuple(dict.fromkeys(reasons)),
                policy_version=self.config.policy_version,
                model_id=inference.model_id,
                timestamp=_isoformat(timestamp),
                session_id=inference.session_id,
                input_source=inference.input_source,
                baseline_version=inference.baseline_version,
                top_drivers=tuple(inference.top_drivers),
                diagnostics={
                    "max_robust_z": quality.max_robust_z,
                    "selective_confidence": selective.confidence,
                },
            )

        try:
            temporal = self.temporal_policy.evaluate(
                session_id=inference.session_id,
                probability=calibrated_probability,
                timestamp=timestamp,
            )
        except ValueError as exc:
            return self._invalid_decision(
                inference,
                timestamp=timestamp,
                reasons=(ReasonCode.TIMESTAMP_OUT_OF_ORDER.value,),
                quality_score=quality.score,
                diagnostics={"error": str(exc)},
            )
        reasons.extend(temporal.reasons)
        if temporal.state == ReliabilityState.RELIABLE_ELEVATED:
            reasons.append(ReasonCode.CALIBRATED_HIGH.value)
            actions = (
                Action.MONITOR,
                Action.ASK_USER,
                Action.SUPPORT,
                Action.PROPOSE_DRY_RUN,
            )
        elif temporal.state == ReliabilityState.MONITOR:
            actions = (Action.MONITOR, Action.ASK_USER)
        else:
            reasons.append(ReasonCode.CALIBRATED_LOW.value)
            actions = (Action.MONITOR, Action.CONTINUE_BY_TEXT)
        return ReliabilityDecision(
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            data_quality_score=quality.score,
            ood_score=ood.score,
            ood_threshold=ood.threshold,
            prediction_set=selective.prediction_set,
            reliability_state=temporal.state,
            allowed_actions=actions,
            reason_codes=tuple(dict.fromkeys(reasons)),
            policy_version=self.config.policy_version,
            model_id=inference.model_id,
            timestamp=_isoformat(timestamp),
            session_id=inference.session_id,
            proactive_notification=temporal.proactive_notification,
            input_source=inference.input_source,
            baseline_version=inference.baseline_version,
            top_drivers=tuple(inference.top_drivers),
            diagnostics={
                "max_robust_z": quality.max_robust_z,
                "selective_confidence": selective.confidence,
                "temporal_elevated_count": temporal.elevated_count,
                "temporal_window_count": temporal.window_count,
                "temporal_active": temporal.active,
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "fitted_at": self.fitted_at,
            "feature_order": list(FEATURE_NAMES),
            "config": self.config.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "quality_profile": self.quality_profile.to_dict(),
            "ood_detector": self.ood_detector.to_dict(),
            "selective_predictor": self.selective_predictor.to_dict(),
            "data_provenance": dict(self.data_provenance),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        digest = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
        envelope = {"manifest": manifest, "manifest_sha256": digest}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ReliabilityPolicy":
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = envelope["manifest"]
        expected = str(envelope["manifest_sha256"])
        actual = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
        if actual != expected:
            raise RuntimeError("policy manifest checksum mismatch")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported policy manifest schema")
        if tuple(manifest.get("feature_order", ())) != FEATURE_NAMES:
            raise ValueError("policy manifest feature order mismatch")
        return cls(
            config=ReliabilityPolicyConfig.from_dict(manifest["config"]),
            calibrator=calibrator_from_dict(manifest["calibrator"]),
            quality_profile=FeatureQualityProfile.from_dict(manifest["quality_profile"]),
            ood_detector=RobustDiagonalOODDetector.from_dict(manifest["ood_detector"]),
            selective_predictor=BinarySelectivePredictor.from_dict(
                manifest["selective_predictor"]
            ),
            fitted_at=str(manifest["fitted_at"]),
            data_provenance=dict(manifest.get("data_provenance", {})),
        )
