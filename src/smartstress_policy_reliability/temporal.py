"""Per-session k-of-m, hysteresis, and notification cooldown policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

from .contracts import ReasonCode, ReliabilityState, TemporalResult


@dataclass(frozen=True)
class TemporalPolicyConfig:
    window_size: int = 5
    required_elevated: int = 3
    threshold_on: float = 0.65
    threshold_off: float = 0.45
    cooldown_seconds: int = 900

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= self.required_elevated <= self.window_size:
            raise ValueError("required_elevated must be within the temporal window")
        if not 0.0 <= self.threshold_off < self.threshold_on <= 1.0:
            raise ValueError("require 0 <= threshold_off < threshold_on <= 1")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "required_elevated": self.required_elevated,
            "threshold_on": self.threshold_on,
            "threshold_off": self.threshold_off,
            "cooldown_seconds": self.cooldown_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalPolicyConfig":
        return cls(
            window_size=int(payload["window_size"]),
            required_elevated=int(payload["required_elevated"]),
            threshold_on=float(payload["threshold_on"]),
            threshold_off=float(payload["threshold_off"]),
            cooldown_seconds=int(payload["cooldown_seconds"]),
        )


@dataclass
class _SessionTemporalState:
    elevated_history: deque[bool]
    active: bool = False
    last_notification: datetime | None = None
    last_timestamp: datetime | None = None


@dataclass
class TemporalPolicy:
    config: TemporalPolicyConfig
    _sessions: dict[str, _SessionTemporalState] = field(default_factory=dict, init=False)

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    def evaluate(
        self,
        *,
        session_id: str,
        probability: float,
        timestamp: datetime,
    ) -> TemporalResult:
        state = self._sessions.get(session_id)
        if state is None:
            state = _SessionTemporalState(
                elevated_history=deque(maxlen=self.config.window_size)
            )
            self._sessions[session_id] = state
        if state.last_timestamp is not None and timestamp < state.last_timestamp:
            raise ValueError("timestamps must be non-decreasing within a session")
        state.last_timestamp = timestamp

        is_elevated = probability >= self.config.threshold_on
        state.elevated_history.append(is_elevated)
        elevated_count = sum(state.elevated_history)
        reasons: list[str] = []

        if state.active and probability <= self.config.threshold_off:
            state.active = False
            reasons.append(ReasonCode.HYSTERESIS_RELEASED.value)

        if not state.active and probability >= self.config.threshold_on:
            if elevated_count >= self.config.required_elevated:
                state.active = True
                reasons.append(ReasonCode.HYSTERESIS_ACTIVE.value)
            else:
                reasons.append(ReasonCode.HYSTERESIS_PENDING.value)

        proactive = False
        if state.active:
            cooldown = timedelta(seconds=self.config.cooldown_seconds)
            if state.last_notification is None or timestamp - state.last_notification >= cooldown:
                proactive = True
                state.last_notification = timestamp
            else:
                reasons.append(ReasonCode.COOLDOWN_ACTIVE.value)
            output_state = ReliabilityState.RELIABLE_ELEVATED
        elif probability > self.config.threshold_off:
            output_state = ReliabilityState.MONITOR
            if not reasons:
                reasons.append(ReasonCode.HYSTERESIS_PENDING.value)
        else:
            output_state = ReliabilityState.RELIABLE_LOW
            if not reasons:
                reasons.append(ReasonCode.CALIBRATED_LOW.value)

        return TemporalResult(
            state=output_state,
            active=state.active,
            proactive_notification=proactive,
            elevated_count=elevated_count,
            window_count=len(state.elevated_history),
            reasons=tuple(reasons),
        )
