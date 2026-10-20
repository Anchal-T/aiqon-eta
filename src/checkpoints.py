"""Availability rules for milestones and non-stage ETA features."""

from __future__ import annotations

from src import config

# Checkpoints are cumulative. Delivery is deliberately never visible.
VISIBLE_STAGES: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("departure",),
    2: ("departure", "hub_arrival"),
    3: ("departure", "hub_arrival", "hub_departure"),
    4: ("departure", "hub_arrival", "hub_departure", "port_arrival"),
    5: ("departure", "hub_arrival", "hub_departure", "port_arrival", "port_done"),
    6: (
        "departure",
        "hub_arrival",
        "hub_departure",
        "port_arrival",
        "port_done",
        "customs",
    ),
}

CONNECTING_CHECKPOINTS = tuple(VISIBLE_STAGES)
DIRECT_CHECKPOINTS = (0, 1, 4, 5, 6)

# Earliest checkpoint for each non-stage feature, ordered as connecting/direct.
# None means the feature does not apply to that route type.
FEATURE_CHECKPOINTS: dict[str, tuple[int | None, int | None]] = {
    "weather_leg1": (2, 4),
    "weather_leg2": (4, None),
    "congestion_hub": (2, None),
    "congestion_port": (4, 4),
    "arrival_dow": (4, 4),
    "slack_remaining": (2, None),
    "slot_margin": (4, 4),
    "cong_x_dwell": (4, 4),
    "cong_x_customs": (4, 4),
    "port_done_dow": (5, 5),
    "port_done_is_weekend": (5, 5),
    "port_done_missing": (5, 5),
    "customs_missing": (6, 6),
}


def for_shipment(is_connecting: bool) -> tuple[int, ...]:
    """Return the checkpoints that apply to a direct or connecting shipment."""
    return CONNECTING_CHECKPOINTS if is_connecting else DIRECT_CHECKPOINTS


def _validate_checkpoint(checkpoint: int, is_connecting: bool) -> None:
    if checkpoint not in VISIBLE_STAGES:
        raise ValueError(f"unknown checkpoint C{checkpoint}")
    if not is_connecting and checkpoint not in DIRECT_CHECKPOINTS:
        raise ValueError(f"checkpoint C{checkpoint} does not apply to direct shipments")


def visible_stages(checkpoint: int, is_connecting: bool) -> tuple[str, ...]:
    """Return applicable observed stages, rejecting impossible direct checkpoints."""
    _validate_checkpoint(checkpoint, is_connecting)
    applicable = config.STAGE_ORDER if is_connecting else config.DIRECT_STAGES
    return tuple(stage for stage in VISIBLE_STAGES[checkpoint] if stage in applicable)


def feature_is_visible(feature: str, checkpoint: int, is_connecting: bool) -> bool:
    """Whether a non-stage feature is available at a checkpoint."""
    _validate_checkpoint(checkpoint, is_connecting)
    try:
        first_visible = FEATURE_CHECKPOINTS[feature][0 if is_connecting else 1]
    except KeyError as error:
        raise ValueError(f"no checkpoint rule for feature {feature}") from error
    return first_visible is not None and checkpoint >= first_visible
