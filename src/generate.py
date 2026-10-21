"""Synthetic shipment generator.

Produces the two raw files the rest of the pipeline reads:

    data/raw/shipments.csv  what a real system sees (the model-facing data)
    data/raw/truth.csv      hidden causes; only evaluate.py and explain.py read it

Delay model
-----------
A forward walk over the stages.  Each stage inherits the lateness of the one
before it, absorbs a fraction of it, and adds its own::

    delay_new = delay_in * (1 - absorb) + own_delay

Absorption shows up in the timestamps: a leg that inherits lateness makes up
part of it by running faster than its nominal transit time.

The hub is the exception.  It either clears the lateness completely - the
cargo makes the connection and the hub buffer absorbs the delay - or it
misses the sailing, and the lateness resets to the wait for the next one.

The raw file is deliberately complete: no missing timestamps, no injected
missingness.  That is a separate step (missingness.py) so the generator stays
the single place where the world is defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.config import Route

# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Latents:
    """Everything the generator draws.

    Kept separate from the propagation rules so that the rules can be run on
    hand-chosen values (see ``_specimen``) instead of only on random draws.
    """

    departure_delay_h: float
    weather_leg1: float
    leg1_noise_h: float
    storm_leg1: bool
    storm_leg1_h: float
    weather_leg2: float
    leg2_noise_h: float
    storm_leg2: bool
    storm_leg2_h: float
    congestion_hub: float
    congestion_port: float
    dwell_noise_h: float
    customs_noise_h: float
    truck_delay_h: float
    hub_noise_h: float


@dataclass(frozen=True)
class Schedule:
    """Planned timestamps.  Hub stages are ``None`` for a direct shipment."""

    departure: pd.Timestamp
    hub_arrival: pd.Timestamp | None
    hub_departure: pd.Timestamp | None
    port_arrival: pd.Timestamp
    port_done: pd.Timestamp
    customs: pd.Timestamp
    delivery: pd.Timestamp
    hub_buffer_h: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _at(base: pd.Timestamp, hours: float) -> pd.Timestamp:
    """Timestamp ``hours`` after ``base``, rounded to whole seconds."""
    return (base + pd.Timedelta(hours=hours)).round("s")


def _delay_hours(actual: pd.Timestamp, planned: pd.Timestamp) -> float:
    """Delay in hours, read off the timestamps."""
    return (actual - planned).total_seconds() / 3600.0


def _weather_delay(weather_index: float) -> float:
    """Extra hours on a leg caused by the weather on it."""
    return config.WEATHER_MAX_DELAY_H * weather_index**config.WEATHER_EXPONENT


def _leg_duration(nominal_h: float, inherited_delay_h: float, absorb: float) -> float:
    """Hours the leg actually takes.

    The leg makes up part of the lateness it inherited, so it runs faster than
    its nominal transit time.  A leg may make up at most
    ``MAX_RECOVERY_FRAC`` of its nominal duration; past that the attenuation
    is capped.
    """
    recovered = absorb * max(inherited_delay_h, 0.0)
    floor = (1.0 - config.MAX_RECOVERY_FRAC) * nominal_h
    return max(nominal_h - recovered, floor)


def _storm_delay(rng: np.random.Generator) -> float:
    return float(rng.uniform(config.STORM_DELAY_MIN_H, config.STORM_DELAY_MAX_H))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def _build_schedule(
    route: Route, hub_buffer_h: float, departure: pd.Timestamp
) -> Schedule:
    """Planned timestamps for one shipment."""
    if route.hub is not None:
        hub_arrival = _at(departure, route.leg1_h)
        hub_departure = _at(hub_arrival, hub_buffer_h)
        port_arrival = _at(hub_departure, route.leg2_h)
    else:
        hub_arrival = hub_departure = None
        port_arrival = _at(departure, route.leg1_h)

    norm = config.PORT_NORMS[route.destination]
    port_done = _at(port_arrival, norm.dwell_base_h)
    customs = _at(port_done, norm.customs_base_h)
    delivery = _at(customs, config.TRUCK_BASE_H)

    return Schedule(
        departure=departure,
        hub_arrival=hub_arrival,
        hub_departure=hub_departure,
        port_arrival=port_arrival,
        port_done=port_done,
        customs=customs,
        delivery=delivery,
        hub_buffer_h=hub_buffer_h,
    )


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def _propagate_departure(
    schedule: Schedule, latents: Latents, actuals: dict[str, pd.Timestamp]
) -> None:
    actuals["departure"] = _at(schedule.departure, latents.departure_delay_h)


def _propagate_leg1(
    route: Route,
    schedule: Schedule,
    latents: Latents,
    actuals: dict[str, pd.Timestamp],
) -> None:
    delay = _delay_hours(actuals["departure"], schedule.departure)
    own = _weather_delay(latents.weather_leg1) + latents.leg1_noise_h
    if latents.storm_leg1:
        own += latents.storm_leg1_h
    duration = _leg_duration(route.leg1_h, delay, config.TRANSIT_ABSORB)
    arrival_1 = _at(actuals["departure"], duration + own)
    if route.hub is not None:
        actuals["hub_arrival"] = arrival_1
    else:
        actuals["port_arrival"] = arrival_1


def _propagate_connection(
    schedule: Schedule,
    latents: Latents,
    actuals: dict[str, pd.Timestamp],
    truth: dict[str, bool],
) -> None:
    delay = _delay_hours(actuals["hub_arrival"], schedule.hub_arrival)
    # The hub absorbs lateness if possible; otherwise the next sailing leaves
    # NEXT_SAILING_WAIT_H later.
    slack_remaining = (schedule.hub_buffer_h - config.MIN_CONNECT_H) - delay
    if slack_remaining < 0:
        truth["missed_connection"] = True
        actuals["hub_departure"] = _at(
            schedule.hub_departure, config.NEXT_SAILING_WAIT_H
        )
    else:
        extra = (
            config.HUB_HANDLING_FACTOR
            * (latents.congestion_hub - config.NORMAL_CONGESTION)
            * config.MIN_CONNECT_H
            + latents.hub_noise_h
        )
        actuals["hub_departure"] = _at(schedule.hub_departure, extra)


def _propagate_leg2(
    route: Route,
    schedule: Schedule,
    latents: Latents,
    actuals: dict[str, pd.Timestamp],
) -> None:
    delay = _delay_hours(actuals["hub_departure"], schedule.hub_departure)
    own = _weather_delay(latents.weather_leg2) + latents.leg2_noise_h
    if latents.storm_leg2:
        own += latents.storm_leg2_h
    duration = _leg_duration(route.leg2_h, delay, config.TRANSIT_ABSORB)
    actuals["port_arrival"] = _at(actuals["hub_departure"], duration + own)


def _propagate_port(
    norm: config.PortNorm,
    schedule: Schedule,
    latents: Latents,
    actuals: dict[str, pd.Timestamp],
    truth: dict[str, bool],
) -> None:
    # The schedule already allows for normal congestion, so only the
    # deviation from the port's own norm costs time.
    delay = _delay_hours(actuals["port_arrival"], schedule.port_arrival)
    truth["missed_slot"] = delay > config.SLOT_WINDOW_H
    dwell_extra = (
        config.CONGESTION_HANDLING_FACTOR
        * (latents.congestion_port - norm.base_congestion)
        * norm.dwell_base_h
        + latents.dwell_noise_h
        + (config.SLOT_MISS_PENALTY_H if truth["missed_slot"] else 0.0)
    )
    actuals["port_done"] = _at(actuals["port_arrival"], norm.dwell_base_h + dwell_extra)


def _propagate_customs(
    norm: config.PortNorm,
    latents: Latents,
    actuals: dict[str, pd.Timestamp],
) -> None:
    customs_extra = (
        config.CONGESTION_HANDLING_FACTOR
        * (latents.congestion_port - norm.base_congestion)
        * norm.customs_base_h
        + latents.customs_noise_h
    )
    actuals["customs"] = _at(actuals["port_done"], norm.customs_base_h + customs_extra)


def _propagate_delivery(
    latents: Latents, actuals: dict[str, pd.Timestamp], schedule: Schedule
) -> None:
    delay = _delay_hours(actuals["customs"], schedule.customs)
    duration = _leg_duration(config.TRUCK_BASE_H, delay, config.TRUCK_ABSORB)
    actuals["delivery"] = _at(actuals["customs"], duration + latents.truck_delay_h)


def _propagate(
    route: Route, schedule: Schedule, latents: Latents
) -> tuple[dict[str, pd.Timestamp], dict[str, bool]]:
    """Propagate actual timestamps through the stages in order."""
    norm = config.PORT_NORMS[route.destination]
    actuals: dict[str, pd.Timestamp] = {}
    truth = {
        "missed_connection": False,
        "missed_slot": False,
        "storm_leg1": latents.storm_leg1,
        "storm_leg2": latents.storm_leg2,
    }

    _propagate_departure(schedule, latents, actuals)
    _propagate_leg1(route, schedule, latents, actuals)
    if route.hub is not None:
        _propagate_connection(schedule, latents, actuals, truth)
        _propagate_leg2(route, schedule, latents, actuals)
    _propagate_port(norm, schedule, latents, actuals, truth)
    _propagate_customs(norm, latents, actuals)
    _propagate_delivery(latents, actuals, schedule)

    return actuals, truth


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_latents(rng: np.random.Generator, route: Route) -> Latents:
    """Draw the latent causes for one shipment.

    A direct shipment has no second leg, so no second-leg causes are drawn for
    it and its truth file must not claim one.
    """
    norm = config.PORT_NORMS[route.destination]
    storm_leg1 = bool(rng.random() < config.STORM_PROB)
    has_leg2 = route.hub is not None
    storm_leg2 = has_leg2 and bool(rng.random() < config.STORM_PROB)
    return Latents(
        departure_delay_h=float(
            rng.normal(config.DEPARTURE_DELAY_MEAN_H, config.DEPARTURE_DELAY_SD_H)
        ),
        weather_leg1=round(
            float(rng.beta(config.WEATHER_BETA_A, config.WEATHER_BETA_B)), 6
        ),
        leg1_noise_h=float(rng.normal(0.0, config.LEG_NOISE_SD_H)),
        storm_leg1=storm_leg1,
        storm_leg1_h=_storm_delay(rng) if storm_leg1 else 0.0,
        weather_leg2=round(
            float(rng.beta(config.WEATHER_BETA_A, config.WEATHER_BETA_B)), 6
        )
        if has_leg2
        else 0.0,
        leg2_noise_h=float(rng.normal(0.0, config.LEG_NOISE_SD_H)) if has_leg2 else 0.0,
        storm_leg2=storm_leg2,
        storm_leg2_h=_storm_delay(rng) if storm_leg2 else 0.0,
        congestion_hub=round(
            float(rng.lognormal(0.0, config.CONGESTION_LOGNORMAL_SD)), 6
        ),
        congestion_port=round(
            float(
                norm.base_congestion
                * rng.lognormal(0.0, config.CONGESTION_LOGNORMAL_SD)
            ),
            6,
        ),
        dwell_noise_h=float(rng.normal(0.0, config.DWELL_NOISE_SD_H)),
        customs_noise_h=float(rng.normal(0.0, config.CUSTOMS_NOISE_SD_H)),
        truck_delay_h=float(
            rng.normal(config.TRUCK_DELAY_MEAN_H, config.TRUCK_DELAY_SD_H)
        ),
        hub_noise_h=float(rng.normal(0.0, config.HUB_DEPARTURE_NOISE_SD_H)),
    )


# ---------------------------------------------------------------------------
# The specimen shipment
# ---------------------------------------------------------------------------


def _specimen() -> tuple[dict, dict]:
    """S000, hand-authored to reproduce the worked example in the LLD.

    The latents below are chosen so that the accumulated delays come out as
    ``3.7, 7.8, 48, 24.3, 54.1, 69.8, 58.5`` hours - the values the LLD
    quotes in sections 7 (T3) and 8.
    """
    route = config.SPECIMEN_ROUTE
    latents = Latents(
        departure_delay_h=3.7,
        weather_leg1=round(
            (5.21 / config.WEATHER_MAX_DELAY_H) ** (1 / config.WEATHER_EXPONENT), 6
        ),
        leg1_noise_h=0.0,
        storm_leg1=False,
        storm_leg1_h=0.0,
        weather_leg2=0.0,
        leg2_noise_h=-9.3,
        storm_leg2=False,
        storm_leg2_h=0.0,
        congestion_hub=1.0,
        congestion_port=3.0,
        dwell_noise_h=0.55,
        customs_noise_h=0.7,
        truck_delay_h=2.66,
        hub_noise_h=0.0,
    )
    schedule = _build_schedule(
        route,
        hub_buffer_h=config.MIN_CONNECT_H + 6.0,
        departure=pd.Timestamp(config.SPECIMEN_DEPARTURE),
    )
    actuals, truth = _propagate(route, schedule, latents)
    return _row(config.SPECIMEN_ID, route, schedule, actuals, latents), {
        "shipment_id": config.SPECIMEN_ID,
        **truth,
    }


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _row(
    shipment_id: str,
    route: Route,
    schedule: Schedule,
    actuals: dict[str, pd.Timestamp],
    latents: Latents,
) -> dict:
    """One row of ``shipments.csv``."""
    row: dict = {
        "shipment_id": shipment_id,
        "journey_id": route.journey_id,
        "origin": route.origin,
        "hub": route.hub,
        "destination": route.destination,
        "weather_leg1": latents.weather_leg1,
        "weather_leg2": latents.weather_leg2 if route.hub is not None else np.nan,
        "congestion_hub": latents.congestion_hub if route.hub is not None else np.nan,
        "congestion_port": latents.congestion_port,
        "hub_buffer_h": schedule.hub_buffer_h if route.hub is not None else np.nan,
        "arrival_dow": actuals["port_arrival"].weekday(),
    }
    for stage in config.STAGE_ORDER:
        planned = getattr(schedule, stage)
        row[f"sched_{stage}"] = planned if planned is not None else pd.NaT
        row[f"act_{stage}"] = actuals.get(stage, pd.NaT)
    return row


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate(
    settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the raw shipments and truth tables."""
    rng = np.random.default_rng(settings.seed)
    rows: list[dict] = []
    truths: list[dict] = []

    specimen_row, specimen_truth = _specimen()
    rows.append(specimen_row)
    truths.append(specimen_truth)

    start = pd.Timestamp(config.DEPARTURE_START)
    for i in range(1, settings.n_shipments):
        route = config.ROUTES[int(rng.integers(len(config.ROUTES)))]
        departure = start + pd.Timedelta(
            days=float(rng.integers(config.DEPARTURE_WINDOW_DAYS))
        )
        hub_buffer_h = (
            config.MIN_CONNECT_H
            + float(rng.uniform(config.NOMINAL_SLACK_MIN_H, config.NOMINAL_SLACK_MAX_H))
            if route.hub is not None
            else 0.0
        )
        latents = _draw_latents(rng, route)
        schedule = _build_schedule(route, hub_buffer_h, departure)
        actuals, truth = _propagate(route, schedule, latents)
        rows.append(_row(f"S{i:03d}", route, schedule, actuals, latents))
        truths.append({"shipment_id": f"S{i:03d}", **truth})

    return pd.DataFrame(rows), pd.DataFrame(truths)
