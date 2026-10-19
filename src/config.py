"""Single source of truth for the Aiqon ETA pipeline.

Three groups of constants (LLD section 4):

* **Model-visible** - a planner would know these, so features may use them.
* **Generator-only** - hidden from the model; it must learn their effect from
  the data.
* **Run settings** - seed, sizes, thresholds.

The split matters.  If the features contained the penalty sizes (48 h wait,
18 h slot penalty) the model would simply read the answer.  Knowing that a
connection window exists is domain knowledge; knowing how big the penalty is
has to be learned.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

STAGE_ORDER = (
    "departure",
    "hub_arrival",
    "hub_departure",
    "port_arrival",
    "port_done",
    "customs",
    "delivery",
)
HUB_STAGES = ("hub_arrival", "hub_departure")
DIRECT_STAGES = ("departure", "port_arrival", "port_done", "customs", "delivery")

# ---------------------------------------------------------------------------
# Model-visible constants
# ---------------------------------------------------------------------------

#: Unload / move / load time a shipment needs at a hub.
MIN_CONNECT_H = 12.0

#: How late a vessel may arrive and still keep its berth slot.
SLOT_WINDOW_H = 12.0


@dataclass(frozen=True)
class PortNorm:
    """Normal handling times for one port."""

    dwell_base_h: float
    customs_base_h: float
    base_congestion: float


PORT_NORMS: dict[str, PortNorm] = {
    "Busan": PortNorm(dwell_base_h=16.0, customs_base_h=14.0, base_congestion=1.2),
    "Chennai": PortNorm(dwell_base_h=18.0, customs_base_h=20.0, base_congestion=1.3),
    "Durban": PortNorm(dwell_base_h=20.0, customs_base_h=22.0, base_congestion=1.3),
    "Hamburg": PortNorm(dwell_base_h=16.0, customs_base_h=18.0, base_congestion=1.2),
    "Jebel_Ali": PortNorm(dwell_base_h=14.0, customs_base_h=12.0, base_congestion=0.9),
    "Los_Angeles": PortNorm(dwell_base_h=20.0, customs_base_h=20.0, base_congestion=1.3),
    "Mumbai": PortNorm(dwell_base_h=16.0, customs_base_h=18.0, base_congestion=1.2),
    "New_York": PortNorm(dwell_base_h=22.0, customs_base_h=22.0, base_congestion=1.4),
    "Rotterdam": PortNorm(dwell_base_h=15.0, customs_base_h=20.0, base_congestion=1.5),
    "Santos": PortNorm(dwell_base_h=22.0, customs_base_h=24.0, base_congestion=1.5),
    "Shanghai": PortNorm(dwell_base_h=18.0, customs_base_h=16.0, base_congestion=1.4),
    "Singapore": PortNorm(dwell_base_h=14.0, customs_base_h=12.0, base_congestion=1.0),
}

# ---------------------------------------------------------------------------
# Generator-only constants
# ---------------------------------------------------------------------------

#: Wait for the next sailing after missing the connecting one.
NEXT_SAILING_WAIT_H = 48.0

#: Penalty for losing the berth slot at the destination port.
SLOT_MISS_PENALTY_H = 18.0

#: Fraction of the inherited lateness an ocean leg makes up.
TRANSIT_ABSORB = 0.3

#: Fraction of the inherited lateness the trucking leg makes up.
TRUCK_ABSORB = 0.2

#: Realised port congestion turns into this much extra handling per hour of
#: nominal handling.
CONGESTION_HANDLING_FACTOR = 0.5

#: Hub congestion turns into this much extra handling per hour of nominal
#: hub handling.
HUB_HANDLING_FACTOR = 0.3

#: The congestion level a port or hub is assumed to have when nothing is
#: going on.  Realised congestion is measured against this.
NORMAL_CONGESTION = 1.0

#: Nominal last-mile trucking time.
TRUCK_BASE_H = 36.0

#: A leg may make up at most this fraction of its nominal duration.  Past
#: that the delay attenuation is capped and the leg runs at its floor.
MAX_RECOVERY_FRAC = 0.75

#: A weather index of 1.0 means this many extra hours on the leg.
WEATHER_MAX_DELAY_H = 12.0

#: The weather index is curved before it becomes hours: most legs are only
#: slightly affected, a few are hit hard.
WEATHER_EXPONENT = 2.0

#: Storm probability per ocean leg, and the extra delay it brings.
STORM_PROB = 0.08
STORM_DELAY_MIN_H = 24.0
STORM_DELAY_MAX_H = 72.0

#: Extra hub layover a shipment may draw on before it misses the connection.
NOMINAL_SLACK_MIN_H = 0.0
NOMINAL_SLACK_MAX_H = 20.0

#: Noise sigmas.
DEPARTURE_DELAY_MEAN_H = 2.0
DEPARTURE_DELAY_SD_H = 2.5
LEG_NOISE_SD_H = 3.0
DWELL_NOISE_SD_H = 2.0
CUSTOMS_NOISE_SD_H = 2.0
TRUCK_DELAY_MEAN_H = 3.0
TRUCK_DELAY_SD_H = 2.0
HUB_DEPARTURE_NOISE_SD_H = 0.5
CONGESTION_LOGNORMAL_SD = 0.25

#: Weather index distribution (beta).  Low mean: most legs are uneventful.
WEATHER_BETA_A = 1.5
WEATHER_BETA_B = 3.0

# ---------------------------------------------------------------------------
# Route universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """One journey template.

    ``hub`` is ``None`` for a direct shipment.  ``leg2_h`` is unused then.
    """

    origin: str
    hub: str | None
    destination: str
    leg1_h: float
    leg2_h: float = 0.0

    @property
    def journey_id(self) -> str:
        return ">".join(p for p in (self.origin, self.hub, self.destination) if p)


ROUTES: tuple[Route, ...] = (
    # Direct (9).
    Route("Chennai", None, "Rotterdam", 168.0),
    Route("Shanghai", None, "Los_Angeles", 288.0),
    Route("Busan", None, "New_York", 300.0),
    Route("Singapore", None, "Rotterdam", 240.0),
    Route("Mumbai", None, "Hamburg", 216.0),
    Route("Jebel_Ali", None, "Santos", 336.0),
    Route("Durban", None, "Singapore", 168.0),
    Route("Rotterdam", None, "New_York", 144.0),
    Route("Shanghai", None, "Rotterdam", 312.0),
    # Connecting via one hub (10).
    Route("Chennai", "Singapore", "Rotterdam", 120.0, 168.0),
    Route("Mumbai", "Jebel_Ali", "Hamburg", 96.0, 216.0),
    Route("Shanghai", "Singapore", "Los_Angeles", 96.0, 288.0),
    Route("Busan", "Shanghai", "New_York", 72.0, 264.0),
    Route("Chennai", "Jebel_Ali", "Hamburg", 144.0, 216.0),
    Route("Singapore", "Rotterdam", "New_York", 240.0, 144.0),
    Route("Durban", "Jebel_Ali", "Rotterdam", 168.0, 240.0),
    Route("Mumbai", "Singapore", "Hamburg", 144.0, 216.0),
    Route("Shanghai", "Busan", "Los_Angeles", 48.0, 264.0),
    Route("Santos", "Rotterdam", "Hamburg", 264.0, 72.0),
)

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

SEED = 42
N_SHIPMENTS = 300
MISSING_RATE = 0.10
DELAY_THRESHOLD_H = 6.0
TEST_FRAC = 0.20
CV_FOLDS = 5

#: Departure window for generated shipments, in days.
DEPARTURE_WINDOW_DAYS = 90
#: First departure date.
DEPARTURE_START = "2026-01-05"

#: The specimen shipment, S000, is hand-authored to reproduce the worked
#: example in the LLD (sections 7 and 8).  It is real data for the model, not
#: a test-only artifact.
SPECIMEN_ID = "S000"
SPECIMEN_ROUTE = Route("Chennai", "Singapore", "Rotterdam", 120.0, 168.0)
SPECIMEN_DEPARTURE = "2026-01-05 08:00:00"
