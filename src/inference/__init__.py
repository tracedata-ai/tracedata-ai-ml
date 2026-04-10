"""
Inference entrypoints — **two separate models** (see ``docs/SCORING_PATHS.md``):

- ``SmoothnessInference``: raw **pings** → 3 features → ``score_trip_from_ping_windows``.
- ``DeviceAggregateTripScorer``: **smoothness_log** / 18 aggregate columns → ``score_trip_at_end``.
"""

from .device_trip_scorer import DeviceAggregateTripScorer
from .smoothness_inference import SmoothnessInference

__all__ = ["DeviceAggregateTripScorer", "SmoothnessInference"]
