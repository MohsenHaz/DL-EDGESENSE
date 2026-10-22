"""
Unified-clock, drift-correction buffer (Section 4.5 / Case Studies:
"sensor synchronisation requires timestamp alignment across
heterogeneous modalities; DL-EgSense addresses this through a
unified clock and a drift-correction buffer.")
"""
from bisect import bisect_left
from dataclasses import dataclass, field


@dataclass
class DriftCorrectionBuffer:
    """
    Maintains a rolling estimate of per-sensor clock offset relative to
    a reference clock, and resamples/aligns incoming (timestamp, value)
    streams onto a common timeline before fusion.
    """
    reference_hz: float = 100.0
    max_buffer_s: float = 5.0
    _offsets: dict = field(default_factory=dict)   # sensor_id -> offset (s)
    _streams: dict = field(default_factory=dict)   # sensor_id -> [(t, value)]

    def register_sync_pulse(self, sensor_id: str, local_ts: float, reference_ts: float):
        """Update the estimated clock offset for `sensor_id` using a
        shared synchronisation event (e.g. a flash/beep detectable
        across modalities)."""
        self._offsets[sensor_id] = reference_ts - local_ts

    def push(self, sensor_id: str, local_ts: float, value):
        offset = self._offsets.get(sensor_id, 0.0)
        corrected_ts = local_ts + offset
        buf = self._streams.setdefault(sensor_id, [])
        buf.append((corrected_ts, value))
        cutoff = corrected_ts - self.max_buffer_s
        while buf and buf[0][0] < cutoff:
            buf.pop(0)

    def aligned_window(self, sensor_ids, window_start: float, window_end: float):
        """Return {sensor_id: [values in [window_start, window_end)]}."""
        out = {}
        for sid in sensor_ids:
            buf = self._streams.get(sid, [])
            times = [t for t, _ in buf]
            lo = bisect_left(times, window_start)
            hi = bisect_left(times, window_end)
            out[sid] = [v for _, v in buf[lo:hi]]
        return out
