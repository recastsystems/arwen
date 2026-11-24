"""A past cycle's fetch line says the file was already up, in days.

The breakage: ``gpuwm fetch --source gdas --cycle 2026-09-01T00`` run on
22 Nov printed "f000 posted 2026-11-22T21:26:09Z (118958.1 min after its
scheduled time; it was already up when first asked)": the fetch's own
clock time presented as a posting time, and an 82-day gap in minutes
(gpuwm 3.0.5, README step 1 of gpuwm-global 0.3.0).
"""
from __future__ import annotations

from datetime import datetime

from gpuwm import fetch_as_posted

#: The times of the logged line: 118958.1 min apart.
SCHEDULED = "2026-09-01T06:48:03Z"
ASKED = "2026-11-22T21:26:09Z"


def test_a_past_cycle_says_it_was_already_up_and_how_old_in_days():
    line = fetch_as_posted.posting_note(SCHEDULED, ASKED, True)
    assert line.startswith(f"was already up when first asked at {ASKED}")
    assert "82.6 days after its scheduled time" in line
    assert "when the host posted it is not known" in line
    assert "min after" not in line
    assert not line.startswith("posted")


def test_a_lead_watched_as_it_posted_keeps_its_posting_time_in_minutes():
    line = fetch_as_posted.posting_note(
        "2026-11-22T03:25:00Z", "2026-11-22T03:37:18Z", False)
    assert line == ("posted 2026-11-22T03:37:18Z (12.3 min after its "
                    "scheduled time)")


def test_spans_read_in_the_unit_a_reader_takes_in():
    assert fetch_as_posted.span_words(90) == "1.5 min"
    assert fetch_as_posted.span_words(377 * 60) == "6.3 h"
    assert fetch_as_posted.span_words(118958.1 * 60) == "82.6 days"
    # Up before its scheduled time is said so, not as a negative lateness.
    line = fetch_as_posted.posting_note(
        "2026-11-22T03:25:00Z", "2026-11-22T03:15:00Z", True)
    assert "10.0 min before its scheduled time" in line


def test_the_verified_line_of_a_published_lead_uses_the_note(tmp_path):
    lines: list[str] = []
    loop = fetch_as_posted.PostingLoop.__new__(fetch_as_posted.PostingLoop)
    # The few fields publish() reads; the rest of the loop is not needed
    # to write one marker.
    import threading
    from types import SimpleNamespace

    loop._lock = threading.Lock()
    loop._tried = {}
    loop._rows = {0: {"lead": 0, "expected_at": SCHEDULED,
                      "first_seen_at": None, "late_at": None,
                      "valid_time": "2026-09-01T00:00:00Z"}}
    loop.window = SimpleNamespace(source="gdas", member=None)
    loop.document = {"cycle": "2026-09-01T00:00:00Z", "leads": []}
    loop.folder = tmp_path
    loop.label = "fetch gdas 2026-09-01T00"
    loop.progress = lines.append
    loop.now = lambda: datetime(2026, 11, 22, 21, 26, 9)
    loop._mark = lambda lead, **fields: loop._rows[lead].update(fields)
    loop.publish(0, [{"name": "gdas.t00z.pgrb2.0p25.f000",
                      "bytes": 471_773_591}])
    assert len(lines) == 1, lines
    assert lines[0] == (
        f"fetch gdas 2026-09-01T00: f000 was already up when first asked at "
        f"{ASKED}, 82.6 days after its scheduled time (when the host posted "
        "it is not known); 472 MB verified")
