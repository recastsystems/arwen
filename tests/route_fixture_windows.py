"""A cycle, a first lead and a window each table route really publishes.

Read from the route's own row, so a test that walks every route asks each one
a question it can answer.  Tests that used one fixture day (2026-08-17) and a
window from f000 for every route went red when routes with other periods
landed: the GEFS v11 archives (2017 to 2020), the 0.1 degree IFS open data
(from 2026-10-20), the GEFS reforecast (no f000: its files start at f003) and
the analysis-only surface donors (ncep-r1-seaice, gefs-reforecast-land: one
time per cycle).  The product refuses each of those windows by name, as it
should; the fixture was what was wrong.
"""
from __future__ import annotations

from datetime import datetime

#: The day every route without dated publication eras is asked about.
FIXTURE_DAY = datetime(2026, 8, 17)


def route_cycle(source: str, hour: int | None = None) -> datetime:
    """A cycle inside SOURCE's newest preparable publication, at HOUR (default:
    the route's first cycle hour), or the fixture day for a route that is
    published at every date."""

    from gpuwm import fetch_routes

    route = fetch_routes.route_for(source)
    hour = route.cycle_hours[0] if hour is None else hour
    if route.publication_eras:
        cycles = fetch_routes.planning_cycles(route)
        matching = [cycle for cycle in cycles if cycle.hour == hour]
        return (matching or list(cycles))[0]
    return FIXTURE_DAY.replace(hour=hour)


def route_start(source: str) -> int:
    """The first lead SOURCE publishes (3 for the GEFS reforecast, else 0;
    0 for a source outside the route table, such as GFS)."""

    from gpuwm import fetch_routes

    route = fetch_routes.table_route(source)
    return 0 if route is None else route.first_lead


def analysis_only(source: str) -> bool:
    """True for a route whose cycles publish a single time."""

    from gpuwm import fetch_routes

    route = fetch_routes.route_for(source)
    return len(fetch_routes.ladder_for(route, route_cycle(source))) == 1


def route_hours(source: str, hours: int | None = None) -> int:
    """HOURS (default: the route's default cadence), or 0 for an
    analysis-only route, whose only window is its one time."""

    from gpuwm import fetch_routes

    if analysis_only(source):
        return 0
    return fetch_routes.route_for(source).default_cadence if hours is None else hours
