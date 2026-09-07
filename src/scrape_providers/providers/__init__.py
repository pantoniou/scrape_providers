"""Provider-specific scrapers. One module per provider."""

from __future__ import annotations

from datetime import date, datetime, timezone


def created_date(raw: dict, field: str = "created") -> date | None:
    """A provider's ``created`` timestamp as a date, or ``None`` if unusable.

    Several providers ship the field without meaning it — OpenCode Zen stamps
    every model with the time of the request and Meta returns a literal ``0`` —
    so a falsy or clearly-bogus value is treated as absent rather than emitted
    as a 1970 date.
    """
    value = raw.get(field)
    if not value:  # missing, or Meta's `0` stub
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None
