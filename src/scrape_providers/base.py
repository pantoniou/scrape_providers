"""The Scraper interface that every provider implementation satisfies.

Keep fetching (network/HTML) separate from normalization: ``scrape`` should
return a fully normalized :class:`Provider` so that output-format changes never
ripple back into provider code.
"""

from __future__ import annotations

import abc

import httpx

from .models import Provider


class Scraper(abc.ABC):
    """Base class for provider scrapers."""

    #: Stable, lowercase provider identifier used on the CLI and as a registry key.
    name: str

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=30.0, follow_redirects=True, headers={"User-Agent": "scrape-providers/0.1"}
        )

    @abc.abstractmethod
    def scrape(self) -> Provider:
        """Fetch and normalize this provider's catalog into a :class:`Provider`."""

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Scraper":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
