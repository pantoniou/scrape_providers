"""Maps provider names to their scraper classes."""

from __future__ import annotations

from .base import Scraper
from .providers.anthropic import AnthropicScraper
from .providers.deepseek import DeepSeekScraper
from .providers.openai import OpenAIScraper
from .providers.openrouter import OpenRouterScraper

_SCRAPERS: dict[str, type[Scraper]] = {
    AnthropicScraper.name: AnthropicScraper,
    DeepSeekScraper.name: DeepSeekScraper,
    OpenAIScraper.name: OpenAIScraper,
    OpenRouterScraper.name: OpenRouterScraper,
}


def available() -> list[str]:
    return sorted(_SCRAPERS)


def get(name: str) -> type[Scraper]:
    try:
        return _SCRAPERS[name]
    except KeyError:
        raise KeyError(f"unknown provider {name!r}; available: {', '.join(available())}") from None
