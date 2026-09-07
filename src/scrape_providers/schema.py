"""JSON Schema for the catalog output, and a validator.

The schema (``catalog.schema.json``, packaged alongside this module) describes the
two-section structure produced by :func:`scrape_providers.emit.build_catalog`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

import jsonschema

SCHEMA_FILE = "catalog.schema.json"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(files("scrape_providers").joinpath(SCHEMA_FILE).read_text("utf-8"))


def validate_catalog(catalog: dict) -> None:
    """Validate a built catalog against the schema; raise ValidationError if invalid.

    ``format`` is annotation-only in JSON Schema unless a checker is supplied, so
    a format checker is passed explicitly — without it a ``format: date`` field
    would accept ``28/05/2026`` or ``2026-13-45`` and the schema's promise to
    catch bad scraped data would be hollow for exactly the fields most likely to
    be misparsed.
    """
    jsonschema.validate(catalog, load_schema(), format_checker=jsonschema.FormatChecker())
