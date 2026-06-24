#!/usr/bin/env bash
# Capture opencode's tool schemas -> agent_schemas/opencode.json
#
# opencode is provider-agnostic; this defaults to whatever provider matches the
# base-URL env you point at the proxy. Defaults to Anthropic — for an
# OpenAI-family model, run with these set instead:
#   CAPTURE_UPSTREAM=https://api.openai.com CAPTURE_BASE_ENV=OPENAI_BASE_URL \
#   CAPTURE_BASE_PATH=/v1 scripts/capture/capture-opencode.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export CAPTURE_UPSTREAM="${CAPTURE_UPSTREAM:-https://api.anthropic.com}"
export CAPTURE_BASE_ENV="${CAPTURE_BASE_ENV:-ANTHROPIC_BASE_URL}"

# Runs `opencode` with any extra args appended; override the binary with OPENCODE_BIN.
run_capture opencode "${OPENCODE_BIN:-opencode}" "$@"
