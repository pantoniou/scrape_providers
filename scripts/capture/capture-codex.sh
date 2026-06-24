#!/usr/bin/env bash
# Capture Codex's tool schemas -> agent_schemas/codex.json
#
# Codex sends the model request to the openai provider's base_url, which defaults
# to the ChatGPT backend (https://chatgpt.com/backend-api/codex) under ChatGPT
# login, or https://api.openai.com/v1 in API-key mode. OPENAI_BASE_URL only
# affects the latter. So instead we override `model_providers.openai.base_url`
# via -c, which wins regardless of auth mode, and reverse-proxy to the real
# backend. We capture the outgoing request (which carries `tools`) even if the
# upstream rejects it.
#
# mitmproxy reverse mode takes a host only (no path), so the upstream is the bare
# host and the API path lives in the base_url override (reverse mode preserves the
# request path). Default = ChatGPT backend. For API-key mode, run with:
#   CAPTURE_UPSTREAM=https://api.openai.com CODEX_BASE_PATH=/v1 ...capture-codex.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export CAPTURE_UPSTREAM="${CAPTURE_UPSTREAM:-https://chatgpt.com}"
# No CAPTURE_BASE_ENV: Codex takes CLI config overrides, not an env var.

# Codex forbids overriding the built-in `openai` provider, so define a custom one
# that points at the proxy but keeps OpenAI/ChatGPT auth (requires_openai_auth),
# and select it. Reuses the existing login token, so no re-auth needed.
proxy="http://127.0.0.1:${MITM_PORT:-8080}${CODEX_BASE_PATH:-/backend-api/codex}"
prov="${CODEX_PROVIDER_ID:-codexcapture}"
run_capture codex "${CODEX_BIN:-codex}" \
  -c "model_provider=\"${prov}\"" \
  -c "model_providers.${prov}.name=\"codex capture\"" \
  -c "model_providers.${prov}.base_url=\"${proxy}\"" \
  -c "model_providers.${prov}.wire_api=\"responses\"" \
  -c "model_providers.${prov}.requires_openai_auth=true" \
  "$@"
