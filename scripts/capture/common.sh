#!/usr/bin/env bash
# Shared helper for the capture-*.sh launchers.
#
# Uses mitmproxy in REVERSE-PROXY mode and points the agent's API base URL at it
# (e.g. ANTHROPIC_BASE_URL). This is far more reliable than a forward proxy:
# the agents' HTTP clients (Node fetch/undici) ignore HTTPS_PROXY, but they DO
# honor base-URL env vars. Plain HTTP to localhost means no CA trust is needed.
#
# Caller sets, then calls run_capture <agent> <command...>:
#   CAPTURE_UPSTREAM  - real API origin, e.g. https://api.anthropic.com
#   CAPTURE_BASE_ENV  - env var the agent reads, e.g. ANTHROPIC_BASE_URL
#   CAPTURE_BASE_PATH - optional path suffix the agent expects, e.g. /v1
set -euo pipefail

_capture_cleanup() {
  [ -n "${MITM_PID:-}" ] && kill "${MITM_PID}" 2>/dev/null || true
  MITM_PID=""
}

run_capture() {
  local agent="$1"; shift   # remaining args = the agent command to run

  : "${CAPTURE_UPSTREAM:?set CAPTURE_UPSTREAM (e.g. https://api.anthropic.com)}"
  # CAPTURE_BASE_ENV is optional: most agents take a base-URL env var, but some
  # (Codex) need a CLI config override instead — those leave it unset and pass
  # the proxy URL ($CAPTURE_PROXY_URL, exported below) in their own args.

  local here repo_root addon out prompt_out port
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$here/../.." && pwd)"
  addon="$here/capture_tools.py"
  out="$repo_root/src/scrape_providers/agent_schemas/${agent}.json"
  prompt_out="$repo_root/src/scrape_providers/agent_prompts/${agent}.txt"
  mkdir -p "$(dirname "$prompt_out")"
  port="${MITM_PORT:-8080}"

  command -v mitmdump >/dev/null 2>&1 || {
    echo "mitmdump not found. Install with: pipx install mitmproxy" >&2
    return 1
  }

  echo "reverse-proxying $CAPTURE_UPSTREAM on :$port  (capturing -> $out)" >&2
  CAPTURE_OUT="$out" CAPTURE_PROMPT_OUT="$prompt_out" mitmdump -q -s "$addon" \
    --mode "reverse:${CAPTURE_UPSTREAM}" --listen-port "$port" &
  MITM_PID=$!
  trap '_capture_cleanup' EXIT INT TERM

  for _ in $(seq 1 50); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break
    sleep 0.2
  done

  # Point the agent at the proxy. Most agents read a base-URL env var; for those
  # that don't, the caller uses $CAPTURE_PROXY_URL in their own CLI args.
  export CAPTURE_PROXY_URL="http://127.0.0.1:${port}${CAPTURE_BASE_PATH:-}"
  if [ -n "${CAPTURE_BASE_ENV:-}" ]; then
    export "${CAPTURE_BASE_ENV}=${CAPTURE_PROXY_URL}"
    echo "${CAPTURE_BASE_ENV}=${CAPTURE_PROXY_URL}" >&2
  fi

  echo "running: $*" >&2
  echo "(do one trivial turn so the agent sends its tools, then quit)" >&2
  "$@" || true

  if [ -f "$out" ]; then
    echo "captured -> $out" >&2
  else
    echo "no tools captured (run again with CAPTURE_DEBUG=1 to see traffic)" >&2
  fi
  [ -f "$prompt_out" ] && echo "captured system prompt -> $prompt_out" >&2
}
