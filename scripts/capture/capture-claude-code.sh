#!/usr/bin/env bash
# Capture Claude Code's tool schemas -> agent_schemas/claude_code.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export CAPTURE_UPSTREAM="https://api.anthropic.com"
export CAPTURE_BASE_ENV="ANTHROPIC_BASE_URL"

# Runs `claude` with any extra args appended; override the binary with CLAUDE_BIN.
run_capture claude_code "${CLAUDE_BIN:-claude}" "$@"
