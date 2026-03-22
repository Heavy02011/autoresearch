#!/usr/bin/env bash
# Entrypoint for autoresearch Docker container.
# Routes command-line arguments to the autoresearch CLI.

set -euo pipefail

COMMAND="${1:-run}"
shift || true

export PYTHONPATH="/autoresearch/src:${PYTHONPATH:-}"

case "$COMMAND" in
    preflight)
        exec python -m autoresearch.preflight "$@"
        ;;
    run)
        # Run preflight first, warn on failure but continue
        python -m autoresearch.preflight --sim-path "${DONKEY_SIM_PATH:-}" 2>&1 || \
            echo "[WARN] Preflight reported issues - continuing anyway"
        exec autoresearch run "$@"
        ;;
    status)
        exec autoresearch status "$@"
        ;;
    rollback)
        exec autoresearch rollback "$@"
        ;;
    export)
        exec autoresearch export "$@"
        ;;
    *)
        exec autoresearch "$COMMAND" "$@"
        ;;
esac
