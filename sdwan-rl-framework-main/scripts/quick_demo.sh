#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$EUID" -ne 0 ]]; then
    echo "Run in an isolated Linux testbed with root privileges and Ryu already listening on 6653/8080." >&2
    exit 1
fi
folder="$(mktemp -d)"
echo "Diagnostic trace: $folder/demo.jsonl"
exec "${PYTHON:-python3}" -m sdwan_rl.runners --agent-type random --mode heuristic --episodes 1 --seed 1 --trace "$folder/demo.jsonl" "$@"
