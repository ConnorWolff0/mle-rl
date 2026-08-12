#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"
if [ "$#" -ne 2 ]; then
    echo "usage: ./run.sh SCENARIO_JSON OUTPUT_JSON" >&2
    exit 2
fi
exec python3 -m serving_sim run "$1" "$2" --policy "$HERE/policy.py"
