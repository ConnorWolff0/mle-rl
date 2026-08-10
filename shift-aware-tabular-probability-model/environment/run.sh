#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
    echo "usage: ./run.sh TRAIN_CSV VALIDATION_CSV TEST_CSV OUTPUT_CSV" >&2
    exit 64
fi

APP="${APP_DIR:-/app}"
exec python3 "$APP/public/solution.py" "$1" "$2" "$3" "$4"
