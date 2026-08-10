#!/bin/sh
set -u
umask 077

APP="${APP_DIR:-/app}"
LOGS="${LOGS_DIR:-/logs/verifier}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DETAILS="$LOGS/details.json"
REWARD="$LOGS/reward.json"

mkdir -p "$LOGS" 2>/dev/null || true
printf '%s\n' '{"reward":0.0}' > "$REWARD"

if [ ! -f "$APP/public/solution.py" ]; then
    printf '%s\n' '{"error":"missing /app/public/solution.py","reward":0.0}' > "$DETAILS"
    exit 0
fi

if ! PYTHONDONTWRITEBYTECODE=1 python3 -I -B "$HERE/contract_checks.py" \
    --app "$APP" --tests "$HERE" > "$LOGS/contract.json" \
    2> "$LOGS/contract.stderr"; then
    printf '%s\n' '{"error":"public contract check failed","reward":0.0}' > "$DETAILS"
    exit 0
fi

if ! PYTHONDONTWRITEBYTECODE=1 python3 -I -B "$HERE/grade.py" \
    "$APP/run.sh" --anchors "$HERE/anchors.json" \
    > "$DETAILS.tmp" 2> "$LOGS/grader.stderr"; then
    printf '%s\n' '{"error":"private grader failed","reward":0.0}' > "$DETAILS"
    exit 0
fi

mv "$DETAILS.tmp" "$DETAILS"
DETAILS="$DETAILS" REWARD="$REWARD" python3 -I -B - <<'PY'
import json
import os
from pathlib import Path

details = json.loads(Path(os.environ["DETAILS"]).read_text(encoding="utf-8"))
reward = float(details.get("reward", 0.0))
if not 0.0 <= reward <= 1.0:
    raise ValueError("reward must be between zero and one")
target = Path(os.environ["REWARD"])
temporary = target.with_suffix(".tmp")
temporary.write_text(
    json.dumps({"reward": reward}, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
os.replace(temporary, target)
PY

exit 0
