#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"
python3 -m compileall -q policy.py serving_sim tests
python3 -m serving_sim --help >/dev/null
