#!/bin/sh
set -eu

cp /solution/golden/solution.py /app/public/solution.py
chmod 0644 /app/public/solution.py
cd /app
./build.sh
./test.sh
