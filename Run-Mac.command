#!/bin/bash
# Double-click to run from source on macOS. First run sets everything up
# (Python virtual environment + dependencies), later runs start instantly.
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Return to close."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up, this takes a minute..."
  python3 -m venv .venv && .venv/bin/python -m pip install --quiet --upgrade pip \
    && .venv/bin/python -m pip install --quiet -e . || { read -r -p "Setup failed. Press Return to close."; exit 1; }
fi

exec .venv/bin/python -m visualizer.main
