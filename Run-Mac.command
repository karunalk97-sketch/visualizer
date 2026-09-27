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

# The helper that hears everything the Mac plays (macOS 14.2+). Needs Apple's
# command line tools; without them the visualizer uses the microphone instead.
if [ ! -x build/mac/SystemAudioTap ] && xcode-select -p >/dev/null 2>&1; then
  mkdir -p build/mac && swiftc -O packaging/mac/SystemAudioTap.swift -o build/mac/SystemAudioTap 2>/dev/null \
    || echo "Couldn't build the system audio helper; using the microphone."
fi

exec .venv/bin/python -m visualizer.main
