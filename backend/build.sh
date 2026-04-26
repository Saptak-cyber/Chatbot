#!/usr/bin/env bash
set -e

echo "=== Installing Python dependencies ==="
pip install --upgrade certifi
pip install -r requirements.txt

echo "=== Build complete ==="
