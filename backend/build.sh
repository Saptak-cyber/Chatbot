#!/usr/bin/env bash
set -e

echo "=== Installing Python dependencies ==="
pip install --upgrade certifi
pip install -r requirements.txt

echo "=== Pre-downloading NLTK models ==="
python3 -c "
import ssl, certifi, nltk
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
print('NLTK models ready.')
"

echo "=== Build complete ==="
