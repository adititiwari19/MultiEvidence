#!/usr/bin/env bash
set -euo pipefail
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
