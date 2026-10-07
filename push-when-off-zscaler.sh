#!/bin/bash
# Push master to GitHub. On Zscaler networks, git push often 403 — fall back to API.
set -euo pipefail
cd "$(dirname "$0")"
if git push origin master; then
  echo "OK (git push): $(git rev-parse --short HEAD) on origin/master"
  exit 0
fi
echo "git push failed; trying github_api_push.py ..."
python3 github_api_push.py
echo "OK (API): $(git rev-parse --short HEAD) on origin/master"
