#!/usr/bin/env bash
set -euo pipefail

# ── EBH Academy AI — Production deploy script ──
# Usage: ssh into EC2, cd to app dir, run this script.
#
# Prerequisites:
#   - Git remote configured
#   - .env file in place
#   - systemd services: academy-gateway, academy-agent

APP_DIR="${APP_DIR:-/home/ubuntu/app/ebh-academy-ai}"
GATEWAY_SERVICE="academy-gateway"
AGENT_SERVICE="academy-agent"

echo "╔══════════════════════════════════════╗"
echo "║  EBH Academy AI — Deploy            ║"
echo "╚══════════════════════════════════════╝"

cd "$APP_DIR"

# 1. Pull latest
echo "[1/4] Pulling latest code..."
git pull --ff-only

# 2. Install deps (if changed)
if git diff HEAD~1 --name-only | grep -q "requirements.txt"; then
    echo "[2/4] requirements.txt changed — installing deps..."
    pip install -r requirements.txt
else
    echo "[2/4] requirements.txt unchanged — skipping pip install"
fi

# 3. Run tests
echo "[3/4] Running tests..."
python -m pytest tests/ -x -q --tb=short
if [ $? -ne 0 ]; then
    echo "TESTS FAILED — aborting deploy"
    exit 1
fi

# 4. Restart services
echo "[4/4] Restarting services..."
sudo systemctl restart "$GATEWAY_SERVICE"
sudo systemctl restart "$AGENT_SERVICE"

echo ""
echo "Deploy complete. Checking service status..."
sleep 2
systemctl is-active --quiet "$GATEWAY_SERVICE" && echo "  $GATEWAY_SERVICE: running" || echo "  $GATEWAY_SERVICE: FAILED"
systemctl is-active --quiet "$AGENT_SERVICE" && echo "  $AGENT_SERVICE: running" || echo "  $AGENT_SERVICE: FAILED"
echo ""
echo "Done."
