#!/usr/bin/env bash
# overnight-tripletex.sh — Keep Mac awake + monitor Tripletex agent overnight
#
# Usage:
#   bash scripts/overnight-tripletex.sh
#
# What it does:
#   1. Prevents Mac from sleeping (caffeinate)
#   2. Starts local server if not already running
#   3. Starts Cloudflare tunnel if not already running
#   4. Monitors Cloud Run logs every 5 minutes
#   5. Fetches learning DB stats periodically
#   6. Logs everything to logs/overnight.log
#
# To stop: Ctrl+C (kills caffeinate + tunnel + server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

OVERNIGHT_LOG="$LOG_DIR/overnight-$(date +%Y%m%d-%H%M%S).log"
CLOUD_RUN_URL="https://tripletex-agent-609915262705.europe-north1.run.app"
LOCAL_PORT=8000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OVERNIGHT_LOG"
}

cleanup() {
    log "Shutting down..."
    # Kill caffeinate
    if [[ -n "${CAFFEINATE_PID:-}" ]]; then
        kill "$CAFFEINATE_PID" 2>/dev/null || true
        log "Caffeinate stopped"
    fi
    # Kill tunnel
    if [[ -n "${TUNNEL_PID:-}" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        log "Tunnel stopped"
    fi
    # Kill local server
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        log "Local server stopped"
    fi
    log "All processes stopped. Log: $OVERNIGHT_LOG"
    exit 0
}
trap cleanup EXIT INT TERM

log "=== OVERNIGHT TRIPLETEX MONITOR ==="
log "Cloud Run URL: $CLOUD_RUN_URL"
log "Log file: $OVERNIGHT_LOG"

# 1. Prevent Mac from sleeping
caffeinate -dims &
CAFFEINATE_PID=$!
log "Caffeinate started (PID $CAFFEINATE_PID) — Mac will not sleep"

# 2. Check Cloud Run health
if curl -sf "$CLOUD_RUN_URL/health" > /dev/null 2>&1; then
    log "Cloud Run: HEALTHY"
else
    log "WARNING: Cloud Run not responding at $CLOUD_RUN_URL"
fi

# 3. Optionally start local server (for Cloudflare tunnel fallback)
if [[ "${START_LOCAL:-false}" == "true" ]]; then
    if lsof -i :$LOCAL_PORT > /dev/null 2>&1; then
        log "Local server already running on port $LOCAL_PORT"
    else
        log "Starting local server on port $LOCAL_PORT..."
        cd "$PROJECT_DIR/oppgave-2-tripletex-agent/ola"
        source "$PROJECT_DIR/env/bin/activate" 2>/dev/null || true
        uvicorn server:app --host 0.0.0.0 --port $LOCAL_PORT >> "$LOG_DIR/server.log" 2>&1 &
        SERVER_PID=$!
        log "Local server started (PID $SERVER_PID)"
        sleep 2
    fi

    # 4. Start Cloudflare tunnel if needed
    if [[ "${START_TUNNEL:-false}" == "true" ]]; then
        log "Starting Cloudflare tunnel..."
        npx cloudflared tunnel --url http://localhost:$LOCAL_PORT >> "$LOG_DIR/tunnel.log" 2>&1 &
        TUNNEL_PID=$!
        sleep 5
        TUNNEL_URL=$(grep -o 'https://[a-z-]*\.trycloudflare\.com' "$LOG_DIR/tunnel.log" 2>/dev/null | tail -1 || echo "unknown")
        log "Tunnel started (PID $TUNNEL_PID): $TUNNEL_URL"
        log "IMPORTANT: Update submission URL on app.ainm.no to: $TUNNEL_URL"
    fi
fi

# 5. Monitor loop
log ""
log "=== MONITORING (Ctrl+C to stop) ==="
log "Checking every 5 minutes..."
log ""

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))

    # Health check
    if curl -sf "$CLOUD_RUN_URL/health" > /dev/null 2>&1; then
        HEALTH="OK"
    else
        HEALTH="DOWN"
        log "ALERT: Cloud Run is DOWN!"
    fi

    # Fetch learning stats
    STATS=$(curl -sf "$CLOUD_RUN_URL/learning" 2>/dev/null || echo '{"stats":{}}')
    TOTAL=$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stats',{}).get('total',0))" 2>/dev/null || echo "?")
    ERRORS=$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stats',{}).get('errors',0))" 2>/dev/null || echo "?")
    RETRIES=$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stats',{}).get('retries_needed',0))" 2>/dev/null || echo "?")

    # Fetch recent logs
    LOGS=$(curl -sf "$CLOUD_RUN_URL/logs" 2>/dev/null || echo '{"count":0}')
    LOG_COUNT=$(echo "$LOGS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo "?")

    log "[#$ITERATION] Health=$HEALTH | Submissions=$TOTAL | Errors=$ERRORS | Retries=$RETRIES | Logs=$LOG_COUNT"

    # Every 30 min, fetch Cloud Run logs
    if [[ $((ITERATION % 6)) -eq 0 ]]; then
        log "--- Cloud Run logs (last 20) ---"
        gcloud run services logs read tripletex-agent --region europe-north1 --limit 20 --project=ainm26osl-745 2>/dev/null | tee -a "$OVERNIGHT_LOG" || true
        log "--- end logs ---"
    fi

    sleep 300  # 5 minutes
done
