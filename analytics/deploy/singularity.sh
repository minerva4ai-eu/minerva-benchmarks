#!/usr/bin/env bash
# Build and run the dashboard with Singularity or Apptainer (whichever is installed).
#
#   deploy/singularity.sh build            build minerva-dashboard.sif   (FAKEROOT=1 if you are not root)
#   deploy/singularity.sh start [PORT]     start it in the background (default port 8080)
#   deploy/singularity.sh stop             stop it
#   deploy/singularity.sh status [PORT]    show the instance and query /healthz
#   deploy/singularity.sh logs             last lines of the instance log
#
# Variables: SIF (image path), NAME (instance name), FAKEROOT=1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the analytics/ folder
SIF="${SIF:-$HERE/minerva-dashboard.sif}"
NAME="${NAME:-minerva-dashboard}"
SING="$(command -v apptainer || command -v singularity || true)"
[ -n "$SING" ] || { echo "Neither apptainer nor singularity was found in PATH." >&2; exit 1; }

case "${1:-help}" in
  build)
    cd "$HERE"
    "$SING" build ${FAKEROOT:+--fakeroot} "$SIF" deploy/minerva-dashboard.def
    ;;
  start)
    port="${2:-8080}"
    [ -f "$SIF" ] || { echo "$SIF not found. Run: $0 build" >&2; exit 1; }
    "$SING" instance start \
      --bind "$HERE/data:/app/data:ro" \
      --bind "$HERE/dashboard/defaults.json:/app/dashboard/defaults.json:ro" \
      "$SIF" "$NAME" --host 0.0.0.0 --port "$port"
    echo "Dashboard starting on http://$(hostname -f 2>/dev/null || hostname):$port  (check: $0 status $port)"
    ;;
  stop)
    "$SING" instance stop "$NAME"
    ;;
  status)
    "$SING" instance list "$NAME" || true
    curl -fsS "http://localhost:${2:-8080}/healthz" || echo "healthz not reachable"
    ;;
  logs)
    for base in "$HOME/.apptainer" "$HOME/.singularity"; do
      for f in "$base/instances/logs/$(hostname)/$USER/$NAME".{out,err}; do
        [ -f "$f" ] && { echo "== $f"; tail -n 40 "$f"; }
      done
    done
    ;;
  *)
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
