#!/usr/bin/env bash
# Run once before the first `docker compose up` (and again after moving the project).
#
# Creates every folder docker-compose.yml mounts from the project, owned by the user the
# containers run as (uid 1000). Without it, Docker creates missing folders as root and the
# services cannot write their logs, scheduler state, snapshots or trained models
# (PermissionError on Docker Engine; Docker Desktop on Windows hides this).
set -euo pipefail
cd "$(dirname "$0")/../.."

QM=Queue-Management-System-v2-main/Queue-Management-System-v2-main
dirs=(.runtime secrets Head-Detector/models Head-Detector/LOGs "$QM/models" "$QM/snapshots" "$QM/LOGs")

mkdir -p "${dirs[@]}"
if [ "$(id -u)" -eq 0 ]; then
    chown -R 1000:1000 "${dirs[@]}"
else
    sudo chown -R 1000:1000 "${dirs[@]}"
fi
echo "Ready: ${dirs[*]} (owner uid 1000)"
[ -f secrets/vapid_private_key.pem ] || echo "Still missing: secrets/vapid_private_key.pem (push key, see deploy/docker/README.md)"
[ -f .env ] || echo "Still missing: .env (settings, see deploy/docker/README.md)"
