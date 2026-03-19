#!/bin/bash
# Hent andres arbeid. Commit/push gjøres via submit.sh.
# Bruk: bash scripts/sync.sh

cd "$(dirname "$0")/.."
git pull --rebase
echo "Synced!"
