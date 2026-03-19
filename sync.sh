#!/bin/bash
# Auto-sync: pull andres endringer, commit og push dine
# Bruk: bash sync.sh "oppgave-1: score 72, XGBoost ensemble"
# Eller uten melding for å bare hente andres arbeid: bash sync.sh

cd "$(dirname "$0")"

# Hent andres arbeid
git pull --rebase 2>/dev/null

# Hvis melding er gitt, commit og push
if [ -n "$1" ]; then
    git add -A
    git commit -m "$1"
    git push
    echo "Pushet: $1"
else
    echo "Synced. Ingen commit (gi melding som argument for å pushe)."
fi
