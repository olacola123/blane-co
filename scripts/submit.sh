#!/bin/bash
# submit.sh — Submit + logg + backup + commit + push
#
# Bruk:  bash scripts/submit.sh <oppgave> <person> <score> "<beskrivelse>"
# Eks:   bash scripts/submit.sh 1 ola 72.3 "XGBoost ensemble med feature engineering"

set -e

OPPGAVE=$1
PERSON=$2
SCORE=$3
BESKRIVELSE=$4

if [ -z "$OPPGAVE" ] || [ -z "$PERSON" ] || [ -z "$SCORE" ]; then
    echo "Bruk: bash scripts/submit.sh <oppgave> <person> <score> \"<beskrivelse>\""
    echo "Eks:  bash scripts/submit.sh 1 ola 72.3 \"XGBoost ensemble\""
    exit 1
fi

DIR="oppgave-${OPPGAVE}/${PERSON}"
SCORES_FILE="oppgave-${OPPGAVE}/scores.jsonl"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S")

if [ ! -d "$DIR" ]; then
    echo "FEIL: Mappe $DIR finnes ikke"
    exit 1
fi

echo "=== Submit: oppgave-${OPPGAVE} | ${PERSON} | score ${SCORE} ==="

# 1. Logg score til jsonl
echo "{\"person\": \"${PERSON}\", \"score\": ${SCORE}, \"approach\": \"${BESKRIVELSE}\", \"timestamp\": \"${TIMESTAMP}\"}" >> "$SCORES_FILE"

# 2. Backup solution hvis den finnes
if [ -f "${DIR}/solution.py" ]; then
    cp "${DIR}/solution.py" "${DIR}/solution.py.best-${SCORE}"
    echo "Backup: ${DIR}/solution.py.best-${SCORE}"
fi

# 3. Oppdater scoreboard
python3 scripts/scoreboard.py 2>/dev/null || echo "(scoreboard ikke oppdatert)"

# 4. Git: pull, add, commit, push
git pull --rebase --quiet 2>/dev/null || true
git add "${DIR}/" "$SCORES_FILE" STATUS.md 2>/dev/null || true
git commit -m "oppgave-${OPPGAVE}: score ${SCORE}, ${BESKRIVELSE} (${PERSON})" --quiet
git push --quiet

echo "=== Done! Score ${SCORE} logget og pushet ==="
