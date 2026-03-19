#!/bin/bash
# Kopier en template til din oppgave-mappe
#
# Bruk:  bash scripts/copy-template.sh <template> <oppgave> <person>
# Eks:   bash scripts/copy-template.sh classifier 1 ola
#
# Templates: classifier, rl_agent, rag_pipeline, segmentation, optimizer, websocket_bot

TEMPLATE=$1
OPPGAVE=$2
PERSON=$3

if [ -z "$TEMPLATE" ] || [ -z "$OPPGAVE" ] || [ -z "$PERSON" ]; then
    echo "Bruk: bash scripts/copy-template.sh <template> <oppgave> <person>"
    echo ""
    echo "Templates:"
    ls templates/*.py | sed 's|templates/||;s|\.py||' | while read t; do echo "  $t"; done
    exit 1
fi

# Map task number to directory name
case "$OPPGAVE" in
    1) OPPGAVE_DIR="oppgave-1-object-detection" ;;
    2) OPPGAVE_DIR="oppgave-2-tripletex-agent" ;;
    3) OPPGAVE_DIR="oppgave-3-astar-island" ;;
    *) echo "FEIL: Ukjent oppgave '$OPPGAVE'. Bruk 1, 2 eller 3."; exit 1 ;;
esac

SRC="templates/${TEMPLATE}.py"
DST="${OPPGAVE_DIR}/${PERSON}/solution.py"

if [ ! -f "$SRC" ]; then
    echo "FEIL: Template '$SRC' finnes ikke"
    echo "Tilgjengelige templates:"
    ls templates/*.py | sed 's|templates/||;s|\.py||' | while read t; do echo "  $t"; done
    exit 1
fi

cp "$SRC" "$DST"
echo "Kopiert: $SRC -> $DST"
