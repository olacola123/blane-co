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

SRC="templates/${TEMPLATE}.py"
DST="oppgave-${OPPGAVE}/${PERSON}/solution.py"

if [ ! -f "$SRC" ]; then
    echo "FEIL: Template '$SRC' finnes ikke"
    echo "Tilgjengelige templates:"
    ls templates/*.py | sed 's|templates/||;s|\.py||' | while read t; do echo "  $t"; done
    exit 1
fi

cp "$SRC" "$DST"
echo "Kopiert: $SRC -> $DST"
