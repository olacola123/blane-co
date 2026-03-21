#!/bin/bash
# Monitor for task 11, 12, 22, 28 hits — run this in a terminal
# Usage: bash monitor.sh

echo "🔍 Monitoring for tasks 11 (salary), 12 (supplier_invoice), 22 (reminder_fee), 28 (receipt_voucher)..."
echo "Press Ctrl+C to stop"
echo ""

WATCHED="salary|supplier_invoice|reminder_fee|receipt_voucher"

while true; do
  gcloud run services logs read tripletex-agent \
    --project=ainm26osl-745 \
    --region=europe-north1 \
    --limit=50 2>&1 \
  | grep -E "(type=($WATCHED))" \
  | while read line; do
    echo "$(date '+%H:%M:%S') ⚡ HIT: $line"
  done
  sleep 30
done
