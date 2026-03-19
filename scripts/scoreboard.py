"""
Leser scores.jsonl fra alle oppgaver og genererer STATUS.md.
Bruk: python3 scripts/scoreboard.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Map task numbers to directory names
TASK_DIRS = {
    1: "oppgave-1-object-detection",
    2: "oppgave-2-tripletex-agent",
    3: "oppgave-3-astar-island",
}

TASK_NAMES = {
    1: "Object Detection",
    2: "Tripletex Agent",
    3: "Astar Island",
}


def read_scores(task_num: int) -> list[dict]:
    path = ROOT / TASK_DIRS[task_num] / "scores.jsonl"
    if not path.exists():
        return []
    scores = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            try:
                scores.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return scores


def generate_status() -> str:
    lines = [
        "# NM i AI 2026 — Scoreboard",
        f"Sist oppdatert: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Lag",
        "- **Ola** (olacola123) — Claude Code",
        "- **Joakim** (joakimotto) — Claude Code",
        "- **Mathea** (matheabrannstorph-commits) — Copilot",
        "",
    ]

    best_per_task = {}

    for task in [1, 2, 3]:
        scores = read_scores(task)
        lines.append(f"## Oppgave {task} — {TASK_NAMES[task]}")

        if not scores:
            lines.append("Ingen scores ennå.\n")
            continue

        # Sorter etter score (høyest først)
        scores.sort(key=lambda s: s.get("score", 0), reverse=True)

        lines.append("| # | Person | Score | Tilnærming | Tid |")
        lines.append("|---|--------|-------|-----------|-----|")

        for i, s in enumerate(scores):
            t = s.get("timestamp", "")
            tid = t[11:16] if len(t) >= 16 else t
            lines.append(
                f"| {i+1} | {s['person']} | {s['score']} | {s.get('approach', '')} | {tid} |"
            )

        best = scores[0]
        best_per_task[task] = best
        lines.append("")

    # Totaltoversikt
    lines.append("## Totalt (beste per oppgave)")
    lines.append("| Oppgave | Score | Person | Tilnærming |")
    lines.append("|---------|-------|--------|-----------|")

    for task in [1, 2, 3]:
        if task in best_per_task:
            b = best_per_task[task]
            lines.append(
                f"| {task} — {TASK_NAMES[task]} | {b['score']} | {b['person']} | {b.get('approach', '')} |"
            )
        else:
            lines.append(f"| {task} — {TASK_NAMES[task]} | — | — | — |")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    status = generate_status()
    (ROOT / "STATUS.md").write_text(status + "\n")
    print("STATUS.md oppdatert!")
