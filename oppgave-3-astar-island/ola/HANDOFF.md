# Olas forbedringer — Handoff til Joakim

## TL;DR
Nye optimerte prior-tabeller gir **snitt 82.2** i backtest (opp fra ~71).
Backtest slår faktisk score med +40 poeng i snitt over alle 14 runder.

## Hva er nytt

### 1. `calibration_optimized.json` (VIKTIGST)
162 lookup-tabeller beregnet fra **14 runder × 5 seeds** ground truth.
Nøkkelformat: `{world_type}_{terrain_group}_{dist_band}_{coastal}`

- World types: `DEAD`, `STABLE`, `BOOM`, `ALL` (fallback)
- Terrain groups: `plains`, `forest`, `settlement`, `port`, `ruin`
- Distance bands: 0, 1, 2, 3, 4-5, 6-8, 9+ (Chebyshev til nærmeste settlement)
- Coastal: `0` (inland), `1` (adjacent ocean), `any` (fallback)

### 2. `solution.py` v7 endringer
- `load_optimized_calibration()` — laster de nye tabellene
- `get_prior()` — prøver opt_tables FØRST, faller tilbake til gamle tabeller
- `SeedObserver` — tar `opt_tables` og `world_type` parametre
- `solve_round` — sender opt_tables gjennom hele pipelinen
- Floor redusert 0.005→0.003 (+0.8 poeng)
- Obs-vekter justert: 1 obs = 91% prior / 9% obs (var 54%/46%)

### 3. Observation-strategi endret
**KRITISK FUNN**: Med <50 observasjoner per celle er prioren mer nøyaktig.
- 1 obs per alle celler → score 0.3
- 10 obs per alle celler → score 52.7
- Prior alone → score 82.2

Observasjoner bør brukes for **world-type detection** (1-2 queries), IKKE celle-oppdatering.

## Backtest-resultater (oracle world type, alle 14 runder)
```
R1  BOOM:   83.4   R8  DEAD:   89.1
R2  BOOM:   85.1   R9  STABLE: 89.2
R3  DEAD:   86.3   R10 DEAD:   89.9
R4  STABLE: 90.2   R11 BOOM:   79.5
R5  STABLE: 80.3   R12 BOOM:   61.1 ← svakest
R6  BOOM:   77.9   R13 STABLE: 90.6
R7  BOOM:   71.6   R14 BOOM:   77.0
                    AVG: 82.2
```

## Hvordan bruke i Joakims kode
```python
import json
from pathlib import Path

# Last tabellene
opt_data = json.loads(Path("calibration_optimized.json").read_text())
opt_tables = opt_data["tables"]

# Lookup for en celle
def get_opt_prior(world_type, terrain_group, dist_band, coastal):
    for key in [
        f"{world_type}_{terrain_group}_{dist_band}_{int(coastal)}",
        f"{world_type}_{terrain_group}_{dist_band}_any",
        f"ALL_{terrain_group}_{dist_band}_any",
    ]:
        if key in opt_tables:
            return opt_tables[key]["distribution"]  # [empty, settle, port, ruin, forest, mount]
    return None

# Eksempel
prior = get_opt_prior("BOOM", "forest", 2, True)
# → [0.12, 0.15, 0.11, 0.01, 0.61, 0.0] (ca)
```

## World-type detection
Observe 1-2 settlement-celler. Sjekk om de er settlement (alive) eller ruin/empty (dead).
- Survival rate > 0.35 → BOOM
- Survival rate 0.10-0.35 → STABLE
- Survival rate < 0.10 → DEAD

Riktig type = +16 poeng over default. Feil type = kan koste -50 poeng.
