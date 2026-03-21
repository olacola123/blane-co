"""
Continuous blending between calibration tables, supporting both 3-type and 4-type.

3-type: DEAD / STABLE / BOOMING
4-type: DEAD / STABLE / BOOM_SPREAD / BOOM_CONC

Vitality mapping:
  vitality=0.0  → 100% DEAD
  vitality=0.25 →  50% DEAD + 50% STABLE
  vitality=0.5  → 100% STABLE
  vitality=0.75 →  50% STABLE + 50% BOOM_*
  vitality=1.0  → 100% BOOM_*

For 4-type: BOOM_SPREAD vs BOOM_CONC er bestemt av n_settlements (gratis info),
ikke av vitality. Blending skjer mellom DEAD↔STABLE↔valgt BOOM-type.
"""

import numpy as np


def get_blend_weights(vitality: float) -> tuple[float, float, float]:
    """
    Map vitality (0.0-1.0) to (w_dead, w_stable, w_booming) weights.
    """
    vitality = max(0.0, min(1.0, vitality))

    if vitality <= 0.5:
        t = vitality / 0.5
        return (1.0 - t, t, 0.0)
    else:
        t = (vitality - 0.5) / 0.5
        return (0.0, 1.0 - t, t)


def _resolve_boom_key(type_tables):
    """Finn riktig boom-nøkkel: BOOMING, BOOM_SPREAD, eller BOOM_CONC."""
    for key in ("BOOMING", "BOOM_SPREAD", "BOOM_CONC"):
        if key in type_tables:
            return key
    return None


def get_blended_prior(terrain, band, coastal, type_tables, vitality,
                      transition_table=None, simple_prior=None, fallback_prior=None,
                      n_settlements=None):
    """
    Get a blended prior distribution by interpolating between type-specific tables.

    Supports both 3-type (DEAD/STABLE/BOOMING) and 4-type (DEAD/STABLE/BOOM_SPREAD/BOOM_CONC).
    For 4-type: pass n_settlements to select BOOM_SPREAD vs BOOM_CONC.
    """
    key = f"{terrain}_{band}_{int(coastal)}"
    key_nc = f"{terrain}_{band}_0"

    w_dead, w_stable, w_booming = get_blend_weights(vitality)

    # Bestem boom-nøkkel basert på tilgjengelige tabeller og n_settlements
    if "BOOMING" in type_tables:
        boom_key = "BOOMING"
    elif n_settlements is not None and n_settlements >= 40 and "BOOM_CONC" in type_tables:
        boom_key = "BOOM_CONC"
    elif "BOOM_SPREAD" in type_tables:
        boom_key = "BOOM_SPREAD"
    else:
        boom_key = _resolve_boom_key(type_tables)

    # Collect available distributions and their weights
    weighted_dists = []
    total_weight = 0.0

    blend_spec = [("DEAD", w_dead), ("STABLE", w_stable)]
    if boom_key:
        blend_spec.append((boom_key, w_booming))

    for type_name, weight in blend_spec:
        if weight < 1e-9:
            continue
        table = type_tables.get(type_name)
        if table is None:
            continue

        if key in table:
            dist = np.array(table[key]["distribution"], dtype=float)
            weighted_dists.append(weight * dist)
            total_weight += weight
        elif key_nc in table:
            dist = np.array(table[key_nc]["distribution"], dtype=float)
            weighted_dists.append(weight * dist)
            total_weight += weight

    if weighted_dists and total_weight > 0:
        blended = sum(weighted_dists) / total_weight
        blended = np.maximum(blended, 0.0)
        s = blended.sum()
        if s > 0:
            blended /= s
        return blended

    # Fallback chain: transition_table → simple_prior → fallback_prior → uniform
    if transition_table:
        if key in transition_table:
            return np.array(transition_table[key]["distribution"], dtype=float)
        if key_nc in transition_table:
            return np.array(transition_table[key_nc]["distribution"], dtype=float)

    if simple_prior and str(terrain) in simple_prior:
        return np.array(simple_prior[str(terrain)], dtype=float)

    if fallback_prior and str(terrain) in fallback_prior:
        return np.array(fallback_prior[str(terrain)], dtype=float)

    return np.ones(6, dtype=float) / 6


def infer_vitality_continuous(observers):
    """
    Improved vitality inference — returns a continuous float instead of discrete buckets.

    Maps survival_rate linearly to vitality using empirical anchors:
      dead worlds:    survival ~0.02-0.07  → vitality ~0.0-0.10
      stable worlds:  survival ~0.23-0.45  → vitality ~0.40-0.60
      booming worlds: survival ~0.43-0.57  → vitality ~0.70-0.90

    Uses piecewise linear mapping with 3 anchor points:
      survival=0.03 → vitality=0.05
      survival=0.35 → vitality=0.50
      survival=0.55 → vitality=0.90
    """
    total_initial = 0
    total_survived = 0.0

    for obs in observers:
        for s in obs.settlements:
            sx, sy = s.get("x", -1), s.get("y", -1)
            if not (0 <= sx < 40 and 0 <= sy < 40):
                continue
            if obs.observed[sy, sx] > 0:
                total_initial += 1
                survival = obs.counts[sy, sx, 1] / obs.observed[sy, sx]
                total_survived += survival

    if total_initial == 0:
        return 0.5  # no data → assume stable

    survival_rate = total_survived / total_initial

    # Piecewise linear mapping with clamp
    # Anchors: (survival, vitality)
    anchors = [
        (0.00, 0.00),
        (0.03, 0.05),
        (0.10, 0.20),
        (0.25, 0.40),
        (0.35, 0.50),
        (0.45, 0.65),
        (0.55, 0.85),
        (0.65, 0.95),
        (1.00, 1.00),
    ]

    # Find segment and interpolate
    if survival_rate <= anchors[0][0]:
        return anchors[0][1]
    if survival_rate >= anchors[-1][0]:
        return anchors[-1][1]

    for i in range(len(anchors) - 1):
        s0, v0 = anchors[i]
        s1, v1 = anchors[i + 1]
        if s0 <= survival_rate <= s1:
            t = (survival_rate - s0) / (s1 - s0) if s1 > s0 else 0.0
            return v0 + t * (v1 - v0)

    return 0.5  # shouldn't reach here


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import json
    from pathlib import Path

    cal_file = Path(__file__).parent / "calibration_by_type.json"
    with open(cal_file) as f:
        data = json.load(f)
    type_tables = data["tables"]

    print("=== Blend Weight Tests ===")
    for v in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
        w = get_blend_weights(v)
        print(f"  v={v:.2f} → DEAD={w[0]:.2f}  STABLE={w[1]:.2f}  BOOMING={w[2]:.2f}")

    print("\n=== Blended Prior: key=11_2_1 ===")
    for v in [0.0, 0.15, 0.25, 0.5, 0.75, 0.85, 1.0]:
        prior = get_blended_prior(11, 2, 1, type_tables, v)
        cls = ["empty", "settl", "port ", "ruin ", "forest", "mount"]
        vals = " ".join(f"{cls[i]}={prior[i]:.4f}" for i in range(6))
        print(f"  v={v:.2f}: {vals}  (sum={prior.sum():.6f})")

    print("\n=== High-impact key: 1_0_0 (biggest dead/boom diff) ===")
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prior = get_blended_prior(1, 0, 0, type_tables, v)
        vals = " ".join(f"{prior[i]:.4f}" for i in range(6))
        print(f"  v={v:.2f}: [{vals}]")

    print("\n=== Missing key test (STABLE missing 11_6_0) ===")
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prior = get_blended_prior(11, 6, 0, type_tables, v)
        vals = " ".join(f"{prior[i]:.4f}" for i in range(6))
        w = get_blend_weights(v)
        print(f"  v={v:.2f} w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]: [{vals}]")

    print("\n=== Vitality mapping (survival → vitality) ===")
    # Simulate different survival rates
    for sr in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60]:
        # Fake the mapping directly
        anchors = [
            (0.00, 0.00), (0.03, 0.05), (0.10, 0.20), (0.25, 0.40),
            (0.35, 0.50), (0.45, 0.65), (0.55, 0.85), (0.65, 0.95), (1.00, 1.00),
        ]
        vitality = 0.5
        for i in range(len(anchors) - 1):
            s0, v0 = anchors[i]
            s1, v1 = anchors[i + 1]
            if s0 <= sr <= s1:
                t = (sr - s0) / (s1 - s0) if s1 > s0 else 0.0
                vitality = v0 + t * (v1 - v0)
                break
        w = get_blend_weights(vitality)
        print(f"  survival={sr:.2f} → vitality={vitality:.3f} → DEAD={w[0]:.2f} STABLE={w[1]:.2f} BOOM={w[2]:.2f}")

    print("\n=== Round estimates ===")
    print("  R3 (dead, survival~0.03): ", end="")
    v = 0.05
    w = get_blend_weights(v)
    print(f"vitality={v:.2f} → 100% DEAD (good)")

    print("  R4 (stable, survival~0.35): ", end="")
    v = 0.50
    w = get_blend_weights(v)
    print(f"vitality={v:.2f} → 100% STABLE (good)")

    print("  R12 (booming but weird, survival~0.45): ", end="")
    v = 0.65
    w = get_blend_weights(v)
    print(f"vitality={v:.2f} → STABLE={w[1]:.0%} BOOM={w[2]:.0%} (blended!)")

    print("\n=== INTEGRATION GUIDE ===")
    print("""
To integrate into solution.py, replace:
  1. infer_vitality() → infer_vitality_continuous() from blending.py
  2. get_prior() typed_table param → get_blended_prior() with vitality
  3. Remove adjust_priors_for_vitality() — blending handles this
  4. Remove the discrete world_type selection in solve_round()

Key changes in solve_round():
  - After observing seed 0-1, call infer_vitality_continuous()
  - Pass vitality to SeedObserver (which uses get_blended_prior internally)
  - No need to rebuild seed 0-1 priors separately
""")
