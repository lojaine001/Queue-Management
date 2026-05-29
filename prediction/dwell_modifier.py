"""
dwell_modifier.py — Fuzzy dwell-time modifier.

Combines four signals into a single multiplier applied to the base
service time estimate:

  - Equipment type   (trolley ↑↑ / basket baseline / personal bag ↓)
  - Customer age     (young ↓ / middle baseline / senior ↑)
  - Gender           (female ↑ / male ↓ / unknown → neutral)
  - Group size       (co-present tracks in same lane → shared checkout ↓)

All four factors are combined with a weighted geometric mean so no single
signal dominates.  The result is a float near 1.0 — multiply it against
the current service_minutes to get a demographically-adjusted estimate.

Example
-------
    from prediction.dwell_modifier import compute_dwell_modifier_from_recent
    factor = compute_dwell_modifier_from_recent(conn)   # e.g. 1.45
    adjusted_dwell = base_dwell_min * factor
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

# ── Combination weights (must sum to 1.0) ────────────────────────────────────
# Equipment is the strongest signal (carries most item volume info).
# Group is the second strongest (shared checkout halves time per person).
# Age and gender are mild secondary modifiers.
_W_EQUIPMENT: float = 0.50
_W_GROUP:      float = 0.23
_W_AGE:        float = 0.17
_W_GENDER:     float = 0.10

# ── Gender multipliers ────────────────────────────────────────────────────────
# Based on Arnaud's observation: female customers tend to take slightly longer.
# Kept mild — no strong statistical evidence yet, revisit once data improves.
_GENDER_FACTOR: dict[str, float] = {
    "female":  1.10,
    "male":    0.92,
    "unknown": 1.00,
}

# ── Equipment multipliers ────────────────────────────────────────────────────
# Trolley → large shop → much longer checkout.
# Personal bag → quick purchase → shorter.
_EQUIPMENT_FACTOR: dict[str, float] = {
    "trolley":      2.30,
    "store_basket": 1.00,
    "personal_bag": 0.70,
    "none":         1.00,
}

# ── Group fuzzy probabilities ─────────────────────────────────────────────────
# p_together: probability that co-present people in the same lane are
# checking out as one group (not just two independent customers queuing).
# Not strict — we never assume certainty.
_GROUP_P_TOGETHER: dict[int, float] = {
    1: 0.00,   # alone → no discount
    2: 0.60,   # pair  → 60% likely together
    3: 0.45,   # trio  → 45% likely together
}
_DEFAULT_GROUP_P = 0.35        # for 4+ people
_GROUP_TOGETHER_FACTOR = 0.55  # when definitely together, each person ~0.55× solo time


# ── Individual factor functions ───────────────────────────────────────────────

def age_factor(age_estimate: Optional[float]) -> float:
    """Triangular membership function.

    young (≤25) → 0.82   (faster checkout — fewer items on average)
    middle (35–48) → 1.00 (baseline)
    senior (≥65) → 1.20  (slower checkout — may need more assistance)
    Linear ramps between zones.
    """
    if age_estimate is None or (isinstance(age_estimate, float) and math.isnan(age_estimate)):
        return 1.0
    a = float(age_estimate)
    if a <= 25:
        return 0.82
    elif a <= 35:
        # ramp up: 0.82 → 1.00 over 10 years
        return 0.82 + (a - 25) / 10.0 * 0.18
    elif a <= 48:
        return 1.00
    elif a <= 65:
        # ramp up: 1.00 → 1.20 over 17 years
        return 1.00 + (a - 48) / 17.0 * 0.20
    else:
        return 1.20


def equipment_factor(equipment_type: Optional[str]) -> float:
    """Direct lookup — equipment type is crisp, not fuzzy."""
    return _EQUIPMENT_FACTOR.get(equipment_type or "none", 1.0)


def gender_factor(gender: Optional[str]) -> float:
    """Direct lookup — female slightly longer, male slightly shorter, unknown neutral."""
    return _GENDER_FACTOR.get((gender or "unknown").lower().strip(), 1.0)


def group_factor(active_heads_in_lane: int) -> float:
    """Fuzzy group discount.

    Uses expected-value blending:
        factor = p_together × together_factor + (1 − p_together) × 1.0

    Two people in a lane are 60% likely to be checking out together.
    When they are together each person's share of the checkout is ~0.55×
    what a solo customer would take.

    This is intentionally non-strict — we never assign 0.0 probability
    to independence, matching Arnaud's 'fuzzy' requirement.
    """
    n = max(int(active_heads_in_lane), 1)
    p = _GROUP_P_TOGETHER.get(n, _DEFAULT_GROUP_P)
    return p * _GROUP_TOGETHER_FACTOR + (1.0 - p) * 1.0


def combine_factors(af: float, ef: float, gf: float, genf: float = 1.0) -> float:
    """Weighted geometric mean of the four factors.

    Geometric mean dampens compounding extremes: a trolley-pushing senior
    couple won't produce an absurd multiplier the way simple multiplication
    would.  Each factor contributes proportionally to its weight.
    """
    return (ef ** _W_EQUIPMENT) * (gf ** _W_GROUP) * (af ** _W_AGE) * (genf ** _W_GENDER)


# ── Queue-level modifier (queries DB) ─────────────────────────────────────────

def compute_dwell_modifier_from_recent(conn, lookback_min: int = 20) -> float:
    """Return a dwell time multiplier based on the current queue's demographics.

    Queries the last `lookback_min` minutes of entrance_events to get age,
    gender, equipment type, and group size for people most likely still in the queue.
    Returns the median combined factor across all matched records.

    Returns 1.0 (neutral — no adjustment) on any DB error or empty result.

    Parameters
    ----------
    conn         : psycopg2 connection (read-only query)
    lookback_min : how far back to look for relevant entrance events

    Returns
    -------
    float — multiplier to apply to base dwell estimate (clamped 0.5 – 2.5)
    """
    try:
        df = pd.read_sql(
            f"""
            SELECT age_estimate,
                   gender,
                   equipment_type,
                   active_head_tracks_in_lane
            FROM entrance_events
            WHERE timestamp >= NOW() - INTERVAL '{int(lookback_min)} minutes'
              AND camera_id NOT LIKE 'SIM_%%'
            ORDER BY timestamp DESC
            LIMIT 60
            """,
            conn,
        )
    except Exception:
        return 1.0

    if df.empty:
        return 1.0

    factors: list[float] = []
    for _, row in df.iterrows():
        af   = age_factor(row.get("age_estimate"))
        ef   = equipment_factor(row.get("equipment_type"))
        gf   = group_factor(int(row.get("active_head_tracks_in_lane") or 1))
        genf = gender_factor(row.get("gender"))
        factors.append(combine_factors(af, ef, gf, genf))

    if not factors:
        return 1.0

    result = float(pd.Series(factors).median())
    return max(0.50, min(2.50, result))  # hard clamp — never absurd