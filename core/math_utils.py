# ============================
# core/math_utils.py
# ============================

import math

# ----------------------------
# Credit calculation
# ----------------------------
def calc_credit(sell_mid, buy_mid, contracts=1):
    """Net credit collected per spread (in dollars)."""
    credit_per_contract = max(sell_mid - buy_mid, 0)
    return credit_per_contract * 100 * contracts


# ----------------------------
# Max loss calculation
# ----------------------------
def calc_max_loss(width, credit, contracts=1):
    """Max loss = (width * 100 * contracts) - credit collected."""
    max_loss_total = (width * 100 * contracts) - credit
    return max(max_loss_total, 0)


# ----------------------------
# Breakeven calculation
# ----------------------------
def calc_breakeven(short_strike, credit, opt_type, contracts=1):
    """Breakeven for a vertical spread."""
    credit_per_share = credit / (100 * contracts) if contracts > 0 else 0

    if opt_type == "put":
        return short_strike - credit_per_share
    elif opt_type == "call":
        return short_strike + credit_per_share
    return None


# ----------------------------
# POP (Probability of Profit)
# ----------------------------
def calc_pop(short_strike, spot, width, credit, max_loss, opt_type, delta=None, contracts=1):
    """
    POP estimation hierarchy:
      1. Use delta if available.
      2. Else, use OTM distance vs. spot.
      3. Else, use credit/(credit+max_loss).
    """
    credit_per_share = credit / (100 * contracts) if contracts > 0 else 0
    max_loss_per_share = max_loss / (100 * contracts) if contracts > 0 else 0

    # 1. Delta-based estimate
    if delta is not None and isinstance(delta, (int, float)):
        prob_itm = -delta if opt_type == "put" else delta
        prob_itm = max(0.0, min(1.0, prob_itm))
        return round((1 - prob_itm) * 100, 1)

    # 2. Distance-based fallback
    distance = abs(short_strike - spot)
    if spot > 0 and width > 0:
        est_pop = (distance / width) * 10
        return round(max(0, min(est_pop, 95)), 1)

    # 3. Credit-loss ratio fallback
    denom = credit_per_share + max_loss_per_share
    if denom > 0:
        return round(100 * (credit_per_share / denom), 1)

    return 0.0
