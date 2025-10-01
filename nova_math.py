# nova_math.py

import math

# ----------------------------
# Credit calculation
# ----------------------------
def calc_credit(sell_mid, buy_mid, contracts=1):
    """
    Net credit collected per spread.
    """
    return max((sell_mid - buy_mid) * contracts * 100, 0)

# ----------------------------
# Max loss calculation
# ----------------------------
def calc_max_loss(width, credit, contracts=1):
    """
    Max loss = width - credit collected.
    """
    return max((width * 100 * contracts) - credit, 0)

# ----------------------------
# Breakeven calculation
# ----------------------------
def calc_breakeven(short_strike, credit, opt_type):
    """
    Breakeven for simple vertical spread.
    """
    if opt_type == "put":
        return short_strike - (credit / 100.0)
    elif opt_type == "call":
        return short_strike + (credit / 100.0)
    return None

# ----------------------------
# POP (Probability of Profit) calculation
# ----------------------------
def calc_pop(short_strike, spot, width, credit, max_loss, opt_type, delta=None):
    """
    POP estimation hierarchy:
      1. Use delta if available.
      2. Else, use OTM distance vs. spot.
      3. Else, use crude credit/(credit+max_loss).
    Returns POP (%) rounded to 1 decimal.
    """

    # 1. Use delta (prob of ITM) → POP = 1 - prob(ITM)
    if delta is not None and isinstance(delta, (int, float)):
        if opt_type == "put":
            prob_itm = -delta  # put deltas are negative
        else:
            prob_itm = delta
        prob_itm = max(0.0, min(1.0, prob_itm))  # clamp to [0,1]
        return round((1 - prob_itm) * 100, 1)

    # 2. Distance-to-spot heuristic
    distance = abs(short_strike - spot)
    if spot > 0 and width > 0:
        est_pop = (distance / width) * 10  # crude scaling
        return round(max(0, min(est_pop, 95)), 1)

    # 3. Fallback: credit vs max loss
    denom = credit + max_loss
    if denom > 0:
        return round(100 * (credit / denom), 1)

    return 0.0


