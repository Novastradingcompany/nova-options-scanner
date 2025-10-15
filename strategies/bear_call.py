import pandas as pd
from core.math_utils import (
    calc_credit,
    calc_max_loss,
    calc_breakeven,
    calc_pop
)

# ----------------------------
# Bear Call Vertical Scanner
# ----------------------------
def scan_bear_call(chain, spot_price, expiry, dte, T,
                   max_width, max_loss, min_pop,
                   raw_mode, contracts=1):
    """
    Scan for bearish call credit spreads (bear calls).
    - Sells a lower strike call, buys a higher strike call
    - Collects premium upfront
    """
    trades = []

    for i in range(len(chain) - 1):
        sell_leg = chain.iloc[i]
        buy_leg = chain.iloc[i + 1]

        # Ensure proper strike order (sell lower, buy higher)
        if sell_leg["strike"] > buy_leg["strike"]:
            sell_leg, buy_leg = buy_leg, sell_leg

        width = abs(buy_leg["strike"] - sell_leg["strike"])
        if width > max_width:
            continue

        # --- mid prices ---
        sell_mid = (sell_leg["bid"] + sell_leg["ask"]) / 2
        buy_mid = (buy_leg["bid"] + buy_leg["ask"]) / 2
        if sell_mid <= 0 or buy_mid <= 0:
            continue

        # --- ensure OTM call (bearish bias) ---
        if sell_leg["strike"] <= spot_price:
            continue

        # --- calculate credit ---
        credit_mid = calc_credit(sell_mid, buy_mid, contracts)
        credit_realistic = calc_credit(sell_leg["bid"], buy_leg["ask"], contracts)
        if credit_mid <= 0 and credit_realistic <= 0:
            continue

        credit_used = credit_realistic if credit_realistic > 0 else credit_mid

        # --- core metrics ---
        max_loss_val = calc_max_loss(width, credit_used, contracts)
        breakeven = calc_breakeven(sell_leg["strike"], credit_used, "call", contracts)
        pop = calc_pop(
            short_strike=sell_leg["strike"],
            spot=spot_price,
            width=width,
            credit=credit_used,
            max_loss=max_loss_val,
            opt_type="call",
            delta=sell_leg.get("delta", None)
        )

        # --- apply filters unless raw_mode ---
        if not raw_mode:
            if max_loss_val > max_loss or pop < min_pop:
                continue

        trades.append({
            "Strategy": "Bear Call Vertical",
            "Expiry": expiry,
            "DTE": dte,
            "Trade": f"Sell {sell_leg['strike']} / Buy {buy_leg['strike']} CALL",
            "Credit ($)": round(credit_mid, 2),
            "Credit (Realistic)": round(credit_realistic, 2),
            "Max Loss ($)": max_loss_val,
            "POP %": pop,
            "Breakeven": breakeven,
            "Distance %": round((abs(sell_leg['strike'] - spot_price) / spot_price) * 100, 2),
            "Delta": sell_leg.get("delta", None),
            "Contracts": contracts,
            "Spot": round(spot_price, 2)
        })

    return pd.DataFrame(trades)
