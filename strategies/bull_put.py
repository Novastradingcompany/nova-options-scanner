import pandas as pd
from core.math_utils import (
    calc_credit,
    calc_max_loss,
    calc_breakeven,
    calc_pop
)

# =========================================================
# 📊 Bull Put Vertical Scanner
# =========================================================
def scan_bull_put(chain, spot_price, expiry, dte, T,
                  max_width, max_loss, min_pop,
                  raw_mode, contracts=1):
    """
    Scans for Bull Put Credit Spreads.
    Logic:
      - Sell higher strike put
      - Buy lower strike put
      - Goal: Collect credit, limited downside risk
    """

    trades = []

    for i in range(len(chain) - 1):
        sell_leg = chain.iloc[i]
        buy_leg = chain.iloc[i + 1]

        # ✅ Ensure proper order (sell higher strike, buy lower)
        if sell_leg["strike"] < buy_leg["strike"]:
            sell_leg, buy_leg = buy_leg, sell_leg

        # ✅ Spread width filter
        width = abs(buy_leg["strike"] - sell_leg["strike"])
        if width > max_width:
            continue

        # ✅ Mid-price calc
        sell_mid = (sell_leg["bid"] + sell_leg["ask"]) / 2
        buy_mid = (buy_leg["bid"] + buy_leg["ask"]) / 2
        if sell_mid <= 0 or buy_mid <= 0:
            continue

        # ✅ Only consider OTM puts (bullish setup)
        if sell_leg["strike"] >= spot_price:
            continue

        # ✅ Credit values
        credit_mid = calc_credit(sell_mid, buy_mid, contracts)
        credit_realistic = calc_credit(sell_leg["bid"], buy_leg["ask"], contracts)
        if credit_mid <= 0 and credit_realistic <= 0:
            continue

        effective_credit = credit_realistic if credit_realistic > 0 else credit_mid

        # ✅ Core metrics
        max_loss_val = calc_max_loss(width, effective_credit, contracts)
        breakeven = calc_breakeven(sell_leg["strike"], effective_credit, "put", contracts)
        pop = calc_pop(
            short_strike=sell_leg["strike"],
            spot=spot_price,
            width=width,
            credit=effective_credit,
            max_loss=max_loss_val,
            opt_type="put",
            delta=sell_leg.get("delta", None)
        )

        # ✅ Apply filters (unless raw mode)
        if not raw_mode:
            if max_loss_val > max_loss or pop < min_pop:
                continue

        # ✅ Record trade
        trades.append({
            "Strategy": "Bull Put Vertical",
            "Expiry": expiry,
            "DTE": dte,
            "Trade": f"Sell {sell_leg['strike']} / Buy {buy_leg['strike']} PUT",
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

    # ✅ Return DataFrame for table display
    return pd.DataFrame(trades)


