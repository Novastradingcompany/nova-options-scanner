import pandas as pd
import math
from nova_math import calc_credit, calc_max_loss, calc_breakeven, calc_pop

# ----------------------------
# Black-Scholes Delta helper
# ----------------------------
def bs_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if option_type == "call":
            return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        else:
            return -0.5 * (1 - math.erf(d1 / math.sqrt(2)))
    except Exception:
        return None

# ----------------------------
# Vertical spread scanner
# ----------------------------
def scan_verticals(chain, spot_price, expiry, dte, T,
                   max_width, max_loss, min_pop,
                   raw_mode, opt_type, contracts=1):

    trades = []
    for i in range(len(chain) - 1):
        sell_leg = chain.iloc[i]
        buy_leg = chain.iloc[i + 1]

        # --- enforce correct strike order ---
        if opt_type == "put":
            if sell_leg["strike"] < buy_leg["strike"]:
                sell_leg, buy_leg = buy_leg, sell_leg
        elif opt_type == "call":
            if sell_leg["strike"] > buy_leg["strike"]:
                sell_leg, buy_leg = buy_leg, sell_leg

        width = abs(buy_leg["strike"] - sell_leg["strike"])
        if width > max_width:
            continue

        # --- mid-prices ---
        sell_mid = (sell_leg["bid"] + sell_leg["ask"]) / 2 if sell_leg["ask"] > 0 else sell_leg["bid"]
        buy_mid  = (buy_leg["bid"] + buy_leg["ask"]) / 2 if buy_leg["ask"] > 0 else buy_leg["ask"]

        if sell_mid <= 0 or buy_mid < 0:
            continue

        # --- OTM requirement ---
        if opt_type == "put" and sell_leg["strike"] >= spot_price:
            continue
        if opt_type == "call" and sell_leg["strike"] <= spot_price:
            continue

        # --- credit calculations ---
        credit_mid = (sell_mid - buy_mid) * contracts * 100
        credit_realistic = max((sell_leg["bid"] - buy_leg["ask"]) * contracts * 100, 0)

        if credit_realistic <= 0 and credit_mid <= 0:
            continue

        # Use realistic credit if positive, otherwise fallback to mid
        effective_credit = credit_realistic if credit_realistic > 0 else credit_mid

        max_loss_val = calc_max_loss(width, effective_credit, contracts)
        breakeven = calc_breakeven(sell_leg["strike"], effective_credit, opt_type)

        # --- POP using nova_math ---
        pop = calc_pop(
            short_strike=sell_leg["strike"],
            spot=spot_price,
            width=width,
            credit=effective_credit,
            max_loss=max_loss_val,
            opt_type=opt_type,
            delta=sell_leg.get("delta", None)
        )

        # --- DEBUG print only if raw_mode checked ---
        if raw_mode:
            print("DEBUG CANDIDATE:", {
                "opt_type": opt_type,
                "sell_strike": float(sell_leg["strike"]),
                "buy_strike": float(buy_leg["strike"]),
                "sell_bid": float(sell_leg["bid"]),
                "sell_ask": float(sell_leg["ask"]),
                "buy_bid": float(buy_leg["bid"]),
                "buy_ask": float(buy_leg["ask"]),
                "sell_mid": round(sell_mid, 3),
                "buy_mid": round(buy_mid, 3),
                "credit_mid": round(credit_mid, 2),
                "credit_realistic": round(credit_realistic, 2),
                "max_loss": round(max_loss_val, 2),
                "pop": round(pop, 2)
            })

        if not raw_mode:
            if max_loss_val > max_loss or pop < min_pop:
                continue

        trades.append({
            "Strategy": f"{'Bull Put' if opt_type == 'put' else 'Bear Call'} Vertical",
            "Expiry": expiry,
            "DTE": dte,
            "Trade": f"Sell {sell_leg['strike']} / Buy {buy_leg['strike']} {opt_type.upper()}",
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

    return pd.DataFrame(trades), None

# ----------------------------
# Dummy Iron Condor scanner
# ----------------------------
def scan_condors(*args, **kwargs):
    return pd.DataFrame(), None

# ----------------------------
# Style table with $ and %
# ----------------------------
def style_table(df, min_pop_val):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "Credit ($)" in df.columns:
        df["Credit ($)"] = df["Credit ($)"].map(lambda x: f"${x:,.2f}")
    if "Credit (Realistic)" in df.columns:
        df["Credit (Realistic)"] = df["Credit (Realistic)"].map(lambda x: f"${x:,.2f}")
    if "Max Loss ($)" in df.columns:
        df["Max Loss ($)"] = df["Max Loss ($)"].map(lambda x: f"${x:,.2f}")
    if "Breakeven" in df.columns:
        df["Breakeven"] = df["Breakeven"].map(lambda x: f"${x:,.2f}")
    if "POP %" in df.columns:
        df["POP %"] = df["POP %"].map(lambda x: f"{x:.1f}%")
    if "Distance %" in df.columns:
        df["Distance %"] = df["Distance %"].map(lambda x: f"{x:.1f}%")

    styler = df.style
    if "POP %" in df.columns:
        styler = styler.map(
            lambda v: "color: green" if isinstance(v, str) and v.endswith("%") and float(v.strip('%')) >= min_pop_val else "",
            subset=["POP %"]
        )
    return styler

















































