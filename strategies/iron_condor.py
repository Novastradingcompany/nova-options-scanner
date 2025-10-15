import pandas as pd
from core.math_utils import calc_credit, calc_max_loss, calc_breakeven, calc_pop

# =========================================================
# 🧩 Iron Condor Scanner (robust to stale bid/ask on one wing)
# =========================================================
def scan_iron_condor(chains, spot_price, expiry, dte, T,
                     max_width, max_loss, min_pop,
                     raw_mode, contracts=1):
    """
    Build Iron Condors from put + call verticals.

    Key rules:
      - Put wing: short higher strike put, long lower strike put (both OTM)
      - Call wing: short lower strike call, long higher strike call (both OTM)
      - Credit side logic is tolerant: if realistic (bid/ask) credit <= 0, fall back to mid.
      - Max loss = (max(put_width, call_width) * 100 * contracts) - total_credit
      - Breakeven range uses TOTAL credit per share, not per-wing credit.
    """
    puts = chains["puts"].copy()
    calls = chains["calls"].copy()

    # guard against missing cols
    for df in (puts, calls):
        for col in ["bid", "ask", "strike"]:
            if col not in df.columns:
                return pd.DataFrame()

    # drop NaNs / zeros that break math
    puts = puts.dropna(subset=["bid", "ask", "strike"])
    calls = calls.dropna(subset=["bid", "ask", "strike"])

    trades = []

    # ----------- iterate put wing -----------
    for i in range(len(puts) - 1):
        short_put = puts.iloc[i]
        long_put  = puts.iloc[i + 1]

        # ensure bull-put structure (short strike higher than long)
        if short_put["strike"] < long_put["strike"]:
            short_put, long_put = long_put, short_put

        put_width = abs(short_put["strike"] - long_put["strike"])
        if put_width <= 0 or put_width > max_width:
            continue

        # require OTM short put
        if short_put["strike"] >= spot_price:
            continue

        # compute mid & realistic credits for put wing
        sp_mid = (short_put["bid"] + short_put["ask"]) / 2
        lp_mid = (long_put["bid"] + long_put["ask"]) / 2
        put_credit_mid = calc_credit(sp_mid, lp_mid, contracts)
        put_credit_real = calc_credit(short_put["bid"], long_put["ask"], contracts)
        put_credit_eff = put_credit_real if put_credit_real > 0 else put_credit_mid

        # if even mid credit isn't positive, skip the wing
        if put_credit_mid <= 0:
            continue

        # ----------- iterate call wing -----------
        for j in range(len(calls) - 1):
            short_call = calls.iloc[j]
            long_call  = calls.iloc[j + 1]

            # ensure bear-call structure (short strike lower than long)
            if short_call["strike"] > long_call["strike"]:
                short_call, long_call = long_call, short_call

            call_width = abs(long_call["strike"] - short_call["strike"])
            if call_width <= 0 or call_width > max_width:
                continue

            # require OTM short call
            if short_call["strike"] <= spot_price:
                continue

            # compute mid & realistic credits for call wing
            sc_mid = (short_call["bid"] + short_call["ask"]) / 2
            lc_mid = (long_call["bid"] + long_call["ask"]) / 2
            call_credit_mid = calc_credit(sc_mid, lc_mid, contracts)
            call_credit_real = calc_credit(short_call["bid"], long_call["ask"], contracts)
            call_credit_eff = call_credit_real if call_credit_real > 0 else call_credit_mid

            # must have some positive mid credit for the call wing
            if call_credit_mid <= 0:
                continue

            # ---------- combine the wings ----------
            total_credit_mid = put_credit_mid + call_credit_mid
            total_credit_eff = put_credit_eff + call_credit_eff

            # In raw mode, allow mid-credit > 0 even if effective ≤ 0
            total_credit = total_credit_eff if (total_credit_eff > 0 or not raw_mode) else total_credit_mid
            if total_credit <= 0:
                continue

            # ✅ correct max loss: only the widest wing counts
            max_width_used = max(put_width, call_width)
            max_loss_val = calc_max_loss(max_width_used, total_credit, contracts)

            # ✅ breakeven range uses TOTAL credit/share
            credit_per_share = total_credit / (100 * contracts)
            lower_breakeven = float(short_put["strike"]) - credit_per_share
            upper_breakeven = float(short_call["strike"]) + credit_per_share

            # ✅ POP estimate: average of wings (simple, consistent with earlier verticals)
            pop_put = calc_pop(short_put["strike"], spot_price, put_width, put_credit_eff,
                               max_loss_val, "put", short_put.get("delta", None), contracts=contracts)
            pop_call = calc_pop(short_call["strike"], spot_price, call_width, call_credit_eff,
                                max_loss_val, "call", short_call.get("delta", None), contracts=contracts)
            pop_avg = round((pop_put + pop_call) / 2, 1)

            # filters unless raw
            if not raw_mode:
                if max_loss_val > max_loss or pop_avg < min_pop:
                    continue

            trades.append({
                "Strategy": "Iron Condor",
                "Expiry": expiry,
                "DTE": dte,
                "Trade": (
                    f"Sell {short_put['strike']} / Buy {long_put['strike']} PUT + "
                    f"Sell {short_call['strike']} / Buy {long_call['strike']} CALL"
                ),
                # keep both credits visible (effective = realistic-fallback-mid)
                "Credit (Realistic)": round(total_credit_eff, 2),
                "Credit ($)": round(total_credit_mid, 2),
                "Max Loss ($)": round(max_loss_val, 2),
                "POP %": pop_avg,
                "Breakeven": round(lower_breakeven, 2),
                "Contracts": contracts,
                "Spot": round(spot_price, 2),
            })
 
    return pd.DataFrame(trades)



