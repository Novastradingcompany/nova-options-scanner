# nova_utils.py
import math
import pandas as pd

# ----------------------------
# Greeks helper
# ----------------------------
def bs_delta(S, K, T, r, sigma, option_type="put"):
    """Very light Black–Scholes delta (normal CDF via math.erf)."""
    try:
        if T <= 0 or not sigma or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        # call delta = N(d1); put delta = N(d1) - 1
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
        return nd1 if option_type == "call" else (nd1 - 1.0)
    except Exception:
        return None


# ----------------------------
# Core scanners
# ----------------------------
def _scan_verticals(options_df, spot_price, expiry, dte, T,
                    max_width, max_loss_cap, min_pop, raw_mode,
                    side, contracts=1, r=0.05):
    """
    Build verticals by pairing consecutive strikes (short i, long i+1).
    Returns a DataFrame of candidate spreads.
    """
    rows = []
    if options_df is None or options_df.empty:
        return pd.DataFrame(rows), []

    # Ensure predictable ordering (ascending strikes)
    df = options_df.sort_values("strike").reset_index(drop=True)

    rejects = []
    for i in range(len(df) - 1):
        short = df.iloc[i]
        long = df.iloc[i + 1]

        # Mid prices
        def mid(x, y):
            try:
                if pd.notnull(x) and pd.notnull(y):
                    return (float(x) + float(y)) / 2.0
            except Exception:
                pass
            return None

        short_mid = mid(short.get("bid"), short.get("ask"))
        long_mid  = mid(long.get("bid"),  long.get("ask"))
        if short_mid is None or long_mid is None:
            rejects.append(("no_mid", (short.get("strike"), long.get("strike"))))
            continue

        width = abs(float(long["strike"]) - float(short["strike"]))
        if width <= 0:
            rejects.append(("bad_width", width))
            continue

        # credit per share
        credit_ps = (short_mid - long_mid)
        # protection: sometimes inverted due to chain anomalies
        if side == "put" and float(short["strike"]) <= float(long["strike"]):
            pass  # OK (sell higher put, buy lower put) – our order below sells lower strike first wording-wise
        if side == "call" and float(short["strike"]) >= float(long["strike"]):
            pass  # OK (sell lower call, buy higher call)

        # total credit in $
        credit_total = float(credit_ps) * 100.0 * int(contracts)
        max_loss = width * 100.0 * int(contracts) - credit_total

        sigma = short.get("impliedVolatility")
        try:
            sigma = float(sigma) if pd.notnull(sigma) else None
        except Exception:
            sigma = None

        delta = bs_delta(
            S=float(spot_price),
            K=float(short["strike"]),
            T=float(T),
            r=float(r),
            sigma=sigma,
            option_type=("put" if side == "put" else "call"),
        )

        pop = None
        if delta is not None:
            # crude POP approximation from delta
            pop = round((1.0 - abs(delta)) * 100.0, 1)
            pop = max(0.0, min(100.0, pop))

        # breakeven per share uses credit per share (not total)
        breakeven = (float(short["strike"]) - credit_ps) if side == "put" else (float(short["strike"]) + credit_ps)

        # distance % from spot to breakeven (positive = further away)
        if side == "put":
            distance_pct = (float(spot_price) - breakeven) / float(spot_price) * 100.0
        else:
            distance_pct = (breakeven - float(spot_price)) / float(spot_price) * 100.0

        # Row
        row = {
            "Strategy": "Bull Put" if side == "put" else "Bear Call",
            "Expiry": expiry,
            "DTE": int(dte),
            "Trade": f"Sell {short['strike']}{'P' if side=='put' else 'C'} / Buy {long['strike']}{'P' if side=='put' else 'C'}",
            "Width ($)": float(width),
            "Credit ($)": float(credit_total),
            "Max Loss ($)": float(max_loss),
            "POP %": float(pop) if pop is not None else None,
            "Breakeven": float(breakeven),
            "Distance %": float(distance_pct),
            "Delta": float(delta) if delta is not None else None,
            "Contracts": int(contracts),
        }

        # Filter if not raw
        if raw_mode is True:
            rows.append(row)
        else:
            if credit_ps <= 0:
                continue
            if max_loss > float(max_loss_cap) * int(contracts):
                continue
            if pop is None or pop < float(min_pop):
                continue
            if width > float(max_width):
                continue
            rows.append(row)

    return pd.DataFrame(rows), rejects


def scan_verticals(options_df, spot_price, expiry, dte, T,
                   max_width, max_loss_cap, min_pop, raw_mode,
                   side, contracts=1):
    return _scan_verticals(options_df, spot_price, expiry, dte, T,
                           max_width, max_loss_cap, min_pop, raw_mode,
                           side, contracts)


def scan_condors(opt_chain, spot_price, expiry, dte, T,
                 max_width, max_loss_cap, min_pop, raw_mode, contracts=1):
    """Simple IC builder from best put/call verticals on opposite sides."""
    if opt_chain is None:
        return pd.DataFrame()

    puts_df = opt_chain.puts
    calls_df = opt_chain.calls

    put_spreads, _ = _scan_verticals(puts_df, spot_price, expiry, dte, T,
                                     max_width, max_loss_cap, min_pop, True, "put", contracts)
    call_spreads, _ = _scan_verticals(calls_df, spot_price, expiry, dte, T,
                                      max_width, max_loss_cap, min_pop, True, "call", contracts)

    rows = []
    if put_spreads.empty or call_spreads.empty:
        return pd.DataFrame(rows)

    for _, p in put_spreads.iterrows():
        if float(p["Breakeven"]) >= float(spot_price):
            continue
        for _, c in call_spreads.iterrows():
            if float(c["Breakeven"]) <= float(spot_price):
                continue
            # only match same width
            if abs(float(p["Width ($)"]) - float(c["Width ($)"])) > 1e-6:
                continue

            total_credit = float(p["Credit ($)"]) + float(c["Credit ($)"])
            width = float(p["Width ($)"])
            max_loss = width * 100.0 * int(contracts) - total_credit

            pop_put = (float(p["POP %"]) / 100.0) if pd.notnull(p["POP %"]) else 0.0
            pop_call = (float(c["POP %"]) / 100.0) if pd.notnull(c["POP %"]) else 0.0
            combo_pop = round(pop_put * pop_call * 100.0, 1)

            # be's (per share credit not needed for condor range here)
            lower_be = float(p["Breakeven"])
            upper_be = float(c["Breakeven"])

            row = {
                "Strategy": "Iron Condor",
                "Expiry": expiry,
                "DTE": int(dte),
                "Trade": f"{p['Trade']}  +  {c['Trade']}",
                "Width ($)": float(width),
                "Credit ($)": float(total_credit),
                "Max Loss ($)": float(max_loss),
                "POP %": float(combo_pop),
                "Breakeven": None,  # range shown via Distance %
                "Distance %": None,  # can compute two-sided distance if desired
                "Delta": None,
                "Contracts": int(contracts),
                "Spot": float(spot_price),
            }

            if raw_mode:
                rows.append(row)
            else:
                if total_credit <= 0:
                    continue
                if max_loss > float(max_loss_cap) * int(contracts):
                    continue
                if combo_pop < float(min_pop):
                    continue
                rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Styling for Streamlit
# ----------------------------
def style_table(df: pd.DataFrame, min_pop: float):
    """
    Returns a pandas Styler with safe numeric formatting and light conditional colors.
    All comparisons are done on numeric columns (no strings).
    """
    # formatting map (assumes numeric columns)
    fmt = {
        "Width ($)": "${:,.2f}".format,
        "Credit ($)": "${:,.2f}".format,
        "Max Loss ($)": "${:,.2f}".format,
        "POP %": "{:,.1f}%".format,
        "Breakeven": "${:,.2f}".format,
        "Distance %": "{:,.1f}%".format,
        "Delta": "{:,.3f}".format,
    }
    # Only include keys that exist in df
    fmt = {k: v for k, v in fmt.items() if k in df.columns}

    styler = df.style.format(fmt)

    # POP coloring
    if "POP %" in df.columns:
        styler = styler.map(
            lambda v: "color: green;" if isinstance(v, (int, float)) and v >= float(min_pop)
            else ("color: orange;" if isinstance(v, (int, float)) and v >= float(min_pop) - 5 else "color: red;"),
            subset=["POP %"],
        )

    # Credit positive = green
    if "Credit ($)" in df.columns:
        styler = styler.map(
            lambda v: "color: green;" if isinstance(v, (int, float)) and v > 0 else "",
            subset=["Credit ($)"],
        )

    # Distance heuristic
    if "Distance %" in df.columns:
        styler = styler.map(
            lambda v: "color: green;" if isinstance(v, (int, float)) and v > 10
            else ("color: orange;" if isinstance(v, (int, float)) and v >= 5 else "color: red;"),
            subset=["Distance %"],
        )

    # Bold Trade
    if "Trade" in df.columns:
        styler = styler.map(lambda _: "font-weight: bold;", subset=["Trade"])

    return styler








