import yfinance as yf
import pandas as pd
import math
from datetime import datetime

def bs_delta(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0:
        return None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if option_type == "call":
            delta = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        else:
            delta = -0.5 * (1 - math.erf(d1 / math.sqrt(2)))
        return delta
    except Exception:
        return None

def fetch_spreads(ticker_symbol, max_width=1.5, max_loss=300, min_pop=80):
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    if not expirations:
        print(f"No options data for {ticker_symbol}")
        return

    expiry = expirations[0]  # nearest expiry
    opt_chain = ticker.option_chain(expiry)
    puts = opt_chain.puts
    spot_price = ticker.history(period="1d")["Close"].iloc[-1]

    expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
    days_to_exp = (expiry_date - datetime.now()).days
    T = days_to_exp / 365.0

    results = []
    rejects = []

    for i in range(len(puts) - 1):
        short = puts.iloc[i]
        long = puts.iloc[i + 1]

        width = long["strike"] - short["strike"]
        credit = short["bid"] - long["ask"]
        max_loss_calc = width * 100 - credit * 100
        sigma = short["impliedVolatility"]

        delta = bs_delta(S=spot_price, K=short["strike"], T=T, r=0.05, sigma=sigma, option_type="put")
        pop = round((1 - abs(delta)) * 100, 2) if delta else None

        # Apply filters
        if width <= 0 or width > max_width:
            rejects.append(("width", short["strike"], width))
            continue
        if credit <= 0:
            rejects.append(("credit", short["strike"], credit))
            continue
        if max_loss_calc > max_loss:
            rejects.append(("max_loss", short["strike"], max_loss_calc))
            continue
        if delta is None:
            rejects.append(("delta", short["strike"], None))
            continue
        if pop < min_pop:
            rejects.append(("POP", short["strike"], pop))
            continue

        results.append({
            "strategy": "Bull Put Spread",
            "expiry": expiry,
            "short_leg": f"{short['strike']}P",
            "long_leg": f"{long['strike']}P",
            "width": round(width, 2),
            "credit_$": round(credit * 100, 2),
            "max_loss_$": round(max_loss_calc, 2),
            "delta": round(delta, 3),
            "POP%": pop
        })

    return pd.DataFrame(results), rejects

if __name__ == "__main__":
    trades, rejects = fetch_spreads("SPY")

    if trades is not None and not trades.empty:
        print("\n✅ Candidate Trades (first 10):\n")
        print(trades.head(10).to_string(index=False))
    else:
        print("⚠️ No trades passed filters.\n")

    print("\n🔍 Sample Rejections (first 10):")
    for r in rejects[:10]:
        print(f"Reason: {r[0]} | Strike: {r[1]} | Value: {r[2]}")




