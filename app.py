import yfinance as yf
import pandas as pd
import math
from datetime import datetime
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI   # ✅ Correct import

# ----------------------------
# Page + API init
# ----------------------------
st.set_page_config(page_title="Nova Options Scanner", layout="centered")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Utils
# ----------------------------
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

def scan_verticals(options_df, spot_price, expiry, dte, T, max_width, max_loss, min_pop, raw_mode, side, contracts=1):
    results = []
    rejects = []

    for i in range(len(options_df) - 1):
        short = options_df.iloc[i]
        long = options_df.iloc[i + 1]

        width = abs(long["strike"] - short["strike"])
        short_mid = (short["bid"] + short["ask"]) / 2
        long_mid = (long["bid"] + long["ask"]) / 2
        credit = short_mid - long_mid

        credit_total = credit * 100 * contracts
        max_loss_calc = width * 100 * contracts - credit_total

        sigma = short["impliedVolatility"]
        delta = bs_delta(S=spot_price, K=short["strike"], T=T, r=0.05, sigma=sigma, option_type=side)
        pop = round((1 - abs(delta)) * 100, 2) if delta else None

        if side == "put":
            breakeven_val = short["strike"] - (credit_total / (100 * contracts))
            distance_pct = (spot_price - breakeven_val) / spot_price * 100
        else:
            breakeven_val = short["strike"] + (credit_total / (100 * contracts))
            distance_pct = (breakeven_val - spot_price) / spot_price * 100

        row = {
            "Strategy": "Bull Put" if side == "put" else "Bear Call",
            "Expiry": expiry,
            "DTE": dte,
            "Trade": f"Sell {short['strike']}{'P' if side=='put' else 'C'} / Buy {long['strike']}{'P' if side=='put' else 'C'}",
            "Width ($)": width,
            "Credit ($)": credit_total,
            "Max Loss ($)": max_loss_calc,
            "POP %": pop,
            "Breakeven": f"{breakeven_val:.2f}",
            "Distance %": f"{distance_pct:.1f}%",
            "Delta": delta,
            "Contracts": contracts
        }

        if raw_mode:
            results.append(row)
        else:
            if width <= 0 or width > max_width: continue
            if credit <= 0: continue
            if max_loss_calc > max_loss * contracts: continue
            if delta is None: continue
            if pop < min_pop: continue
            results.append(row)

    return pd.DataFrame(results), rejects

def scan_condors(opt_chain, spot_price, expiry, dte, T, max_width, max_loss, min_pop, raw_mode, contracts=1):
    puts_df = opt_chain.puts
    calls_df = opt_chain.calls

    put_spreads, _ = scan_verticals(puts_df, spot_price, expiry, dte, T, max_width, max_loss, min_pop, True, "put", contracts)
    call_spreads, _ = scan_verticals(calls_df, spot_price, expiry, dte, T, max_width, max_loss, min_pop, True, "call", contracts)

    condors = []
    for _, put in pd.DataFrame(put_spreads).iterrows():
        put_strike = float(put["Trade"].split()[1][:-1])
        if put_strike >= spot_price: continue
        for _, call in pd.DataFrame(call_spreads).iterrows():
            call_strike = float(call["Trade"].split()[1][:-1])
            if call_strike <= spot_price: continue
            if abs(put["Width ($)"] - call["Width ($)"]) > 0.01: continue

            total_credit = put["Credit ($)"] + call["Credit ($)"]
            total_width = put["Width ($)"]
            max_loss_calc = total_width * 100 * contracts - total_credit

            pop_put = put["POP %"] / 100 if put["POP %"] else 0
            pop_call = call["POP %"] / 100 if call["POP %"] else 0
            combined_pop = round(pop_put * pop_call * 100, 2)

            lower_breakeven_val = put_strike - (total_credit / (100 * contracts))
            upper_breakeven_val = call_strike + (total_credit / (100 * contracts))

            condor = {
                "Strategy": "Iron Condor",
                "Expiry": expiry,
                "DTE": dte,
                "Trade": f"Sell {put['Trade']} + Sell {call['Trade']}",
                "Width ($)": total_width,
                "Credit ($)": total_credit,
                "Max Loss ($)": max_loss_calc,
                "POP %": combined_pop,
                "Breakeven": f"{lower_breakeven_val:.2f} / {upper_breakeven_val:.2f}",
                "Distance %": f"{(spot_price - lower_breakeven_val)/spot_price*100:.1f}% / {(upper_breakeven_val-spot_price)/spot_price*100:.1f}%",
                "Contracts": contracts,
                "Spot": spot_price
            }

            if raw_mode:
                condors.append(condor)
            else:
                if max_loss_calc > max_loss * contracts: continue
                if total_credit <= 0: continue
                if combined_pop < min_pop: continue
                condors.append(condor)

    return pd.DataFrame(condors)

def style_table(df, min_pop):
    def color_credit(val): return "color: green;" if val > 0 else ""
    def color_loss(val): return "color: red;" if val > 0 else ""
    def color_pop(val):
        if val is None: return ""
        if val >= min_pop: return "color: green;"
        elif val >= min_pop - 5: return "color: orange;"
        else: return "color: red;"
    def color_distance(val):
        try:
            nums = [float(v.replace("%","").strip()) for v in val.split("/")]
            worst = min(nums)
            if worst > 10: return "color: green;"
            elif worst >= 5: return "color: orange;"
            else: return "color: red;"
        except: return ""
    def bold_trade(val): return "font-weight: bold;"
    return (
        df.style
          .map(color_credit, subset=["Credit ($)"])
          .map(color_loss, subset=["Max Loss ($)"])
          .map(color_pop, subset=["POP %"])
          .map(color_distance, subset=["Distance %"])
          .map(bold_trade, subset=["Trade"])
          .format({
              "Width ($)": "${:,.2f}",
              "Credit ($)": "${:,.2f}",
              "Max Loss ($)": "${:,.2f}",
              "POP %": "{:,.1f}%",
              "Delta": "{:,.3f}",
              "Spot": "${:,.2f}",
              "Breakeven": lambda x: (
                  "$" + x if "/" not in x else " / ".join(["$" + v.strip() for v in x.split("/")])
              )
          })
    )

# ----------------------------
# Session state
# ----------------------------
if "nova_chat" not in st.session_state:
    st.session_state.nova_chat = []  # [{"role":"user"/"assistant","content":str}]
if "last_trades_df" not in st.session_state:
    st.session_state.last_trades_df = None  # pandas DataFrame
if "last_trades_records" not in st.session_state:
    st.session_state.last_trades_records = None  # list of dicts for OpenAI
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None  # {"ticker","expiry","dte","min_pop","spot","strategy"}

# ----------------------------
# Scanner UI (unchanged controls)
# ----------------------------
st.title("📊 Nova Options Scanner")

ticker_input = st.text_input("Enter Ticker Symbol", "NVDA").upper()

exp_dates, spot_price = [], None
if ticker_input:
    try:
        ticker_obj = yf.Ticker(ticker_input)
        exp_dates = ticker_obj.options
        spot_price = ticker_obj.history(period="1d")["Close"].iloc[-1]
        st.subheader(f"📈 {ticker_input} Current Price: ${spot_price:,.2f}")
    except Exception:
        exp_dates, spot_price = [], None

if exp_dates:
    expiry_selected = st.selectbox("Select Expiration Date", exp_dates)
else:
    expiry_selected = None
    st.warning("⚠️ Could not fetch expiration dates for this ticker.")

max_width = st.slider("Max Spread Width ($)", 0.5, 5.0, 2.5, 0.5)
max_loss  = st.slider("Max Loss ($)", 50, 1000, 550, 50)
min_pop   = st.slider("Minimum POP (%)", 50, 95, 70, 1)
contracts = st.slider("Contracts", 1, 10, 1)

raw_mode   = st.checkbox("🔎 Show Raw (ignore filters)")
spread_type = st.radio("Choose Strategy", ["Bull Put", "Bear Call", "Iron Condor"])

# ----------------------------
# Run Scan
# ----------------------------
just_scanned = False
if st.button("Scan") and expiry_selected:
    with st.spinner("🔍 Scanning options..."):
        opt_chain = ticker_obj.option_chain(expiry_selected)
        expiry_date = datetime.strptime(expiry_selected, "%Y-%m-%d")
        dte = (expiry_date - datetime.now()).days
        T = dte / 365.0

        if spread_type == "Bull Put":
            trades, rejects = scan_verticals(opt_chain.puts, spot_price, expiry_selected, dte, T,
                                             max_width, max_loss, min_pop, raw_mode, "put", contracts)
        elif spread_type == "Bear Call":
            trades, rejects = scan_verticals(opt_chain.calls, spot_price, expiry_selected, dte, T,
                                             max_width, max_loss, min_pop, raw_mode, "call", contracts)
        else:
            trades = scan_condors(opt_chain, spot_price, expiry_selected, dte, T,
                                  max_width, max_loss, min_pop, raw_mode, contracts)
            rejects = []

        if trades is not None and not trades.empty:
            # Persist results for future reruns (so Send button won’t “clear” the table)
            st.session_state.last_trades_df = trades.copy()
            st.session_state.last_trades_records = trades.to_dict(orient="records")
            st.session_state.last_meta = {
                "ticker": ticker_input,
                "expiry": expiry_selected,
                "dte": int(dte),
                "min_pop": int(min_pop),
                "spot": float(spot_price) if spot_price is not None else None,
                "strategy": spread_type,
                "contracts": int(contracts)
            }
            just_scanned = True
        else:
            st.warning("⚠️ No trades passed filters.")
            if rejects:
                st.write(pd.DataFrame(rejects[:10], columns=["Reason", "Strike", "Value"]))

# ----------------------------
# Results display (persists across reruns)
# ----------------------------
def render_results(trades_df, min_pop_val):
    # Column ordering logic preserved
    column_order = ["Strategy","Expiry","DTE","Trade","Width ($)","Credit ($)","Max Loss ($)","POP %","Breakeven","Distance %","Delta","Contracts"]
    if "Spot" in trades_df.columns:
        column_order.append("Spot")
    trades_show = trades_df[column_order]
    st.success(f"✅ Found {len(trades_show)} {st.session_state.last_meta['strategy']} candidates")
    # Keep original styling approach
    st.dataframe(style_table(trades_show, min_pop_val), width="stretch")

def build_summary_prompt():
    meta = st.session_state.last_meta
    recs = st.session_state.last_trades_records
    if not meta or not recs:
        return "You are Nova. No scan results available. Ask the user to run a scan."

    # Keep content tight—Nova must give a decisive call using our rules
    header = f"""
You are Nova, Denny’s senior trading partner. Apply Rebuild Phase rules strictly:
- POP ≥ 80%
- Max Loss ≤ 300
- Prefer narrow widths ($1–$1.50) and closest expiry with safe distance
- Risk cap per trade: $200–$300 (prefer ≤ $200)
Meta: Ticker={meta['ticker']}, Expiry={meta['expiry']} ({meta['dte']} DTE), Spot={meta['spot']}, Strategy={meta['strategy']}, Min POP slider={meta['min_pop']}%, Contracts={meta['contracts']}
Return a single decisive recommendation with:
- Exact trade (short/long strikes), credit, POP, max loss, breakeven(s), distance
- Why this is the safest fit for the rules
- Clear entry/exit plan and risk trigger
"""
    # Limit to top 20 records to control token size
    trimmed = recs[:20]
    body = f"Latest scanner results (up to 20 rows): {trimmed}"
    return header + "\n" + body

# If we just scanned, render now; otherwise render from session_state if available
if just_scanned and st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, min_pop)

elif st.session_state.last_trades_df is not None:
    # Re-render persisted results so other button clicks (like Send) don't clear the table
    render_results(st.session_state.last_trades_df, st.session_state.last_meta["min_pop"])

# ----------------------------
# Auto-summary (shown when results exist)
# ----------------------------
if st.session_state.last_trades_df is not None:
    summary_prompt = build_summary_prompt()

    st.markdown("### 🧠 Nova’s Auto Take")
    # Only call on fresh scan to avoid spamming; otherwise show cached or let user refresh
    if just_scanned:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": summary_prompt}]
            )
            nova_reply = response.choices[0].message.content
            st.session_state["auto_summary"] = nova_reply
        except Exception as e:
            st.session_state["auto_summary"] = f"(Nova API error: {e})"

    # Display whatever we have (fresh or cached)
    if "auto_summary" in st.session_state and st.session_state["auto_summary"]:
        st.write(st.session_state["auto_summary"])
    else:
        st.info("Run a scan to generate Nova’s auto-summary.")

    if st.button("🔄 Ask Nova Again (refresh summary)"):
        try:
            response = client.chat_completions.create(  # fallback if SDK alias exists
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": summary_prompt}]
            )
            nova_reply = response.choices[0].message.content
        except Exception:
            # Standard path
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": summary_prompt}]
            )
            nova_reply = response.choices[0].message.content
        st.session_state["auto_summary"] = nova_reply
        st.write(nova_reply)

# ----------------------------
# 💬 Always-Visible Nova Chat
# ----------------------------
st.markdown("---")
st.header("💬 Talk with Nova")

# Render history
for msg in st.session_state.nova_chat:
    role = "You" if msg["role"] == "user" else "Nova"
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(f"**{role}:** {msg['content']}")

# Chat input (auto-clears on send)
user_msg = st.chat_input("Type to Nova…")

if user_msg:
    # Add user message
    st.session_state.nova_chat.append({"role": "user", "content": user_msg})

    # Build context using the latest scan (if any)
    if st.session_state.last_trades_records and st.session_state.last_meta:
        meta = st.session_state.last_meta
        # Keep to 20 rows max to control token size
        trades_context = st.session_state.last_trades_records[:20]
        system_ctx = [
            {"role": "system",
             "content": (
                 "You are Nova, Denny's senior trading partner. Use the latest scan data (below) with Rebuild Phase rules: "
                 "POP ≥ 80%, Max Loss ≤ $300 (prefer ≤ $200), narrow widths, closest expiry with safe distance. "
                 "Be decisive and concrete—give specific strikes, credits, POP, max loss, breakevens, and risk triggers."
             )},
            {"role": "system",
             "content": f"Scan meta: {meta}"},
            {"role": "system",
             "content": f"Latest scanner results (up to 20 rows): {trades_context}"}
        ]
    else:
        system_ctx = [
            {"role": "system",
             "content": "You are Nova. No scan results available. Ask the user to run a scan and state exactly what you need (ticker, expiry)."}
        ]

    # Compose full message list (system context + full running chat history)
    messages = system_ctx + st.session_state.nova_chat

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        nova_reply = resp.choices[0].message.content
    except Exception:
        # Try an alternate alias if present in the SDK
        resp = client.chat_completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        nova_reply = resp.choices[0].message.content

    st.session_state.nova_chat.append({"role": "assistant", "content": nova_reply})
    with st.chat_message("assistant"):
        st.markdown(f"**Nova:** {nova_reply}")




































































































