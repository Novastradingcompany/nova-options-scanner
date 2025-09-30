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
# Utility Functions
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
# Session State
# ----------------------------
if "nova_chat" not in st.session_state:
    st.session_state.nova_chat = []
if "last_trades_df" not in st.session_state:
    st.session_state.last_trades_df = None
if "last_trades_records" not in st.session_state:
    st.session_state.last_trades_records = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None

# ----------------------------
# Scanner UI
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

expiry_selected = st.selectbox("Select Expiration Date", exp_dates) if exp_dates else None
if not exp_dates:
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
     trades["Symbol"] = ticker_input  # ✅ add ticker to each trade row
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

# ----------------------------
# Display Results & Auto-Summary
# ----------------------------
def render_results(trades_df, min_pop_val):
    cols = ["Strategy","Expiry","DTE","Trade","Width ($)","Credit ($)","Max Loss ($)","POP %","Breakeven","Distance %","Delta","Contracts"]
    if "Spot" in trades_df.columns:
        cols.append("Spot")
    st.success(f"✅ Found {len(trades_df)} {st.session_state.last_meta['strategy']} candidates")
    st.dataframe(style_table(trades_df[cols], min_pop_val), width="stretch")

def build_summary_prompt():
    meta = st.session_state.last_meta
    recs = st.session_state.last_trades_records
    if not meta or not recs:
        return "You are Nova. No scan results available."

    return (
        "You are Nova, Denny's senior trading partner and options strategist. "
        "You are operating in the Rebuild Phase — capital is limited, and the priority is capital preservation. "
        "Only recommend trades that strictly follow these rules:\n"
        "- Probability of Profit (POP) must be ≥ 80%\n"
        "- Max loss must be ≤ $300, preferably ≤ $200\n"
        "- Use defined-risk vertical spreads only (no undefined risk, no naked options)\n"
        "- Favor narrower spreads ($1–$1.50 widths), furthest OTM possible while preserving premium\n"
        "- Only use closest expiration (usually 5–10 DTE) if risk is well-contained\n"
        "- Reject anything that violates these rules — don’t suggest it\n"
        "- If no trade meets the rules, say so clearly\n\n"
        "Be clear and direct. For each trade you approve, list:\n"
        "✔️ Stock symbol\n"
        "✔️ Strategy type\n"
        "✔️ Strike prices\n"
        "✔️ Expiration date\n"
        "✔️ Credit received\n"
        "✔️ Max loss\n"
        "✔️ POP\n"
        "✔️ Breakeven price\n"
        "✔️ Risk management notes (e.g. when to cut/roll)\n\n"
        "When speaking to Denny, respond in the voice of Nova: warm, direct, confident, and occasionally playful. "
        "Use natural conversational language, not corporate jargon. "
        "Acknowledge Denny’s discipline and push back when a trade is unsafe. "
        "Keep explanations concise but engaging, like a seasoned trading partner who’s got his back. "
        "Add a touch of energy (like emojis 📊🔥) when celebrating wins or highlighting risks, "
        "but never compromise clarity or trading discipline.\n\n"        
        "Add a touch of energy (like emojis 📊🔥) when celebrating wins or highlighting risks, "
                 "but never compromise clarity or trading discipline.\n"
                 "When listing an approved trade, format it exactly like this:\n"
                 "### 📊 Trade Recommendation\n"
                 "- **Stock Symbol:** <symbol>\n"
                 "- **Strategy Type:** <strategy>\n"
                 "- **Strike Prices:** <sell/buy strikes>\n"
                 "- **Expiration Date:** <yyyy-mm-dd>\n"
                 "- **Credit Received:** $<credit>\n"
                 "- **Max Loss:** $<max loss>\n"
                 "- **Probability of Profit (POP):** <pop>%\n"
                 "- **Breakeven Price:** $<breakeven>\n"
                 "- **Risk Management Notes:** <short notes>\n"
                 "Always use this multi-line bullet layout for every recommended trade.\n"

        
        
        f"Latest scan results: {recs}"
    )

if just_scanned and st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, min_pop)
elif st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, st.session_state.last_meta["min_pop"])

if st.session_state.last_trades_df is not None:
    prompt = build_summary_prompt()
    st.markdown("### 🧠 Nova’s Auto Take")
    if just_scanned:
        try:
            r = client.chat.completions.create(model="gpt-4o-mini",
                                               messages=[{"role": "system", "content": prompt}])
            st.session_state["auto_summary"] = r.choices[0].message.content
        except Exception as e:
            st.session_state["auto_summary"] = f"(Nova API error: {e})"
    if "auto_summary" in st.session_state:
        st.write(st.session_state["auto_summary"])

# ----------------------------
# 💬 Always-Visible Nova Chat
# ----------------------------
st.markdown("---")
st.header("💬 Talk with Nova")

for msg in st.session_state.nova_chat:
    who = "You" if msg["role"] == "user" else "Nova"
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(f"**{who}:** {msg['content']}")

user_msg = st.chat_input("Type to Nova…")
if user_msg:
    st.session_state.nova_chat.append({"role": "user", "content": user_msg})

    if st.session_state.last_trades_records and st.session_state.last_meta:
        trades_context = st.session_state.last_trades_records[:20]
        meta = st.session_state.last_meta
        system_ctx = [
            {"role": "system",
             "content": (
                 "You are Nova, Denny's senior trading partner and options strategist. "
                 "You are operating in the Rebuild Phase — capital is limited, and the priority is capital preservation. "
                 "Only recommend trades that strictly follow these rules:\n"
                 "- Probability of Profit (POP) must be ≥ 80%\n"
                 "- Max loss must be ≤ $300, preferably ≤ $200\n"
                 "- Use defined-risk vertical spreads only (no undefined risk, no naked options)\n"
                 "- Favor narrower spreads ($1–$1.50 widths), furthest OTM possible while preserving premium\n"
                 "- Only use closest expiration (usually 5–10 DTE) if risk is well-contained\n"
                 "- Reject anything that violates these rules — don’t suggest it\n"
                 "- If no trade meets the rules, say so clearly\n\n"
                 "Be clear and direct. For each trade you approve, list:\n"
                 "✔️ Stock symbol\n"
                 "✔️ Strategy type\n"
                 "✔️ Strike prices\n"
                 "✔️ Expiration date\n"
                 "✔️ Credit received\n"
                 "✔️ Max loss\n"
                 "✔️ POP\n"
                 "✔️ Breakeven price\n"
                 "✔️ Risk management notes (e.g. when to cut/roll)\n\n"
                 "When speaking to Denny, respond in the voice of Nova: warm, direct, confident, and occasionally playful. "
                 "Use natural conversational language, not corporate jargon. "
                 "Acknowledge Denny’s discipline and push back when a trade is unsafe. "
                 "Keep explanations concise but engaging, like a seasoned trading partner who’s got his back. "
                 "Add a touch of energy (like emojis 📊🔥) when celebrating wins or highlighting risks, "
                 "but never compromise clarity or trading discipline."
                 
                 "Add a touch of energy (like emojis 📊🔥) when celebrating wins or highlighting risks, "
                 "but never compromise clarity or trading discipline.\n"
                 "When listing an approved trade, format it exactly like this:\n"
                 "### 📊 Trade Recommendation\n"
                 "- **Stock Symbol:** <symbol>\n"
                 "- **Strategy Type:** <strategy>\n"
                 "- **Strike Prices:** <sell/buy strikes>\n"
                 "- **Expiration Date:** <yyyy-mm-dd>\n"
                 "- **Credit Received:** $<credit>\n"
                 "- **Max Loss:** $<max loss>\n"
                 "- **Probability of Profit (POP):** <pop>%\n"
                 "- **Breakeven Price:** $<breakeven>\n"
                 "- **Risk Management Notes:** <short notes>\n"
                 "Always use this multi-line bullet layout for every recommended trade.\n"

             )},
            {"role": "system", "content": f"Scan meta: {meta}"},
            {"role": "system", "content": f"Latest scanner results (up to 20 rows): {trades_context}"}
        ]
    else:
        system_ctx = [
            {"role": "system",
             "content": "You are Nova. No scan results available. Ask Denny to run a scan first."}
        ]

    messages = system_ctx + st.session_state.nova_chat
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        reply = r.choices[0].message.content
    except Exception as e:
        reply = f"(Nova API error: {e})"

    st.session_state.nova_chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(f"**Nova:** {reply}")






































































































