# app.py
import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from nova_rules import NOVA_RULES
from nova_utils import scan_verticals, scan_condors, style_table

# ----------------------------
# Page + API init
# ----------------------------
st.set_page_config(page_title="Nova Options Scanner", layout="centered")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Session State
# ----------------------------
for key, default in [
    ("nova_chat", []),
    ("last_trades_df", None),
    ("last_trades_records", None),
    ("last_meta", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------
# Scanner UI
# ----------------------------
st.title("📊 Nova Options Scanner")

ticker_input = st.text_input("Enter Ticker Symbol", "NVDA").upper()
exp_dates, spot_price, ticker_obj = [], None, None
if ticker_input:
    try:
        ticker_obj = yf.Ticker(ticker_input)
        exp_dates = ticker_obj.options or []
        hist = ticker_obj.history(period="1d")
        spot_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if spot_price:
            st.subheader(f"📈 {ticker_input} Current Price: ${spot_price:,.2f}")
    except Exception:
        exp_dates, spot_price, ticker_obj = [], None, None

expiry_selected = st.selectbox("Select Expiration Date", exp_dates) if exp_dates else None
if not exp_dates:
    st.warning("⚠️ Could not fetch expiration dates for this ticker.")

max_width = st.slider("Max Spread Width ($)", 0.5, 5.0, 2.5, 0.5)
max_loss  = st.slider("Max Loss ($)", 50, 1000, 300, 50)
min_pop   = st.slider("Minimum POP (%)", 50, 95, 80, 1)
contracts = st.slider("Contracts", 1, 10, 1)

raw_mode   = st.checkbox("🔎 Show Raw (ignore filters)")
spread_type = st.radio("Choose Strategy", ["Bull Put", "Bear Call", "Iron Condor"])

# ----------------------------
# Run Scan
# ----------------------------
just_scanned = False
if st.button("Scan") and expiry_selected and ticker_obj and spot_price:
    with st.spinner("🔍 Scanning options..."):
        opt_chain = ticker_obj.option_chain(expiry_selected)
        expiry_date = datetime.strptime(expiry_selected, "%Y-%m-%d")
        dte = (expiry_date - datetime.now()).days
        T = max(dte, 0) / 365.0

        if spread_type == "Bull Put":
            trades, _ = scan_verticals(opt_chain.puts, spot_price, expiry_selected,
                                       dte, T, max_width, max_loss, min_pop,
                                       raw_mode, "put", contracts)
        elif spread_type == "Bear Call":
            trades, _ = scan_verticals(opt_chain.calls, spot_price, expiry_selected,
                                       dte, T, max_width, max_loss, min_pop,
                                       raw_mode, "call", contracts)
        else:
            trades = scan_condors(opt_chain, spot_price, expiry_selected,
                                  dte, T, max_width, max_loss, min_pop,
                                  raw_mode, contracts)

        if trades is not None and not trades.empty:
            trades = trades.copy()
            trades["Symbol"] = ticker_input
            # ensure numeric columns are numeric
            for col in ["Width ($)", "Credit ($)", "Max Loss ($)", "POP %",
                        "Breakeven", "Distance %", "Delta"]:
                if col in trades.columns:
                    trades[col] = pd.to_numeric(trades[col], errors="coerce")

            st.session_state.last_trades_df = trades
            st.session_state.last_trades_records = trades.to_dict(orient="records")
            st.session_state.last_meta = {
                "ticker": ticker_input,
                "expiry": expiry_selected,
                "dte": int(dte),
                "min_pop": int(min_pop),
                "spot": float(spot_price),
                "strategy": spread_type,
                "contracts": int(contracts),
            }
            just_scanned = True
        else:
            st.warning("⚠️ No trades passed filters.")

# ----------------------------
# Display Results & Auto-Summary
# ----------------------------
def render_results(trades_df: pd.DataFrame, min_pop_val: float):
    cols = ["Strategy", "Expiry", "DTE", "Trade", "Width ($)", "Credit ($)",
            "Max Loss ($)", "POP %", "Breakeven", "Distance %", "Delta", "Contracts", "Symbol"]
    available = [c for c in cols if c in trades_df.columns]
    st.success(f"✅ Found {len(trades_df)} {st.session_state.last_meta['strategy']} candidates")
    st.dataframe(style_table(trades_df[available], min_pop_val), width="stretch")

def build_summary_prompt():
    meta = st.session_state.last_meta
    recs = st.session_state.last_trades_records
    if not meta or not recs:
        return "You are Nova. No scan results available."
    sample = recs[:12]  # keep it small to avoid token limits
    return NOVA_RULES + f"\nLatest scanner snapshot (up to 12 rows): {sample}"

if just_scanned and st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, min_pop)
elif st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, st.session_state.last_meta["min_pop"])

if st.session_state.last_trades_df is not None:
    st.markdown("### 🧠 Nova’s Auto Take")
    if just_scanned:
        try:
            prompt = build_summary_prompt()
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
            )
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
        trades_context = st.session_state.last_trades_records[:12]
        meta = st.session_state.last_meta
        messages = [
            {"role": "system", "content": NOVA_RULES},
            {"role": "system", "content": f"Scan meta: {meta}"},
            {"role": "system", "content": f"Latest scanner results (up to 12 rows): {trades_context}"},
        ] + st.session_state.nova_chat
    else:
        messages = [{"role": "system", "content": "You are Nova. No scan results available. Ask Denny to run a scan first."}]
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        reply = r.choices[0].message.content
    except Exception as e:
        reply = f"(Nova API error: {e})"
    st.session_state.nova_chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(f"**Nova:** {reply}")














































































































