import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# 🔗 Nova shared modules
from nova_rules import NOVA_RULES
from nova_utils import scan_verticals, scan_condors, style_table

# ----------------------------
# Clean option chain data
# ----------------------------
def clean_chain(df):
    df = df.copy()
    df = df[(df["bid"] >= 0.01) & (df["ask"] >= 0.01)]  # remove zero/garbage
    df = df[df["ask"] >= df["bid"]]  # skip inverted markets
    if "openInterest" in df.columns:
        df = df[df["openInterest"] > 0]
    return df

# ----------------------------
# Page + API init
# ----------------------------
st.set_page_config(
    page_title="Nova Options Scanner",
    page_icon="favicon.png",   # ✅ uses your favicon
    layout="centered"
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
if "auto_summary" not in st.session_state:
    st.session_state.auto_summary = None

# ----------------------------
# Scanner UI
# ----------------------------
st.title("📊 Nova Options Scanner")
st.caption("Version 2025-10-04")

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
min_pop   = st.slider("Minimum POP (%)", 0, 95, 70, 1)
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
            puts = clean_chain(opt_chain.puts)
            trades, _ = scan_verticals(
                puts, spot_price, expiry_selected,
                dte, T, max_width, max_loss, min_pop,
                raw_mode, "put", contracts
            )
        elif spread_type == "Bear Call":
            calls = clean_chain(opt_chain.calls)
            trades, _ = scan_verticals(
                calls, spot_price, expiry_selected,
                dte, T, max_width, max_loss, min_pop,
                raw_mode, "call", contracts
            )
        else:  # ✅ Iron Condor
            puts = clean_chain(opt_chain.puts)
            calls = clean_chain(opt_chain.calls)
            trades, _ = scan_condors(
                {"puts": puts, "calls": calls},  # pass clean legs
                spot_price, expiry_selected,
                dte, T, max_width, max_loss, min_pop,
                raw_mode, contracts
            )

        if trades is not None and not trades.empty:
            trades["Symbol"] = ticker_input
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
            st.session_state.last_trades_df = None
            st.session_state.last_trades_records = None
            st.session_state.last_meta = None
            st.session_state.auto_summary = None
            st.warning("⚠️ No trades passed filters.")

# ----------------------------
# Display Results & Auto-Summary
# ----------------------------
def render_results(trades_df, min_pop_val):
    available = [c for c in [
        "Strategy","Expiry","DTE","Trade",
        "Credit (Realistic)","Credit ($)","Max Loss ($)",
        "POP %","Breakeven","Distance %","Contracts","Spot"
    ] if c in trades_df.columns]

    if "Credit (Realistic)" in trades_df.columns:
        trades_df = trades_df.sort_values(by="Credit (Realistic)", ascending=False)

    st.success(f"✅ Found {len(trades_df)} {st.session_state.last_meta['strategy']} candidates")
    st.dataframe(style_table(trades_df[available], min_pop_val), width="stretch")

def build_summary_prompt():
    meta = st.session_state.last_meta
    recs = st.session_state.last_trades_records
    if not meta or not recs:
        return "You are Nova. No scan results available."
    return NOVA_RULES + f"\nLatest scan results: {recs}"

if just_scanned and st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, min_pop)
elif st.session_state.last_trades_df is not None:
    render_results(st.session_state.last_trades_df, st.session_state.last_meta["min_pop"])

if st.session_state.last_trades_df is not None:
    prompt = build_summary_prompt()
    st.markdown("### 🧠 Nova’s Auto Take")
    if just_scanned:
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}]
            )
            st.session_state.auto_summary = r.choices[0].message.content
        except Exception as e:
            st.session_state.auto_summary = f"(Nova API error: {e})"
    if st.session_state.auto_summary:
        st.write(st.session_state.auto_summary)

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
        messages = [
            {"role": "system", "content": NOVA_RULES},
            {"role": "system", "content": f"Scan meta: {meta}"},
            {"role": "system", "content": f"Latest scanner results (up to 20 rows): {trades_context}"}
        ] + st.session_state.nova_chat
    else:
        messages = [{"role": "system",
                     "content": "You are Nova. No scan results available. Ask Denny to run a scan first."}]
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        reply = r.choices[0].message.content
    except Exception as e:
        reply = f"(Nova API error: {e})"
    st.session_state.nova_chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(f"**Nova:** {reply}")



























































































































