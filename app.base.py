import yfinance as yf
import pandas as pd
import math
from datetime import datetime
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI   # ✅ Correct import

# --- Load API key ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Black-Scholes Delta ---
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

# --- Vertical Spread Scanner ---
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

# --- Iron Condor Scanner ---
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

# --- Styling ---
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
            nums = [float(v.replace("%","").strip()) for v in val.split("/") ]
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

# --- Streamlit UI ---
st.set_page_config(page_title="Nova Options Scanner", layout="centered")
st.title("📊 Nova Options Scanner")

# --- Session State for Nova Chat ---
if "nova_chat" not in st.session_state:
    st.session_state.nova_chat = []
if "last_trades" not in st.session_state:
    st.session_state.last_trades = None

ticker_input = st.text_input("Enter Ticker Symbol", "NVDA").upper()

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
    max_loss = st.slider("Max Loss ($)", 50, 1000, 550, 50)
    min_pop = st.slider("Minimum POP (%)", 50, 95, 70, 1)
    contracts = st.slider("Contracts", 1, 10, 1)

    raw_mode = st.checkbox("🔎 Show Raw (ignore filters)")
    spread_type = st.radio("Choose Strategy", ["Bull Put", "Bear Call", "Iron Condor"])

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
                column_order = ["Strategy","Expiry","DTE","Trade","Width ($)","Credit ($)","Max Loss ($)","POP %","Breakeven","Distance %","Delta","Contracts"]
                if "Spot" in trades.columns: 
                    column_order.append("Spot")
                trades = trades[column_order]
                st.success(f"✅ Found {len(trades)} {spread_type} candidates")
                st.dataframe(style_table(trades, min_pop), width="stretch")

                # Store the latest trades for Nova chat context
                st.session_state.last_trades = trades.to_dict(orient="records")

                # --- Nova Auto-Summary ---
                summary_prompt = f"""
                You are Nova, Denny’s senior trading partner.
                Trades scanned for {ticker_input} expiring {expiry_selected} ({dte} DTE).
                Apply Rebuild Phase rules strictly:
                - POP ≥ 80%
                - Max Loss ≤ 300
                - Favor closest expiry with good distance
                Give a decisive recommendation from these trades:
                {st.session_state.last_trades}
                """
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":summary_prompt}]
                    )
                    nova_reply = response.choices[0].message.content
                    st.markdown("### 🧠 Nova’s Auto Take")
                    st.write(nova_reply)
                except Exception as e:
                    st.error(f"Nova API error: {e}")

                # --- Ask Nova Again Button ---
                if st.button("Ask Nova Again"):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"system","content":summary_prompt}]
                        )
                        nova_reply = response.choices[0].message.content
                        st.markdown("### 🔄 Nova’s Fresh Take")
                        st.write(nova_reply)
                    except Exception as e:
                        st.error(f"Nova API error: {e}")
            else:
                st.warning("⚠️ No trades passed filters.")
                if rejects:
                    st.write(pd.DataFrame(rejects[:10], columns=["Reason", "Strike", "Value"]))

# --- 💬 Always-Visible Nova Chat ---
st.markdown("---")
st.header("💬 Talk with Nova")

# Display chat history
for msg in st.session_state.nova_chat:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Nova:** {msg['content']}")

# Input box for new message
user_msg = st.text_area("Your message to Nova", key="nova_input", height=80)

if st.button("Send", key="send_nova"):
    if user_msg.strip():
        st.session_state.nova_chat.append({"role": "user", "content": user_msg})

        context = "No scan yet." if not st.session_state.last_trades else st.session_state.last_trades
        messages = [
            {"role": "system",
             "content": "You are Nova, Denny's senior trading partner. Use the latest scan data to help."},
            {"role": "system",
             "content": f"Latest scanner results: {context}"},
        ] + st.session_state.nova_chat

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            nova_reply = resp.choices[0].message.content
            st.session_state.nova_chat.append({"role": "assistant", "content": nova_reply})
        except Exception as e:
            st.error(f"Nova API error: {e}")



































































































