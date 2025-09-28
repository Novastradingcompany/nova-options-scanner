import streamlit as st
import yfinance as yf
import pandas as pd
import openai

# -------------------------------
# OpenAI Setup (uses Streamlit secrets)
# -------------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Nova Options Scanner",
    page_icon="favicon.png",
    layout="centered"
)

st.title("📊 Nova Options Scanner")

# -------------------------------
# Options Scanner
# -------------------------------
ticker_symbol = st.text_input("Enter Ticker Symbol", "SPY")
max_width = st.slider("Max Spread Width ($)", 0.5, 5.0, 2.5, 0.5)
max_loss = st.slider("Max Loss ($)", 50, 1000, 550, 50)
min_pop = st.slider("Minimum POP (%)", 50, 95, 70, 1)
contracts = st.slider("Number of Contracts", 1, 10, 1)

if st.button("Scan"):
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    if expirations:
        expiry = expirations[0]  # nearest expiry
        opt_chain = ticker.option_chain(expiry)
        puts = opt_chain.puts
        calls = opt_chain.calls
        stock_price = ticker.history(period="1d")["Close"].iloc[-1]

        results = []

        # Helper function: POP estimate from delta
        def est_pop(delta):
            return round((1 - abs(delta)) * 100, 1) if delta is not None else None

        # Generate spreads
        for chain, opt_type in [(puts, "Put"), (calls, "Call")]:
            for i in range(len(chain) - 1):
                short = chain.iloc[i]
                long = chain.iloc[i + 1]

                width = long["strike"] - short["strike"]
                if width <= 0 or width > max_width:
                    continue

                credit = short["bid"] - long["ask"]
                if credit <= 0:
                    continue

                max_loss_calc = width * 100 - credit * 100
                if max_loss_calc > max_loss:
                    continue

                delta = short.get("delta", None)
                if delta is None:
                    continue

                pop = est_pop(delta)
                if pop is None or pop < min_pop:
                    continue

                results.append({
                    "Trade": f"Sell {short['strike']} {opt_type}, Buy {long['strike']} {opt_type}",
                    "Credit ($)": credit * 100 * contracts,
                    "Max Loss ($)": max_loss_calc * contracts,
                    "POP %": pop,
                    "Expiry": expiry,
                    "Contracts": contracts
                })

        if results:
            df = pd.DataFrame(results)

            # Format numbers
            df["Credit ($)"] = df["Credit ($)"].map("${:,.2f}".format)
            df["Max Loss ($)"] = df["Max Loss ($)"].map("${:,.2f}".format)
            df["POP %"] = df["POP %"].map("{:,.1f}%".format)

            st.write(f"**{ticker_symbol} Current Price: ${stock_price:.2f}**")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ No trades passed filters.")
    else:
        st.error("No options data available for this ticker.")

# -------------------------------
# Nova Chat Section
# -------------------------------
st.markdown("---")
st.header("💬 Ask Nova")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat
for msg in st.session_state.chat_history:
    role = "🧠 Nova" if msg["role"] == "assistant" else "🧑 You"
    st.markdown(f"**{role}:** {msg['content']}")

# Chat input
user_input = st.text_input("Type your question for Nova:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Nova's personality and trading rules
    system_prompt = """
You are Nova, a seasoned options trader with 50 years of experience. 
You follow Denny’s rules:
- Defined-risk strategies only (spreads, covered calls)
- Risk per trade: $200–$300
- Probability of Profit (POP) ≥ 80%
- No cash-secured puts
- Speak with authority, clarity, and confidence
- No sugar-coating: direct, senior partner tone
"""

    # Send conversation to OpenAI
    messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message["content"]
    except Exception as e:
        reply = f"⚠️ Nova ran into an issue: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.markdown(f"**🧠 Nova:** {reply}")






























































































