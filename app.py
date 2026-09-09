"""
app.py — Professional Streamlit Dashboard for the Trading Engine
================================================================
B.Tech Project: "Predicting Cross-Sectional Stock Returns Using
Technical Indicators: A Machine Learning Approach"

Authors: Som Nainwal, Nachiket Chondhikar, Atharva Mahajan

Launch:  streamlit run app.py
"""

import os, sys, io, tempfile, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="TI Signal Engine — BTP Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core_engine import run_engine, load_data

# ═══════════════════════════════════════════════════════════════════
#  CUSTOM CSS — dark professional trading-platform theme
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f1128 50%, #0a0a1a 100%);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a3e, #12122b);
    border: 1px solid rgba(100, 100, 255, 0.15);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #8888cc !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e0e0ff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d24, #111133) !important;
    border-right: 1px solid #222255;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #aaaaee;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4040cc, #6060ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #5050dd, #7070ff);
    box-shadow: 0 4px 15px rgba(80,80,255,0.4);
    transform: translateY(-1px);
}

/* DataFrames */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 2px dashed #3333aa;
    border-radius: 12px;
    padding: 10px;
}

/* Footer */
.footer-credit {
    text-align: center;
    color: #555580;
    font-size: 0.78rem;
    padding: 30px 0 10px;
    border-top: 1px solid #1a1a44;
    margin-top: 40px;
}
.footer-credit a { color: #7777cc; text-decoration: none; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1a4e 0%, #0d0d2a 100%);
    border: 1px solid rgba(100,100,255,0.12);
    border-radius: 16px;
    padding: 30px 35px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-banner h1 {
    color: #d0d0ff;
    font-size: 1.8rem;
    margin: 0 0 4px;
    font-weight: 700;
}
.hero-banner p {
    color: #8888bb;
    font-size: 0.95rem;
    margin: 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #15153a;
    border-radius: 8px 8px 0 0;
    color: #9999cc;
    border: 1px solid #222255;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1e1e55 !important;
    color: #ccccff !important;
    border-bottom: 2px solid #6060ff;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 TI Signal Engine")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload OHLCV Dataset",
        type=["csv", "xlsx", "xls"],
        help="Any CSV or Excel file with Open, High, Low, Close columns"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Engine Parameters")
    train_p = st.slider("Train window (rows)", 60, 500, 120, 10)
    test_p  = st.slider("Test window (rows)", 6, 60, 12, 3)
    dz_hi   = st.slider("Dead zone HIGH", 0.50, 0.80, 0.65, 0.05)
    dz_lo   = st.slider("Dead zone LOW", 0.20, 0.50, 0.35, 0.05)

    st.markdown("---")
    run_btn = st.button("🚀  Run Engine", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <h1>📈 Technical Indicator Signal Engine</h1>
    <p>Predicting Cross-Sectional Stock Returns Using Technical Indicators
    — A Machine Learning Approach</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════

if uploaded is None and not run_btn:
    # Landing state
    st.info("👈  Upload an OHLCV dataset (CSV or Excel) and press **Run Engine** to begin.")
    st.markdown("""
    **How it works:**
    1. Upload any CSV/Excel file with `Open, High, Low, Close` columns
    2. The engine auto-detects the timeframe and computes ~30 technical indicators
    3. An expanding-window backtest runs the **ensemble** strategy
       alongside top **individual** indicator strategies
    4. View comparative metrics, equity curves, and download the full report
    """)

elif uploaded is not None:
    # Save uploaded file to a temp location
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()

    # Show auto-detected info
    try:
        preview_df = load_data(tmp.name)
        freq = pd.infer_freq(preview_df.index) or "Unknown"
    except Exception:
        freq = "Unknown"
        preview_df = None

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("📄 File", uploaded.name)
    col_b.metric("⏱ Detected Freq", freq)
    col_c.metric("📊 Rows", f"{len(preview_df):,}" if preview_df is not None else "—")

    if run_btn:
        # Patch CFG with sidebar values
        from core_engine import CFG
        CFG["TRAIN_PERIODS"] = train_p
        CFG["TEST_PERIODS"]  = test_p
        CFG["DEAD_ZONE_HIGH"]= dz_hi
        CFG["DEAD_ZONE_LOW"] = dz_lo

        with st.spinner("⏳ Running engine — computing indicators & backtesting …"):
            try:
                result = run_engine(tmp.name)
            except Exception as e:
                st.error(f"Engine failed: {e}")
                st.stop()

        st.success("✅ Engine run complete!")
        st.markdown("---")

        # ── TABS ──────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "📊 Metrics", "📈 Equity Curve", "📋 Trade Log"
        ])

        comp = result["comparison_df"]
        ens_trades = result["ensemble_trades"]

        # ── TAB 1: METRICS ────────────────────────────────────────
        with tab1:
            st.subheader("Comparative Performance")

            # Top KPI cards for ensemble
            if "Ensemble" in comp.index:
                ens = comp.loc["Ensemble"]
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Win Rate", f"{ens['Win Rate']:.1%}")
                k2.metric("Sharpe Ratio", f"{ens['Sharpe']:.3f}")
                k3.metric("Max Drawdown", f"{ens['Max DD']:.2%}")
                k4.metric("Total Return", f"{ens['Total Return']:+.2%}")
                k5.metric("Dir. Accuracy", f"{ens['Dir. Accuracy']:.1%}")

            st.markdown("")

            # Full table with colour formatting
            styled = comp.style.format({
                "Win Rate": "{:.2%}",
                "Sharpe": "{:.3f}",
                "Max DD": "{:.2%}",
                "Total Return": "{:+.2%}",
                "Dir. Accuracy": "{:.1%}",
            }).background_gradient(
                subset=["Win Rate"], cmap="RdYlGn", vmin=0.4, vmax=0.7
            ).background_gradient(
                subset=["Sharpe"], cmap="RdYlGn", vmin=-0.5, vmax=1.5
            )
            st.dataframe(styled, use_container_width=True, height=400)

        # ── TAB 2: EQUITY CURVE ───────────────────────────────────
        with tab2:
            st.subheader("Equity Curve — Ensemble vs Buy & Hold")

            if ens_trades is not None and len(ens_trades):
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ens_trades["date"], y=ens_trades["cum_strategy"],
                    name="Ensemble Strategy",
                    line=dict(color="#00d4ff", width=2.5),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
                ))
                fig.add_trace(go.Scatter(
                    x=ens_trades["date"], y=ens_trades["cum_buyhold"],
                    name="Buy & Hold",
                    line=dict(color="#ff6b6b", width=2, dash="dot"),
                ))

                # Add top individual strategies
                colors = ["#00e676","#ffd740","#e040fb","#ff9100","#40c4ff"]
                for i, (name, t) in enumerate(result["individual_trades"].items()):
                    if t is not None and len(t) and "cum_strategy" in t.columns:
                        fig.add_trace(go.Scatter(
                            x=t["date"], y=t["cum_strategy"],
                            name=name,
                            line=dict(color=colors[i % len(colors)],
                                      width=1.3, dash="dash"),
                            opacity=0.7,
                        ))
                    if i >= 4:
                        break

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0a1a",
                    plot_bgcolor="#0f0f2a",
                    height=520,
                    margin=dict(l=50, r=30, t=40, b=40),
                    legend=dict(
                        orientation="h", y=-0.12,
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11),
                    ),
                    yaxis_title="Growth of $1",
                    xaxis_title="Date",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Drawdown subplot
                cum = ens_trades["cum_strategy"]
                dd = (cum - cum.cummax()) / cum.cummax()
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=ens_trades["date"], y=dd,
                    fill="tozeroy", fillcolor="rgba(255,23,68,0.3)",
                    line=dict(color="#ff1744", width=1),
                    name="Drawdown",
                ))
                fig_dd.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0a1a",
                    plot_bgcolor="#0f0f2a",
                    height=200,
                    margin=dict(l=50, r=30, t=10, b=30),
                    yaxis_title="Drawdown",
                    yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.warning("No trades generated — try a larger dataset.")

        # ── TAB 3: TRADE LOG ──────────────────────────────────────
        with tab3:
            st.subheader("Ensemble Trade Log")
            if ens_trades is not None and len(ens_trades):
                display = ens_trades[["date","signal","weighted_sum",
                    "actual_return","portfolio_return","correct",
                    "cum_strategy"]].copy()
                display["signal"] = display["signal"].map(
                    {1: "🟢 BUY", 0: "🔴 SELL"})
                st.dataframe(display, use_container_width=True, height=500)
            else:
                st.info("No trade data available.")

        # ── EXPORT BUTTON ─────────────────────────────────────────
        st.markdown("---")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            comp.to_excel(w, sheet_name="Metrics")
            if ens_trades is not None and len(ens_trades):
                ens_trades.to_excel(w, sheet_name="Trades", index=False)
        buf.seek(0)
        st.download_button(
            "📥 Download Full Excel Report",
            data=buf,
            file_name="engine_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Clean up temp file
    try:
        os.unlink(tmp.name)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer-credit">
    B.Tech Project — <b>Som Nainwal</b>, <b>Nachiket Chondhikar</b>,
    <b>Atharva Mahajan</b><br>
    "Predicting Cross-Sectional Stock Returns Using Technical Indicators:
    A Machine Learning Approach"
</div>
""", unsafe_allow_html=True)
