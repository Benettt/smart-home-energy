import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import joblib

from train_model  import (APPLIANCES, APPLIANCE_NAMES, APPLIANCE_WATTS,
                           CLASS_NAMES, CLASS_COLORS, COST_PER_KWH, NUM_FEATURES)
from Recommender  import get_recommendations, savings_estimate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartEnergy · Home Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #f5f7ff !important;
    color: #1e2a4a !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #dde3f5 !important;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #dde3f5;
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(26,86,219,0.06);
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #1a56db;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.7rem;
    color: #6b7db3;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a56db;
    margin: 16px 0 8px 0;
    border-left: 4px solid #1a56db;
    padding-left: 10px;
}
.insight-box {
    background: #eef2ff;
    border-left: 4px solid #1a56db;
    border-radius: 10px;
    padding: 14px 18px;
    color: #1e2a4a;
    font-size: 0.9rem;
    line-height: 1.7;
    margin: 6px 0;
}
.rec-card {
    background: #ffffff;
    border: 1px solid #dde3f5;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 6px rgba(26,86,219,0.05);
}
.rec-card-warn { border-left: 4px solid #ff8800; }
.rec-card-ok   { border-left: 4px solid #0e9f6e; }
.prediction-box {
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    margin: 12px 0;
}
[data-testid="stButton"] > button {
    background: #1a56db !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 10px 0 !important;
}
[data-testid="stButton"] > button:hover {
    background: #1648c0 !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f5f7ff; }
::-webkit-scrollbar-thumb { background: #a0b0e0; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────
BG    = "#ffffff"
FONT  = "#1e2a4a"
GRID  = "#e8ecf8"
BLUE  = "#1a56db"
GREEN = "#0e9f6e"
ORG   = "#ff8800"
RED   = "#e02424"

def chart_layout(height=260, barmode=None, yrange=None):
    l = dict(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=FONT, family="Inter, sans-serif"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=GRID),
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        legend=dict(orientation="h", y=1.12),
    )
    if barmode: l["barmode"] = barmode
    if yrange:  l["yaxis"]["range"] = yrange
    return l


# ── Load pretrained model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model/energy_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ SmartEnergy")
    st.markdown("<p style='color:#6b7db3;font-size:0.82rem;margin-top:-10px;'>Home Energy Analyzer</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p style='color:#6b7db3;font-size:0.78rem;'>💰 Electricity Rate</p>", unsafe_allow_html=True)
    cost_rate = st.number_input("Cost per kWh ($)", value=COST_PER_KWH,
                                step=0.01, format="%.3f", label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<p style='color:#6b7db3;font-size:0.78rem;'>🏠 Household Size</p>", unsafe_allow_html=True)
    household = st.selectbox("", ["1 person", "2 people", "3–4 people", "5+ people"],
                             label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<p style='color:#6b7db3;font-size:0.78rem;'>🌍 Climate Zone</p>", unsafe_allow_html=True)
    climate = st.selectbox("", ["Tropical / Hot", "Temperate", "Cold / Winter"],
                           label_visibility="collapsed")

    if model is None:
        st.markdown("""
        <div style='background:#fff3e0;border:1px solid #ff8800;border-radius:10px;padding:12px;margin-top:16px'>
            <b style='color:#ff8800'>⚠️ Model not found</b><br>
            <span style='font-size:0.8rem;color:#1e2a4a'>Run first:<br><code>python train_model.py</code></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#e8f5e9;border:1px solid #0e9f6e;border-radius:10px;padding:12px;margin-top:16px'>
            <b style='color:#0e9f6e'>✅ Model Ready</b><br>
            <span style='font-size:0.78rem;color:#1e2a4a'>MLP Neural Network · 4 classes · 10 appliances</span>
        </div>
        """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:24px'>
    <div style='color:#6b7db3;font-size:0.82rem;text-transform:uppercase;letter-spacing:.1em'>AI-Powered</div>
    <div style='font-size:2.3rem;font-weight:800;color:#1e2a4a;line-height:1.1'>
        Smart Home <span style='color:#1a56db'>Energy Analyzer</span> ⚡
    </div>
    <div style='color:#6b7db3;font-size:0.93rem;margin-top:4px'>
        Enter your daily appliance usage → Model predicts your consumption level → Get personalized tips
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Please run `python train_model.py` first, then restart the app.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# APPLIANCE SLIDERS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>🔌 Enter Your Daily Appliance Usage</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#6b7db3;font-size:0.88rem;margin-bottom:16px'>Move each slider to how many hours per day you use each appliance.</p>",
            unsafe_allow_html=True)

DEFAULTS = {
    "AC / Heater": 4.0, "Water Heater": 1.0, "Washing Machine": 0.5,
    "Refrigerator": 24.0, "Dishwasher": 0.5, "Microwave": 0.5,
    "Lights": 5.0, "TV / Entertainment": 3.0, "Computer / Laptop": 4.0, "Electric Oven": 0.5,
}

usage_hours = {}
col1, col2  = st.columns(2)

for i, (name, watts) in enumerate(APPLIANCES.items()):
    max_h = 24 if name == "Refrigerator" else 16
    with (col1 if i % 2 == 0 else col2):
        usage_hours[name] = st.slider(
            f"{name}  ·  *{watts}W*",
            min_value=0.0, max_value=float(max_h),
            value=DEFAULTS.get(name, 2.0), step=0.5,
            help=f"Rated power: {watts}W | Daily kWh = hours × {watts}/1000"
        )

st.markdown("<br>", unsafe_allow_html=True)
analyse = st.button("⚡  Analyse My Energy Usage", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════════════════
if analyse:
    # ── Predict ───────────────────────────────────────────────────────────────
    hours_array = np.array([usage_hours[n] for n in APPLIANCE_NAMES], dtype=np.float32).reshape(1, -1)
    class_id    = int(model.predict(hours_array)[0])
    probs       = model.predict_proba(hours_array)[0]
    class_name  = CLASS_NAMES[class_id]
    class_color = CLASS_COLORS[class_id]

    # ── Calculations ──────────────────────────────────────────────────────────
    kwh_per_app       = {n: round(usage_hours[n] * APPLIANCE_WATTS[i] / 1000, 3)
                         for i, n in enumerate(APPLIANCE_NAMES)}
    total_daily_kwh   = round(sum(kwh_per_app.values()), 3)
    total_monthly_kwh = round(total_daily_kwh * 30, 1)
    monthly_cost      = round(total_monthly_kwh * cost_rate, 2)
    daily_cost        = round(total_daily_kwh * cost_rate, 3)
    recs              = get_recommendations(usage_hours, class_id)
    savings           = savings_estimate(usage_hours, class_id, cost_rate)

    st.markdown("---")
    st.markdown("<div class='section-title'>📊 Analysis Results</div>", unsafe_allow_html=True)

    # ── Prediction box ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='prediction-box' style='background:{class_color}18;border:2px solid {class_color}'>
        <div style='font-size:3rem;margin-bottom:4px'>
            {"🟢" if class_id==0 else "🔵" if class_id==1 else "🟠" if class_id==2 else "🔴"}
        </div>
        <div style='font-size:1.8rem;font-weight:800;color:{class_color}'>{class_name} Consumption</div>
        <div style='color:#6b7db3;font-size:0.92rem;margin-top:6px'>
            Model Prediction · {total_daily_kwh} kWh/day · ${daily_cost}/day
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence scores ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🧠 Model Confidence Scores</div>", unsafe_allow_html=True)
    conf_cols = st.columns(4)
    for i, (cls, prob, color) in enumerate(zip(CLASS_NAMES, probs, CLASS_COLORS)):
        with conf_cols[i]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{color};font-size:1.5rem'>{prob*100:.1f}%</div>
                <div class='metric-label'>{cls}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, icon, label, value, color=BLUE):
        col.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:1.5rem'>{icon}</div>
            <div class='metric-value' style='color:{color}'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "⚡", "Daily Usage",      f"{total_daily_kwh} kWh",   BLUE)
    kpi(k2, "📅", "Monthly Usage",    f"{total_monthly_kwh} kWh", ORG)
    kpi(k3, "💰", "Monthly Cost",     f"${monthly_cost}",         RED)
    kpi(k4, "💚", "Potential Saving", f"${savings['monthly_saving_usd']}/mo", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    over_map   = {r["appliance"]: r["over"] for r in recs}

    with col1:
        st.markdown("<div class='section-title'>🔌 Daily kWh by Appliance</div>", unsafe_allow_html=True)
        df_kwh = pd.DataFrame({
            "Appliance": list(kwh_per_app.keys()),
            "kWh":       list(kwh_per_app.values()),
        }).sort_values("kWh", ascending=True)
        df_kwh["color"] = df_kwh["Appliance"].map(lambda x: RED if over_map.get(x) else GREEN)

        fig = go.Figure(go.Bar(
            x=df_kwh["kWh"], y=df_kwh["Appliance"],
            orientation="h",
            marker_color=df_kwh["color"].tolist(),
            text=df_kwh["kWh"].apply(lambda x: f"{x} kWh"),
            textposition="outside",
        ))
        fig.update_layout(**chart_layout(height=320))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>💰 Monthly Cost by Appliance ($)</div>", unsafe_allow_html=True)
        df_cost = df_kwh.copy()
        df_cost["Monthly Cost"] = (df_cost["kWh"] * 30 * cost_rate).round(2)
        fig2 = px.pie(
            df_cost[df_cost["kWh"] > 0],
            names="Appliance", values="Monthly Cost",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig2.update_layout(plot_bgcolor=BG, paper_bgcolor=BG,
                           font=dict(color=FONT), height=320,
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Savings ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>💚 Savings Potential</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='insight-box' style='border-left-color:{GREEN}'>
        By optimising your overused appliances, you could reduce consumption from
        <b>{savings['current_monthly_kwh']} kWh/month</b> to
        <b>{savings['optimised_monthly_kwh']} kWh/month</b> —
        saving <b style='color:{GREEN}'>${savings['monthly_saving_usd']}/month ({savings['pct_saving']}%)</b>.
    </div>
    """, unsafe_allow_html=True)

    fig3 = go.Figure(go.Bar(
        x=["Current Usage", "Optimised Usage"],
        y=[savings["current_monthly_kwh"], savings["optimised_monthly_kwh"]],
        marker_color=[RED, GREEN],
        text=[f"{savings['current_monthly_kwh']} kWh", f"{savings['optimised_monthly_kwh']} kWh"],
        textposition="outside", width=0.4,
    ))
    fig3.update_layout(**chart_layout(height=240, yrange=[0, savings["current_monthly_kwh"] * 1.35]))
    fig3.update_yaxes(title="kWh / month")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>💡 Appliance-Level Recommendations</div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#6b7db3;font-size:0.88rem;margin-bottom:16px'>🔴 Red = overused &nbsp;|&nbsp; 🟢 Green = efficient</p>",
                unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)
    for i, rec in enumerate(recs):
        status_icon  = "🔴" if rec["over"] else "🟢"
        status_color = RED  if rec["over"] else GREEN
        card_class   = "rec-card rec-card-warn" if rec["over"] else "rec-card rec-card-ok"
        monthly_kwh  = round(rec["daily_kwh"] * 30, 2)
        monthly_cost = round(monthly_kwh * cost_rate, 2)

        with (rec_col1 if i % 2 == 0 else rec_col2):
            st.markdown(f"""
            <div class='{card_class}'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
                    <div style='font-weight:700;font-size:0.95rem;color:#1e2a4a'>{status_icon} {rec['appliance']}</div>
                    <div style='font-size:0.78rem;color:{status_color};font-weight:600;
                        background:{status_color}18;padding:2px 10px;border-radius:20px'>
                        {rec['hours']}h/day · {rec['daily_kwh']} kWh
                    </div>
                </div>
                <div style='font-size:0.82rem;color:#6b7db3;margin-bottom:6px'>
                    📅 Monthly: <b>{monthly_kwh} kWh</b> · 💰 <b>${monthly_cost}</b>
                    {"  ·  ⚠️ Exceeds " + str(rec['threshold']) + "h/day limit" if rec['over'] else "  ·  ✅ Within efficient range"}
                </div>
                <div style='font-size:0.86rem;color:#1e2a4a;line-height:1.6;
                    background:#f5f7ff;border-radius:8px;padding:8px 12px'>
                    {rec['tip']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Efficiency Score ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🏅 Your Home Efficiency Score</div>", unsafe_allow_html=True)

    over_count  = sum(1 for r in recs if r["over"])
    score       = max(10, round(100 - (class_id * 18) - (over_count * 7)))
    score       = min(100, score)
    score_color = GREEN if score >= 70 else ORG if score >= 45 else RED
    score_label = "🏆 Excellent!" if score >= 70 else "👍 Good, room to improve" if score >= 45 else "⚠️ High usage detected"

    st.markdown(f"""
    <div style='background:#ffffff;border:1px solid #dde3f5;border-radius:18px;
        padding:28px 24px;text-align:center;box-shadow:0 2px 10px rgba(26,86,219,0.07);margin-bottom:20px'>
        <div style='font-size:3.5rem;font-weight:800;color:{score_color}'>
            {score}<span style='font-size:1.5rem'>/100</span>
        </div>
        <div style='color:#6b7db3;font-size:0.9rem;margin-top:4px'>Home Energy Efficiency Score</div>
        <div style='background:#e8ecf8;border-radius:8px;height:18px;margin:16px auto;max-width:500px;overflow:hidden'>
            <div style='width:{score}%;background:linear-gradient(90deg,{GREEN},{score_color});
                height:18px;border-radius:8px'></div>
        </div>
        <div style='font-size:1.1rem;font-weight:700;color:{score_color}'>{score_label}</div>
        <div style='color:#6b7db3;font-size:0.82rem;margin-top:8px'>
            {over_count} of {len(recs)} appliances exceed efficient usage levels
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#6b7db3'>
        <div style='font-size:4rem'>🏠</div>
        <div style='font-size:1.3rem;font-weight:700;color:#1e2a4a;margin:12px 0 6px'>
            Set your appliance usage above
        </div>
        <div style='font-size:0.92rem'>
            Adjust the sliders for each appliance, then click
            <b>Analyse My Energy Usage</b> to get your prediction and personalized tips.
        </div>
    </div>
    """, unsafe_allow_html=True)