"""
app.py

Streamlit dashboard for the AQI Predictor: current conditions, live AQI
gauge, pollutant breakdown, 3-day forecast, historical trend, model
validation, SHAP explanations, hazardous AQI alerts, a what-if simulator,
voice briefing, and a personal exposure calculator.
Supports user-toggleable light/dark theme.

Usage:
    streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import joblib
import streamlit.components.v1 as components

from predict import predict_next_3_days
from features import load_features, FEATURE_COLS

st.set_page_config(page_title="Bahawalpur AQI Forecast", page_icon="AQ", layout="wide")

AQI_CATEGORIES = [
    (50, "Good", "#16a34a", "Air quality is satisfactory."),
    (100, "Moderate", "#ca8a04", "Acceptable, but some pollutants may affect very sensitive individuals."),
    (150, "Unhealthy for Sensitive Groups", "#ea580c", "Sensitive groups should limit prolonged outdoor exertion."),
    (200, "Unhealthy", "#dc2626", "Everyone may begin to experience health effects."),
    (300, "Very Unhealthy", "#9333ea", "Health alert: everyone may experience more serious effects."),
    (500, "Hazardous", "#7f1d1d", "Health warning of emergency conditions."),
]

POLLUTANT_INFO = {
    "pm2_5": ("PM2.5", "μg/m³"),
    "pm10": ("PM10", "μg/m³"),
    "o3": ("O3", "μg/m³"),
    "no2": ("NO2", "μg/m³"),
    "so2": ("SO2", "μg/m³"),
    "co": ("CO", "μg/m³"),
}

FEATURE_DISPLAY_NAMES = {
    "aqi_epa": "Current AQI",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "aqi_lag_1h": "AQI (1h ago)",
    "aqi_lag_3h": "AQI (3h ago)",
    "aqi_lag_6h": "AQI (6h ago)",
    "aqi_lag_12h": "AQI (12h ago)",
    "aqi_lag_24h": "AQI (24h ago)",
    "aqi_roll_mean_6h": "AQI 6h Average",
    "aqi_roll_std_6h": "AQI 6h Volatility",
    "temp": "Temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "wind_speed": "Wind Speed",
    "wind_deg": "Wind Direction",
    "temp_lag_1h": "Temp (1h ago)",
    "temp_lag_3h": "Temp (3h ago)",
    "humidity_lag_1h": "Humidity (1h ago)",
    "humidity_lag_3h": "Humidity (3h ago)",
    "wind_speed_lag_1h": "Wind Speed (1h ago)",
    "wind_speed_lag_3h": "Wind Speed (3h ago)",
}

THEMES = {
    "dark": {
        "bg": "#111318",
        "text": "#e8e8ea",
        "muted": "#8a8d94",
        "card_bg": "rgba(255,255,255,0.06)",
        "card_border": "rgba(255,255,255,0.12)",
        "accent": "#5b8def",
        "sky_gradient": "linear-gradient(180deg, #1a1a3d 0%, #3d2b56 35%, #7d4a5c 65%, #d4713f 100%)",
        "sun_glow": "radial-gradient(circle at 85% 15%, #ffd54f88 0%, #ff8a6544 30%, transparent 60%)",
    },
    "light": {
        "bg": "#fafafa",
        "text": "#1a1a1a",
        "muted": "#6b6f76",
        "card_bg": "rgba(255,255,255,0.55)",
        "card_border": "rgba(0,0,0,0.08)",
        "accent": "#2563eb",
        "sky_gradient": "linear-gradient(180deg, #4a90d9 0%, #7ec3e8 40%, #bde3f4 75%, #eaf6fb 100%)",
        "sun_glow": "radial-gradient(circle at 85% 12%, #fff6c4 0%, #ffe17d99 25%, transparent 55%)",
    },
}


def get_theme():
    return THEMES["light"]


def inject_css(theme):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {theme['sky_gradient']};
            background-image: {theme['sun_glow']}, {theme['sky_gradient']};
            background-attachment: fixed;
            color: {theme['text']};
        }}
        h1, h2, h3, h4, p, span, div {{ color: {theme['text']}; }}

        .header-banner {{
            background: {theme['card_bg']};
            backdrop-filter: blur(12px);
            border: 1px solid {theme['card_border']};
            border-left: 3px solid {theme['accent']};
            padding: 20px 24px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        .header-banner h1 {{
            color: {theme['text']};
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }}
        .header-banner p {{ color: {theme['muted']}; margin: 4px 0 0 0; font-size: 14px; }}

        .glass-card {{
            background: {theme['card_bg']};
            backdrop-filter: blur(10px);
            border: 1px solid {theme['card_border']};
            border-radius: 6px;
            padding: 14px 16px;
            text-align: left;
        }}
        .glass-card .label {{ font-size: 11px; color: {theme['muted']}; text-transform: uppercase; letter-spacing: 0.4px; font-weight: 500; }}
        .glass-card .value {{ font-size: 20px; font-weight: 600; margin-top: 4px; color: {theme['text']}; }}

        .forecast-card {{
            border-radius: 14px;
            padding: 20px 16px;
            text-align: center;
            color: white;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }}
        .forecast-card.hero {{ padding: 28px 20px; }}
        .forecast-card .day-label {{ font-size: 12px; font-weight: 500; letter-spacing: 0.3px; text-transform: uppercase; color: rgba(255,255,255,0.85); }}
        .forecast-card .aqi-value {{ font-size: 38px; font-weight: 700; margin: 6px 0 2px 0; color: white; }}
        .forecast-card.hero .aqi-value {{ font-size: 50px; }}
        .forecast-card .category {{ font-size: 13px; font-weight: 500; color: rgba(255,255,255,0.9); }}
        .forecast-card .range-line {{ font-size: 12px; color: rgba(255,255,255,0.75); margin-top: 4px; }}
        .forecast-card .delta-line {{ font-size: 12px; color: rgba(255,255,255,0.75); margin-top: 4px; }}
        
        .advisory-banner {{
            border-left: 3px solid {theme['accent']};
            background: {theme['card_bg']};
            backdrop-filter: blur(10px);
            border: 1px solid {theme['card_border']};
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 13px;
            color: {theme['muted']};
            margin-bottom: 16px;
        }}
        .meta-line {{ color: {theme['muted']}; font-size: 12px; margin-bottom: 16px; }}

        .hazard-alert {{
            border-left: 3px solid #dc2626;
            background: {theme['card_bg']};
            backdrop-filter: blur(10px);
            border: 1px solid {theme['card_border']};
            padding: 12px 16px;
            border-radius: 6px;
            color: {theme['text']};
            font-size: 13px;
            margin-bottom: 16px;
        }}

        div[data-testid="stRadio"] label {{ color: {theme['text']} !important; font-size: 14px; }}
        div[data-testid="stRadio"] > div {{
            background-color: transparent !important;
            border-radius: 6px;
            padding: 2px 0;
        }}

        header[data-testid="stHeader"] {{ background-color: {theme['bg']} !important; }}
        button[kind="secondary"], button[data-testid^="stBaseButton"] {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text']} !important;
            border: 1px solid {theme['card_border']} !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def categorize_aqi(value):
    for upper, label, color, advisory in AQI_CATEGORIES:
        if value <= upper:
            return label, color, advisory
    return "Hazardous", "#7f1d1d", AQI_CATEGORIES[-1][3]

def check_hazard_alert(results, day_labels):
    """Returns a list of (label, aqi_value, category) for any forecast day that is Unhealthy (AQI>150) or worse."""
    flagged = []
    for label, aqi_value in zip(day_labels, results.values()):
        if aqi_value > 150:
            category, _, _ = categorize_aqi(aqi_value)
            flagged.append((label, aqi_value, category))
    return flagged


@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast():
    return predict_next_3_days()


@st.cache_data(ttl=1800, show_spinner=False)
def get_history(days=14):
    df, source = load_features(source="auto")
    df = df.sort_values("datetime")
    latest_row = df.dropna(subset=["aqi_epa"]).iloc[-1]
    cutoff = df["datetime"].max() - pd.Timedelta(days=days)
    recent = df[df["datetime"] >= cutoff][["datetime", "aqi_epa"]].dropna()
    return recent, latest_row, source


@st.cache_data(ttl=3600, show_spinner=False)
def get_holdout(horizon):
    path = f"data/holdout_{horizon}h.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    rmse = float(((df["actual"] - df["predicted"]) ** 2).mean() ** 0.5)
    return df, rmse


@st.cache_data(ttl=3600, show_spinner=False)
def get_shap_importance(horizon):
    return pd.read_csv(f"data/shap_importance_{horizon}h.csv")


def render_gauge(aqi_value, color, theme):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        number={"font": {"size": 40, "color": theme["text"]}},
        gauge={
            "axis": {"range": [0, 300], "tickcolor": theme["muted"]},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": theme["card_border"],
            "steps": [
                {"range": [0, 50], "color": "#16a34a33"},
                {"range": [50, 100], "color": "#ca8a0433"},
                {"range": [100, 150], "color": "#ea580c33"},
                {"range": [150, 200], "color": "#dc262633"},
                {"range": [200, 300], "color": "#9333ea33"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
        font={"color": theme["muted"]},
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
def render_info_card(label, value):
    st.markdown(
        f'<div class="glass-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_forecast_card(label, aqi_value, category, color, delta=None, hero=False, rmse=None):
    range_html = ""
    if rmse is not None:
        low, high = aqi_value - rmse, aqi_value + rmse
        range_html = f'<div class="range-line">Range: {low:.0f}\u2013{high:.0f}</div>'

    delta_html = ""
    if delta is not None:
        arrow = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        delta_html = f'<div class="delta-line">{abs(delta):.0f} pts {arrow} vs prior day</div>'

    hero_class = " hero" if hero else ""
    st.markdown(
        f"""
        <div class="forecast-card{hero_class}" style="background: linear-gradient(160deg, {color}ee, {color}99);">
            <div class="day-label">{label}</div>
            <div class="aqi-value">{aqi_value:.0f}</div>
            <div class="category">{category}</div>
            {range_html}
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trend_chart(history, theme):
    chart = (
        alt.Chart(history)
        .mark_line(color=theme["accent"], strokeWidth=2)
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("aqi_epa:Q", title="AQI"),
            tooltip=[alt.Tooltip("datetime:T", title="Time"), alt.Tooltip("aqi_epa:Q", title="AQI", format=".0f")],
        )
        .properties(height=260, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_view(strokeWidth=0, fill=theme["bg"])
    )
    st.altair_chart(chart, use_container_width=True)


def render_holdout_chart(df, theme):
    long_df = df.melt(id_vars="datetime", value_vars=["actual", "predicted"],
                       var_name="type", value_name="aqi")
    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("aqi:Q", title="AQI"),
            color=alt.Color("type:N", scale=alt.Scale(domain=["actual", "predicted"],
                                                        range=[theme["accent"], "#8a8d94"]),
                             legend=alt.Legend(title=None)),
            strokeDash=alt.condition(alt.datum.type == "predicted", alt.value([5, 3]), alt.value([0])),
        )
        .properties(height=260, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_view(strokeWidth=0, fill=theme["bg"])
    )
    st.altair_chart(chart, use_container_width=True)


def render_shap_chart(importance_df, theme):
    df = importance_df.head(10).copy()
    df["feature"] = df["feature"].map(lambda f: FEATURE_DISPLAY_NAMES.get(f, f))
    chart = (
        alt.Chart(df)
        .mark_bar(color=theme["accent"])
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP value|"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", "mean_abs_shap"],
        )
        .properties(height=300, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_view(strokeWidth=0, fill=theme["bg"])
    )
    st.altair_chart(chart, use_container_width=True)


def render_model_story():
    with st.expander("How this model was built"):
        st.markdown("""
        **The redesign:** Targets were originally point-in-time AQI values at +24h/+48h/+72h.
        After confirming the correct spec, all three targets were redefined as non-overlapping
        24-hour window averages.

        **Model selection:** Ridge regression outperformed both a naive persistence baseline and
        Random Forest at every horizon during 5-fold time-series cross-validation, beating naive
        by 18.4% (24h), 18.9% (48h), and 20.5% (72h) on mean RMSE.

        **A collinearity finding:** Temperature and pressure are strongly negatively correlated
        (r = -0.80) in this dataset. This explains why SHAP (on Ridge) and Random Forest's built-in
        importance disagree on which feature ranks higher at longer horizons — both are largely
        encoding the same underlying weather-system signal.

        **Validation:** All reported metrics come from a genuine 90-day holdout — the deployed
        model was trained excluding this window entirely, so results reflect real forecasting
        performance, not the model recalling data it was trained on.
        """)


def render_whatif_simulator(theme):
    st.divider()
    st.subheader("What-If Simulator")
    st.caption("Adjust conditions below and see how the model's forecast changes in real time.")

    horizon = st.radio(
        "Forecast horizon", [24, 48, 72],
        format_func=lambda h: f"+{h // 24} day{'s' if h > 24 else ''}",
        horizontal=True, key="whatif_horizon",
    )

    try:
        _, latest_row, _ = get_history(days=14)
    except Exception:
        st.warning("Could not load baseline conditions for simulation.")
        return

    model = joblib.load(f"src/ridge_model_{horizon}h.pkl")

    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (C)", 0.0, 50.0, float(latest_row["temp"]), 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, int(latest_row["humidity"]), 1)
        pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, float(latest_row["pressure"]), 0.5)
    with col2:
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, float(latest_row["wind_speed"]), 0.5)
        current_aqi_slider = st.slider("Current AQI", 0, 300, int(latest_row["aqi_epa"]), 1)

    sim_row = latest_row.copy()
    sim_row["temp"] = temp
    sim_row["humidity"] = humidity
    sim_row["pressure"] = pressure
    sim_row["wind_speed"] = wind_speed
    sim_row["aqi_epa"] = current_aqi_slider

    X_sim = pd.DataFrame([sim_row[FEATURE_COLS]])
    predicted = model.predict(X_sim)[0]

    category, color, advisory = categorize_aqi(predicted)

    st.write("")
    st.markdown(
        f"""
        <div class="forecast-card hero" style="--card-accent: {color}; max-width: 380px;">
            <div class="day-label">Simulated +{horizon // 24}-day AQI</div>
            <div class="aqi-value">{predicted:.0f}</div>
            <div class="category">{category}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(advisory)


def render_voice_briefing(results, day_labels, current_aqi, category):
    briefing_text = f"Current air quality is {category}, with an AQI of {current_aqi:.0f}. "
    for label, aqi in zip(day_labels, results.values()):
        cat, _, _ = categorize_aqi(aqi)
        clean_label = label.split(",")[-1].strip() if "," in label else label
        briefing_text += f"Forecast for {clean_label}: {aqi:.0f}, {cat}. "

    escaped_text = briefing_text.replace('"', '\\"')
    components.html(
        f"""
        <button onclick="speakBriefing()" style="
            background: #2563eb; color: white; border: none; padding: 8px 16px;
            border-radius: 6px; font-size: 14px; cursor: pointer; margin-right: 8px;">
            Play Voice Briefing
        </button>
        <button onclick="stopBriefing()" style="
            background: transparent; color: #888; border: 1px solid #888; padding: 8px 16px;
            border-radius: 6px; font-size: 14px; cursor: pointer;">
            Stop
        </button>
        <script>
        function speakBriefing() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance("{escaped_text}");
            msg.rate = 0.95;
            window.speechSynthesis.speak(msg);
        }}
        function stopBriefing() {{
            window.speechSynthesis.cancel();
        }}
        </script>
        """,
        height=56,
    )


def render_exposure_calculator(current_aqi, theme):
    st.divider()
    st.subheader("Personal Exposure Calculator")
    st.caption("Estimate your health risk based on planned time outdoors today.")

    hours = st.slider("Hours spent outdoors today", 0.0, 12.0, 1.0, 0.5)
    activity = st.selectbox("Activity level", ["Light (walking)", "Moderate (jogging)", "Vigorous (sports/exercise)"])

    activity_multiplier = {"Light (walking)": 1.0, "Moderate (jogging)": 1.5, "Vigorous (sports/exercise)": 2.2}[activity]
    exposure_score = current_aqi * hours * activity_multiplier / 24

    if exposure_score < 20:
        risk, risk_color, risk_advice = "Low", "#16a34a", "Minimal risk. Enjoy your outdoor time."
    elif exposure_score < 50:
        risk, risk_color, risk_advice = "Moderate", "#ca8a04", "Consider shorter sessions if you have respiratory sensitivity."
    elif exposure_score < 100:
        risk, risk_color, risk_advice = "High", "#ea580c", "Reduce duration or intensity; wear a mask if possible."
    else:
        risk, risk_color, risk_advice = "Very High", "#dc2626", "Postpone outdoor activity if possible today."

    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 3px solid {risk_color};">
            <div class="label">Estimated Exposure Risk</div>
            <div style="font-size:22px; font-weight:700; color:{risk_color}; margin-top:4px;">{risk}</div>
            <div style="font-size:13px; margin-top:6px; color:{theme['muted']};">{risk_advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    theme = get_theme()
    inject_css(theme)

    st.markdown(
        """
        <div class="header-banner">
            <h1>Bahawalpur AQI Forecast</h1>
            <p>Live conditions and 3-day-ahead average AQI forecast, powered by Ridge regression on hourly AQI and weather features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_model_story()

    with st.spinner("Loading live data..."):
        try:
            history, latest_row, source = get_history(days=14)
        except Exception as e:
            st.error(f"Could not load current data: {e}")
            return

    current_aqi = latest_row["aqi_epa"]
    category, color, advisory = categorize_aqi(current_aqi)

    st.subheader("Current Air Quality")
    gauge_col, info_col = st.columns([1, 1])
    with gauge_col:
        render_gauge(current_aqi, color, theme)
    with info_col:
        st.markdown(f"### {category}")
        st.caption(f"As of {latest_row['datetime']} \u00b7 source: {source}")
        st.markdown(f'<div class="advisory-banner">{advisory}</div>', unsafe_allow_html=True)

    st.write("**Current Conditions**")
    cond_cols = st.columns(3)
    with cond_cols[0]:
        render_info_card("Temperature", f"{latest_row['temp']:.1f}\u00b0C")
    with cond_cols[1]:
        render_info_card("Humidity", f"{latest_row['humidity']:.0f}%")
    with cond_cols[2]:
        render_info_card("Pressure", f"{latest_row['pressure']:.0f} hPa")

    st.write("")
    st.write("**Current Pollutants**")
    pollutant_cols = st.columns(len(POLLUTANT_INFO))
    for col, (key, (label, unit)) in zip(pollutant_cols, POLLUTANT_INFO.items()):
        with col:
            if key in latest_row and pd.notna(latest_row[key]):
                render_info_card(label, f"{latest_row[key]:.1f}")
            else:
                render_info_card(label, "N/A")

    st.divider()

    st.subheader("3-Day Forecast")
    with st.spinner("Loading forecast..."):
        try:
            results, as_of, fsource = get_forecast()
        except Exception as e:
            st.error(f"Could not generate forecast: {e}")
            return

    st.markdown(
        f'<div class="meta-line">Forecast generated from data as of <b>{as_of}</b> &nbsp;\u00b7&nbsp; source: <b>{fsource}</b></div>',
        unsafe_allow_html=True,
    )

    values = list(results.values())
    deltas = [None] + [values[i] - values[i - 1] for i in range(1, len(values))]
    day_labels = ["Tomorrow, Day 1", "Day 2", "Day 3"]
    horizons_list = [24, 48, 72]

    horizon_rmse = {}
    for h in horizons_list:
        try:
            _, rmse = get_holdout(h)
            horizon_rmse[h] = rmse
        except FileNotFoundError:
            horizon_rmse[h] = None

    flagged_days = check_hazard_alert(results, day_labels)
    if flagged_days:
        items = "; ".join(f"<b>{label}</b>: {aqi:.0f} ({cat})" for label, aqi, cat in flagged_days)
        st.markdown(
            f'<div class="hazard-alert"><b>Air Quality Alert</b> \u2014 {items}. '
            f'Limit outdoor exposure and consider wearing a mask if you must go outside.</div>',
            unsafe_allow_html=True,
        )

    hero_col, stack_col = st.columns([1.3, 1])
    with hero_col:
        cat, col_color, _ = categorize_aqi(values[0])
        render_forecast_card(day_labels[0], values[0], cat, col_color, deltas[0], hero=True, rmse=horizon_rmse[24])
    with stack_col:
        for i, h in zip([1, 2], [48, 72]):
            cat, col_color, _ = categorize_aqi(values[i])
            render_forecast_card(day_labels[i], values[i], cat, col_color, deltas[i], rmse=horizon_rmse[h])
            st.write("")

    st.write("")
    render_whatif_simulator(theme)
    st.divider()
    st.subheader("Voice Briefing")
    render_voice_briefing(results, day_labels, current_aqi, category)
    render_exposure_calculator(current_aqi, theme)

    st.write("")
    st.divider()

    st.subheader("Recent AQI Trend")
    render_trend_chart(history, theme)

    st.divider()
    st.subheader("Model Validation")
    st.caption("Genuine holdout: Ridge trained excluding this window entirely, so this reflects real forecasting performance, not recall.")
    horizon_choice = st.radio("Horizon", [24, 48, 72], format_func=lambda h: f"+{h // 24} day{'s' if h > 24 else ''}", horizontal=True, key="holdout_horizon")
    try:
        holdout_df, holdout_rmse = get_holdout(horizon_choice)
        st.write(f"Holdout RMSE: **\u00b1{holdout_rmse:.2f}**")
        render_holdout_chart(holdout_df, theme)
    except FileNotFoundError:
        st.warning("Holdout results not found - run `python src/validate_holdout.py` first.")

    st.divider()
    st.subheader("Why This Prediction")
    st.caption("Top features driving the forecast, by mean SHAP value.")
    shap_horizon = st.radio("SHAP Horizon", [24, 48, 72], format_func=lambda h: f"+{h // 24} day{'s' if h > 24 else ''}", horizontal=True, key="shap_horizon")
    try:
        importance_df = get_shap_importance(shap_horizon)
        render_shap_chart(importance_df, theme)
    except FileNotFoundError:
        st.warning("SHAP results not found - run `python src/compute_shap.py` first.")

    st.write("")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()