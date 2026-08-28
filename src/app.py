"""
app.py

Streamlit dashboard for the AQI Predictor: current conditions, live AQI
gauge, pollutant breakdown, 3-day forecast, historical trend, model
validation, SHAP explanations, and hazardous AQI alerts.
Supports user-toggleable light/dark theme with a gradient-mesh background.

Usage:
    streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go

from predict import predict_next_3_days
from features import load_features

st.set_page_config(page_title="Bahawalpur AQI Forecast", page_icon="🌫️", layout="wide")

AQI_CATEGORIES = [
    (50, "Good", "#00c853", "🟢", "Air quality is satisfactory."),
    (100, "Moderate", "#ffd600", "🟡", "Acceptable, but some pollutants may affect very sensitive individuals."),
    (150, "Unhealthy for Sensitive Groups", "#ff9100", "🟠", "Sensitive groups should limit prolonged outdoor exertion."),
    (200, "Unhealthy", "#ff3d00", "🔴", "Everyone may begin to experience health effects."),
    (300, "Very Unhealthy", "#9c27b0", "🟣", "Health alert: everyone may experience more serious effects."),
    (500, "Hazardous", "#6d1b1b", "⚫", "Health warning of emergency conditions."),
]

POLLUTANT_INFO = {
    "pm2_5": ("PM2.5", "μg/m³"),
    "pm10": ("PM10", "μg/m³"),
    "o3": ("O₃", "μg/m³"),
    "no2": ("NO₂", "μg/m³"),
    "so2": ("SO₂", "μg/m³"),
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
        "bg": "#0b0e14",
        "text": "#f1f3f6",
        "muted": "#9aa5b1",
        "card_bg": "rgba(255,255,255,0.06)",
        "card_border": "rgba(255,255,255,0.12)",
        "blob_colors": ["#8a7a6655", "#5c4a3344", "#a89f8f55"],
    },
    "light": {
        "bg": "#f4f6fb",
        "text": "#1a1f29",
        "muted": "#5c6773",
        "card_bg": "rgba(255,255,255,0.55)",
        "card_border": "rgba(0,0,0,0.08)",
        "blob_colors": ["#d4c4a866", "#b8a68844", "#e8dcc866"],
    },
}


def get_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    return THEMES[st.session_state.theme]


def inject_css(theme):
    blob1, blob2, blob3 = theme["blob_colors"]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme['bg']};
            background-image:
                radial-gradient(circle at 15% 20%, {blob1} 0%, transparent 45%),
                radial-gradient(circle at 85% 10%, {blob2} 0%, transparent 40%),
                radial-gradient(circle at 50% 90%, {blob3} 0%, transparent 50%);
            background-attachment: fixed;
            color: {theme['text']};
        }}
        h1, h2, h3, h4, p, span, div {{ color: {theme['text']}; }}

        .header-banner {{
            background: linear-gradient(135deg, #1e3c72cc 0%, #2a5298cc 100%);
            backdrop-filter: blur(12px);
            padding: 28px 32px;
            border-radius: 20px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.15);
        }}
        .header-banner h1 {{ color: white; margin: 0; font-size: 32px; }}
        .header-banner p {{ color: #e2e8f5; margin: 6px 0 0 0; font-size: 15px; }}

        .glass-card {{
            background: {theme['card_bg']};
            border: 1px solid {theme['card_border']};
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 16px 18px;
            text-align: center;
        }}
        .glass-card .label {{ font-size: 12px; color: {theme['muted']}; text-transform: uppercase; letter-spacing: 0.5px; }}
        .glass-card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}

        .forecast-card {{
            border-radius: 18px;
            padding: 22px 18px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .forecast-card.hero {{ padding: 32px 20px; }}
        .forecast-card .day-label {{ font-size: 13px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: rgba(255,255,255,0.85); }}
        .forecast-card .aqi-value {{ font-size: 42px; font-weight: 800; margin: 6px 0; color: white; }}
        .forecast-card.hero .aqi-value {{ font-size: 56px; }}
        .forecast-card .category {{ font-size: 14px; font-weight: 600; color: rgba(255,255,255,0.95); }}

        .advisory-banner {{
            border-left: 5px solid #ff9100;
            background: {theme['card_bg']};
            backdrop-filter: blur(10px);
            padding: 12px 18px;
            border-radius: 10px;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .meta-line {{ color: {theme['muted']}; font-size: 13px; margin-bottom: 20px; }}

        .hazard-alert {{
            background: linear-gradient(135deg, #b71c1c 0%, #7f0000 100%);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 20px;
            color: white;
            font-size: 15px;
            box-shadow: 0 6px 20px rgba(183,28,28,0.4);
        }}
        .hazard-alert b {{ color: white; }}

        /* Theme radio toggle */
        div[data-testid="stRadio"] label {{
            color: {theme['text']} !important;
        }}
        div[data-testid="stRadio"] > div {{
            background-color: {theme['card_bg']} !important;
            border-radius: 10px;
            padding: 4px 8px;
        }}

        /* Top app header (Deploy / menu bar) */
        header[data-testid="stHeader"] {{
            background-color: {theme['bg']} !important;
        }}

        /* Buttons (e.g. Refresh) */
        button[kind="secondary"], button[data-testid^="stBaseButton"] {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text']} !important;
            border: 1px solid {theme['card_border']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def categorize_aqi(value):
    for upper, label, color, icon, advisory in AQI_CATEGORIES:
        if value <= upper:
            return label, color, icon, advisory
    return "Hazardous", "#6d1b1b", "⚫", AQI_CATEGORIES[-1][4]


def check_hazard_alert(results, day_labels):
    """Returns a list of (label, aqi_value, category) for any forecast day that is Unhealthy (AQI>150) or worse."""
    flagged = []
    for label, aqi_value in zip(day_labels, results.values()):
        if aqi_value > 150:
            category, _, _, _ = categorize_aqi(aqi_value)
            flagged.append((label, aqi_value, category))
    return flagged


@st.cache_data(ttl=1800)
def get_forecast():
    return predict_next_3_days()


@st.cache_data(ttl=1800)
def get_history(days=14):
    df, source = load_features(source="auto")
    df = df.sort_values("datetime")
    latest_row = df.dropna(subset=["aqi_epa"]).iloc[-1]
    cutoff = df["datetime"].max() - pd.Timedelta(days=days)
    recent = df[df["datetime"] >= cutoff][["datetime", "aqi_epa"]].dropna()
    return recent, latest_row, source


def render_gauge(aqi_value, color, theme):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        number={"font": {"size": 48, "color": theme["text"]}},
        gauge={
            "axis": {"range": [0, 300], "tickcolor": theme["muted"]},
            "bar": {"color": color},
            "bgcolor": theme["bg"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#00c85333"},
                {"range": [50, 100], "color": "#ffd60033"},
                {"range": [100, 150], "color": "#ff910033"},
                {"range": [150, 200], "color": "#ff3d0033"},
                {"range": [200, 300], "color": "#9c27b033"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["bg"],
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        font={"color": theme["muted"]},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_info_card(label, value):
    st.markdown(
        f'<div class="glass-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_forecast_card(label, aqi_value, category, color, icon, delta=None, hero=False):
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "―")
        delta_html = f'<div style="font-size:12px; color:rgba(255,255,255,0.8); margin-top:6px;">{arrow} {abs(delta):.0f} vs prior day</div>'
    hero_class = " hero" if hero else ""
    st.markdown(
        f"""
        <div class="forecast-card{hero_class}" style="background: linear-gradient(160deg, {color}ee, {color}99);">
            <div class="day-label">{label}</div>
            <div class="aqi-value">{icon} {aqi_value:.0f}</div>
            <div class="category">{category}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trend_chart(history, theme):
    chart = (
        alt.Chart(history)
        .mark_area(
            line={"color": "#4fc3f7", "strokeWidth": 2},
            interpolate="monotone",
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#4fc3f7", offset=0),
                    alt.GradientStop(color="rgba(79,195,247,0.02)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("aqi_epa:Q", title="AQI"),
            tooltip=[alt.Tooltip("datetime:T", title="Time"), alt.Tooltip("aqi_epa:Q", title="AQI", format=".0f")],
        )
        .properties(height=280, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_view(strokeWidth=0, fill=theme["bg"])
    )
    st.altair_chart(chart, use_container_width=True)


@st.cache_data(ttl=3600)
def get_holdout(horizon):
    path = f"data/holdout_{horizon}h.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    rmse = float(((df["actual"] - df["predicted"]) ** 2).mean() ** 0.5)
    return df, rmse


@st.cache_data(ttl=3600)
def get_shap_importance(horizon):
    return pd.read_csv(f"data/shap_importance_{horizon}h.csv")


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
                                                        range=["#4fc3f7", "#ff9100"]),
                             legend=alt.Legend(title=None)),
            strokeDash=alt.condition(alt.datum.type == "predicted", alt.value([5, 3]), alt.value([0])),
        )
        .properties(height=280, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_view(strokeWidth=0, fill=theme["bg"])
    )
    st.altair_chart(chart, use_container_width=True)


def render_shap_chart(importance_df, theme):
    df = importance_df.head(10).copy()
    df["feature"] = df["feature"].map(lambda f: FEATURE_DISPLAY_NAMES.get(f, f))
    chart = (
        alt.Chart(df)
        .mark_bar(color="#4fc3f7")
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP value|"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", "mean_abs_shap"],
        )
        .properties(height=320, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_view(strokeWidth=0, fill=theme["bg"])
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    theme = get_theme()
    inject_css(theme)

    top_l, top_r = st.columns([4, 1])
    with top_r:
        choice = st.radio(
            "Theme",
            ["Dark", "Light"],
            index=0 if st.session_state.theme == "dark" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        new_theme = "dark" if choice == "Dark" else "light"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

    st.markdown(
        """
        <div class="header-banner">
            <h1>🌫️ Bahawalpur AQI Forecast</h1>
            <p>Live conditions and 3-day-ahead average AQI forecast, powered by Ridge regression on hourly AQI + weather features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading live data..."):
        try:
            history, latest_row, source = get_history(days=14)
        except Exception as e:
            st.error(f"Could not load current data: {e}")
            return

    current_aqi = latest_row["aqi_epa"]
    category, color, icon, advisory = categorize_aqi(current_aqi)

    st.subheader("Current Air Quality")
    gauge_col, info_col = st.columns([1, 1])
    with gauge_col:
        render_gauge(current_aqi, color, theme)
    with info_col:
        st.markdown(f"### {icon} {category}")
        st.caption(f"As of {latest_row['datetime']} · source: {source}")
        st.markdown(f'<div class="advisory-banner">{advisory}</div>', unsafe_allow_html=True)

    st.write("**Current Conditions**")
    cond_cols = st.columns(3)
    with cond_cols[0]:
        render_info_card("Temperature", f"{latest_row['temp']:.1f}°C")
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
        f'<div class="meta-line">Forecast generated from data as of <b>{as_of}</b> &nbsp;·&nbsp; source: <b>{fsource}</b></div>',
        unsafe_allow_html=True,
    )

    values = list(results.values())
    deltas = [None] + [values[i] - values[i - 1] for i in range(1, len(values))]
    day_labels = ["Tomorrow · Day 1", "Day 2", "Day 3"]

    flagged_days = check_hazard_alert(results, day_labels)
    if flagged_days:
        items = "; ".join(f"<b>{label}</b>: {aqi:.0f} ({cat})" for label, aqi, cat in flagged_days)
        st.markdown(
            f'<div class="hazard-alert">⚠️ <b>Air Quality Alert</b> — {items}. '
            f'Limit outdoor exposure and consider wearing a mask if you must go outside.</div>',
            unsafe_allow_html=True,
        )

    # Bento layout: hero card for tomorrow, two stacked smaller cards for day 2/3
    hero_col, stack_col = st.columns([1.3, 1])
    with hero_col:
        cat, col_color, ic, _ = categorize_aqi(values[0])
        render_forecast_card(day_labels[0], values[0], cat, col_color, ic, deltas[0], hero=True)
    with stack_col:
        for i in [1, 2]:
            cat, col_color, ic, _ = categorize_aqi(values[i])
            render_forecast_card(day_labels[i], values[i], cat, col_color, ic, deltas[i])
            st.write("")

    st.write("")
    st.divider()

    st.subheader("📈 Recent AQI Trend")
    render_trend_chart(history, theme)

    st.divider()
    st.subheader("🎯 Model Validation")
    st.caption("Genuine holdout: Ridge trained excluding this window entirely, so this reflects real forecasting performance, not recall.")
    horizon_choice = st.radio("Horizon", [24, 48, 72], format_func=lambda h: f"+{h//24} day{'s' if h > 24 else ''}", horizontal=True, key="holdout_horizon")
    try:
        holdout_df, holdout_rmse = get_holdout(horizon_choice)
        st.write(f"Holdout RMSE: **±{holdout_rmse:.2f}**")
        render_holdout_chart(holdout_df, theme)
    except FileNotFoundError:
        st.warning("Holdout results not found - run `python src/validate_holdout.py` first.")

    st.divider()
    st.subheader("🔍 Why This Prediction")
    st.caption("Top features driving the forecast, by mean SHAP value.")
    shap_horizon = st.radio("SHAP Horizon", [24, 48, 72], format_func=lambda h: f"+{h//24} day{'s' if h > 24 else ''}", horizontal=True, key="shap_horizon")
    try:
        importance_df = get_shap_importance(shap_horizon)
        render_shap_chart(importance_df, theme)
    except FileNotFoundError:
        st.warning("SHAP results not found - run `python src/compute_shap.py` first.")

    st.write("")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()