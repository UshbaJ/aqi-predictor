"""
app.py

Streamlit dashboard for the AQI Predictor: current conditions, live AQI
gauge, pollutant breakdown, 3-day forecast, historical trend, model
validation, SHAP explanations, hazardous AQI alerts, a what-if simulator,
voice briefing, and a personal exposure calculator.
Supports a city selector (Bahawalpur, Lahore, Islamabad) and
user-toggleable light/dark theme.

Usage:
    streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import joblib
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

from predict import predict_next_3_days
from features import load_features, FEATURE_COLS, CITIES
from train import model_filename, results_path_for_city
from validate_holdout import holdout_output_path
from compute_shap import shap_output_path

st.set_page_config(page_title="AQI Forecast", page_icon="AQ", layout="wide")

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
            :root {{
            color-scheme: light !important;
        }}
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

        .hazard-alert-v2 {{
            border-left: 4px solid #dc2626;
            background: {theme['card_bg']};
            backdrop-filter: blur(10px);
            border: 1px solid {theme['card_border']};
            border-left: 4px solid #dc2626;
            padding: 16px 18px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .hazard-alert-v2 .alert-title {{
            font-size: 15px; font-weight: 700; color: #dc2626;
            display: flex; align-items: center; gap: 6px; margin-bottom: 10px;
        }}
        .hazard-chip-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .hazard-chip {{
            background: rgba(220,38,38,0.12);
            border: 1px solid rgba(220,38,38,0.3);
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 13px;
            color: {theme['text']};
        }}
        .hazard-chip b {{ color: #dc2626; }}
        .hazard-footer {{ font-size: 12.5px; color: {theme['muted']}; margin-top: 10px; }}

        button[data-baseweb="tab"] {{
            font-size: 15px !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            color: {theme['muted']} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {theme['accent']} !important;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 2px solid {theme['card_border']} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {theme['accent']} !important;
            height: 3px !important;
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
        
                .react-aria-ComboBox input {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text']} !important;
        }}
        .react-aria-Popover {{
            background-color: {theme['bg']} !important;
        }}
        .react-aria-ListBox {{
            background-color: {theme['bg']} !important;
        }}
        .react-aria-ListBoxItem {{
            background-color: {theme['bg']} !important;
            color: {theme['text']} !important;
        }}
        .react-aria-ListBoxItem[data-hovered="true"],
        .react-aria-ListBoxItem[data-focused="true"] {{
            background-color: {theme['card_bg']} !important;
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
def get_forecast(city):
    return predict_next_3_days(city=city)


@st.cache_data(ttl=1800, show_spinner=False)
def get_history(city, days=14):
    df, source = load_features(city=city, source="auto")
    df = df.sort_values("datetime")
    latest_row = df.dropna(subset=["aqi_epa"]).iloc[-1]
    cutoff = df["datetime"].max() - pd.Timedelta(days=days)
    recent = df[df["datetime"] >= cutoff][["datetime", "aqi_epa"]].dropna()
    return recent, latest_row, source


@st.cache_data(ttl=3600, show_spinner=False)
def get_holdout(city, horizon):
    path = holdout_output_path(city, horizon)
    df = pd.read_csv(path, parse_dates=["datetime"])
    rmse = float(((df["actual"] - df["predicted"]) ** 2).mean() ** 0.5)
    return df, rmse


@st.cache_data(ttl=3600, show_spinner=False)
def get_shap_importance(city, horizon):
    return pd.read_csv(shap_output_path(city, horizon))


@st.cache_data(ttl=3600, show_spinner=False)
def get_cv_results(city):
    return pd.read_csv(results_path_for_city(city))


def render_model_findings(theme):
    st.subheader("Model Performance by City")
    st.caption("Ridge and Random Forest, 5-fold time-series CV, vs. the naive persistence baseline.")

    tabs = st.tabs([c.title() for c in CITIES])
    for tab, c in zip(tabs, CITIES):
        with tab:
            try:
                df = get_cv_results(c)
            except FileNotFoundError:
                st.warning(f"No CV results found for {c.title()}.")
                continue
            for _, row in df.iterrows():
                horizon = int(row["horizon"])
                naive_rmse, ridge_rmse, rf_rmse = row["naive_rmse_mean"], row["ridge_rmse_mean"], row["rf_rmse_mean"]
                best_model = "Ridge" if ridge_rmse <= rf_rmse else "Random Forest"
                best_rmse = min(ridge_rmse, rf_rmse)
                if best_rmse < naive_rmse:
                    improvement = (1 - best_rmse / naive_rmse) * 100
                    verdict = f"**{best_model}** beats the naive baseline by **{improvement:.1f}%** (RMSE)."
                else:
                    verdict = "Neither model beat the naive baseline at this horizon."
                st.markdown(f"**+{horizon}h horizon** — {verdict}")
                cols = st.columns(3)
                cols[0].metric("Naive RMSE", f"{naive_rmse:.1f}")
                cols[1].metric("Ridge RMSE", f"{ridge_rmse:.1f}")
                cols[2].metric("RF RMSE", f"{rf_rmse:.1f}")
                st.write("")


@st.cache_data(ttl=1800, show_spinner=False)
def get_city_history_all(days=14):
    data = {}
    for c in CITIES:
        try:
            recent, latest_row, _ = get_history(c, days=days)
            data[c] = (recent, latest_row)
        except Exception:
            data[c] = (None, None)
    return data

POLLUTANT_COLS = ["aqi_epa", "co", "no2", "o3", "so2", "pm2_5", "pm10"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_raw_aqi_data(city):
    df, _ = load_features(city=city, source="auto")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def render_city_eda(city, theme):
    try:
        df = get_raw_aqi_data(city)
    except FileNotFoundError:
        st.warning(f"No raw data found for {city.title()}.")
        return

    st.caption(f"{len(df):,} hourly readings · {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])
    ax.plot(df["datetime"], df["aqi_epa"], color="#2563eb", linewidth=0.8)
    ax.axhline(150, color="orange", linestyle="--", label="Unhealthy (150)")
    ax.axhline(200, color="red", linestyle="--", label="Very Unhealthy (200)")
    ax.set_title(f"EPA AQI Over Time — {city.title()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(theme["bg"])
        sns.heatmap(df[POLLUTANT_COLS].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Pollutant Correlation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        hourly_avg = df.assign(hour=df["datetime"].dt.hour).groupby("hour")["aqi_epa"].mean()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(theme["bg"])
        ax.set_facecolor(theme["bg"])
        hourly_avg.plot(kind="bar", ax=ax, color="#2563eb")
        ax.set_xticklabels(hourly_avg.index, rotation=0)
        ax.set_title("Average AQI by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Average AQI")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_cross_city_hourly_pattern(theme):
    frames = []
    for c in CITIES:
        try:
            df = get_raw_aqi_data(c)
        except FileNotFoundError:
            continue
        hourly = df.assign(hour=df["datetime"].dt.hour).groupby("hour")["aqi_epa"].mean().reset_index()
        hourly["city"] = c.title()
        frames.append(hourly)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    chart = (
        alt.Chart(combined)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("hour:O", title="Hour of Day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("aqi_epa:Q", title="Average AQI"),
            color=alt.Color("city:N", legend=alt.Legend(title=None)),
            tooltip=["city", "hour", "aqi_epa"],
        )
        .properties(height=280, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_legend(labelColor=theme["text"], titleColor=theme["text"], labelFontSize=12)
        .configure_view(strokeWidth=0, fill=theme["bg"])
    )
    st.altair_chart(chart, use_container_width=True)

def render_eda_section(theme):
    st.subheader("City Comparison")
    st.caption("Current conditions and recent 14-day AQI trends across all monitored cities.")

    city_data = get_city_history_all(days=14)

    compare_rows = [
        {"city": c.title(), "aqi": latest_row["aqi_epa"]}
        for c in CITIES
        if (latest_row := city_data[c][1]) is not None
    ]
    if compare_rows:
        compare_df = pd.DataFrame(compare_rows)
        chart = (
            alt.Chart(compare_df)
            .mark_bar()
            .encode(
                x=alt.X("city:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("aqi:Q", title="Current AQI"),
                color=alt.Color("aqi:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
                tooltip=["city", "aqi"],
            )
            .properties(height=240, background=theme["bg"])
            .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
            .configure_view(strokeWidth=0, fill=theme["bg"])
        )
        st.altair_chart(chart, use_container_width=True)

    trend_frames = []
    for c in CITIES:
        recent, _ = city_data[c]
        if recent is not None:
            tmp = recent.copy()
            tmp["city"] = c.title()
            trend_frames.append(tmp)
    if trend_frames:
        trend_df = pd.concat(trend_frames, ignore_index=True)
        chart = (
            alt.Chart(trend_df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("datetime:T", title=None, axis=alt.Axis(format="%b %d, %H:%M")),
                y=alt.Y("aqi_epa:Q", title="AQI"),
                color=alt.Color("city:N", legend=alt.Legend(title=None)),
                tooltip=["city", "datetime:T", "aqi_epa:Q"],
            )
            .properties(height=280, background=theme["bg"])
            .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
            .configure_legend(labelColor=theme["text"], titleColor=theme["text"], labelFontSize=12)
            .configure_view(strokeWidth=0, fill=theme["bg"])
        )
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("Hourly AQI Pattern by City")
    st.caption("Average AQI by hour of day — reveals daily pollution rhythms across cities.")
    render_cross_city_hourly_pattern(theme)

    st.divider()
    st.subheader("Per-City Exploratory Analysis")
    eda_tabs = st.tabs([c.title() for c in CITIES])
    for tab, c in zip(eda_tabs, CITIES):
        with tab:
            render_city_eda(c, theme)

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
            x=alt.X("datetime:T", title=None, axis=alt.Axis(format="%b %d, %H:%M")),
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
            x=alt.X("datetime:T", title=None, axis=alt.Axis(format="%b %d, %H:%M")),
            y=alt.Y("aqi:Q", title="AQI"),
            color=alt.Color("type:N", scale=alt.Scale(domain=["actual", "predicted"],
                                                        range=[theme["accent"], "#8a8d94"]),
                             legend=alt.Legend(title=None)),
            strokeDash=alt.condition(alt.datum.type == "predicted", alt.value([5, 3]), alt.value([0])),
        )
        .properties(height=260, background=theme["bg"])
        .configure_axis(grid=True, gridColor=theme["card_border"], labelColor=theme["muted"], titleColor=theme["muted"])
        .configure_legend(labelColor=theme["text"], titleColor=theme["text"], labelFontSize=12)
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


def render_whatif_simulator(theme, city):
    st.subheader("What-If Simulator")
    st.caption("Adjust conditions below and see how the model's forecast changes in real time.")

    horizon = st.radio(
        "Forecast horizon", [24, 48, 72],
        format_func=lambda h: f"+{h // 24} day{'s' if h > 24 else ''}",
        horizontal=True, key="whatif_horizon",
    )

    try:
        _, latest_row, _ = get_history(city, days=14)
    except Exception:
        st.warning("Could not load baseline conditions for simulation.")
        return

    model = joblib.load(model_filename(city, "ridge", horizon))

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
        <div class="forecast-card hero" style="background: linear-gradient(160deg, {color}ee, {color}99); max-width: 380px;">
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

    city = st.selectbox(
        "City",
        CITIES,
        format_func=lambda c: c.title(),
        key="selected_city",
    )

    st.markdown(
        f"""
        <div class="header-banner">
            <h1>{city.title()} AQI Forecast</h1>
            <p>Live conditions and 3-day-ahead average AQI forecast, powered by Ridge regression on hourly AQI and weather features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    with st.spinner("Loading live data..."):
        try:
            history, latest_row, source = get_history(city, days=14)
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
            results, as_of, fsource = get_forecast(city)
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
            _, rmse = get_holdout(city, h)
            horizon_rmse[h] = rmse
        except FileNotFoundError:
            horizon_rmse[h] = None

    flagged_days = check_hazard_alert(results, day_labels)
    if flagged_days:
        chips = "".join(
            f'<div class="hazard-chip">{label}: <b>{aqi:.0f}</b> ({cat})</div>'
            for label, aqi, cat in flagged_days
        )
        st.markdown(
            f"""
            <div class="hazard-alert-v2">
                <div class="alert-title">⚠ Air Quality Alert</div>
                <div class="hazard-chip-row">{chips}</div>
                <div class="hazard-footer">Limit outdoor exposure and consider wearing a mask if you must go outside.</div>
            </div>
            """,
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
    tab_forecast, tab_model, tab_compare = st.tabs(["Forecast Details", "Model & Validation", "City Comparison"])

    with tab_forecast:
        render_whatif_simulator(theme, city)
        st.divider()
        st.subheader("Voice Briefing")
        render_voice_briefing(results, day_labels, current_aqi, category)
        render_exposure_calculator(current_aqi, theme)
        st.divider()
        st.subheader("Recent AQI Trend")
        render_trend_chart(history, theme)

    with tab_model:
        render_model_findings(theme)
        st.divider()
        st.subheader("Model Validation")
        st.caption("Genuine holdout: Ridge trained excluding this window entirely, so this reflects real forecasting performance, not recall.")
        horizon_choice = st.radio("Horizon", [24, 48, 72], format_func=lambda h: f"+{h // 24} day{'s' if h > 24 else ''}", horizontal=True, key="holdout_horizon")
        try:
            holdout_df, holdout_rmse = get_holdout(city, horizon_choice)
            st.write(f"Holdout RMSE: **\u00b1{holdout_rmse:.2f}**")
            render_holdout_chart(holdout_df, theme)
        except FileNotFoundError:
            st.warning(f"Holdout results not found for {city.title()}.")
        st.divider()
        st.subheader("Why This Prediction")
        st.caption("Top features driving the forecast, by mean SHAP value.")
        shap_horizon = st.radio("SHAP Horizon", [24, 48, 72], format_func=lambda h: f"+{h // 24} day{'s' if h > 24 else ''}", horizontal=True, key="shap_horizon")
        try:
            importance_df = get_shap_importance(city, shap_horizon)
            render_shap_chart(importance_df, theme)
        except FileNotFoundError:
            st.warning(f"SHAP results not found for {city.title()}.")

    with tab_compare:
        render_eda_section(theme)

    st.write("")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()
        

if __name__ == "__main__":
    main()