import datetime
import importlib.util
import os
import time
from pathlib import Path

import folium
import holidays
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from folium.plugins import Fullscreen, HeatMap
from streamlit_folium import st_folium

try:
    st_autorefresh = importlib.import_module('streamlit_autorefresh').st_autorefresh
except Exception:
    st_autorefresh = None

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHICAGO_TZ = pytz.timezone('America/Chicago')

FEATURE_COLUMNS = [
    'Hour',
    'DayOfWeek',
    'Month',
    'Risk_Zone',
    'Temperature',
    'Precipitation',
    'Is_Holiday',
    'Spatial_Lag',
]

MODEL_FEATURE_COLUMNS = [
    'Precipitation',
    'Risk_Zone',
    'Temperature',
    'Hour',
    'DayOfWeek',
    'Month',
    'Is_Holiday',
    'Spatial_Lag',
]

PRECINCTS = [
    {'name': '1st District - Central', 'Latitude': 41.8853, 'Longitude': -87.6270},
    {'name': '6th District - Gresham', 'Latitude': 41.7559, 'Longitude': -87.6217},
    {'name': '7th District - Englewood', 'Latitude': 41.7772, 'Longitude': -87.6411},
    {'name': '8th District - Chicago Lawn', 'Latitude': 41.7740, 'Longitude': -87.7237},
    {'name': '19th District - Town Hall', 'Latitude': 41.9946, 'Longitude': -87.6908},
]

st.set_page_config(
    page_title='The Machine v2.0 - Predictive Policing',
    page_icon='🚨',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
        .stApp {
            background: #050505;
            color: #f2f5f7;
            font-family: 'Courier New', monospace;
        }
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            font-family: 'Courier New', monospace;
        }
        section[data-testid='stSidebar'] {
            background: linear-gradient(180deg, #0d0d0d 0%, #050505 100%);
            border-right: 1px solid rgba(255, 0, 0, 0.35);
        }
        div[data-testid='stMetric'] {
            background: rgba(9, 9, 9, 0.96);
            border: 1px solid #ff0000;
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 0 14px rgba(255, 0, 0, 0.45), 0 0 30px rgba(255, 0, 0, 0.18);
        }
        .clock-box {
            padding: 0.75rem 0.9rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 0, 0, 0.55);
            color: #d8e6ef;
            font-family: 'Courier New', monospace;
            font-size: 1rem;
            text-align: center;
            box-shadow: 0 0 12px rgba(255, 0, 0, 0.18);
        }
        div[data-testid='stDateInput'] {
            border: 1px solid rgba(255, 0, 0, 0.9);
            border-radius: 12px;
            padding: 0.25rem 0.5rem 0.35rem 0.5rem;
            box-shadow: 0 0 14px rgba(255, 0, 0, 0.35);
            background: rgba(255, 255, 255, 0.03);
        }
        div[data-testid='stDateInput']:hover,
        div[data-testid='stDateInput']:focus-within {
            box-shadow: 0 0 18px rgba(255, 0, 0, 0.55), 0 0 36px rgba(255, 0, 0, 0.2);
            border-color: rgba(255, 120, 120, 1);
        }
        div[data-testid='stDateInput'] button {
            cursor: pointer;
        }
        .section-title {
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #f2f5f7;
            margin-bottom: 0.25rem;
        }
        .intel-box {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 0, 0, 0.55);
            box-shadow: 0 0 12px rgba(255, 0, 0, 0.18);
            color: #f2f5f7;
            margin-top: 0.5rem;
        }
        div[data-testid='stTabs'] button {
            font-family: 'Courier New', monospace;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join(APP_DIR, 'the_machine_model.pkl'))
        scaler = joblib.load(os.path.join(APP_DIR, 'scaler.pkl'))
        centroids = pd.read_csv(os.path.join(APP_DIR, 'centroids.csv'))
        processed_data = pd.read_csv(os.path.join(APP_DIR, 'processed_crime_data.csv'))

        spatial_lag_map = processed_data.groupby('Risk_Zone')['Spatial_Lag'].mean().to_dict()
        centroids['Spatial_Lag_Mean'] = centroids['Risk_Zone'].map(spatial_lag_map).fillna(0)
        return model, scaler, centroids
    except FileNotFoundError as exc:
        st.error(f'Error loading assets: {exc}')
        st.error(f'Attempted to load from directory: {APP_DIR}')
        return None, None, None


def chicago_now():
    return datetime.datetime.now(CHICAGO_TZ)


def fahrenheit_to_celsius(value):
    return (value - 32.0) * 5.0 / 9.0


def manhattan_distance_miles(lat1, lon1, lat2, lon2):
    lat_miles = abs(lat1 - lat2) * 69.0
    lon_miles = abs(lon1 - lon2) * 52.0
    return lat_miles + lon_miles


def find_nearest_precinct(lat, lon):
    nearest_precinct = None
    nearest_distance = None
    for precinct in PRECINCTS:
        distance_miles = manhattan_distance_miles(lat, lon, precinct['Latitude'], precinct['Longitude'])
        if nearest_distance is None or distance_miles < nearest_distance:
            nearest_precinct = precinct
            nearest_distance = distance_miles
    estimated_minutes = round(4.0 + (nearest_distance / 25.0) * 60.0, 1)
    return nearest_precinct, nearest_distance, estimated_minutes


def attach_precinct_metrics(df):
    precinct_names = []
    precinct_distances = []
    response_minutes = []

    for _, row in df.iterrows():
        precinct, distance_miles, estimated_minutes = find_nearest_precinct(row['Latitude'], row['Longitude'])
        precinct_names.append(precinct['name'])
        precinct_distances.append(distance_miles)
        response_minutes.append(estimated_minutes)

    df = df.copy()
    df['Nearest_Precinct'] = precinct_names
    df['Manhattan_Distance_Miles'] = precinct_distances
    df['Estimated_Response_Minutes'] = response_minutes
    return df


def build_feature_frame(base_df, prediction_date, prediction_hour, temperature_c, precipitation, lag_weight, is_holiday):
    future_df = base_df.copy()
    future_df['Precipitation'] = precipitation
    future_df['Temperature'] = temperature_c
    future_df['Hour'] = prediction_hour
    future_df['DayOfWeek'] = prediction_date.weekday()
    future_df['Month'] = prediction_date.month
    future_df['Is_Holiday'] = 1 if is_holiday else 0
    future_df['Spatial_Lag'] = future_df['Spatial_Lag_Mean'] * lag_weight
    return future_df


def explain_prediction(sample_row, model_obj, scaler_obj):
    ordered_features = SCALER_FEATURE_COLUMNS
    sample_frame = pd.DataFrame([sample_row[ordered_features].to_dict()], columns=ordered_features)
    scaled_sample = scaler_obj.transform(sample_frame)

    if not hasattr(model_obj, 'coefs_') or len(model_obj.coefs_) == 0:
        explanation = pd.DataFrame({'Feature': ordered_features, 'Importance': np.zeros(len(ordered_features))})
        explanation['Percent'] = 0.0
        return explanation

    layer_strength = np.abs(model_obj.coefs_[0])
    for layer_weights in model_obj.coefs_[1:]:
        layer_strength = layer_strength @ np.abs(layer_weights)

    if layer_strength.ndim > 1:
        layer_strength = layer_strength[:, 0]

    raw_importance = np.abs(scaled_sample[0]) * layer_strength
    raw_importance = np.nan_to_num(raw_importance, nan=0.0, posinf=0.0, neginf=0.0)
    if float(raw_importance.sum()) <= 0:
        raw_importance = np.ones_like(raw_importance)

    explanation = pd.DataFrame({
        'Feature': ordered_features,
        'Importance': raw_importance,
    })
    explanation['Percent'] = explanation['Importance'] / explanation['Importance'].sum() * 100.0
    explanation = explanation.sort_values('Importance', ascending=True)
    return explanation


def run_boot_sequence():
    if st.session_state.get('boot_sequence_complete'):
        return

    boot_slot = st.empty()
    boot_messages = [
        'Establishing Secure Connection to Chicago Mainframe...',
        'Running Recursive Heuristics...',
        'Synchronizing Tactical Intelligence Layers...',
        'Calibrating Predictive Engine...',
    ]

    with st.spinner('Initializing The Machine...'):
        for message in boot_messages:
            boot_slot.markdown(f"<div class='intel-box'>{message}</div>", unsafe_allow_html=True)
            time.sleep(0.35)
    boot_slot.empty()
    st.session_state.boot_sequence_complete = True
    st.session_state.system_logs.append(f"Boot sequence completed at {chicago_now().strftime('%Y-%m-%d %H:%M:%S')} Chicago time.")


@st.cache_resource
def get_cached_assets():
    return load_assets()


if 'system_logs' not in st.session_state:
    st.session_state.system_logs = []
if 'dashboard_result' not in st.session_state:
    st.session_state.dashboard_result = None
if 'last_analysis_error' not in st.session_state:
    st.session_state.last_analysis_error = None
if 'clicked_zone_info' not in st.session_state:
    st.session_state.clicked_zone_info = None
if 'boot_sequence_complete' not in st.session_state:
    st.session_state.boot_sequence_complete = False

run_boot_sequence()

with st.spinner('Loading model, scaler, and tactical intelligence layers...'):
    model, scaler, centroids = get_cached_assets()

us_holidays = holidays.US()
SCALER_FEATURE_COLUMNS = list(getattr(scaler, 'feature_names_in_', MODEL_FEATURE_COLUMNS)) if scaler is not None else MODEL_FEATURE_COLUMNS

st.sidebar.header('Prediction Parameters')
if st_autorefresh is not None:
    st_autorefresh(interval=1000, key='chicago-clock-refresh')
clock_placeholder = st.sidebar.empty()
chicago_time = chicago_now()
clock_placeholder.markdown(
    f"<div class='clock-box'>Chicago Time<br><strong>{chicago_time.strftime('%Y-%m-%d %H:%M:%S')}</strong></div>",
    unsafe_allow_html=True,
)

prediction_date = st.sidebar.date_input('Select a Date', chicago_time.date())
prediction_hour = st.sidebar.slider('Select an Hour (24-hour format)', 0, 23, chicago_time.hour)

st.sidebar.subheader('Stochastic Controls')
temp_unit = st.sidebar.radio('Temperature Unit', ['Fahrenheit (°F)', 'Celsius (°C)'], index=0)
if temp_unit == 'Fahrenheit (°F)':
    temp_slider = st.sidebar.slider('Temperature (°F)', 14, 104, 50)
else:
    temp_slider = st.sidebar.slider('Temperature (°C)', -10, 40, 10)
precip_slider = st.sidebar.slider('Precipitation (mm)', 0.0, 50.0, 0.0, 0.1)
lag_weight_slider = st.sidebar.slider('Spatial Lag Weight', 0.5, 2.0, 1.0, 0.1)
crime_type = st.sidebar.selectbox('Crime Type', ['All', 'Theft', 'Assault', 'Battery', 'Robbery', 'Other'])
is_payday = st.sidebar.toggle('Is Payday?', value=False)
zone_options = centroids['Risk_Zone'].astype(int).tolist() if centroids is not None else [0]
selected_zone = st.sidebar.selectbox('Trend Zone', zone_options)
predict_button = st.sidebar.button('Analyze & Predict Risk', type='primary')

st.title('🚨 The Machine v2.0')
st.subheader('Predictive Crime Risk Intelligence Dashboard for Chicago')

if model is None or scaler is None or centroids is None:
    st.error('Missing model assets. Please confirm the model, scaler, and centroid files exist in the project folder.')
    st.stop()


def run_analysis(pred_date, pred_hour, temp_unit_value, temp_value, precip_value, lag_weight_value, crime_type_value, payday_flag, selected_zone_value):
    selected_hour = int(pred_hour)
    selected_zone_int = int(selected_zone_value)
    temp_celsius = temp_value if temp_unit_value == 'Celsius (°C)' else fahrenheit_to_celsius(temp_value)

    future_df = build_feature_frame(
        centroids,
        pred_date,
        selected_hour,
        temp_celsius,
        precip_value,
        lag_weight_value,
        pred_date in us_holidays,
    )

    missing_columns = [column for column in SCALER_FEATURE_COLUMNS if column not in future_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required model columns: {missing_columns}")

    X_future = future_df.loc[:, SCALER_FEATURE_COLUMNS].copy()
    X_future_scaled = scaler.transform(X_future)
    risk_probabilities = model.predict_proba(X_future_scaled)[:, 1]

    if payday_flag:
        risk_probabilities = np.clip(risk_probabilities * 1.10, 0.0, 1.0)

    future_df['Predicted_Risk_Probability'] = risk_probabilities
    future_df = attach_precinct_metrics(future_df)

    top_idx = future_df['Predicted_Risk_Probability'].idxmax()
    top_row = future_df.loc[top_idx]

    explanation_df = explain_prediction(top_row, model, scaler)
    top_factors = explanation_df.sort_values('Importance', ascending=False).head(3)['Feature'].tolist()

    trend_rows = []
    for hour in range(24):
        trend_frame = build_feature_frame(
            centroids,
            pred_date,
            hour,
            temp_celsius,
            precip_value,
            lag_weight_value,
            pred_date in us_holidays,
        )
        trend_frame = trend_frame[trend_frame['Risk_Zone'] == selected_zone_int].copy()
        if trend_frame.empty:
            continue
        trend_missing = [column for column in SCALER_FEATURE_COLUMNS if column not in trend_frame.columns]
        if trend_missing:
            raise ValueError(f"Missing required model columns in trend frame: {trend_missing}")
        trend_scaled = scaler.transform(trend_frame.loc[:, SCALER_FEATURE_COLUMNS].copy())
        trend_prob = float(model.predict_proba(trend_scaled)[:, 1][0])
        if payday_flag:
            trend_prob = min(trend_prob * 1.10, 1.0)
        trend_rows.append({'Hour': hour, 'Risk': trend_prob})

    trend_df = pd.DataFrame(trend_rows)
    clicked_zone = st.session_state.clicked_zone_info

    st.session_state.system_logs.append(
        f"Prediction run for {pred_date} {selected_hour:02d}:00 | top zone {int(top_row['Risk_Zone'])} | risk {float(top_row['Predicted_Risk_Probability']):.2%}"
    )

    return {
        'future_df': future_df,
        'trend_df': trend_df,
        'selected_zone_int': selected_zone_int,
        'temp_slider': temp_value,
        'temp_unit': temp_unit_value,
        'temp_celsius': temp_celsius,
        'crime_type': crime_type_value,
        'is_payday': payday_flag,
        'highest_risk': float(top_row['Predicted_Risk_Probability']),
        'top_target_zone': int(top_row['Risk_Zone']),
        'top_row': top_row,
        'explanation_df': explanation_df,
        'top_factors': top_factors,
        'clicked_zone': clicked_zone,
    }


if predict_button:
    with st.spinner(f'Running analysis for {prediction_date} at {int(prediction_hour):02d}:00...'):
        try:
            st.session_state.dashboard_result = run_analysis(
                prediction_date,
                prediction_hour,
                temp_unit,
                temp_slider,
                precip_slider,
                lag_weight_slider,
                crime_type,
                is_payday,
                selected_zone,
            )
            st.session_state.last_analysis_error = None
            st.rerun()
        except Exception as exc:
            st.session_state.dashboard_result = None
            st.session_state.last_analysis_error = str(exc)
            st.error(f'Analysis failed: {exc}')


if st.session_state.dashboard_result is None:
    if st.session_state.last_analysis_error is not None:
        st.error(f'Last analysis error: {st.session_state.last_analysis_error}')
    st.info('Select prediction parameters and click "Analyze & Predict Risk" to begin.')
    st.stop()

result = st.session_state.dashboard_result
st.success('Analysis complete. Results loaded.')
future_df = result['future_df']
trend_df = result['trend_df']
selected_zone_int = result['selected_zone_int']
temp_slider = result['temp_slider']
temp_unit = result['temp_unit']
temp_celsius = result['temp_celsius']
crime_type = result['crime_type']
is_payday = result['is_payday']
highest_risk = result['highest_risk']
top_target_zone = result['top_target_zone']
top_row = result['top_row']
explanation_df = result['explanation_df']
top_factors = result['top_factors']
clicked_zone = result['clicked_zone']

live_tab, trend_tab, log_tab = st.tabs(['Live Tactical Map', 'Risk Trends', 'System Logs'])

with live_tab:
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric('Current Highest Risk %', f'{highest_risk:.2%}')
    metric_col2.metric('Top Target Zone', f'Zone {top_target_zone}')

    st.caption(
        f'Crime Type Filter: {crime_type} | Payday Heuristic: {"On" if is_payday else "Off"} | '
        f'Temperature Input: {temp_slider} {"°F" if temp_unit == "Fahrenheit (°F)" else "°C"} '
        f'({temp_celsius:.1f}°C used by the model)'
    )

    map_col, info_col = st.columns([2, 1])

    with map_col:
        st.markdown("<div class='section-title'>Predicted Crime Risk Map</div>", unsafe_allow_html=True)
        map_center = [41.8781, -87.6298]
        tactical_map = folium.Map(location=map_center, zoom_start=11, tiles='CartoDB dark_matter')
        Fullscreen(position='topright').add_to(tactical_map)

        heat_data = [
            [row['Latitude'], row['Longitude'], row['Predicted_Risk_Probability']]
            for _, row in future_df.iterrows()
        ]
        HeatMap(heat_data, radius=18, blur=15, min_opacity=0.25, name='Risk Heat').add_to(tactical_map)

        crime_color_map = {
            'Assault': 'red',
            'Theft': 'orange',
            'Battery': 'purple',
            'Robbery': 'cadetblue',
            'Other': 'green',
        }

        for _, row in future_df.iterrows():
            risk_level = float(row['Predicted_Risk_Probability'])
            if crime_type == 'All':
                if risk_level > 0.6:
                    marker_color = 'red'
                elif risk_level > 0.3:
                    marker_color = 'orange'
                else:
                    marker_color = 'green'
            else:
                marker_color = crime_color_map.get(crime_type, 'orange')

            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.75 if int(row['Risk_Zone']) == top_target_zone else 0.5,
                popup=(
                    f"Zone {int(row['Risk_Zone'])}<br>"
                    f"Risk: {risk_level:.2%}<br>"
                    f"Nearest Precinct: {row['Nearest_Precinct']}<br>"
                    f"Estimated Response: {float(row['Estimated_Response_Minutes']):.1f} min"
                ),
                tooltip=f"Zone {int(row['Risk_Zone'])}"
            ).add_to(tactical_map)

        for precinct in PRECINCTS:
            folium.Marker(
                location=[precinct['Latitude'], precinct['Longitude']],
                tooltip=precinct['name'],
                popup=precinct['name'],
                icon=folium.Icon(color='blue', icon='shield', prefix='fa'),
            ).add_to(tactical_map)

        map_state = st_folium(
            tactical_map,
            width=740,
            height=580,
            returned_objects=['last_clicked', 'last_object_clicked'],
            key='tactical-map',
        )

        clicked_point = map_state.get('last_clicked') or map_state.get('last_object_clicked')
        if clicked_point:
            clicked_lat = clicked_point.get('lat') or clicked_point.get('latitude')
            clicked_lon = clicked_point.get('lng') or clicked_point.get('lon') or clicked_point.get('longitude')
            if clicked_lat is not None and clicked_lon is not None:
                nearest_zone_idx = future_df.apply(
                    lambda row: manhattan_distance_miles(clicked_lat, clicked_lon, row['Latitude'], row['Longitude']),
                    axis=1,
                ).idxmin()
                clicked_zone_row = future_df.loc[nearest_zone_idx]
                precinct, distance_miles, response_minutes = find_nearest_precinct(
                    float(clicked_zone_row['Latitude']),
                    float(clicked_zone_row['Longitude']),
                )
                st.session_state.clicked_zone_info = {
                    'zone': int(clicked_zone_row['Risk_Zone']),
                    'risk': float(clicked_zone_row['Predicted_Risk_Probability']),
                    'nearest_precinct': precinct['name'],
                    'distance_miles': distance_miles,
                    'response_minutes': response_minutes,
                }
                st.session_state.system_logs.append(
                    f"Map clicked -> zone {int(clicked_zone_row['Risk_Zone'])}, response {response_minutes:.1f} min to {precinct['name']}"
                )

    with info_col:
        st.markdown("<div class='section-title'>Target Intelligence</div>", unsafe_allow_html=True)
        st.metric('Top Target Zone', f'Zone {top_target_zone}')
        st.metric('Nearest Precinct', top_row['Nearest_Precinct'])
        st.metric('Estimated Response Time', f"{float(top_row['Estimated_Response_Minutes']):.1f} min")

        if st.session_state.clicked_zone_info is not None:
            clicked = st.session_state.clicked_zone_info
            st.markdown("<div class='intel-box'>Clicked Zone Response</div>", unsafe_allow_html=True)
            st.metric('Clicked Zone', f"Zone {clicked['zone']}")
            st.metric('Clicked Risk', f"{clicked['risk']:.2%}")
            st.metric('Nearest Precinct', clicked['nearest_precinct'])
            st.metric('Estimated Response Time', f"{clicked['response_minutes']:.1f} min")

    st.markdown("<div class='section-title'>Decision Intelligence (Why?)</div>", unsafe_allow_html=True)
    explanation_display = explanation_df.sort_values('Importance', ascending=True)
    xai_fig = go.Figure(
        go.Bar(
            x=explanation_display['Percent'],
            y=explanation_display['Feature'],
            orientation='h',
            marker=dict(color='#FF0000'),
            hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
        )
    )
    xai_fig.update_layout(
        template='plotly_dark',
        height=380,
        margin=dict(l=40, r=20, t=20, b=30),
        paper_bgcolor='#050505',
        plot_bgcolor='#050505',
        xaxis_title='Contribution (%)',
        yaxis_title='',
    )
    st.plotly_chart(xai_fig, use_container_width=True)

    st.markdown(
        f"<div class='intel-box'>Primary drivers: {', '.join(top_factors)} | Model explanation based on scaled feature contribution path.</div>",
        unsafe_allow_html=True,
    )

with trend_tab:
    st.markdown("<div class='section-title'>Daily Risk Trend</div>", unsafe_allow_html=True)
    if not trend_df.empty:
        st.line_chart(trend_df.set_index('Hour'))
    else:
        st.info('No trend data available for the selected zone.')

    st.markdown("<div class='section-title'>Zone Intelligence Snapshot</div>", unsafe_allow_html=True)
    snapshot = future_df[future_df['Risk_Zone'] == selected_zone_int]
    if not snapshot.empty:
        snapshot_row = snapshot.iloc[0]
        st.metric('Selected Trend Zone', f'Zone {selected_zone_int}')
        st.metric('Nearest Precinct', snapshot_row['Nearest_Precinct'])
        st.metric('Estimated Response Time', f"{float(snapshot_row['Estimated_Response_Minutes']):.1f} min")

with log_tab:
    st.markdown("<div class='section-title'>System Logs</div>", unsafe_allow_html=True)
    st.code('\n'.join(st.session_state.system_logs[-20:]) or 'No logs yet.')
    st.markdown("<div class='section-title'>Feature Schema Integrity</div>", unsafe_allow_html=True)
    st.code(str(FEATURE_COLUMNS))
    st.markdown("<div class='section-title'>Precinct Registry</div>", unsafe_allow_html=True)
    precinct_df = pd.DataFrame(PRECINCTS)
    st.dataframe(precinct_df, use_container_width=True)
    if st.session_state.clicked_zone_info is not None:
        st.markdown("<div class='section-title'>Clicked Zone Response</div>", unsafe_allow_html=True)
        st.json(st.session_state.clicked_zone_info)
