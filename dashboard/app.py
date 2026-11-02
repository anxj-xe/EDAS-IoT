"""EDAS-IoT Telemetry & Deployment Readiness Dashboard.

Advanced engineering console for environmental anomaly detection, statistical
monitoring, and target IoT hardware deployment readiness assessment.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.assessment.readiness import ReadinessAssessor
from src.common.config import get_config
from src.common.schemas import AnomalySeverity, DeviceProfile, ReadinessStatus
from src.detection.baseline import BaselineCalibrator, BaselineModel
from src.detection.isolation_forest import IsolationForestDetector
from src.ingestion.csv_reader import CSVReader
from src.ingestion.simulator import SensorSimulator
from src.processing.pipeline import EDASPipeline
from src.storage.repository import DatabaseRepository


# Page Setup
st.set_page_config(
    page_title="EDAS-IoT | Deployment Readiness & Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = get_config()
root_dir = config.root_dir

# Unified Enterprise Slate / Obsidian Theme CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f1f5f9;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #131c2e;
            border: 1px solid #22324e;
            border-radius: 8px;
            padding: 14px 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        div[data-testid="stMetricLabel"] > div {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94a3b8;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] > div {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.55rem;
            font-weight: 600;
            color: #f8fafc;
        }

        /* Status Verdict Callouts */
        .status-box {
            padding: 1.1rem 1.3rem;
            border-radius: 8px;
            border: 1px solid #22324e;
            margin-bottom: 1rem;
        }

        .status-box-ready {
            background-color: rgba(16, 185, 129, 0.08);
            border-color: rgba(16, 185, 129, 0.35);
            color: #34d399;
        }

        .status-box-warning {
            background-color: rgba(245, 158, 11, 0.08);
            border-color: rgba(245, 158, 11, 0.35);
            color: #fbbf24;
        }

        .status-box-not-ready {
            background-color: rgba(244, 63, 94, 0.09);
            border-color: rgba(244, 63, 94, 0.40);
            color: #fda4af;
        }

        .status-box-insufficient {
            background-color: rgba(148, 163, 184, 0.08);
            border-color: rgba(148, 163, 184, 0.25);
            color: #cbd5e1;
        }

        /* Alert Episode Cards */
        .alert-card {
            border-left: 4px solid #64748b;
            background: #131c2e;
            padding: 0.85rem 1.1rem;
            border-radius: 0 6px 6px 0;
            margin-bottom: 0.65rem;
            border-top: 1px solid #1e293b;
            border-right: 1px solid #1e293b;
            border-bottom: 1px solid #1e293b;
        }

        .alert-card-critical {
            border-left-color: #f43f5e;
            background: rgba(244, 63, 94, 0.06);
            color: #fecdd3;
        }

        .alert-card-warning {
            border-left-color: #f59e0b;
            background: rgba(245, 158, 11, 0.06);
            color: #fde68a;
        }

        .alert-card-info {
            border-left-color: #38bdf8;
            background: rgba(56, 189, 248, 0.06);
            color: #bae6fd;
        }

        .alert-card-quality {
            border-left-color: #a855f7;
            background: rgba(168, 85, 247, 0.06);
            color: #e9d5ff;
        }

        /* DataFrames & Table elements */
        div[data-testid="stDataFrame"] {
            border: 1px solid #22324e;
            border-radius: 6px;
        }

        /* Clean tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 8px 16px;
            color: #94a3b8;
        }

        .stTabs [aria-selected="true"] {
            color: #38bdf8 !important;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_repository() -> DatabaseRepository:
    return DatabaseRepository()


@st.cache_resource
def load_baseline_and_model():
    models_dir = root_dir / "models"
    baseline_file = models_dir / "baseline.json"
    model_file = models_dir / "isolation_forest.joblib"

    baseline = BaselineModel()
    if baseline_file.exists():
        calibrator = BaselineCalibrator()
        baseline = calibrator.load_baseline(baseline_file)

    detector = IsolationForestDetector()
    if model_file.exists():
        detector.load_model(model_file)

    return baseline, detector


repo = get_repository()
baseline, detector = load_baseline_and_model()

# Setup Session State
if "device_profiles" not in st.session_state:
    st.session_state.device_profiles = config.get_device_profiles()

if "selected_profile_key" not in st.session_state:
    st.session_state.selected_profile_key = "Standard_Indoor_Gateway"

if "pipeline" not in st.session_state:
    profile = st.session_state.device_profiles.get(st.session_state.selected_profile_key)
    st.session_state.pipeline = EDASPipeline(baseline=baseline, ml_detector=detector, device_profile=profile)

if "simulator" not in st.session_state:
    st.session_state.simulator = SensorSimulator(
        pipeline=st.session_state.pipeline,
        repository=repo,
        delay_seconds=1.0,
    )

if "active_dataset_df" not in st.session_state:
    sample_eval = root_dir / "data" / "samples" / "evaluation_labeled.csv"
    if sample_eval.exists():
        reader = CSVReader(pipeline=st.session_state.pipeline)
        df_res, rd, summ = reader.run_csv_pipeline(sample_eval, profile)
        st.session_state.active_dataset_df = df_res
        st.session_state.active_readiness = rd
        st.session_state.active_summary = summ
    else:
        st.session_state.active_dataset_df = pd.DataFrame()
        st.session_state.active_readiness = None
        st.session_state.active_summary = {}


# --- Sidebar Navigation & Hardware Profile Selection ---
with st.sidebar:
    st.markdown("### EDAS-IoT Console")
    st.caption("Environmental Assessment & Anomaly Detection")

    st.markdown("---")
    st.markdown("**TARGET HARDWARE PROFILE**")
    profile_keys = list(st.session_state.device_profiles.keys())
    selected_key = st.selectbox(
        "Deployment Profile:",
        options=profile_keys,
        index=profile_keys.index(st.session_state.selected_profile_key) if st.session_state.selected_profile_key in profile_keys else 0,
        label_visibility="collapsed",
    )

    if selected_key != st.session_state.selected_profile_key:
        st.session_state.selected_profile_key = selected_key
        new_prof = st.session_state.device_profiles[selected_key]
        st.session_state.pipeline.set_device_profile(new_prof)
        sample_file = root_dir / "data" / "samples" / "evaluation_labeled.csv"
        if sample_file.exists():
            reader = CSVReader(pipeline=st.session_state.pipeline)
            df_res, rd, summ = reader.run_csv_pipeline(sample_file, new_prof)
            st.session_state.active_dataset_df = df_res
            st.session_state.active_readiness = rd
            st.session_state.active_summary = summ
        st.rerun()

    active_profile = st.session_state.device_profiles[st.session_state.selected_profile_key]
    st.markdown(f"**{active_profile.name}**")
    st.caption(active_profile.description)

    with st.expander("Profile Specifications", expanded=False):
        st.markdown(f"- Temp Envelope: `{active_profile.min_temperature_c}°C` to `{active_profile.max_temperature_c}°C`")
        st.markdown(f"- Humidity Envelope: `{active_profile.min_humidity_pct}%` to `{active_profile.max_humidity_pct}%`")
        if active_profile.min_pressure_hpa:
            st.markdown(f"- Pressure Envelope: `{active_profile.min_pressure_hpa}` to `{active_profile.max_pressure_hpa} hPa`")
        st.markdown(f"- Required Metrics: `{', '.join(active_profile.required_parameters)}`")
        st.markdown(f"- Safety Margin: `{active_profile.safety_margins.get('temperature_c', 0)}°C` / `{active_profile.safety_margins.get('humidity_pct', 0)}%`")

    st.markdown("---")
    st.markdown("**SUBSYSTEM STATUS**")
    if baseline.is_ready():
        st.markdown(
            '<div style="font-size:0.85rem; color:#34d399; margin-bottom:5px;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#10b981;margin-right:6px;"></span>Baseline: Calibrated ('
            + str(baseline.sample_count)
            + " samples)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.85rem; color:#f87171; margin-bottom:5px;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#ef4444;margin-right:6px;"></span>Baseline: Uncalibrated</div>',
            unsafe_allow_html=True,
        )

    if detector.is_fitted:
        st.markdown(
            '<div style="font-size:0.85rem; color:#34d399; margin-bottom:5px;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#10b981;margin-right:6px;"></span>Isolation Forest: Active (v'
            + str(detector.version)
            + ")</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.85rem; color:#fbbf24; margin-bottom:5px;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#f59e0b;margin-right:6px;"></span>Isolation Forest: Untrained</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**NAVIGATION**")
    menu = st.radio(
        "Navigation Menu:",
        [
            "System Overview",
            "Telemetry & Time-Series",
            "Anomaly Intelligence",
            "Deployment Readiness",
            "Model Performance",
            "Data & Simulation Hub",
        ],
        label_visibility="collapsed",
    )


# --- 1. SYSTEM OVERVIEW ---
if menu == "System Overview":
    st.subheader("System Overview & Operational Health")
    st.caption("Live telemetry metrics, environmental compliance, and active event indicators.")

    df = st.session_state.active_dataset_df
    latest_reading = df.iloc[-1] if not df.empty else None

    # Top Row Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temp_val = f"{latest_reading['temperature_c']:.1f} °C" if latest_reading is not None and pd.notna(latest_reading['temperature_c']) else "N/A"
        st.metric("Latest Temperature", temp_val)
    with col2:
        hum_val = f"{latest_reading['humidity_pct']:.1f} %" if latest_reading is not None and pd.notna(latest_reading['humidity_pct']) else "N/A"
        st.metric("Latest Humidity", hum_val)
    with col3:
        press_val = f"{latest_reading['pressure_hpa']:.1f} hPa" if latest_reading is not None and pd.notna(latest_reading['pressure_hpa']) else "N/A"
        st.metric("Latest Pressure", press_val)
    with col4:
        readiness = st.session_state.active_readiness
        status_val = readiness.status.value if readiness else "INSUFFICIENT_DATA"
        comp = f"{readiness.in_bounds_ratio * 100:.1f}%" if readiness else "0.0%"
        st.metric("Readiness Verdict", status_val, delta=f"{comp} compliant")

    st.markdown("---")

    # Secondary Summary Grid
    summary = st.session_state.active_summary or {}
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Total Observations", summary.get("total_records", 0))
    with c_m2:
        st.metric("Flagged Outliers", summary.get("overall_anomalies", 0))
    with c_m3:
        st.metric("Critical Limit Alerts", summary.get("critical_alerts", 0))
    with c_m4:
        st.metric("Warning Alerts", summary.get("warning_alerts", 0))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Active Telemetry Episodes & Alerts")
    active_alerts = repo.get_active_alerts()
    if active_alerts:
        for alert in active_alerts:
            sev = alert["max_severity"]
            sev_class = "alert-card-info"
            if sev == "CRITICAL":
                sev_class = "alert-card-critical"
            elif sev == "WARNING":
                sev_class = "alert-card-warning"
            elif sev == "DATA_QUALITY":
                sev_class = "alert-card-quality"

            st.markdown(
                f"""
                <div class="alert-card {sev_class}">
                    <div style="font-weight:600; font-size:0.9rem;">
                        [{sev}] Episode {alert['event_id']} on Node {alert['device_id']}
                    </div>
                    <div style="font-size:0.85rem; color:#cbd5e1; margin-top:3px;">
                        {alert['summary']} (Duration: {alert['observation_count']} samples | First: {alert['first_detected']} | Last: {alert['last_detected']})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="alert-card alert-card-info">
                <div style="font-weight:600; font-size:0.85rem; color:#38bdf8;">ALL SYSTEMS NOMINAL</div>
                <div style="font-size:0.82rem; color:#94a3b8;">No active alerts. Telemetry is within calibrated baseline and device limits.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Baseline Parameter Profile")
    if baseline.is_ready():
        param_rows = []
        for param, stats in baseline.parameters.items():
            param_rows.append({
                "Parameter": param,
                "Sample Count": stats.count,
                "Mean": f"{stats.mean:.2f}",
                "Std Dev": f"{stats.std:.2f}",
                "Min Observed": f"{stats.min:.2f}",
                "Max Observed": f"{stats.max:.2f}",
                "Median": f"{stats.median:.2f}",
                "IQR [Q25, Q75]": f"[{stats.q25:.2f}, {stats.q75:.2f}]",
            })
        st.table(pd.DataFrame(param_rows))
    else:
        st.info("Baseline has not been fitted. Calibrate baseline from Data & Simulation Hub.")


# --- 2. TELEMETRY & TIME-SERIES ---
elif menu == "Telemetry & Time-Series":
    st.subheader("Telemetry & Time-Series Monitoring")
    st.caption("Time-series sensor observations with target hardware limits and detected anomaly flags.")

    df = st.session_state.active_dataset_df
    if df.empty:
        st.warning("No sensor telemetry loaded. Please load or generate a dataset in Data & Simulation Hub.")
    else:
        anom_mask = df["is_overall_anomaly"]

        # Helper for clean dark chart layout
        def apply_clean_chart_layout(fig, y_title: str):
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131c2e",
                plot_bgcolor="#0d1424",
                height=320,
                margin=dict(l=20, r=20, t=25, b=20),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, title=y_title),
            )

        # Temperature Plot
        st.markdown("**Temperature (°C)**")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=df["timestamp"], y=df["temperature_c"],
            mode="lines", name="Temperature (°C)",
            line=dict(color="#38bdf8", width=1.8),
        ))
        if active_profile.min_temperature_c is not None:
            fig_temp.add_hline(y=active_profile.min_temperature_c, line_dash="dash", line_color="#f59e0b",
                               annotation_text=f"Min Limit ({active_profile.min_temperature_c}°C)")
        if active_profile.max_temperature_c is not None:
            fig_temp.add_hline(y=active_profile.max_temperature_c, line_dash="dash", line_color="#f43f5e",
                               annotation_text=f"Max Limit ({active_profile.max_temperature_c}°C)")
        if anom_mask.any():
            fig_temp.add_trace(go.Scatter(
                x=df.loc[anom_mask, "timestamp"], y=df.loc[anom_mask, "temperature_c"],
                mode="markers", name="Flagged Outlier",
                marker=dict(color="#f43f5e", size=6, symbol="x"),
                text=df.loc[anom_mask, "explanation"],
            ))
        apply_clean_chart_layout(fig_temp, "°C")
        st.plotly_chart(fig_temp, use_container_width=True)

        # Humidity Plot
        st.markdown("**Relative Humidity (%)**")
        fig_hum = go.Figure()
        fig_hum.add_trace(go.Scatter(
            x=df["timestamp"], y=df["humidity_pct"],
            mode="lines", name="Humidity (%)",
            line=dict(color="#10b981", width=1.8),
        ))
        if active_profile.min_humidity_pct is not None:
            fig_hum.add_hline(y=active_profile.min_humidity_pct, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"Min Limit ({active_profile.min_humidity_pct}%)")
        if active_profile.max_humidity_pct is not None:
            fig_hum.add_hline(y=active_profile.max_humidity_pct, line_dash="dash", line_color="#f43f5e",
                              annotation_text=f"Max Limit ({active_profile.max_humidity_pct}%)")
        if anom_mask.any():
            fig_hum.add_trace(go.Scatter(
                x=df.loc[anom_mask, "timestamp"], y=df.loc[anom_mask, "humidity_pct"],
                mode="markers", name="Flagged Outlier",
                marker=dict(color="#f43f5e", size=6, symbol="x"),
                text=df.loc[anom_mask, "explanation"],
            ))
        apply_clean_chart_layout(fig_hum, "%")
        st.plotly_chart(fig_hum, use_container_width=True)

        # Pressure Plot
        if "pressure_hpa" in df.columns and df["pressure_hpa"].notna().any():
            st.markdown("**Barometric Pressure (hPa)**")
            fig_press = go.Figure()
            fig_press.add_trace(go.Scatter(
                x=df["timestamp"], y=df["pressure_hpa"],
                mode="lines", name="Pressure (hPa)",
                line=dict(color="#a855f7", width=1.8),
            ))
            if anom_mask.any():
                fig_press.add_trace(go.Scatter(
                    x=df.loc[anom_mask, "timestamp"], y=df.loc[anom_mask, "pressure_hpa"],
                    mode="markers", name="Flagged Outlier",
                    marker=dict(color="#f43f5e", size=6, symbol="x"),
                    text=df.loc[anom_mask, "explanation"],
                ))
            apply_clean_chart_layout(fig_press, "hPa")
            st.plotly_chart(fig_press, use_container_width=True)


# --- 3. ANOMALY INTELLIGENCE ---
elif menu == "Anomaly Intelligence":
    st.subheader("Anomaly Intelligence & Root-Cause Inspector")
    st.caption("Granular diagnostic telemetry with rule-based and machine-learning attribution.")

    df = st.session_state.active_dataset_df
    if df.empty:
        st.warning("No records to display. Load telemetry in Data & Simulation Hub.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            severity_filter = st.multiselect(
                "Filter Severity:",
                options=["CRITICAL", "WARNING", "DATA_QUALITY", "INFO", "NORMAL"],
                default=["CRITICAL", "WARNING", "DATA_QUALITY", "INFO"],
            )
        with f2:
            show_anomalies_only = st.checkbox("Flagged Outliers Only", value=True)
        with f3:
            search_query = st.text_input("Search In Reasons:", placeholder="e.g. spike, stuck, drift, multivariate")

        filtered_df = df.copy()
        if severity_filter:
            filtered_df = filtered_df[filtered_df["severity"].isin(severity_filter)]
        if show_anomalies_only:
            filtered_df = filtered_df[filtered_df["is_overall_anomaly"]]
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["explanation"].str.lower().str.contains(q)
                | filtered_df["suspected_cause"].str.lower().str.contains(q)
            ]

        st.caption(f"Displaying {len(filtered_df)} of {len(df)} observations.")

        disp_cols = [
            "timestamp", "device_id", "temperature_c", "humidity_pct", "pressure_hpa",
            "severity", "explanation", "suspected_cause", "is_statistical_anomaly", "is_ml_anomaly", "ground_truth_label"
        ]
        st.dataframe(filtered_df[disp_cols], use_container_width=True, height=400)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Diagnostic Record Inspection")
        if not filtered_df.empty:
            sel_idx = st.selectbox("Select Observation Row:", options=filtered_df.index.tolist())
            row = filtered_df.loc[sel_idx]

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"**Timestamp:** `{row['timestamp']}`")
                st.markdown(f"**Device ID:** `{row['device_id']}`")
                st.markdown(f"**Severity Class:** `{row['severity']}`")
                st.markdown(f"**Ground Truth:** `{row['ground_truth_label']}`")
            with r2:
                st.markdown(f"**Temperature:** `{row['temperature_c']} °C` (Z: `{row['temp_z_score']}`)")
                st.markdown(f"**Humidity:** `{row['humidity_pct']} %` (Z: `{row['humidity_z_score']}`)")
                st.markdown(f"**Pressure:** `{row['pressure_hpa']} hPa` (Z: `{row['pressure_z_score']}`)")
            with r3:
                st.markdown(f"**Statistical Anomaly:** `{row['is_statistical_anomaly']}`")
                st.markdown(f"**Isolation Forest Outlier:** `{row['is_ml_anomaly']}`")
                st.markdown(f"**ML Decision Score:** `{row['ml_decision_score']}`")
                st.markdown(f"**Hardware Violations:** `{row['violated_device_limits']}`")

            st.markdown(
                f"""
                <div class="status-box status-box-insufficient" style="margin-top:0.75rem;">
                    <div style="font-weight:600; font-size:0.85rem; color:#f8fafc; margin-bottom:4px;">SYNTHESIZED OBSERVATION EVIDENCE</div>
                    <div style="font-size:0.85rem; color:#cbd5e1;">{row['explanation']}</div>
                    <div style="font-weight:600; font-size:0.85rem; color:#f8fafc; margin-top:10px; margin-bottom:4px;">SUSPECTED REASONING</div>
                    <div style="font-size:0.85rem; color:#94a3b8;">{row['suspected_cause']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --- 4. DEPLOYMENT READINESS ---
elif menu == "Deployment Readiness":
    st.subheader("Environmental Deployment Readiness Assessment")
    st.caption("Window-based operational suitability check against target IoT device constraints.")

    readiness = st.session_state.active_readiness
    if readiness is None:
        st.warning("No assessment available. Run ingestion on a dataset.")
    else:
        status_val = readiness.status.value

        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown(f"#### Profile: `{active_profile.name}`")
            st.caption(active_profile.description)
            st.markdown("---")
            st.markdown("**Manufacturer Operating Envelope**")
            st.markdown(f"- Temp Envelope: `[{active_profile.min_temperature_c}, {active_profile.max_temperature_c}] °C`")
            st.markdown(f"- Humidity Envelope: `[{active_profile.min_humidity_pct}, {active_profile.max_humidity_pct}] %`")
            if active_profile.min_pressure_hpa:
                st.markdown(f"- Pressure Envelope: `[{active_profile.min_pressure_hpa}, {active_profile.max_pressure_hpa}] hPa`")
            st.markdown(f"- Safety Margin: `{active_profile.safety_margins.get('temperature_c', 0)} °C` / `{active_profile.safety_margins.get('humidity_pct', 0)} %`")

        with col_right:
            if status_val == "READY":
                box_class = "status-box-ready"
                header_text = "VERDICT: READY FOR DEPLOYMENT"
                desc_text = "Measured environmental observations satisfy all manufacturer operating limits throughout the assessment window."
            elif status_val == "WARNING":
                box_class = "status-box-warning"
                header_text = "VERDICT: WARNING - APPROACHING LIMITS"
                desc_text = "Conditions are currently within bounds but frequently approach safety margins or exhibited brief excursions."
            elif status_val == "NOT_READY":
                box_class = "status-box-not-ready"
                header_text = "VERDICT: NOT READY FOR DEPLOYMENT"
                desc_text = "The monitored location violates device operational limits. Permanent deployment risks device degradation or outage."
            elif status_val == "INSUFFICIENT_DATA":
                box_class = "status-box-insufficient"
                header_text = "VERDICT: INSUFFICIENT DATA"
                desc_text = "Observation window does not contain enough valid data samples to establish confident deployment readiness."
            else:
                box_class = "status-box-not-ready"
                header_text = "VERDICT: SENSOR DATA INVALID"
                desc_text = "Telemetry quality issues prevent trustworthy environmental assessment."

            st.markdown(
                f"""
                <div class="status-box {box_class}">
                    <div style="font-weight:700; font-size:1.15rem; letter-spacing:0.02em;">{header_text}</div>
                    <div style="font-size:0.9rem; margin-top:4px;">{desc_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Window Metrics
            w1, w2, w3 = st.columns(3)
            with w1:
                st.metric("Window Size", f"{readiness.window_size} samples")
            with w2:
                st.metric("Valid Telemetry", f"{readiness.valid_sample_count} samples")
            with w3:
                st.metric("Compliance Rate", f"{readiness.in_bounds_ratio * 100:.1f}%")

            st.markdown("#### Assessment Findings & Evidence")
            for reason in readiness.reasons:
                st.markdown(f"- {reason}")

            if readiness.warnings:
                st.markdown("#### Advisory Warnings")
                for w in readiness.warnings:
                    st.markdown(f"- Advisory: {w}")

            if readiness.violated_parameters:
                st.error(f"Hardware Constraint Breaches: {', '.join(readiness.violated_parameters)}")


# --- 5. MODEL PERFORMANCE ---
elif menu == "Model Performance":
    st.subheader("Model Performance & Benchmark Evaluation")
    st.caption("Quantitative metrics on labelled evaluation datasets with 10 controlled anomaly types.")

    report_path = root_dir / "data" / "processed" / "evaluation_report.json"
    report = None
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

    if report is None:
        st.info("Evaluation report not found. Click below to generate.")
        if st.button("Execute Model Evaluation"):
            from scripts.evaluate_model import evaluate_pipeline_on_dataset
            report = evaluate_pipeline_on_dataset()
            st.success("Evaluation completed.")
            st.rerun()
    else:
        st.markdown(f"**Dataset:** `{report['dataset']}` | Total Samples: `{report['total_records']}` (Normal: `{report['normal_records']}`, Anomalous: `{report['anomalous_records']}`)")

        ov = report["overall_pipeline"]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Precision", f"{ov['precision'] * 100:.2f}%")
        with m2:
            st.metric("Recall", f"{ov['recall'] * 100:.2f}%")
        with m3:
            st.metric("F1-Score", f"{ov['f1_score']:.4f}")
        with m4:
            st.metric("False Positive Rate", f"{ov['false_positive_rate'] * 100:.2f}%")

        st.markdown("---")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown("#### Confusion Matrix")
            cm = ov["confusion_matrix"]
            cm_data = [[cm["true_negatives"], cm["false_positives"]], [cm["false_negatives"], cm["true_positives"]]]
            fig_cm = px.imshow(
                cm_data,
                labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count"),
                x=["Normal (0)", "Anomaly (1)"],
                y=["Normal (0)", "Anomaly (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_dark",
                paper_bgcolor="#131c2e",
                plot_bgcolor="#0d1424",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_right:
            st.markdown("#### Subsystem Comparison")
            comp_rows = [
                {"Pipeline Layer": "Overall Pipeline (Validation + Stat + ML)", "Precision": f"{ov['precision']*100:.1f}%", "Recall": f"{ov['recall']*100:.1f}%", "F1": f"{ov['f1_score']:.3f}"},
                {"Pipeline Layer": "Statistical Detector Alone", "Precision": f"{report['statistical_detector']['precision']*100:.1f}%", "Recall": f"{report['statistical_detector']['recall']*100:.1f}%", "F1": f"{report['statistical_detector']['f1_score']:.3f}"},
                {"Pipeline Layer": "Isolation Forest Alone", "Precision": f"{report['isolation_forest_ml']['precision']*100:.1f}%", "Recall": f"{report['isolation_forest_ml']['recall']*100:.1f}%", "F1": f"{report['isolation_forest_ml']['f1_score']:.3f}"},
            ]
            st.table(pd.DataFrame(comp_rows))

        st.markdown("---")
        st.markdown("#### Anomaly-Type Detection Breakdown")
        breakdown = report.get("anomaly_type_breakdown", {})
        breakdown_rows = []
        for atype, stats in breakdown.items():
            if atype != "NORMAL":
                breakdown_rows.append({
                    "Anomaly Type": atype,
                    "Injected Total": stats["total_samples"],
                    "Detected Total": stats["overall_detected"],
                    "Detection Rate": f"{stats['overall_detection_rate'] * 100:.1f}%",
                    "Caught by ML": stats["ml_detected"],
                    "Caught by Stat": stats["stat_detected"],
                })
        st.table(pd.DataFrame(breakdown_rows))


# --- 6. DATA & SIMULATION HUB ---
elif menu == "Data & Simulation Hub":
    st.subheader("Data & Simulation Hub")
    st.caption("File batch ingestion, live stream simulation controls, and baseline calibration.")

    tab1, tab2, tab3 = st.tabs(["CSV Batch Ingestion", "Simulated Telemetry Stream", "Baseline Calibration"])

    with tab1:
        st.markdown("#### Upload Sensor Telemetry (CSV)")
        uploaded_file = st.file_uploader("Upload CSV formatted data:", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                reader = CSVReader(pipeline=st.session_state.pipeline)
                df_res, rd, summ = reader.run_csv_pipeline(uploaded_file, active_profile)
                st.session_state.active_dataset_df = df_res
                st.session_state.active_readiness = rd
                st.session_state.active_summary = summ
                st.success(f"Processed {len(df_res)} records successfully.")
                st.dataframe(df_res.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

        st.markdown("---")
        if not st.session_state.active_dataset_df.empty:
            csv_bytes = st.session_state.active_dataset_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Processed Records (CSV)",
                data=csv_bytes,
                file_name=f"edas_iot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with tab2:
        st.markdown("#### Real-Time Telemetry Stream Simulation")
        sim = st.session_state.simulator
        sim_progress = sim.get_progress()

        s1, s2, s3 = st.columns(3)
        with s1:
            if st.button("Step Simulation (1 Sample)"):
                step_res = sim.step()
                if step_res:
                    st.success(f"Streamed reading at {step_res.raw.timestamp}: Temp={step_res.raw.temperature_c}°C, Hum={step_res.raw.humidity_pct}%")
                    recent = repo.get_recent_readings(100)
                    if recent:
                        st.session_state.active_dataset_df = pd.DataFrame(recent)
                st.rerun()

        with s2:
            delay = st.slider("Step Interval (seconds):", 0.1, 5.0, float(sim_progress["delay_seconds"]))
            sim.set_delay(delay)

        with s3:
            st.markdown(f"Status: **{'RUNNING' if sim.is_streaming() else 'IDLE'}**")
            st.caption(f"Stream Pointer: {sim_progress['current_index']} / {sim_progress['total_records']}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Controlled Anomaly Injection")
        st.caption("Schedule an on-the-fly anomaly to be injected into subsequent stream steps:")
        i1, i2, i3 = st.columns(3)
        with i1:
            if st.button("Inject Temperature Spike (+15°C)"):
                sim.inject_anomaly("TEMPERATURE_SPIKE", duration_steps=5)
                st.success("Scheduled Temperature Spike.")
            if st.button("Inject Stuck Sensor"):
                sim.inject_anomaly("STUCK_SENSOR", duration_steps=6)
                st.success("Scheduled Stuck Sensor.")
        with i2:
            if st.button("Inject Out-of-Range Humidity (98%)"):
                sim.inject_anomaly("HUMIDITY_OUT_OF_RANGE", duration_steps=5)
                st.success("Scheduled Out-of-Range Humidity.")
            if st.button("Inject Missing Data Dropout"):
                sim.inject_anomaly("MISSING_DATA", duration_steps=4)
                st.success("Scheduled Sensor Dropout.")
        with i3:
            if st.button("Inject Multivariate Anomaly"):
                sim.inject_anomaly("MULTIVARIATE_ANOMALY", duration_steps=6)
                st.success("Scheduled Multivariate Anomaly.")
            if st.button("Inject Implausible Measurement"):
                sim.inject_anomaly("INVALID_MEASUREMENT", duration_steps=3)
                st.success("Scheduled Implausible Reading.")

    with tab3:
        st.markdown("#### Baseline Calibration & Model Retraining")
        st.caption("Calibrate statistical parameters and refit the Isolation Forest model on normal data.")

        train_file = root_dir / "data" / "samples" / "baseline_normal.csv"
        if train_file.exists():
            st.markdown(f"Normal Reference Dataset: `{train_file.name}`")
            if st.button("Calibrate Baseline & Retrain Model"):
                with st.spinner("Calibrating and fitting Isolation Forest..."):
                    from scripts.train_model import train_and_calibrate
                    train_and_calibrate()
                    st.success("Baseline calibration and model training complete.")
                    st.rerun()
        else:
            st.warning("Baseline reference file not found. Run scripts/generate_data.py.")
