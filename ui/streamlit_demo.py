"""
🎯 Streamlit Demo for Explainable Anomaly Detection System
=========================================================

Interactive demonstration for staff panel presentation showcasing:
- 96.2% accuracy ML anomaly detection
- SHAP-based explainable AI
- Root cause analysis
- Professional visualization

Author: ML Implementation Team
Date: September 28, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add ml_implementation directory to path to import our ML modules
ml_implementation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml_implementation')
sys.path.append(ml_implementation_path)

try:
    from anomaly_detector import AnomalyDetector
    from explainability_rca import ExplainabilityRCA
except ImportError:
    st.error("❌ Could not import ML modules. Make sure you're running from the correct directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🧠 Explainable Anomaly Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .normal { 
        background: linear-gradient(135deg, #d4edda, #c3e6cb); 
        color: #155724; 
        border: 2px solid #28a745; 
    }
    .cpu_spike { 
        background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
        color: #856404; 
        border: 2px solid #ffc107; 
    }
    .memory_leak { 
        background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
        color: #721c24; 
        border: 2px solid #dc3545; 
    }
    .service_crash { 
        background: linear-gradient(135deg, #d1ecf1, #bee5eb); 
        color: #0c5460; 
        border: 2px solid #17a2b8; 
    }
    .rca-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .recommendation-item {
        margin: 0.8rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 4px solid #2196f3;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .evidence-item {
        margin: 0.4rem 0;
        padding: 0.6rem;
        background-color: #fff3cd;
        border-left: 3px solid #ffc107;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load ML models with caching for performance"""
    try:
        # Get the path to ml_implementation directory
        ml_implementation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml_implementation')
        
        detector = AnomalyDetector()
        # Load model from ml_implementation directory
        model_path = os.path.join(ml_implementation_path, 'anomaly_detection_model.pkl')
        detector.load_model(model_path)
        
        explainer = ExplainabilityRCA()
        # Load model from ml_implementation directory
        explainer.load_model(model_path)
        
        return detector, explainer
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error(f"💡 Make sure the ML models are trained first by running integrated_demo.py in ml_implementation/")
        return None, None

def create_metrics_input():
    """Create the metrics input interface"""
    st.sidebar.markdown("## 🔧 System Metrics Input")
    
    # Predefined scenarios
    st.sidebar.markdown("### 📋 Quick Scenarios")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🟢 Normal", use_container_width=True):
            set_scenario("normal")
    
    with col2:
        if st.button("🔴 CPU Spike", use_container_width=True):
            set_scenario("cpu_spike")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("🟡 Memory Leak", use_container_width=True):
            set_scenario("memory_leak")
    
    with col4:
        if st.button("🟠 Service Crash", use_container_width=True):
            set_scenario("service_crash")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Manual Input")
    
    # CPU Metrics
    st.sidebar.markdown("**CPU Usage (%)**")
    notification_cpu = st.sidebar.slider("Notification Service CPU", 0, 100, st.session_state.get('notification_cpu', 20), key="notification_cpu")
    web_api_cpu = st.sidebar.slider("Web API CPU", 0, 100, st.session_state.get('web_api_cpu', 25), key="web_api_cpu")
    processor_cpu = st.sidebar.slider("Processor CPU", 0, 100, st.session_state.get('processor_cpu', 22), key="processor_cpu")
    
    # Memory Metrics
    st.sidebar.markdown("**Memory Usage (%)**")
    notification_memory = st.sidebar.slider("Notification Service Memory", 0, 100, st.session_state.get('notification_memory', 45), key="notification_memory")
    web_api_memory = st.sidebar.slider("Web API Memory", 0, 100, st.session_state.get('web_api_memory', 50), key="web_api_memory")
    processor_memory = st.sidebar.slider("Processor Memory", 0, 100, st.session_state.get('processor_memory', 48), key="processor_memory")
    
    # Response Times
    st.sidebar.markdown("**Response Times (ms)**")
    notification_response_time = st.sidebar.slider("Notification Response Time P95", 50, 1000, st.session_state.get('notification_response_time_p95', 120), key="notification_response_time_p95")
    web_api_response_time = st.sidebar.slider("Web API Response Time P95", 50, 1000, st.session_state.get('web_api_response_time_p95', 150), key="web_api_response_time_p95")
    processor_response_time = st.sidebar.slider("Processor Response Time P95", 50, 1000, st.session_state.get('processor_response_time_p95', 100), key="processor_response_time_p95")
    
    # Error Rates
    st.sidebar.markdown("**Error Rates**")
    notification_error_rate = st.sidebar.slider("Notification Error Rate", 0.0, 1.0, st.session_state.get('notification_error_rate', 0.01), step=0.01, key="notification_error_rate")
    
    return {
        'notification_cpu': notification_cpu,
        'web_api_cpu': web_api_cpu,
        'processor_cpu': processor_cpu,
        'notification_memory': notification_memory,
        'web_api_memory': web_api_memory,
        'processor_memory': processor_memory,
        'notification_response_time_p95': notification_response_time,
        'web_api_response_time_p95': web_api_response_time,
        'processor_response_time_p95': processor_response_time,
        'notification_error_rate': notification_error_rate,
        # Set defaults for remaining features
        'processor_error_rate': 0.005,
        'web_api_errors': 0.5,
        'notification_api_health': 1.0,
        'processor_redis_health': 1.0,
        'web_api_redis_health': 1.0,
        'notification_delivery_success': 0.99,
        'notification_message_rate': 50.0,
        'processor_processing_rate': 100.0,
        'web_api_requests': 200.0,
        'web_api_requests_per_second': 45.0,
        'notification_queue': 5.0,
        'notification_queue_depth': 2.0,
        'processor_queue': 3.0,
        'processor_queue_depth': 1.5,
        'web_api_queue_depth': 2.0,
        'processor_db_connections': 8.0,
        'web_api_db_connections': 10.0,
        'notification_thread_count': 10.0,
        'processor_thread_count': 15.0,
        'web_api_thread_count': 20.0,
        'processor_memory_growth': 0.1
    }

def set_scenario(scenario_type):
    """Set predefined scenario values"""
    scenarios = {
        "normal": {
            'notification_cpu': 15, 'web_api_cpu': 25, 'processor_cpu': 20,
            'notification_memory': 45, 'web_api_memory': 50, 'processor_memory': 48,
            'notification_response_time_p95': 120, 'web_api_response_time_p95': 150, 'processor_response_time_p95': 100,
            'notification_error_rate': 0.01
        },
        "cpu_spike": {
            'notification_cpu': 85, 'web_api_cpu': 90, 'processor_cpu': 88,
            'notification_memory': 47, 'web_api_memory': 52, 'processor_memory': 49,
            'notification_response_time_p95': 350, 'web_api_response_time_p95': 400, 'processor_response_time_p95': 300,
            'notification_error_rate': 0.05
        },
        "memory_leak": {
            'notification_cpu': 18, 'web_api_cpu': 28, 'processor_cpu': 22,
            'notification_memory': 85, 'web_api_memory': 88, 'processor_memory': 92,
            'notification_response_time_p95': 250, 'web_api_response_time_p95': 280, 'processor_response_time_p95': 200,
            'notification_error_rate': 0.03
        },
        "service_crash": {
            'notification_cpu': 25, 'web_api_cpu': 30, 'processor_cpu': 28,
            'notification_memory': 55, 'web_api_memory': 60, 'processor_memory': 63,
            'notification_response_time_p95': 850, 'web_api_response_time_p95': 950, 'processor_response_time_p95': 800,
            'notification_error_rate': 0.75
        }
    }
    
    for key, value in scenarios[scenario_type].items():
        st.session_state[key] = value

def create_prediction_display(prediction):
    """Create the prediction display with styling"""
    predicted_class = prediction['predicted_class'].lower()
    confidence = prediction['confidence']
    
    # Determine styling based on prediction
    if predicted_class == 'normal':
        css_class = "normal"
        icon = "🟢"
        status = "NORMAL OPERATION"
    elif predicted_class == 'cpu_spike':
        css_class = "cpu_spike"
        icon = "🔴"
        status = "CPU SPIKE DETECTED"
    elif predicted_class == 'memory_leak':
        css_class = "memory_leak"
        icon = "🟡"
        status = "MEMORY LEAK DETECTED"
    else:
        css_class = "service_crash"
        icon = "🟠"
        status = "SERVICE CRASH DETECTED"
    
    st.markdown(f"""
    <div class="prediction-box {css_class}">
        <div style="font-size: 2rem;">{icon}</div>
        <div style="font-size: 1.5rem; margin: 0.5rem 0;">{status}</div>
        <div style="font-size: 1.2rem;">Confidence: {confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    return predicted_class

def create_feature_importance_chart(explanation):
    """Create interactive feature importance chart"""
    if explanation.get('shap_explanation') and explanation['shap_explanation']:
        shap_data = explanation['shap_explanation']['all_values']
        
        # Get top 10 most important features
        sorted_features = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        # Better color scheme for positive/negative values
        colors = ['#dc3545' if v < 0 else '#28a745' for v in values]  # Red for negative, Green for positive
        
        fig = go.Figure(data=go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.3f}<br>' +
                         '%{customdata}<extra></extra>',
            customdata=['Supports this prediction' if v > 0 else 'Against this prediction' for v in values]
        ))
        
        fig.update_layout(
            title="🧠 SHAP Feature Importance (Top 10)<br><sub>🟢 Green: Supports prediction | 🔴 Red: Against prediction</sub>",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Features",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add vertical line at x=0 for better reference
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
        
        return fig
    return None

def create_metrics_overview_chart(metrics):
    """Create metrics overview radar chart"""
    # Select key metrics for visualization
    key_metrics = {
        'CPU (Avg)': (metrics['notification_cpu'] + metrics['web_api_cpu'] + metrics['processor_cpu']) / 3,
        'Memory (Avg)': (metrics['notification_memory'] + metrics['web_api_memory'] + metrics['processor_memory']) / 3,
        'Response Time (Avg)': (metrics['notification_response_time_p95'] + metrics['web_api_response_time_p95'] + metrics['processor_response_time_p95']) / 3 / 10,  # Scale down
        'Error Rate': metrics['notification_error_rate'] * 100,
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(key_metrics.values()),
        theta=list(key_metrics.keys()),
        fill='toself',
        name='Current Metrics',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="📊 System Metrics Overview",
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<div class="main-header">🧠 Explainable AI Anomaly Detection System</div>', unsafe_allow_html=True)
    
    # Load models
    detector, explainer = load_models()
    if not detector or not explainer:
        st.error("❌ Failed to load ML models. Please check the model files.")
        return
    
    # System info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Model Accuracy", "96.2%", "4.2%")
    with col2:
        st.metric("🔢 Features", "29", "")
    with col3:
        st.metric("📊 Classes", "4", "")
    with col4:
        st.metric("🧠 Explainability", "SHAP", "✅")
    
    st.markdown("---")
    
    # Get user input
    metrics = create_metrics_input()
    
    # Main content area
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.markdown("## 📊 System Metrics Visualization")
        
        # Metrics overview chart
        metrics_chart = create_metrics_overview_chart(metrics)
        st.plotly_chart(metrics_chart, use_container_width=True)
        
        # Key metrics display
        st.markdown("### 📈 Current Values")
        
        cpu_avg = (metrics['notification_cpu'] + metrics['web_api_cpu'] + metrics['processor_cpu']) / 3
        memory_avg = (metrics['notification_memory'] + metrics['web_api_memory'] + metrics['processor_memory']) / 3
        response_avg = (metrics['notification_response_time_p95'] + metrics['web_api_response_time_p95'] + metrics['processor_response_time_p95']) / 3
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("CPU Average", f"{cpu_avg:.1f}%", f"{cpu_avg-25:.1f}%")
            st.metric("Memory Average", f"{memory_avg:.1f}%", f"{memory_avg-50:.1f}%")
        with col_m2:
            st.metric("Response Time Avg", f"{response_avg:.0f}ms", f"{response_avg-150:.0f}ms")
            st.metric("Error Rate", f"{metrics['notification_error_rate']:.2%}", f"{metrics['notification_error_rate']-0.01:.2%}")
    
    with col_right:
        st.markdown("## 🎯 AI Prediction & Analysis")
        
        # Make prediction
        try:
            prediction = detector.predict(metrics)
            predicted_class = create_prediction_display(prediction)
            
            # Get explanation
            explanation = explainer.explain_prediction(metrics, prediction)
            
            # Feature importance chart
            if explanation.get('shap_explanation'):
                st.markdown("#### 🧠 SHAP Explainability")
                st.markdown("""
                <div style='background-color: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #17a2b8; color: #212529;'>
                    <small style='color: #495057;'><strong>📊 How to Read This Chart:</strong><br>
                    🟢 <strong>Green bars</strong>: Features that <em>support</em> the current prediction<br>
                    🔴 <strong>Red bars</strong>: Features that <em>work against</em> the current prediction<br>
                    📏 <strong>Bar length</strong>: Strength of the feature's influence</small>
                </div>
                """, unsafe_allow_html=True)
                
                shap_chart = create_feature_importance_chart(explanation)
                if shap_chart:
                    st.plotly_chart(shap_chart, use_container_width=True)
            
            # Root cause analysis with improved styling
            st.markdown("### 🔍 Root Cause Analysis")
            rca = explanation['root_cause_analysis']
            
            # Severity with better styling
            severity_colors = {
                'low': ('🟢', '#28a745'), 
                'medium': ('🟡', '#ffc107'), 
                'high': ('🔴', '#dc3545'), 
                'critical': ('🚨', '#6f42c1')
            }
            severity = rca['severity']
            icon, color = severity_colors.get(severity, ('⚪', '#6c757d'))
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 0.8rem; border-radius: 8px; border-left: 4px solid {color}; margin: 1rem 0;">
                <strong style="font-size: 1.1rem;">Severity: {icon} {severity.upper()}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Evidence section with better formatting
            if rca['evidence']:
                st.markdown("**📋 Evidence:**")
                for evidence in rca['evidence']:
                    st.markdown(f"""
                    <div style='margin: 0.4rem 0; padding: 0.6rem; background-color: #fff3cd; border-left: 3px solid #ffc107; border-radius: 0 5px 5px 0; color: #856404;'>
                        • {evidence}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Root causes with improved numbering
            st.markdown("**🎯 Likely Root Causes:**")
            for i, cause in enumerate(rca['likely_root_causes'][:3], 1):
                st.markdown(f"""
                <div style='margin: 0.5rem 0; padding: 0.8rem; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid #6c757d; color: #212529;'>
                    <strong style='color: #495057;'>{i}.</strong> {cause}
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations with enhanced styling
            st.markdown("### 💡 Actionable Recommendations")
            recommendations = explanation['recommendations']
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style='margin: 0.7rem 0; padding: 1rem; background: linear-gradient(135deg, #e3f2fd, #bbdefb); border-left: 4px solid #2196f3; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); color: #0d47a1;'>
                    <strong style='color: #1976d2; font-size: 1.1rem;'>{i}.</strong> 
                    <span style='margin-left: 0.5rem; color: #0d47a1;'>{rec}</span>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📋 System Status")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.success("✅ ML Model: Loaded")
    with col_s2:
        st.success("✅ Explainability: Active")
    with col_s3:
        st.success("✅ Real-time Analysis: Ready")

if __name__ == "__main__":
    main()