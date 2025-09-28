# 🎯 Streamlit Demo Instructions

## Quick Start for Staff Panel Presentation

### 1. Launch the Demo
```bash
# Option 1: Direct launch (recommended)
cd ml_implementation/ui
python launch_demo.py

# Option 2: Manual launch
cd ml_implementation/ui
streamlit run streamlit_demo.py
```

### 2. Demo Features

#### 🎛️ Interactive Controls (Left Sidebar)
- **Quick Scenarios**: Pre-configured system states
  - 🟢 Normal Operation
  - 🔴 CPU Spike 
  - 🟡 Memory Leak
  - 🟠 Service Crash

- **Manual Sliders**: Fine-tune metrics manually
  - CPU usage for all 3 services
  - Memory usage for all 3 services  
  - Response times (P95)
  - Error rates

#### 📊 Main Dashboard (Right Side)
- **System Metrics Visualization**: Radar chart showing current state
- **AI Prediction**: Real-time classification with confidence
- **SHAP Explanations**: Top 10 feature importance with impact values
- **Root Cause Analysis**: Evidence-based diagnosis
- **Actionable Recommendations**: Specific remediation steps

### 3. Presentation Flow for Staff Panel

#### Opening (30 seconds)
1. Start with "Normal" scenario → Show 96.2% model accuracy
2. Explain: "This is our explainable AI system for microservice anomaly detection"

#### Core Demo (2-3 minutes)  
1. **CPU Spike Scenario**: 
   - Click "CPU Spike" button
   - Show immediate detection and explanation
   - Highlight SHAP values showing CPU metrics as key factors

2. **Memory Leak Scenario**:
   - Switch to "Memory Leak" 
   - Show different root cause analysis
   - Demonstrate actionable recommendations

3. **Interactive Capability**:
   - Adjust a few sliders manually
   - Show real-time prediction updates
   - Emphasize transparency via SHAP explanations

#### Technical Highlights (1 minute)
- **96.2% Accuracy**: Validated on realistic dataset
- **29 Features**: Comprehensive system monitoring
- **4 Anomaly Classes**: Normal, CPU Spike, Memory Leak, Service Crash
- **SHAP Integration**: Full explainability for every prediction
- **Real-time Analysis**: Instant classification and recommendations

### 4. Key Talking Points

#### Business Value
- "Reduces mean time to resolution by providing immediate root cause analysis"
- "Proactive anomaly detection prevents service outages"
- "Explainable AI ensures trust and transparency in automated decisions"

#### Technical Innovation
- "SHAP-based explainability shows exactly which metrics triggered each prediction"
- "Rule-based root cause analysis provides actionable insights"
- "Real-time processing enables immediate response to system anomalies"

#### Academic Contribution
- "Novel integration of SHAP explainability with microservice monitoring"
- "Comprehensive dataset with realistic class overlap and noise"
- "Production-ready implementation with professional visualization"

### 5. Troubleshooting

If the demo doesn't load:
1. Check that you're in the `ml_implementation/ui/` directory
2. Verify models are trained: `../anomaly_detector.py` should exist
3. Ensure packages are installed: `pip install streamlit plotly`

### 6. Demo URLs
- **Local URL**: http://localhost:8501
- **Network URL**: Will be displayed in terminal after launch

### 7. Advanced Features to Mention

- **Timestamped Results**: All predictions saved with explanations
- **Professional Visualizations**: Publication-ready charts and metrics  
- **Modular Architecture**: Easy integration with existing monitoring systems
- **Scalable Design**: Can handle multiple services and custom metrics

---

**💡 Pro Tip**: Practice the scenario switches beforehand - the transitions are instant and impressive for live demos!

**🎯 Success Metrics**: 
- Model accuracy: 96.2%
- Feature count: 29 comprehensive metrics
- Explanation coverage: 100% of predictions explained
- Real-time performance: <1 second response time