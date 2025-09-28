# 📁 Results Folder Structure

This folder contains all outputs from the Integrated Anomaly Detection & Explainability Demo runs.

## 📂 Folder Naming Convention

Each demo run creates a timestamped folder: `demo_YYYYMMDD_HHMMSS`

## 📄 Generated Files

Each demo run produces:

### 📊 **Visualizations**

- `confusion_matrix.png` - Model performance heatmap
- `feature_importance.png` - Feature analysis chart

### 📝 **Explanation Reports**

- `explanation_normal_operations.txt` - Normal system state analysis
- `explanation_cpu_spike_emergency.txt` - CPU spike anomaly analysis
- `explanation_memory_leak_detected.txt` - Memory leak anomaly analysis

## 🎯 Usage

These results provide:

- ✅ **Model Validation**: Confusion matrix and performance metrics
- 🔍 **Feature Analysis**: Which metrics are most important for predictions
- 💡 **Explainability**: SHAP values and root cause analysis for each anomaly type
- 📋 **Actionable Insights**: Specific recommendations for each detected issue

## 📈 Integration

Results can be used for:

- **Production Monitoring**: Baseline normal operations patterns
- **Alert Tuning**: Understanding feature thresholds for each anomaly type
- **DevOps Training**: Real examples of explainable AI for incident response
- **System Documentation**: Evidence-based troubleshooting guides
