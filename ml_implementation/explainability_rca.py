"""
🔍 Explainable AI & Root Cause Analysis Module
==============================================

This module provides explainability for anomaly detection results using:
1. Feature importance analysis
2. SHAP (SHapley Additive exPlanations) values
3. Root cause analysis with actionable insights
4. Visualization of explanations

Author: ML Implementation Team
Date: September 28, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib  # Changed from pickle to joblib
import warnings
warnings.filterwarnings('ignore')

class ExplainabilityRCA:
    """
    Explainable AI and Root Cause Analysis for Anomaly Detection
    """
    
    def __init__(self):
        """Initialize the explainability engine"""
        self.model = None
        self.feature_columns = None
        self.explainer = None
        self.shap_values = None
        self.feature_importance = None
        self.rca_rules = self._initialize_rca_rules()
        print("🔍 Explainability & RCA Engine initialized")
    
    def load_model(self, model_path='anomaly_detection_model.pkl'):
        """Load the trained anomaly detection model"""
        try:
            # Load using joblib instead of pickle
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"📊 Model features: {len(self.feature_columns)}")
            
            # Initialize SHAP explainer
            self._initialize_shap_explainer()
            
        except FileNotFoundError:
            print(f"❌ Error: Model file {model_path} not found")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for the model"""
        try:
            # For Random Forest, use Tree explainer
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize SHAP explainer: {e}")
    
    def _initialize_rca_rules(self):
        """Initialize root cause analysis rules"""
        return {
            'cpu_spike': {
                'primary_indicators': ['processor_cpu', 'web_api_cpu', 'notification_cpu'],
                'secondary_indicators': ['processor_response_time_p95', 'web_api_response_time_p95'],
                'thresholds': {'cpu': 70, 'response_time': 200},
                'root_causes': [
                    "High CPU utilization across services",
                    "Resource contention between processes",
                    "Inefficient algorithms or infinite loops",
                    "Sudden spike in request volume"
                ],
                'recommendations': [
                    "Scale up CPU resources or add more instances",
                    "Implement CPU-based auto-scaling",
                    "Profile code for CPU-intensive operations",
                    "Add request rate limiting and load balancing"
                ]
            },
            'memory_leak': {
                'primary_indicators': ['processor_memory', 'web_api_memory', 'notification_memory'],
                'secondary_indicators': ['processor_memory_growth', 'processor_response_time_p95'],
                'thresholds': {'memory': 80, 'memory_growth': 2.0},
                'root_causes': [
                    "Memory not being properly released",
                    "Accumulating objects in memory",
                    "Large data structures not being cleaned up",
                    "Connection pooling issues"
                ],
                'recommendations': [
                    "Implement proper memory cleanup and garbage collection",
                    "Review object lifecycle management",
                    "Add memory monitoring and alerts",
                    "Implement connection pooling limits"
                ]
            },
            'service_crash': {
                'primary_indicators': ['notification_error_rate', 'web_api_response_time_p95'],
                'secondary_indicators': ['notification_api_health', 'processor_redis_health'],
                'thresholds': {'error_rate': 0.3, 'response_time': 400},
                'root_causes': [
                    "Service becoming unresponsive",
                    "Database connection failures",
                    "Unhandled exceptions causing crashes",
                    "Resource exhaustion leading to failures"
                ],
                'recommendations': [
                    "Implement health checks and circuit breakers",
                    "Add proper error handling and logging",
                    "Set up service restart policies",
                    "Monitor and alert on critical service metrics"
                ]
            },
            'normal': {
                'primary_indicators': ['processor_cpu', 'processor_memory'],
                'secondary_indicators': ['processor_response_time_p95'],
                'thresholds': {'cpu': 50, 'memory': 70, 'response_time': 150},
                'root_causes': [
                    "System operating within normal parameters",
                    "All services healthy and responsive"
                ],
                'recommendations': [
                    "Continue monitoring for any changes",
                    "Maintain current resource allocation"
                ]
            }
        }
    
    def explain_prediction(self, data, prediction_result):
        """
        Provide comprehensive explanation for a prediction
        
        Args:
            data: Input data dictionary
            prediction_result: Result from anomaly detector
            
        Returns:
            Dictionary containing explanation details
        """
        predicted_class = prediction_result['predicted_class'].lower()
        confidence = prediction_result['confidence']
        
        print(f"\n🔍 EXPLAINABILITY ANALYSIS")
        print("=" * 50)
        print(f"🎯 Predicted Anomaly: {predicted_class.upper()}")
        print(f"📊 Confidence: {confidence:.1%}")
        
        # Prepare data for analysis
        df_input = pd.DataFrame([data])
        
        # Ensure all required features are present
        for feature in self.feature_columns:
            if feature not in df_input.columns:
                df_input[feature] = 0.0
        
        # Reorder columns to match model training
        df_input = df_input[self.feature_columns]
        
        # Get SHAP explanations
        shap_explanation = self._get_shap_explanation(df_input)
        
        # Get feature importance analysis
        feature_analysis = self._analyze_features(data, predicted_class)
        
        # Get root cause analysis
        rca_analysis = self._perform_rca(data, predicted_class)
        
        explanation = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'shap_explanation': shap_explanation,
            'feature_analysis': feature_analysis,
            'root_cause_analysis': rca_analysis,
            'recommendations': self.rca_rules[predicted_class]['recommendations']
        }
        
        # Display comprehensive explanation
        self._display_explanation(explanation)
        
        return explanation
    
    def _get_shap_explanation(self, df_input):
        """Get SHAP explanation for the prediction"""
        if self.explainer is None:
            return None
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(df_input)
            
            if isinstance(shap_values, list):
                # Multi-class returns list of arrays, one per class
                # Get predicted class to select appropriate SHAP values
                prediction = self.model.predict(df_input)[0]
                classes = list(self.model.classes_)
                class_idx = classes.index(prediction) if prediction in classes else 0
                
                shap_vals = shap_values[class_idx][0]  # [class][sample] 
                
            elif isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) == 3:
                    # Shape is (samples, features, classes) - get first sample and predicted class
                    prediction = self.model.predict(df_input)[0]
                    classes = list(self.model.classes_)
                    class_idx = classes.index(prediction) if prediction in classes else 0
                    
                    shap_vals = shap_values[0, :, class_idx]  # [sample=0, all_features, predicted_class]
                    
                elif len(shap_values.shape) == 2:
                    # Shape is (features, classes) or (samples, features)
                    if shap_values.shape[0] == len(self.feature_columns):
                        # Shape is (features, classes) - use predicted class
                        prediction = self.model.predict(df_input)[0]
                        classes = list(self.model.classes_)
                        class_idx = classes.index(prediction) if prediction in classes else 0
                        shap_vals = shap_values[:, class_idx]
                    else:
                        # Shape is (samples, features) - take first sample
                        shap_vals = shap_values[0]
                else:
                    # Shape is (features,) - direct use
                    shap_vals = shap_values
            else:
                return None
            
            # Ensure we have a 1D array of the right length
            shap_vals = np.array(shap_vals).flatten()
            if len(shap_vals) != len(self.feature_columns):
                return None
            
            # Create feature importance from SHAP values
            feature_shap = dict(zip(self.feature_columns, shap_vals))
            
            # Sort by absolute SHAP value - now each value is a scalar
            sorted_shap = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
            
            return {
                'top_positive': [(f, v) for f, v in sorted_shap if v > 0][:5],
                'top_negative': [(f, v) for f, v in sorted_shap if v < 0][:5],
                'all_values': feature_shap
            }
            
        except Exception as e:
            print(f"⚠️ Could not generate SHAP explanation: {e}")
            return None
    
    def _analyze_features(self, data, predicted_class):
        """Analyze key features for the prediction"""
        analysis = {
            'key_features': [],
            'anomalous_values': [],
            'normal_ranges': {}
        }
        
        # Get rules for the predicted class
        rules = self.rca_rules.get(predicted_class, {})
        primary_indicators = rules.get('primary_indicators', [])
        thresholds = rules.get('thresholds', {})
        
        # Analyze primary indicators
        for feature in primary_indicators:
            if feature in data:
                value = data[feature]
                importance = self.feature_importance.get(feature, 0)
                
                # Determine if value is anomalous
                is_anomalous = False
                if 'cpu' in feature and value > thresholds.get('cpu', 70):
                    is_anomalous = True
                elif 'memory' in feature and value > thresholds.get('memory', 80):
                    is_anomalous = True
                elif 'response_time' in feature and value > thresholds.get('response_time', 200):
                    is_anomalous = True
                elif 'error_rate' in feature and value > thresholds.get('error_rate', 0.1):
                    is_anomalous = True
                
                feature_info = {
                    'name': feature,
                    'value': value,
                    'importance': importance,
                    'is_anomalous': is_anomalous
                }
                
                analysis['key_features'].append(feature_info)
                
                if is_anomalous:
                    analysis['anomalous_values'].append(feature_info)
        
        return analysis
    
    def _perform_rca(self, data, predicted_class):
        """Perform root cause analysis"""
        rules = self.rca_rules.get(predicted_class, {})
        
        rca = {
            'anomaly_type': predicted_class,
            'likely_root_causes': rules.get('root_causes', []),
            'evidence': [],
            'severity': self._assess_severity(data, predicted_class)
        }
        
        # Collect evidence
        primary_indicators = rules.get('primary_indicators', [])
        thresholds = rules.get('thresholds', {})
        
        for feature in primary_indicators:
            if feature in data:
                value = data[feature]
                
                if 'cpu' in feature and value > thresholds.get('cpu', 70):
                    rca['evidence'].append(f"High CPU usage: {feature} = {value:.1f}%")
                elif 'memory' in feature and value > thresholds.get('memory', 80):
                    rca['evidence'].append(f"High memory usage: {feature} = {value:.1f}%")
                elif 'response_time' in feature and value > thresholds.get('response_time', 200):
                    rca['evidence'].append(f"Slow response time: {feature} = {value:.1f}ms")
                elif 'error_rate' in feature and value > thresholds.get('error_rate', 0.1):
                    rca['evidence'].append(f"High error rate: {feature} = {value:.1%}")
        
        return rca
    
    def _assess_severity(self, data, predicted_class):
        """Assess severity of the anomaly"""
        if predicted_class == 'normal':
            return 'low'
        elif predicted_class == 'service_crash':
            return 'critical'
        elif predicted_class == 'memory_leak':
            return 'high'
        elif predicted_class == 'cpu_spike':
            return 'medium'
        else:
            return 'medium'
    
    def _display_explanation(self, explanation):
        """Display comprehensive explanation"""
        print(f"\n📋 FEATURE ANALYSIS")
        print("-" * 30)
        
        feature_analysis = explanation['feature_analysis']
        
        # Display key features
        print("🔑 Key Features:")
        for feature_info in feature_analysis['key_features'][:5]:
            status = "⚠️" if feature_info['is_anomalous'] else "✅"
            print(f"   {status} {feature_info['name']}: {feature_info['value']:.2f} "
                  f"(importance: {feature_info['importance']:.3f})")
        
        # Display anomalous values
        if feature_analysis['anomalous_values']:
            print(f"\n🚨 Anomalous Values Detected:")
            for feature_info in feature_analysis['anomalous_values']:
                print(f"   ⚠️ {feature_info['name']}: {feature_info['value']:.2f}")
        
        # Display SHAP explanation if available
        shap_explanation = explanation['shap_explanation']
        if shap_explanation:
            print(f"\n🧠 SHAP EXPLANATION")
            print("-" * 30)
            
            if shap_explanation['top_positive']:
                print("📈 Features pushing towards this prediction:")
                for feature, shap_val in shap_explanation['top_positive'][:3]:
                    print(f"   ↗️ {feature}: +{shap_val:.3f}")
            
            if shap_explanation['top_negative']:
                print("📉 Features pushing against this prediction:")
                for feature, shap_val in shap_explanation['top_negative'][:3]:
                    print(f"   ↘️ {feature}: {shap_val:.3f}")
        
        # Display Root Cause Analysis
        rca = explanation['root_cause_analysis']
        print(f"\n🎯 ROOT CAUSE ANALYSIS")
        print("-" * 30)
        print(f"🚨 Severity: {rca['severity'].upper()}")
        
        if rca['evidence']:
            print(f"\n🔍 Evidence:")
            for evidence in rca['evidence']:
                print(f"   • {evidence}")
        
        print(f"\n🧭 Likely Root Causes:")
        for i, cause in enumerate(rca['likely_root_causes'][:3], 1):
            print(f"   {i}. {cause}")
        
        # Display Recommendations
        recommendations = explanation['recommendations']
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        print("-" * 30)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def create_explanation_report(self, explanation, save_path='explanation_report.txt'):
        """Create a detailed explanation report"""
        with open(save_path, 'w') as f:
            f.write("ANOMALY DETECTION EXPLANATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Predicted Anomaly: {explanation['predicted_class'].upper()}\n")
            f.write(f"Confidence: {explanation['confidence']:.1%}\n")
            f.write(f"Severity: {explanation['root_cause_analysis']['severity'].upper()}\n\n")
            
            # Feature Analysis
            f.write("FEATURE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            for feature_info in explanation['feature_analysis']['key_features']:
                status = "ANOMALOUS" if feature_info['is_anomalous'] else "NORMAL"
                f.write(f"{feature_info['name']}: {feature_info['value']:.2f} ({status})\n")
            f.write("\n")
            
            # Root Causes
            f.write("ROOT CAUSES\n")
            f.write("-" * 20 + "\n")
            for i, cause in enumerate(explanation['root_cause_analysis']['likely_root_causes'], 1):
                f.write(f"{i}. {cause}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(explanation['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"✅ Explanation report saved to {save_path}")

def demo_explainability():
    """
    Demonstrate explainability with sample predictions
    """
    print("\n🔍 Demo: Explainability & Root Cause Analysis")
    print("=" * 55)
    
    # Initialize explainability engine
    explainer = ExplainabilityRCA()
    explainer.load_model()
    
    # Test cases with different anomaly types
    test_cases = [
        {
            'name': 'CPU Spike Scenario',
            'data': {
                'notification_cpu': 85.0, 'web_api_cpu': 90.0, 'processor_cpu': 88.0,
                'notification_memory': 47.0, 'web_api_memory': 52.0, 'processor_memory': 49.0,
                'notification_response_time_p95': 350.0, 'web_api_response_time_p95': 400.0,
                'processor_response_time_p95': 300.0, 'notification_error_rate': 0.05,
                'processor_error_rate': 0.03, 'web_api_errors': 2.0,
                'notification_api_health': 1.0, 'processor_redis_health': 1.0, 'web_api_redis_health': 1.0,
                'notification_delivery_success': 0.95, 'notification_message_rate': 30.0,
                'processor_processing_rate': 60.0, 'web_api_requests': 150.0, 'web_api_requests_per_second': 25.0,
                'notification_queue': 15.0, 'notification_queue_depth': 12.0, 'processor_queue': 20.0,
                'processor_queue_depth': 18.0, 'web_api_queue_depth': 15.0, 'processor_db_connections': 12.0,
                'web_api_db_connections': 15.0, 'notification_thread_count': 25.0,
                'processor_thread_count': 30.0, 'web_api_thread_count': 40.0, 'processor_memory_growth': 0.1
            },
            'prediction': {'predicted_class': 'cpu_spike', 'confidence': 0.85}
        }
    ]
    
    for case in test_cases:
        print(f"\n📋 Analyzing: {case['name']}")
        print("-" * 40)
        
        explanation = explainer.explain_prediction(case['data'], case['prediction'])
        
        # Save explanation report
        report_name = f"explanation_{case['name'].lower().replace(' ', '_')}.txt"
        explainer.create_explanation_report(explanation, report_name)

if __name__ == "__main__":
    demo_explainability()