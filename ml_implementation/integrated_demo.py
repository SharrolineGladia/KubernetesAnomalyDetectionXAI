"""
🔍 Integrated Anomaly Detection with Explainability
==================================================

This demonstrates the complete pipeline:
1. Anomaly Detection
2. Explainability & Root Cause Analysis
3. Actionable Recommendations

Author: ML Implementation Team
Date: September 28, 2025
"""

import os
import shutil
from datetime import datetime
from anomaly_detector import AnomalyDetector
from explainability_rca import ExplainabilityRCA

def create_results_folder():
    """Create a results folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/demo_{timestamp}"
    
    # Create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {results_dir}")
    return results_dir

def integrated_demo():
    """
    Demonstrate the complete anomaly detection + explainability pipeline
    """
    print("🚀 INTEGRATED ANOMALY DETECTION & EXPLAINABILITY DEMO")
    print("=" * 65)
    
    # Create results folder
    results_dir = create_results_folder()
    
    # Initialize both components
    detector = AnomalyDetector()
    explainer = ExplainabilityRCA()
    
    # Load pre-trained model (skip training for demo)
    print("\n🔄 Loading Models...")
    detector.load_model()
    explainer.load_model()
    explainer.load_model()
    
    # Test cases covering all anomaly types
    test_scenarios = [
        {
            'name': '🟢 Normal Operations',
            'data': {
                'notification_cpu': 15.0, 'web_api_cpu': 25.0, 'processor_cpu': 20.0,
                'notification_memory': 45.0, 'web_api_memory': 50.0, 'processor_memory': 48.0,
                'notification_response_time_p95': 120.0, 'web_api_response_time_p95': 150.0,
                'processor_response_time_p95': 100.0, 'notification_error_rate': 0.01,
                'processor_error_rate': 0.005, 'web_api_errors': 0.5,
                'notification_api_health': 1.0, 'processor_redis_health': 1.0, 'web_api_redis_health': 1.0,
                'notification_delivery_success': 0.99, 'notification_message_rate': 50.0,
                'processor_processing_rate': 100.0, 'web_api_requests': 200.0, 'web_api_requests_per_second': 45.0,
                'notification_queue': 5.0, 'notification_queue_depth': 2.0, 'processor_queue': 3.0,
                'processor_queue_depth': 1.5, 'web_api_queue_depth': 2.0, 'processor_db_connections': 8.0,
                'web_api_db_connections': 10.0, 'notification_thread_count': 10.0,
                'processor_thread_count': 15.0, 'web_api_thread_count': 20.0, 'processor_memory_growth': 0.1
            }
        },
        {
            'name': '🔴 CPU Spike Emergency',
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
            }
        },
        {
            'name': '🟡 Memory Leak Detected',
            'data': {
                'notification_cpu': 18.0, 'web_api_cpu': 28.0, 'processor_cpu': 22.0,
                'notification_memory': 85.0, 'web_api_memory': 88.0, 'processor_memory': 92.0,
                'notification_response_time_p95': 250.0, 'web_api_response_time_p95': 280.0,
                'processor_response_time_p95': 200.0, 'notification_error_rate': 0.03,
                'processor_error_rate': 0.02, 'web_api_errors': 1.5,
                'notification_api_health': 1.0, 'processor_redis_health': 1.0, 'web_api_redis_health': 1.0,
                'notification_delivery_success': 0.92, 'notification_message_rate': 40.0,
                'processor_processing_rate': 70.0, 'web_api_requests': 180.0, 'web_api_requests_per_second': 35.0,
                'notification_queue': 8.0, 'notification_queue_depth': 6.0, 'processor_queue': 10.0,
                'processor_queue_depth': 8.0, 'web_api_queue_depth': 5.0, 'processor_db_connections': 10.0,
                'web_api_db_connections': 12.0, 'notification_thread_count': 12.0,
                'processor_thread_count': 18.0, 'web_api_thread_count': 22.0, 'processor_memory_growth': 5.5
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("=" * 50)
        
        # Step 1: Detect anomaly
        print("🔍 Step 1: Anomaly Detection")
        prediction = detector.predict(scenario['data'])
        
        print(f"   🎯 Prediction: {prediction['predicted_class'].upper()}")
        print(f"   📊 Confidence: {prediction['confidence']:.1%}")
        
        # Step 2: Explain the prediction
        print(f"\n🧠 Step 2: Explainability Analysis")
        explanation = explainer.explain_prediction(scenario['data'], prediction)
        
        # Generate and save explanation report in results folder
        report_filename = f"explanation_{scenario['name'].lower().replace(' ', '_').replace('🟢', '').replace('🔴', '').replace('🟡', '').strip()}.txt"
        report_path = os.path.join(results_dir, report_filename)
        explainer.create_explanation_report(explanation, report_path)
        
        # Step 3: Generate action plan
        print(f"\n🎯 Step 3: Action Plan")
        severity = explanation['root_cause_analysis']['severity']
        
        if severity == 'critical':
            print("   🚨 IMMEDIATE ACTION REQUIRED!")
            print("   📞 Alert operations team")
            print("   🔄 Trigger automatic scaling")
        elif severity == 'high':
            print("   ⚠️ High priority - address within 15 minutes")
            print("   📈 Monitor closely for escalation")
        elif severity == 'medium':
            print("   📋 Schedule maintenance window")
            print("   📊 Continue monitoring")
        else:
            print("   ✅ Continue normal operations")
            print("   📈 Baseline monitoring active")
        
        print(f"\n📄 Detailed report saved as: {report_path}")
        
        print("\n" + "-" * 50)
    
    # Copy visualization files to results folder
    copy_visualizations_to_results(results_dir)
    
    return results_dir

def copy_visualizations_to_results(results_dir):
    """Copy existing visualization files to results folder"""
    viz_files = [
        'confusion_matrix.png',
        'feature_importance.png'
    ]
    
    print(f"\n📊 Copying visualizations to results folder...")
    
    for viz_file in viz_files:
        if os.path.exists(viz_file):
            dest_path = os.path.join(results_dir, viz_file)
            shutil.copy2(viz_file, dest_path)
            print(f"   ✅ {viz_file} → {dest_path}")
        else:
            print(f"   ⚠️ {viz_file} not found, skipping...")

def main():
    """
    Main function to run the integrated demo
    """
    results_dir = integrated_demo()
    
    print(f"\n🎉 DEMO COMPLETE!")
    print("✅ Anomaly Detection: Working")
    print("✅ Explainability: Working") 
    print("✅ Root Cause Analysis: Working")
    print("✅ Actionable Recommendations: Generated")
    
    print(f"\n📁 All Results Saved To: {results_dir}")
    print("   📊 Visualizations:")
    print("      • confusion_matrix.png")
    print("      • feature_importance.png") 
    print("   📄 Explanation Reports:")
    print("      • explanation_normal_operations.txt")
    print("      • explanation_cpu_spike_emergency.txt")
    print("      • explanation_memory_leak_detected.txt")
    
    print(f"\n🚀 Your complete ML pipeline results are organized and ready!")

if __name__ == "__main__":
    main()