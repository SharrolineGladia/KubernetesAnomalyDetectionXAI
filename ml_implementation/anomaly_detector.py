"""
Anomaly Detection Engine - Production Ready with Visualizations
Clean implementation for anomaly detection with performance visualization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Production-Ready Anomaly Detection Engine
    
    Features:
    - Multi-class anomaly classification (normal, cpu_spike, memory_leak, service_crash)
    - High-performance Random Forest classifier
    - Simple prediction interface
    - Model persistence
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.model_metrics = {}
        self.is_trained = False
        
    def load_and_prepare_data(self, dataset_path='metrics_dataset.csv'):
        """
        Load and prepare the dataset for training
        
        Args:
            dataset_path (str): Path to the metrics dataset
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"🔄 Loading dataset from {dataset_path}")
        
        # Load the dataset
        df = pd.read_csv(dataset_path)
        print(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Remove non-numeric columns
        columns_to_drop = []
        if 'timestamp' in df.columns:
            columns_to_drop.append('timestamp')
        if 'anomaly_label' in df.columns:
            columns_to_drop.append('anomaly_label')
            
        # Determine target column
        if 'anomaly_type' in df.columns:
            target_column = 'anomaly_type'
            columns_to_drop.append(target_column)
        else:
            raise ValueError("No target column found. Expected 'anomaly_type'")
            
        # Separate features and target
        X = df.drop(columns=columns_to_drop)
        y = df[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        print(f"🔧 Using {len(self.feature_columns)} features for training")
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Show class distribution
        class_distribution = pd.Series(y).value_counts()
        print(f"📈 Class distribution:")
        for class_name, count in class_distribution.items():
            percentage = (count / len(y)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_encoded
        )
        
        print(f"✅ Data prepared - Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model
        """
        print("🚀 Training Random Forest model...")
        
        # Initialize Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Store feature importance for visualization
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance with detailed analysis and visualizations
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
            
        print("📊 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.model_metrics['accuracy'] = accuracy
        
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        print("\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate detailed performance analysis
        self._generate_performance_analysis(cm, class_names, y_test, y_pred)
        
        # Create visualizations
        self._create_confusion_matrix_plot(cm, class_names)
        self._create_feature_importance_plot()
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def _generate_performance_analysis(self, cm, class_names, y_test, y_pred):
        """
        Generate detailed text analysis of model performance
        """
        print(f"\n" + "="*60)
        print(f"📈 DETAILED PERFORMANCE ANALYSIS")
        print(f"="*60)
        
        # Overall metrics
        total_samples = len(y_test)
        correct_predictions = np.sum(y_pred == y_test)
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Total test samples: {total_samples}")
        print(f"   Correct predictions: {correct_predictions}")
        print(f"   Incorrect predictions: {total_samples - correct_predictions}")
        print(f"   Overall accuracy: {correct_predictions/total_samples:.1%}")
        
        # Per-class analysis
        print(f"\n📊 PER-CLASS PERFORMANCE:")
        print(f"   {'Class':<15} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'Support':<8}")
        print(f"   {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
        
        class_performance = []
        
        for i, class_name in enumerate(class_names):
            # Calculate metrics for each class
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            support = cm[i, :].sum()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"   {class_name:<15} {precision:<10.1%} {recall:<8.1%} {f1_score:<10.1%} {support:<8}")
            
            class_performance.append({
                'class': class_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
        
        # Identify best and worst performing classes
        best_class = max(class_performance, key=lambda x: x['f1_score'])
        worst_class = min(class_performance, key=lambda x: x['f1_score'])
        
        print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
        print(f"   Best performing class: {best_class['class']} (F1: {best_class['f1_score']:.1%})")
        print(f"   Most challenging class: {worst_class['class']} (F1: {worst_class['f1_score']:.1%})")
        
        # Identify common misclassifications
        print(f"\n⚠️  COMMON MISCLASSIFICATIONS:")
        misclassifications = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append((class_names[i], class_names[j], cm[i, j]))
        
        # Sort by frequency
        misclassifications.sort(key=lambda x: x[2], reverse=True)
        
        if misclassifications:
            for actual, predicted, count in misclassifications[:3]:  # Top 3
                percentage = count / total_samples * 100
                print(f"   {actual} → {predicted}: {count} cases ({percentage:.1f}%)")
        else:
            print(f"   No misclassifications found!")
        
        # Model confidence analysis
        print(f"\n📊 MODEL CONFIDENCE ANALYSIS:")
        if hasattr(self, '_last_predictions_proba'):
            max_probas = np.max(self._last_predictions_proba, axis=1)
            print(f"   Average confidence: {np.mean(max_probas):.1%}")
            print(f"   High confidence (>90%): {np.sum(max_probas > 0.9)/len(max_probas):.1%} of predictions")
            print(f"   Low confidence (<70%): {np.sum(max_probas < 0.7)/len(max_probas):.1%} of predictions")
    
    def _create_confusion_matrix_plot(self, cm, class_names):
        """
        Create and save confusion matrix heatmap
        """
        plt.figure(figsize=(14, 10))
        
        # Create heatmap with better formatting
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Number of Predictions'},
                   annot_kws={'size': 16, 'weight': 'bold'},
                   linewidths=2,
                   linecolor='white')
        
        # Customize the plot with larger fonts
        plt.title('Anomaly Detection Model - Confusion Matrix', 
                 fontsize=20, fontweight='bold', pad=30)
        plt.xlabel('Predicted Class', fontsize=16, fontweight='bold', labelpad=15)
        plt.ylabel('Actual Class', fontsize=16, fontweight='bold', labelpad=15)
        
        # Increase tick label size
        plt.xticks(fontsize=14, rotation=45, ha='right')
        plt.yticks(fontsize=14, rotation=0)
        
        # Add accuracy text with better positioning
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.1%}', 
                   ha='center', va='bottom',
                   fontsize=18, fontweight='bold', 
                   color='darkgreen',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Add sample counts for each class
        total_samples = np.sum(cm)
        plt.figtext(0.02, 0.98, f'Total Test Samples: {total_samples}', 
                   ha='left', va='top',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95)
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved as 'confusion_matrix.png'")
    
    def _create_feature_importance_plot(self):
        """
        Create and save feature importance chart
        """
        if not hasattr(self, 'feature_importance'):
            print("⚠️  Feature importance not available")
            return
        
        # Get top 15 most important features
        top_features = self.feature_importance.head(15)
        
        plt.figure(figsize=(14, 10))
        
        # Create horizontal bar chart with better colors
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the plot
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=12)
        plt.xlabel('Feature Importance Score', fontsize=14, fontweight='bold', labelpad=15)
        plt.title('Top 15 Most Important Features for Anomaly Detection', 
                 fontsize=18, fontweight='bold', pad=25)
        
        # Add value labels on bars with better formatting
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.4f}', 
                    va='center', ha='left', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Add grid for better readability
        plt.grid(axis='x', alpha=0.4, linestyle='--', color='gray')
        
        # Improve layout
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.xlim(0, max(top_features['importance']) * 1.15)  # Add space for labels
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.1)
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Feature importance chart saved as 'feature_importance.png'")
        
        # Print top features analysis
        print(f"\n🔍 TOP FEATURE ANALYSIS:")
        print(f"   Most important feature: {top_features.iloc[0]['feature']} ({top_features.iloc[0]['importance']:.3f})")
        
        # Group features by category
        cpu_features = top_features[top_features['feature'].str.contains('cpu', case=False)]
        memory_features = top_features[top_features['feature'].str.contains('memory', case=False)]
        response_features = top_features[top_features['feature'].str.contains('response', case=False)]
        
        if len(cpu_features) > 0:
            print(f"   Top CPU metric: {cpu_features.iloc[0]['feature']} (importance: {cpu_features.iloc[0]['importance']:.3f})")
        if len(memory_features) > 0:
            print(f"   Top Memory metric: {memory_features.iloc[0]['feature']} (importance: {memory_features.iloc[0]['importance']:.3f})")
        if len(response_features) > 0:
            print(f"   Top Response metric: {response_features.iloc[0]['feature']} (importance: {response_features.iloc[0]['importance']:.3f})")
    
    def predict(self, features):
        """
        Make predictions on new data
        
        Args:
            features (dict or pd.DataFrame): Feature values
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(features_df.columns)
        if missing_features:
            # Fill missing features with 0 (could be improved with better defaults)
            for feature in missing_features:
                features_df[feature] = 0
        
        # Reorder columns to match training
        features_df = features_df[self.feature_columns]
        
        # Make prediction
        prediction_encoded = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Store for confidence analysis
        self._last_predictions_proba = probabilities.reshape(1, -1)
        
        # Decode prediction
        predicted_class = self.label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = max(probabilities)
        
        # Create probability dictionary
        all_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            all_probabilities[class_name] = probabilities[i]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
    
    def save_model(self, filepath='anomaly_detection_model.pkl'):
        """
        Save the trained model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics,
            'feature_importance': getattr(self, 'feature_importance', None),
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='anomaly_detection_model.pkl'):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data['model_metrics']
        self.feature_importance = model_data.get('feature_importance', None)
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Model loaded from {filepath}")
        print(f"📊 Model accuracy: {self.model_metrics.get('accuracy', 'Unknown'):.2%}")
    
    def get_model_info(self):
        """
        Get information about the trained model
        """
        if not self.is_trained:
            return {"status": "Not trained"}
        
        return {
            "status": "Trained",
            "n_features": len(self.feature_columns),
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "accuracy": self.model_metrics.get('accuracy', None),
            "n_trees": self.model.n_estimators if self.model else None
        }

def train_new_model():
    """
    Train a new anomaly detection model
    """
    print("🚀 Anomaly Detection Model Training")
    print("=" * 50)
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = detector.load_and_prepare_data()
    
    # Train the model
    detector.train_model(X_train, y_train)
    
    # Evaluate the model
    results = detector.evaluate_model(X_test, y_test)
    
    # Save the model
    detector.save_model()
    
    # Show summary
    model_info = detector.get_model_info()
    print(f"\n📋 Final Model Summary:")
    print(f"   Features: {model_info['n_features']}")
    print(f"   Classes: {model_info['n_classes']}")
    print(f"   Accuracy: {model_info['accuracy']:.2%}")
    print(f"   Status: Ready for production")
    
    return detector

def demo_prediction():
    """
    Demonstrate prediction with comprehensive sample data
    """
    print("\n🧪 Demo: Sample Predictions with Full Feature Set")
    print("-" * 50)
    
    # Load the trained model
    detector = AnomalyDetector()
    detector.load_model()
    
    # Get feature names from the model
    print(f"📊 Model expects {len(detector.feature_columns)} features")
    
    # Create comprehensive test cases with all features
    test_cases = [
        {
            'name': 'Normal Operation',
            'data': {
                # CPU metrics - Normal levels
                'notification_cpu': 15.0,
                'web_api_cpu': 25.0,
                'processor_cpu': 20.0,
                
                # Memory metrics - Normal levels
                'notification_memory': 45.0,
                'web_api_memory': 50.0,
                'processor_memory': 48.0,
                
                # Response time metrics - Normal
                'notification_response_time_p95': 120.0,
                'web_api_response_time_p95': 150.0,
                'processor_response_time_p95': 100.0,
                
                # Error rates - Low for normal
                'notification_error_rate': 0.01,
                'processor_error_rate': 0.005,
                'web_api_errors': 0.5,
                
                # Health indicators
                'notification_api_health': 1.0,
                'processor_redis_health': 1.0,
                'web_api_redis_health': 1.0,
                
                # Success rates
                'notification_delivery_success': 0.99,
                
                # Throughput metrics
                'notification_message_rate': 50.0,
                'processor_processing_rate': 100.0,
                'web_api_requests': 200.0,
                'web_api_requests_per_second': 45.0,
                
                # Queue metrics
                'notification_queue': 5.0,
                'notification_queue_depth': 2.0,
                'processor_queue': 3.0,
                'processor_queue_depth': 1.5,
                'web_api_queue_depth': 2.0,
                
                # Connection and thread metrics
                'processor_db_connections': 8.0,
                'web_api_db_connections': 10.0,
                'notification_thread_count': 10.0,
                'processor_thread_count': 15.0,
                'web_api_thread_count': 20.0,
                
                # Memory growth (normal)
                'processor_memory_growth': 0.1
            }
        },
        {
            'name': 'CPU Spike Anomaly',
            'data': {
                # CPU metrics - HIGH for CPU spike
                'notification_cpu': 85.0,
                'web_api_cpu': 90.0,
                'processor_cpu': 88.0,
                
                # Memory metrics - Slightly elevated
                'notification_memory': 47.0,
                'web_api_memory': 52.0,
                'processor_memory': 49.0,
                
                # Response time metrics - HIGH due to CPU stress
                'notification_response_time_p95': 350.0,
                'web_api_response_time_p95': 400.0,
                'processor_response_time_p95': 300.0,
                
                # Error rates - Higher due to stress
                'notification_error_rate': 0.05,
                'processor_error_rate': 0.03,
                'web_api_errors': 2.0,
                
                # Health indicators - Still healthy
                'notification_api_health': 1.0,
                'processor_redis_health': 1.0,
                'web_api_redis_health': 1.0,
                
                # Success rates - Slightly lower
                'notification_delivery_success': 0.95,
                
                # Throughput metrics - Reduced due to high CPU
                'notification_message_rate': 30.0,
                'processor_processing_rate': 60.0,
                'web_api_requests': 150.0,
                'web_api_requests_per_second': 25.0,
                
                # Queue metrics - Higher due to slower processing
                'notification_queue': 15.0,
                'notification_queue_depth': 12.0,
                'processor_queue': 20.0,
                'processor_queue_depth': 18.0,
                'web_api_queue_depth': 15.0,
                
                # Connection and thread metrics - Higher load
                'processor_db_connections': 12.0,
                'web_api_db_connections': 15.0,
                'notification_thread_count': 25.0,
                'processor_thread_count': 30.0,
                'web_api_thread_count': 40.0,
                
                # Memory growth (normal)
                'processor_memory_growth': 0.1
            }
        },
        {
            'name': 'Memory Leak Anomaly',
            'data': {
                # CPU metrics - Normal to slightly elevated
                'notification_cpu': 18.0,
                'web_api_cpu': 28.0,
                'processor_cpu': 22.0,
                
                # Memory metrics - HIGH for memory leak
                'notification_memory': 85.0,
                'web_api_memory': 88.0,
                'processor_memory': 92.0,
                
                # Response time metrics - Elevated due to memory pressure
                'notification_response_time_p95': 250.0,
                'web_api_response_time_p95': 280.0,
                'processor_response_time_p95': 200.0,
                
                # Error rates - Moderate
                'notification_error_rate': 0.03,
                'processor_error_rate': 0.02,
                'web_api_errors': 1.5,
                
                # Health indicators
                'notification_api_health': 1.0,
                'processor_redis_health': 1.0,
                'web_api_redis_health': 1.0,
                
                # Success rates - Slightly affected
                'notification_delivery_success': 0.92,
                
                # Throughput metrics
                'notification_message_rate': 40.0,
                'processor_processing_rate': 70.0,
                'web_api_requests': 180.0,
                'web_api_requests_per_second': 35.0,
                
                # Queue metrics
                'notification_queue': 8.0,
                'notification_queue_depth': 6.0,
                'processor_queue': 10.0,
                'processor_queue_depth': 8.0,
                'web_api_queue_depth': 5.0,
                
                # Connection and thread metrics
                'processor_db_connections': 10.0,
                'web_api_db_connections': 12.0,
                'notification_thread_count': 12.0,
                'processor_thread_count': 18.0,
                'web_api_thread_count': 22.0,
                
                # Memory growth - HIGH indicating leak
                'processor_memory_growth': 5.5
            }
        },
        {
            'name': 'Service Crash Anomaly',
            'data': {
                # CPU metrics - Moderate levels (based on actual data patterns)
                'notification_cpu': 35.0,
                'web_api_cpu': 40.0,
                'processor_cpu': 44.0,
                
                # Memory metrics - Moderate to high
                'notification_memory': 55.0,
                'web_api_memory': 60.0,
                'processor_memory': 63.0,
                
                # Response time metrics - VERY HIGH web_api_response_time_p95 (most important feature!)
                'notification_response_time_p95': 280.0,
                'web_api_response_time_p95': 450.0,  # Very high - top feature for service crash
                'processor_response_time_p95': 300.0,
                
                # Error rates - KEY PATTERN: Very high notification_error_rate for service crash!
                'notification_error_rate': 0.65,
                'processor_error_rate': 0.06,
                'web_api_errors': 4.0,
                
                # Health indicators - Degraded but not completely failed
                'notification_api_health': 0.75,
                'processor_redis_health': 0.85,
                'web_api_redis_health': 0.90,
                
                # Success rates - Good but slightly degraded
                'notification_delivery_success': 0.95,
                
                # Throughput metrics - Reduced performance
                'notification_message_rate': 25.0,
                'processor_processing_rate': 45.0,
                'web_api_requests': 80.0,
                'web_api_requests_per_second': 15.0,
                
                # Queue metrics - Elevated
                'notification_queue': 12.0,
                'notification_queue_depth': 10.0,
                'processor_queue': 18.0,
                'processor_queue_depth': 15.0,
                'web_api_queue_depth': 8.0,
                
                # Connection and thread metrics - Higher load
                'processor_db_connections': 15.0,
                'web_api_db_connections': 18.0,
                'notification_thread_count': 30.0,
                'processor_thread_count': 35.0,
                'web_api_thread_count': 45.0,
                
                # Memory growth - Moderate
                'processor_memory_growth': 1.2
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n📋 {case['name']}:")
        prediction = detector.predict(case['data'])
        
        # Display results with confidence analysis
        confidence = prediction['confidence']
        predicted_class = prediction['predicted_class']
        
        print(f"   🎯 Predicted: {predicted_class.upper()}")
        print(f"   📊 Confidence: {confidence:.1%}")
        
        # Confidence level analysis
        if confidence >= 0.9:
            conf_level = "Very High ✅"
        elif confidence >= 0.8:
            conf_level = "High ✅"
        elif confidence >= 0.7:
            conf_level = "Good ⚠️"
        else:
            conf_level = "Low ❌"
            
        print(f"   📈 Confidence Level: {conf_level}")
        
        # Show top probabilities
        sorted_probs = sorted(prediction['all_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)
        print(f"   📋 All Probabilities:")
        for class_name, prob in sorted_probs:
            indicator = "👑" if class_name == predicted_class else "  "
            print(f"     {indicator} {class_name:>13}: {prob:.1%}")
    
    print(f"\n💡 Note: High confidence indicates the model is very certain about its prediction!")
    print(f"🎯 These samples use all {len(detector.feature_columns)} features for accurate predictions.")

if __name__ == "__main__":
    # Train a new model
    detector = train_new_model()
    
    # Demo predictions
    demo_prediction()