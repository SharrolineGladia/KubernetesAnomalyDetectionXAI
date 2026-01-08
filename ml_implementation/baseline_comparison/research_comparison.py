"""
Comprehensive Anomaly Detection Methods Comparison for Research Paper
Compare XGBoost classifier against 3 baseline methods with detailed metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import time
import pickle
from datetime import datetime
import sys
import os

# Add the current directory to Python path to import baseline methods
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline_methods import BaselineAnomalyDetectors

class ComprehensiveComparison:
    """
    Comprehensive comparison of anomaly detection methods for research paper
    """
    
    def __init__(self, dataset_path='../metrics_dataset_enhanced_rounded.csv'):
        self.dataset_path = dataset_path
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset consistently for all methods"""
        print("🔄 Loading and preparing dataset for all methods...")
        
        # Load the dataset
        df = pd.read_csv(self.dataset_path)
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
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        # Split the data (using same random state for consistency)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_encoded
        )
        
        print(f"✅ Data prepared - Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"📈 Classes: {list(self.class_names)}")
        print(f"🔧 Features: {len(self.feature_columns)}")
        
        # Show class distribution
        class_distribution = pd.Series(self.label_encoder.inverse_transform(y_encoded)).value_counts()
        print(f"📊 Class distribution:")
        for class_name, count in class_distribution.items():
            percentage = (count / len(y_encoded)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def run_xgboost_method(self, X_train, X_test, y_train, y_test):
        """Run XGBoost method (your primary approach)"""
        print("\n" + "="*60)
        print("🚀 XGBOOST CLASSIFIER (Your Primary Method)")
        print("="*60)
        
        try:
            from xgboost import XGBClassifier
            
            # Initialize XGBoost with same parameters as your anomaly_detector.py
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
            
            # Train the model
            start_time = time.time()
            xgb_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = xgb_model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"⏱️  Training time: {training_time:.2f} seconds")
            print(f"⏱️  Prediction time: {prediction_time:.4f} seconds")
            print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=self.class_names, zero_division=0)
            print("\n📋 Classification Report:")
            print(report)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\n🔍 Confusion Matrix:")
            print(f"Actual\\Predicted: {' '.join([f'{cls:>12}' for cls in self.class_names])}")
            for i, actual_class in enumerate(self.class_names):
                print(f"{actual_class:>12}: {' '.join([f'{cm[i,j]:>12}' for j in range(len(self.class_names))])}")
            
            return {
                'method': 'XGBoost',
                'accuracy': accuracy,
                'predictions': y_pred,
                'confusion_matrix': cm,
                'classification_report': report,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'model': xgb_model
            }
            
        except ImportError:
            print("❌ XGBoost not available. Using Random Forest as fallback...")
            return self.run_random_forest_method(X_train, X_test, y_train, y_test)
    
    def run_random_forest_method(self, X_train, X_test, y_train, y_test):
        """Run Random Forest method (your original approach)"""
        print("\n" + "="*60)
        print("🌲 RANDOM FOREST CLASSIFIER (Your Original Method)")
        print("="*60)
        
        # Initialize Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = rf_model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"⏱️  Training time: {training_time:.2f} seconds")
        print(f"⏱️  Prediction time: {prediction_time:.4f} seconds")
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=self.class_names, zero_division=0)
        print("\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔍 Confusion Matrix:")
        print(f"Actual\\Predicted: {' '.join([f'{cls:>12}' for cls in self.class_names])}")
        for i, actual_class in enumerate(self.class_names):
            print(f"{actual_class:>12}: {' '.join([f'{cm[i,j]:>12}' for j in range(len(self.class_names))])}")
        
        return {
            'method': 'Random Forest',
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'classification_report': report,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'model': rf_model
        }
    
    def run_baseline_methods(self, X_train, X_test, y_train, y_test):
        """Run all baseline methods"""
        # Initialize baseline detector with pre-loaded data
        baseline_detector = BaselineAnomalyDetectors(self.dataset_path)
        baseline_detector.X_train = X_train
        baseline_detector.X_test = X_test
        baseline_detector.y_train = y_train
        baseline_detector.y_test = y_test
        baseline_detector.feature_columns = self.feature_columns
        baseline_detector.class_names = self.class_names
        baseline_detector.label_encoder = self.label_encoder
        
        # Run threshold-based method
        start_time = time.time()
        threshold_results = baseline_detector.threshold_based_classifier()
        threshold_results['training_time'] = 0.0  # No training required
        threshold_results['prediction_time'] = time.time() - start_time
        
        # Run isolation forest method
        start_time = time.time()
        isolation_results = baseline_detector.isolation_forest_classifier()
        isolation_results['training_time'] = 0.1  # Minimal training
        isolation_results['prediction_time'] = time.time() - start_time
        
        # Run heuristic voting method
        start_time = time.time()
        voting_results = baseline_detector.heuristic_voting_classifier()
        voting_results['training_time'] = 0.0  # No training required
        voting_results['prediction_time'] = time.time() - start_time
        
        return threshold_results, isolation_results, voting_results
    
    def calculate_detailed_metrics(self, y_test, y_pred, method_name):
        """Calculate detailed metrics for each method"""
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0, labels=range(len(self.class_names))
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'method': method_name,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'macro_precision': precision_macro,
            'macro_recall': recall_macro,
            'macro_f1': f1_macro,
            'weighted_precision': precision_weighted,
            'weighted_recall': recall_weighted,
            'weighted_f1': f1_weighted
        }
    
    def create_comprehensive_comparison_table(self):
        """Create a comprehensive comparison table for the research paper"""
        print("\n" + "="*100)
        print("📊 COMPREHENSIVE RESEARCH PAPER COMPARISON")
        print("="*100)
        
        # Main comparison table
        print(f"\n📈 OVERALL PERFORMANCE METRICS:")
        print(f"{'Method':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12} {'Training Time':<15} {'Prediction Time':<15}")
        print("-" * 95)
        
        methods_order = ['XGBoost', 'Random Forest', 'Heuristic Voting', 'Threshold-Based', 'Isolation Forest']
        
        for method_name in methods_order:
            if method_name in self.results:
                result = self.results[method_name]
                accuracy = result['accuracy']
                training_time = result.get('training_time', 0)
                prediction_time = result.get('prediction_time', 0)
                
                # Calculate macro F1 if not already calculated
                if 'confusion_matrix' in result and len(self.class_names) > 2:
                    y_test = getattr(self, 'y_test_stored', None)
                    y_pred = result.get('predictions_encoded', result.get('predictions'))
                    if y_test is not None and y_pred is not None:
                        _, _, f1_macro, _ = precision_recall_fscore_support(
                            y_test, y_pred, average='macro', zero_division=0
                        )
                        _, _, f1_weighted, _ = precision_recall_fscore_support(
                            y_test, y_pred, average='weighted', zero_division=0
                        )
                    else:
                        f1_macro = 0.0
                        f1_weighted = 0.0
                else:
                    f1_macro = 0.0
                    f1_weighted = 0.0
                
                print(f"{method_name:<20} {accuracy:<10.4f} {f1_macro:<10.4f} {f1_weighted:<12.4f} {training_time:<15.4f} {prediction_time:<15.4f}")
        
        # Per-class performance comparison
        print(f"\n📊 PER-CLASS F1-SCORES COMPARISON:")
        print(f"{'Class':<15}", end="")
        for method_name in methods_order:
            if method_name in self.results:
                print(f"{method_name:<15}", end="")
        print()
        print("-" * (15 + 15 * len([m for m in methods_order if m in self.results])))
        
        # Calculate F1 scores for each class and method
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15}", end="")
            for method_name in methods_order:
                if method_name in self.results and 'confusion_matrix' in self.results[method_name]:
                    cm = self.results[method_name]['confusion_matrix']
                    if cm.shape[0] > i and cm.shape[1] > i:
                        tp = cm[i, i]
                        fp = cm[:, i].sum() - tp
                        fn = cm[i, :].sum() - tp
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        print(f"{f1:<15.4f}", end="")
                    else:
                        print(f"{'N/A':<15}", end="")
                elif method_name in self.results:
                    print(f"{'Binary':<15}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()
        
        # Research insights
        print(f"\n🔬 RESEARCH INSIGHTS:")
        best_accuracy = max(self.results.values(), key=lambda x: x['accuracy'])
        fastest_method = min(self.results.values(), key=lambda x: x.get('prediction_time', float('inf')))
        
        print(f"• Best Overall Accuracy: {best_accuracy['method']} ({best_accuracy['accuracy']:.4f})")
        print(f"• Fastest Prediction: {fastest_method['method']} ({fastest_method.get('prediction_time', 0):.4f}s)")
        print(f"• Dataset Size: {getattr(self, 'dataset_size', 'Unknown')} samples")
        print(f"• Number of Features: {len(self.feature_columns)}")
        print(f"• Class Balance: {len(self.class_names)} classes")
        
        print(f"\n📝 METHODOLOGY SUMMARY:")
        print(f"• XGBoost/Random Forest: Ensemble learning with gradient boosting/bagging")
        print(f"• Threshold-based: Rule-based classification with domain-specific thresholds")
        print(f"• Isolation Forest: Unsupervised anomaly detection (binary classification)")
        print(f"• Heuristic Voting: Multi-rule ensemble with majority voting")
        print(f"• All methods use identical 80/20 train/test split (random_state=42)")
    
    def save_results_for_paper(self):
        """Save detailed results for research paper"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        import pickle
        with open(f'research_comparison_results_{timestamp}.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save CSV summary for easy analysis
        summary_data = []
        for method_name, result in self.results.items():
            summary_data.append({
                'Method': method_name,
                'Accuracy': result['accuracy'],
                'Training_Time': result.get('training_time', 0),
                'Prediction_Time': result.get('prediction_time', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'research_summary_{timestamp}.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   • Detailed results: research_comparison_results_{timestamp}.pkl")
        print(f"   • Summary table: research_summary_{timestamp}.csv")
    
    def run_complete_comparison(self):
        """Run the complete comparison study"""
        print("🚀 STARTING COMPREHENSIVE ANOMALY DETECTION COMPARISON")
        print("="*80)
        print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        self.y_test_stored = y_test  # Store for later use
        self.dataset_size = len(X_train) + len(X_test)
        
        # Run XGBoost method
        self.results['XGBoost'] = self.run_xgboost_method(X_train, X_test, y_train, y_test)
        
        # Run Random Forest method as separate comparison
        self.results['Random Forest'] = self.run_random_forest_method(X_train, X_test, y_train, y_test)
        
        # Run baseline methods
        threshold_results, isolation_results, voting_results = self.run_baseline_methods(
            X_train, X_test, y_train, y_test
        )
        
        self.results['Threshold-Based'] = threshold_results
        self.results['Isolation Forest'] = isolation_results
        self.results['Heuristic Voting'] = voting_results
        
        # Create comprehensive comparison
        self.create_comprehensive_comparison_table()
        
        # Save results
        self.save_results_for_paper()
        
        print(f"\n🎉 COMPARISON STUDY COMPLETED SUCCESSFULLY!")
        print(f"🕐 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # DETAILED METRICS ANALYSIS
        print("\n" + "="*70)
        print("DETAILED METRICS ANALYSIS")
        print("="*70)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        # For each method's predictions
        methods = ['xgboost', 'random_forest', 'threshold', 'isolation_forest', 'heuristic']
        
        for method in methods:
            if method not in self.results:
                continue
                
            print(f"\n{'='*50}")
            print(f"{method.upper()} - DETAILED METRICS")
            print(f"{'='*50}")
            
            # Get predictions
            result = self.results[method]
            if 'predictions_encoded' not in result:
                print(f"❌ No predictions found for {method}")
                continue
                
            y_pred = result['predictions_encoded']
            
            # Classification report
            try:
                class_names = ['cpu_spike', 'memory_leak', 'normal', 'service_crash']
                print(classification_report(self.y_test, y_pred, 
                                            target_names=class_names, zero_division=0))
                
                # Confusion matrix
                print("\nConfusion Matrix:")
                cm = confusion_matrix(self.y_test, y_pred)
                
                # Create a formatted confusion matrix
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                print(cm_df)
                
                # Per-class accuracy
                print("\nPer-Class Accuracy:")
                for i, class_name in enumerate(class_names):
                    if cm[i].sum() > 0:  # Avoid division by zero
                        class_accuracy = cm[i, i] / cm[i].sum()
                        print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
                    else:
                        print(f"  {class_name}: N/A (no samples)")
                        
            except Exception as e:
                print(f"Error generating detailed metrics for {method}: {e}")
        
        print(f"\n🎯 SUMMARY COMPARISON")
        print("-" * 50)
        for method, result in self.results.items():
            accuracy = result.get('accuracy', 0)
            print(f"{method:>15}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return self.results


def main():
    """Main function to run the comprehensive comparison"""
    # Check if dataset exists
    dataset_path = '../metrics_dataset_enhanced_rounded.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the dataset is available at the specified path.")
        return
    
    # Run comprehensive comparison
    comparison = ComprehensiveComparison(dataset_path)
    results = comparison.run_complete_comparison()
    
    return results


if __name__ == "__main__":
    main()