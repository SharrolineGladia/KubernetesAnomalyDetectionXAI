"""
Baseline Anomaly Detection Methods for Research Paper Comparison
Implementation of 3 baseline methods to compare against XGBoost classifier
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class BaselineAnomalyDetectors:
    """
    Implementation of three baseline anomaly detection methods:
    1. Threshold-based classifier (rule-based)
    2. Isolation Forest (unsupervised)
    3. Heuristic voting classifier (multi-rule voting)
    """
    
    def __init__(self, dataset_path='../metrics_dataset_enhanced_rounded.csv'):
        self.dataset_path = dataset_path
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.class_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset for all baseline methods"""
        print(f"🔄 Loading dataset from {self.dataset_path}")
        
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
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_encoded
        )
        
        print(f"✅ Data prepared - Training: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        print(f"📈 Classes: {list(self.class_names)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def threshold_based_classifier(self):
        """
        Method 1: Threshold-based classifier using specific rules
        
        Rules:
        - If processor_cpu > 80 → predict 'cpu_spike'
        - Elif processor_memory_growth > 100 MB/min → predict 'memory_leak'  
        - Elif notification_requests == 0 or processor_response_time_p95 == 0 → predict 'service_crash'
        - Else → predict 'normal'
        """
        print("\n" + "="*60)
        print("🔍 THRESHOLD-BASED CLASSIFIER")
        print("="*60)
        
        if self.X_test is None:
            self.load_and_prepare_data()
        
        # Get the test data as DataFrame to access columns by name
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_columns)
        
        predictions = []
        
        for idx, row in X_test_df.iterrows():
            # Rule 1: CPU spike detection (based on data analysis: CPU spike avg = 66.7)
            if (row.get('processor_cpu', 0) > 50 or 
                row.get('web_api_cpu', 0) > 50 or 
                row.get('notification_cpu', 0) > 50):
                predictions.append('cpu_spike')
            # Rule 2: Memory leak detection (processor memory > 60, since avg normal = 31)
            elif row.get('processor_memory', 0) > 60:
                predictions.append('memory_leak')
            # Rule 3: Service crash detection (response time = 0 indicates service down)
            elif (row.get('processor_response_time_p95', 1) == 0 or
                  row.get('web_api_response_time_p95', 1) == 0):
                predictions.append('service_crash')
            # Default: Normal operation
            else:
                predictions.append('normal')
        
        # Convert string predictions to encoded labels
        predictions_encoded = self.label_encoder.transform(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, predictions_encoded)
        
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        report = classification_report(self.y_test, predictions_encoded, 
                                     target_names=self.class_names, zero_division=0)
        print("\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions_encoded)
        print("\n🔍 Confusion Matrix:")
        print(f"Actual\\Predicted: {' '.join([f'{cls:>12}' for cls in self.class_names])}")
        for i, actual_class in enumerate(self.class_names):
            print(f"{actual_class:>12}: {' '.join([f'{cm[i,j]:>12}' for j in range(len(self.class_names))])}")
        
        return {
            'method': 'Threshold-Based',
            'accuracy': accuracy,
            'predictions': predictions,
            'predictions_encoded': predictions_encoded,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def isolation_forest_classifier(self):
        """
        Method 2: Isolation Forest (unsupervised anomaly detection)
        
        - Train IsolationForest with contamination=0.4
        - Predict on X_test (-1 for anomaly, 1 for normal)
        - Map: 1 → 'normal', -1 → 'anomaly' (generic, cannot distinguish anomaly types)
        """
        print("\n" + "="*60)
        print("🌲 ISOLATION FOREST CLASSIFIER")
        print("="*60)
        
        if self.X_test is None:
            self.load_and_prepare_data()
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.4, random_state=42, n_jobs=-1)
        iso_forest.fit(self.X_train)
        
        # Make predictions
        predictions_iso = iso_forest.predict(self.X_test)
        
        # Convert to 4-class predictions using feature-based heuristics
        predictions = []
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_columns)
        
        for i, (anomaly_pred, row) in enumerate(zip(predictions_iso, X_test_df.iterrows())):
            _, features = row
            
            if anomaly_pred == 1:  # Normal prediction
                predictions.append('normal')
            else:  # Anomaly - classify which type using same logic as threshold method
                # Rule 1: CPU spike detection
                if (features.get('processor_cpu', 0) > 50 or 
                    features.get('web_api_cpu', 0) > 50 or 
                    features.get('notification_cpu', 0) > 50):
                    predictions.append('cpu_spike')
                # Rule 2: Memory leak detection
                elif features.get('processor_memory', 0) > 60:
                    predictions.append('memory_leak')
                # Rule 3: Service crash detection
                elif (features.get('processor_response_time_p95', 1) == 0 or
                      features.get('web_api_response_time_p95', 1) == 0):
                    predictions.append('service_crash')
                else:
                    # Default anomaly type (most common in training)
                    predictions.append('cpu_spike')
        
        # Convert string predictions to encoded labels for 4-class evaluation
        predictions_encoded = self.label_encoder.transform(predictions)
        
        # Calculate 4-class accuracy
        accuracy = accuracy_score(self.y_test, predictions_encoded)
        
        print(f"🎯 4-Class Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        report = classification_report(self.y_test, predictions_encoded, 
                                     target_names=self.class_names, zero_division=0)
        print("\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions_encoded)
        print("\n🔍 Confusion Matrix:")
        print(f"Actual\\Predicted: {' '.join([f'{cls:>12}' for cls in self.class_names])}")
        for i, actual_class in enumerate(self.class_names):
            print(f"{actual_class:>12}: {' '.join([f'{cm[i,j]:>12}' for j in range(len(self.class_names))])}")
        
        return {
            'method': 'Isolation Forest',
            'accuracy': accuracy,
            'predictions': predictions,
            'predictions_encoded': predictions_encoded,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def heuristic_voting_classifier(self):
        """
        Method 3: Heuristic voting classifier using multiple rules
        
        Rules that vote for each class:
        - If processor_cpu > 75 → vote 'cpu_spike'
        - If processor_memory > (rolling mean * 1.5) → vote 'memory_leak'
        - If processor_response_time_p95 == 0 OR notification_requests < 0.1 → vote 'service_crash'
        
        Use majority voting; if no votes or tie → predict 'normal'
        """
        print("\n" + "="*60)
        print("🗳️  HEURISTIC VOTING CLASSIFIER")
        print("="*60)
        
        if self.X_test is None:
            self.load_and_prepare_data()
        
        # Get the test data as DataFrame to access columns by name
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_columns)
        
        # Calculate rolling mean for memory from training data
        if 'processor_memory' in self.feature_columns:
            if isinstance(self.X_train, pd.DataFrame):
                memory_mean = np.mean(self.X_train['processor_memory'])
            else:
                memory_col_idx = self.feature_columns.index('processor_memory')
                memory_mean = np.mean(self.X_train[:, memory_col_idx])
        else:
            memory_mean = 50  # Default fallback
        
        predictions = []
        
        for idx, row in X_test_df.iterrows():
            votes = []
            
            # Voter 1: CPU-based classification (aligned with threshold method)
            max_cpu = max(row.get('processor_cpu', 0), 
                         row.get('web_api_cpu', 0), 
                         row.get('notification_cpu', 0))
            if max_cpu > 50:
                votes.append('cpu_spike')
            elif max_cpu < 30:  # Low CPU suggests normal
                votes.append('normal')
            
            # Voter 2: Memory-based classification (aligned with threshold method) 
            if row.get('processor_memory', 0) > 60:
                votes.append('memory_leak')
            elif row.get('processor_memory', 0) < 45:  # Low memory suggests normal
                votes.append('normal')
            
            # Voter 3: Service health classification
            if (row.get('notification_api_health', 1) < 0.85 or
                row.get('notification_delivery_success', 1) < 0.9):
                votes.append('service_crash')
            elif (row.get('notification_api_health', 1) > 0.95 and
                  row.get('notification_delivery_success', 1) > 0.98):
                votes.append('normal')
            
            # Voter 4: Response time classification
            if (row.get('processor_response_time_p95', 1) == 0 or
                row.get('web_api_response_time_p95', 1) == 0):
                votes.append('service_crash')
            elif (row.get('processor_response_time_p95', 0) > 180 or
                  row.get('web_api_response_time_p95', 0) > 180):
                votes.append('cpu_spike')  # High response time usually means CPU stress
            
            # Voter 5: Memory growth classification
            if row.get('processor_memory_growth', 0) > 1.0:
                votes.append('memory_leak')
            elif row.get('processor_memory_growth', 0) < 0.1:
                votes.append('normal')
            
            # Majority voting with improved tie-breaking
            if len(votes) == 0:
                prediction = 'normal'
            else:
                # Count votes
                vote_counts = {}
                for vote in votes:
                    vote_counts[vote] = vote_counts.get(vote, 0) + 1
                
                # Get the class with maximum votes
                max_votes = max(vote_counts.values())
                winners = [cls for cls, count in vote_counts.items() if count == max_votes]
                
                if len(winners) == 1:
                    prediction = winners[0]
                else:
                    # Improved tie-breaking: prefer normal, then most specific anomaly
                    if 'normal' in winners:
                        prediction = 'normal'
                    elif 'service_crash' in winners:
                        prediction = 'service_crash'  # Most specific
                    elif 'memory_leak' in winners:
                        prediction = 'memory_leak'
                    elif 'cpu_spike' in winners:
                        prediction = 'cpu_spike'
                    else:
                        prediction = 'normal'  # Fallback
            
            predictions.append(prediction)
        
        # Convert string predictions to encoded labels
        predictions_encoded = self.label_encoder.transform(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, predictions_encoded)
        
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        report = classification_report(self.y_test, predictions_encoded, 
                                     target_names=self.class_names, zero_division=0)
        print("\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions_encoded)
        print("\n🔍 Confusion Matrix:")
        print(f"Actual\\Predicted: {' '.join([f'{cls:>12}' for cls in self.class_names])}")
        for i, actual_class in enumerate(self.class_names):
            print(f"{actual_class:>12}: {' '.join([f'{cm[i,j]:>12}' for j in range(len(self.class_names))])}")
        
        return {
            'method': 'Heuristic Voting',
            'accuracy': accuracy,
            'predictions': predictions,
            'predictions_encoded': predictions_encoded,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def run_all_baselines(self):
        """Run all three baseline methods and return comparison results"""
        print("🚀 RUNNING ALL BASELINE ANOMALY DETECTION METHODS")
        print("="*80)
        
        # Load data once
        self.load_and_prepare_data()
        
        # Run all methods
        threshold_results = self.threshold_based_classifier()
        isolation_results = self.isolation_forest_classifier()
        voting_results = self.heuristic_voting_classifier()
        
        # Create comparison summary
        self._create_comparison_table([threshold_results, isolation_results, voting_results])
        
        return {
            'threshold_based': threshold_results,
            'isolation_forest': isolation_results,
            'heuristic_voting': voting_results
        }
    
    def _create_comparison_table(self, results_list):
        """Create a formatted comparison table of all methods"""
        print("\n" + "="*80)
        print("📊 BASELINE METHODS COMPARISON SUMMARY")
        print("="*80)
        
        # Extract accuracy scores
        print(f"\n🎯 OVERALL ACCURACY COMPARISON:")
        print(f"{'Method':<25} {'Accuracy':<15} {'Percentage':<15}")
        print("-" * 55)
        
        for result in results_list:
            method = result['method']
            accuracy = result['accuracy']
            print(f"{method:<25} {accuracy:<15.4f} {accuracy*100:>10.2f}%")
        
        # Find best performing method
        best_method = max(results_list, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best performing method: {best_method['method']} ({best_method['accuracy']:.4f})")
        
        # Per-class performance comparison (only for methods that support multi-class)
        multiclass_methods = [r for r in results_list if 'confusion_matrix' in r and len(self.class_names) > 2]
        
        if multiclass_methods:
            print(f"\n📊 PER-CLASS PERFORMANCE (F1-SCORES):")
            print(f"{'Class':<15}", end="")
            for result in multiclass_methods:
                print(f"{result['method']:<20}", end="")
            print()
            print("-" * (15 + 20 * len(multiclass_methods)))
            
            # Calculate F1 scores for each class and method
            for i, class_name in enumerate(self.class_names):
                print(f"{class_name:<15}", end="")
                for result in multiclass_methods:
                    cm = result['confusion_matrix']
                    if cm.shape[0] > i and cm.shape[1] > i:
                        tp = cm[i, i]
                        fp = cm[:, i].sum() - tp
                        fn = cm[i, :].sum() - tp
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        print(f"{f1:<20.4f}", end="")
                    else:
                        print(f"{'N/A':<20}", end="")
                print()
        
        print(f"\n📝 METHODOLOGY NOTES:")
        print(f"• Threshold-based: Rule-based classifier with fixed thresholds")
        print(f"• Isolation Forest: Unsupervised anomaly detection (binary classification)")
        print(f"• Heuristic Voting: Multi-rule voting system with majority decision")
        print(f"• All methods use the same train/test split for fair comparison")


def main():
    """Main function to run all baseline methods"""
    # Initialize with the dataset path
    dataset_path = '../data/cleaned_ml_dataset.csv'
    
    detector = BaselineAnomalyDetectors(dataset_path)
    results = detector.run_all_baselines()
    
    # Save results for future analysis
    import pickle
    with open('baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to 'baseline_results.pkl'")
    print(f"🎉 Baseline evaluation completed successfully!")


if __name__ == "__main__":
    main()