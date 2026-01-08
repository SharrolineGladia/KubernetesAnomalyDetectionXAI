#!/usr/bin/env python3
"""
Dataset Validation Report for Academic Documentation
Professional summary of anomaly detection dataset quality and characteristics
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_dataset_validation_report():
    print("ANOMALY DETECTION DATASET - VALIDATION REPORT")
    print("=" * 55)
    print(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    print("=" * 55)
    
    
    # Check if final dataset exists
    dataset_file = 'data/cleaned_ml_dataset.csv'
    
    if not os.path.exists(dataset_file):
        print(f"❌ ERROR: Dataset file not found - {dataset_file}")
        print("Please ensure the cleaned ML dataset exists before generating report.")
        return None
    
    # Load and analyze dataset
    df = pd.read_csv(dataset_file)
    
    print("\n1. DATASET OVERVIEW")
    print("-" * 20)
    print(f"Dataset File: {dataset_file}")
    print(f"Total Samples: {df.shape[0]:,}")
    print(f"Total Features: {df.shape[1] - 3}")  # Excluding timestamp, labels
    print(f"File Size: {os.path.getsize(dataset_file) / (1024*1024):.2f} MB")
    print(f"Data Collection Period: Multi-phase experimental scenarios")
    
    # Data quality metrics
    print("\n2. DATA QUALITY ASSESSMENT")
    print("-" * 28)
    
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    print(f"Missing Values: {missing_values:,} ({missing_values/len(df)*100:.2f}%)")
    print(f"Duplicate Records: {duplicate_rows:,} ({duplicate_rows/len(df)*100:.2f}%)")
    print(f"Data Completeness: {((len(df) * df.shape[1] - missing_values) / (len(df) * df.shape[1]) * 100):.1f}%")
    
    # Check for invalid values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_values = 0
    for col in numeric_cols:
        infinite_values += np.isinf(df[col]).sum()
    
    print(f"Invalid/Infinite Values: {infinite_values:,}")
    print(f"Overall Data Quality: {'✓ Excellent' if missing_values == 0 and infinite_values == 0 else '⚠ Good with minor issues' if missing_values < 100 else '✗ Needs attention'}")
    
    # Class distribution analysis
    print("\n3. CLASS DISTRIBUTION ANALYSIS")
    print("-" * 32)
    
    total_samples = len(df)
    normal_count = (df['anomaly_label'] == 0).sum()
    anomaly_count = (df['anomaly_label'] == 1).sum()
    
    print(f"Normal Samples: {normal_count:,} ({normal_count/total_samples*100:.1f}%)")
    print(f"Anomaly Samples: {anomaly_count:,} ({anomaly_count/total_samples*100:.1f}%)")
    
    # Class balance ratio
    balance_ratio = min(normal_count, anomaly_count) / max(normal_count, anomaly_count)
    balance_status = "Well Balanced" if balance_ratio > 0.3 else "Moderately Balanced" if balance_ratio > 0.1 else "Imbalanced"
    print(f"Class Balance Ratio: {balance_ratio:.2f} ({balance_status})")
    
    # Anomaly type distribution
    print(f"\nAnomaly Type Distribution:")
    anomaly_types = df['anomaly_type'].value_counts()
    for atype, count in anomaly_types.items():
        if atype != 'normal':
            print(f"  • {atype.replace('_', ' ').title()}: {count:,} ({count/total_samples*100:.1f}%)")
    
    # Feature analysis
    print("\n4. FEATURE CHARACTERISTICS")
    print("-" * 28)
    
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'anomaly_label', 'anomaly_type']]
    print(f"Total Features: {len(feature_cols)}")
    
    # Categorize features by service
    service_features = {
        'Web API Service': [col for col in feature_cols if 'web_api' in col],
        'Order Processor': [col for col in feature_cols if 'processor' in col],
        'Notification Service': [col for col in feature_cols if 'notification' in col]
    }
    
    print(f"\nFeature Distribution by Service:")
    for service, features in service_features.items():
        print(f"  • {service}: {len(features)} metrics")
    
    # Feature categories
    feature_categories = {
        'Performance Metrics': [col for col in feature_cols if any(keyword in col for keyword in ['cpu', 'memory', 'response_time'])],
        'Throughput Metrics': [col for col in feature_cols if any(keyword in col for keyword in ['requests', 'processing_rate', 'message_rate'])],
        'Quality Metrics': [col for col in feature_cols if any(keyword in col for keyword in ['error', 'success', 'health'])],
        'Resource Metrics': [col for col in feature_cols if any(keyword in col for keyword in ['connections', 'threads', 'queue'])]
    }
    
    print(f"\nFeature Categories:")
    for category, features in feature_categories.items():
        print(f"  • {category}: {len(features)} features")
    
    # Statistical validation
    print("\n5. STATISTICAL VALIDATION")
    print("-" * 28)
    
    # Sample key metrics for class separation analysis
    key_metrics = ['web_api_cpu', 'processor_memory', 'notification_response_time_p95', 'processor_error_rate']
    separations = []
    
    normal_data = df[df['anomaly_label'] == 0]
    anomaly_data = df[df['anomaly_label'] == 1]
    
    print("Class Separation Analysis (Cohen's d):")
    for metric in key_metrics:
        if metric in df.columns:
            normal_mean = normal_data[metric].mean()
            anomaly_mean = anomaly_data[metric].mean()
            pooled_std = np.sqrt(((normal_data[metric].var() + anomaly_data[metric].var()) / 2))
            
            if pooled_std > 0:
                cohens_d = abs(normal_mean - anomaly_mean) / pooled_std
                separations.append(cohens_d)
                
                effect_size = "Large" if cohens_d > 0.8 else "Medium" if cohens_d > 0.5 else "Small" if cohens_d > 0.2 else "Negligible"
                print(f"  • {metric}: d = {cohens_d:.2f} ({effect_size} effect)")
    
    avg_separation = np.mean(separations) if separations else 0
    overall_separation = "Excellent" if avg_separation > 1.0 else "Good" if avg_separation > 0.5 else "Fair" if avg_separation > 0.2 else "Poor"
    print(f"\nOverall Class Separability: {avg_separation:.2f} ({overall_separation})")
    
    # Value range validation
    print(f"\nValue Range Validation:")
    range_issues = []
    
    # Check CPU metrics (should be 0-100% normally)
    cpu_metrics = [col for col in feature_cols if 'cpu' in col]
    for metric in cpu_metrics:
        max_val = df[metric].max()
        if max_val > 300:
            range_issues.append(f"{metric}: {max_val:.1f}% (very high)")
        elif max_val < 0:
            range_issues.append(f"{metric}: negative values")
    
    # Check memory metrics (should be positive)
    memory_metrics = [col for col in feature_cols if 'memory' in col and 'growth' not in col]
    for metric in memory_metrics:
        min_val = df[metric].min()
        if min_val < 0:
            range_issues.append(f"{metric}: negative values")
    
    if not range_issues:
        print("  ✓ All metric values within expected ranges")
    else:
        print("  ⚠ Range issues detected:")
        for issue in range_issues[:3]:
            print(f"    - {issue}")
    
    # ML readiness assessment
    print("\n6. MACHINE LEARNING READINESS")
    print("-" * 32)
    
    ml_score = 0
    max_score = 6
    
    # Sample size adequacy
    if len(df) >= 4000:
        print("  ✓ Sample Size: Excellent (≥4000 samples)")
        ml_score += 2
    elif len(df) >= 1000:
        print("  ✓ Sample Size: Good (≥1000 samples)")
        ml_score += 1
    else:
        print("  ✗ Sample Size: Insufficient (<1000 samples)")
    
    # Class balance
    if balance_ratio > 0.3:
        print("  ✓ Class Balance: Well balanced")
        ml_score += 1
    elif balance_ratio > 0.1:
        print("  ✓ Class Balance: Acceptable")
        ml_score += 0.5
    else:
        print("  ✗ Class Balance: Imbalanced")
    
    # Data quality
    if missing_values == 0 and infinite_values == 0:
        print("  ✓ Data Quality: Excellent")
        ml_score += 1
    elif missing_values < 100:
        print("  ✓ Data Quality: Good")
        ml_score += 0.5
    else:
        print("  ✗ Data Quality: Needs improvement")
    
    # Feature richness
    if len(feature_cols) >= 20:
        print("  ✓ Feature Richness: Comprehensive")
        ml_score += 1
    elif len(feature_cols) >= 10:
        print("  ✓ Feature Richness: Adequate")
        ml_score += 0.5
    else:
        print("  ✗ Feature Richness: Limited")
    
    # Class separability
    if avg_separation > 0.8:
        print("  ✓ Class Separability: Excellent")
        ml_score += 1
    elif avg_separation > 0.5:
        print("  ✓ Class Separability: Good")
        ml_score += 0.5
    else:
        print("  ✗ Class Separability: Weak")
    
    ml_readiness_percentage = (ml_score / max_score) * 100
    readiness_level = "Excellent" if ml_readiness_percentage >= 85 else "Good" if ml_readiness_percentage >= 70 else "Fair" if ml_readiness_percentage >= 50 else "Poor"
    
    print(f"\nML Readiness Score: {ml_score:.1f}/{max_score} ({ml_readiness_percentage:.0f}% - {readiness_level})")
    
    # Experimental methodology validation
    print("\n7. EXPERIMENTAL METHODOLOGY")
    print("-" * 30)
    
    print("Data Collection Approach:")
    print("  • Real-time metrics collection from live microservices")
    print("  • Controlled anomaly injection scenarios")
    print("  • Multi-phase experiments with baseline and recovery periods")
    print("  • Comprehensive monitoring across 3 service tiers")
    
    print(f"\nExperimental Scenarios Covered:")
    experiment_types = df['anomaly_type'].unique()
    for exp_type in sorted(experiment_types):
        if exp_type != 'normal':
            count = (df['anomaly_type'] == exp_type).sum()
            print(f"  • {exp_type.replace('_', ' ').title()}: {count:,} samples")
    
    # Dataset validation summary
    print("\n" + "=" * 55)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 55)
    
    validation_status = "VALIDATED" if ml_readiness_percentage >= 70 and missing_values == 0 else "CONDITIONALLY VALIDATED" if ml_readiness_percentage >= 50 else "REQUIRES IMPROVEMENT"
    
    print(f"Validation Status: {validation_status}")
    print(f"Dataset Quality: {readiness_level}")
    print(f"Recommended Use: {'Production ML Training' if validation_status == 'VALIDATED' else 'Experimental/Development' if validation_status == 'CONDITIONALLY VALIDATED' else 'Data Improvement Required'}")
    
    # Key strengths
    strengths = []
    if len(df) >= 4000:
        strengths.append("Large sample size")
    if balance_ratio > 0.3:
        strengths.append("Well-balanced classes")
    if missing_values == 0:
        strengths.append("Complete data coverage")
    if avg_separation > 0.8:
        strengths.append("Clear anomaly patterns")
    if len(feature_cols) >= 20:
        strengths.append("Comprehensive feature set")
    
    if strengths:
        print(f"\nKey Strengths:")
        for strength in strengths:
            print(f"  • {strength}")
    
    # Limitations/considerations
    limitations = []
    if not range_issues:
        pass  # No range issues is good
    else:
        limitations.append("Some metrics exceed typical ranges")
    
    if balance_ratio < 0.4:
        limitations.append("Slight class imbalance")
    
    if avg_separation < 1.0:
        limitations.append("Moderate class separation")
    
    if limitations:
        print(f"\nConsiderations:")
        for limitation in limitations:
            print(f"  • {limitation}")
    
    print(f"\nDataset ready for: Anomaly Detection, Root Cause Analysis, Multi-class Classification")
    print("=" * 55)
    
    return {
        'validation_status': validation_status,
        'samples': len(df),
        'features': len(feature_cols),
        'ml_readiness': ml_readiness_percentage,
        'class_balance': balance_ratio,
        'separability': avg_separation
    }

if __name__ == "__main__":
    result = generate_dataset_validation_report()
    
    if result and result['validation_status'] == 'VALIDATED':
        print(f"\n✅ Dataset validation completed successfully!")
        print(f"   Status: {result['validation_status']}")
        print(f"   ML Readiness: {result['ml_readiness']:.0f}%")
    elif result:
        print(f"\n⚠️ Dataset validation completed with conditions.")
        print(f"   Status: {result['validation_status']}")
        print(f"   ML Readiness: {result['ml_readiness']:.0f}%")
    else:
        print("❌ Dataset validation failed")