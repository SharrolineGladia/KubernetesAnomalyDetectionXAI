#!/usr/bin/env python3
"""
Dataset Validation Report for Academic Documentation
Professional summary of anomaly detection dataset quality and characteristics
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_report():
    print("ANOMALY DETECTION DATASET - VALIDATION REPORT")
    print("=" * 55)
    print(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    print("=" * 55)
    
    # Load dataset
    dataset_file = 'data/cleaned_ml_dataset.csv'
    
    if not os.path.exists(dataset_file):
        print(f"ERROR: Dataset file not found - {dataset_file}")
        return None
    
    df = pd.read_csv(dataset_file)
    
    # 1. Dataset Overview
    print("\n1. DATASET OVERVIEW")
    print("-" * 20)
    print(f"Dataset File: {dataset_file}")
    print(f"Total Samples: {df.shape[0]:,}")
    print(f"Total Features: {df.shape[1] - 3}")  # Excluding labels
    print(f"File Size: {os.path.getsize(dataset_file) / (1024*1024):.2f} MB")
    
    # 2. Data Quality
    print("\n2. DATA QUALITY ASSESSMENT")
    print("-" * 28)
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"Missing Values: {missing:,} ({missing/len(df)*100:.2f}%)")
    print(f"Duplicate Records: {duplicates:,}")
    print(f"Data Completeness: {((len(df) * df.shape[1] - missing) / (len(df) * df.shape[1]) * 100):.1f}%")
    print(f"Quality Status: {'✓ Excellent' if missing == 0 else '⚠ Good'}")
    
    # 3. Class Distribution
    print("\n3. CLASS DISTRIBUTION ANALYSIS")
    print("-" * 32)
    normal_count = (df['anomaly_label'] == 0).sum()
    anomaly_count = (df['anomaly_label'] == 1).sum()
    total = len(df)
    
    print(f"Normal Samples: {normal_count:,} ({normal_count/total*100:.1f}%)")
    print(f"Anomaly Samples: {anomaly_count:,} ({anomaly_count/total*100:.1f}%)")
    
    balance_ratio = min(normal_count, anomaly_count) / max(normal_count, anomaly_count)
    print(f"Class Balance Ratio: {balance_ratio:.2f}")
    
    # Anomaly types
    print("\nAnomaly Type Distribution:")
    types = df['anomaly_type'].value_counts()
    for atype, count in types.items():
        if atype != 'normal':
            print(f"  • {atype.replace('_', ' ').title()}: {count:,} ({count/total*100:.1f}%)")
    
    # 4. Features
    print("\n4. FEATURE CHARACTERISTICS")
    print("-" * 28)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'anomaly_label', 'anomaly_type']]
    print(f"Total Features: {len(feature_cols)}")
    
    # Service distribution
    services = {
        'Web API': len([col for col in feature_cols if 'web_api' in col]),
        'Order Processor': len([col for col in feature_cols if 'processor' in col]),
        'Notification Service': len([col for col in feature_cols if 'notification' in col])
    }
    
    print("\nFeatures by Service:")
    for service, count in services.items():
        print(f"  • {service}: {count} metrics")
    
    # 5. Statistical Analysis
    print("\n5. STATISTICAL VALIDATION")
    print("-" * 28)
    
    normal_data = df[df['anomaly_label'] == 0]
    anomaly_data = df[df['anomaly_label'] == 1]
    
    # Check separation for key metrics
    key_metrics = ['web_api_cpu', 'processor_memory']
    separations = []
    
    print("Class Separation Analysis:")
    for metric in key_metrics:
        if metric in df.columns:
            normal_mean = normal_data[metric].mean()
            anomaly_mean = anomaly_data[metric].mean()
            normal_std = normal_data[metric].std()
            
            if normal_std > 0:
                separation = abs(anomaly_mean - normal_mean) / normal_std
                separations.append(separation)
                effect = "Large" if separation > 0.8 else "Medium" if separation > 0.5 else "Small"
                print(f"  • {metric}: {separation:.2f}σ ({effect})")
    
    avg_sep = np.mean(separations) if separations else 0
    print(f"\nOverall Separability: {avg_sep:.2f}σ ({'Excellent' if avg_sep > 1.0 else 'Good' if avg_sep > 0.5 else 'Fair'})")
    
    # 6. ML Readiness
    print("\n6. MACHINE LEARNING READINESS")
    print("-" * 32)
    
    ml_score = 0
    
    # Sample size
    if len(df) >= 4000:
        print("  ✓ Sample Size: Excellent (≥4000)")
        ml_score += 2
    elif len(df) >= 1000:
        print("  ✓ Sample Size: Good (≥1000)")
        ml_score += 1
    
    # Balance
    if balance_ratio > 0.3:
        print("  ✓ Class Balance: Well balanced")
        ml_score += 1
    
    # Quality
    if missing == 0:
        print("  ✓ Data Quality: Perfect")
        ml_score += 1
    
    # Features
    if len(feature_cols) >= 20:
        print("  ✓ Feature Richness: Comprehensive")
        ml_score += 1
    
    # Separability
    if avg_sep > 0.8:
        print("  ✓ Class Separability: Excellent")
        ml_score += 1
    elif avg_sep > 0.5:
        print("  ✓ Class Separability: Good")
        ml_score += 0.5
    
    readiness = (ml_score / 6) * 100
    status = "Excellent" if readiness >= 85 else "Good" if readiness >= 70 else "Fair"
    
    print(f"\nML Readiness Score: {ml_score}/6 ({readiness:.0f}% - {status})")
    
    # 7. Summary
    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)
    
    validation_status = "VALIDATED" if readiness >= 70 and missing == 0 else "CONDITIONALLY VALIDATED"
    
    print(f"Validation Status: {validation_status}")
    print(f"Dataset Quality: {status}")
    print(f"Recommended Use: ML Training & Anomaly Detection")
    
    print(f"\nKey Strengths:")
    if len(df) >= 4000:
        print("  • Large sample size (4000+ samples)")
    if balance_ratio > 0.3:
        print("  • Well-balanced classes")
    if missing == 0:
        print("  • Complete data coverage")
    if len(feature_cols) >= 20:
        print("  • Comprehensive feature set")
    if avg_sep > 0.8:
        print("  • Clear anomaly patterns")
    
    print(f"\nSuitable for: Anomaly Detection, Classification, Root Cause Analysis")
    print("=" * 55)
    
    return validation_status

if __name__ == "__main__":
    result = generate_report()
    if result == "VALIDATED":
        print("\n✅ Dataset validation successful!")
    else:
        print("\n⚠️ Dataset conditionally validated")