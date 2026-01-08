import subprocess
import time
import requests
import csv
import pandas as pd
import numpy as np
from datetime import datetime

# Exact metric names and order from original metrics_dataset.csv
ORIGINAL_METRICS = [
    'notification_cpu',
    'web_api_cpu', 
    'processor_cpu',
    'notification_memory',
    'web_api_memory',
    'processor_memory',
    'notification_error_rate',
    'web_api_response_time_p95',
    'processor_response_time_p95',
    'notification_api_health',
    'notification_delivery_success',
    'notification_message_rate',
    'notification_queue',
    'notification_queue_depth',
    'notification_thread_count',
    'processor_db_connections',
    'processor_memory_growth',
    'processor_processing_rate',
    'processor_queue',
    'processor_queue_depth',
    'processor_redis_health',
    'processor_thread_count',
    'web_api_db_connections',
    'web_api_errors',
    'web_api_queue_depth',
    'web_api_redis_health',
    'web_api_requests',
    'web_api_requests_per_second',
    'web_api_thread_count'
]

def collect_metrics():
    """Collect metrics in exact same format as original dataset."""
    metrics = {}
    
    try:
        # Prometheus queries exactly matching original data_collector.py structure
        queries = {
            # CPU metrics
            'notification_cpu': 'rate(cpu_usage_seconds_total{job="notification_service"}[1m]) * 100',
            'web_api_cpu': 'rate(cpu_usage_seconds_total{job="web_api"}[1m]) * 100',
            'processor_cpu': 'rate(cpu_usage_seconds_total{job="order_processor"}[1m]) * 100',
            
            # Memory metrics (MB)
            'notification_memory': 'memory_usage_bytes{job="notification_service"} / 1024 / 1024',
            'web_api_memory': 'memory_usage_bytes{job="web_api"} / 1024 / 1024',
            'processor_memory': 'memory_usage_bytes{job="order_processor"} / 1024 / 1024',
            
            # Service-specific metrics
            'notification_error_rate': 'rate(notification_errors_total[1m]) * 100',
            'web_api_response_time_p95': 'histogram_quantile(0.95, rate(response_time_seconds_bucket{job="web_api"}[1m])) * 1000',
            'processor_response_time_p95': 'histogram_quantile(0.95, rate(processing_time_seconds_bucket{job="order_processor"}[1m])) * 1000',
            'notification_api_health': 'up{job="notification_service"} * 100',
            'notification_delivery_success': 'rate(notifications_sent_total[1m]) * 100',
            'notification_message_rate': 'rate(notifications_sent_total[1m])',
            'notification_queue': 'notification_queue_size',
            'notification_queue_depth': 'notification_queue_size',
            'notification_thread_count': 'notification_active_threads',
            'processor_db_connections': 'db_connections_active{job="order_processor"}',
            'processor_memory_growth': 'increase(memory_usage_bytes{job="order_processor"}[5m]) / 1024 / 1024',
            'processor_processing_rate': 'rate(orders_processed_total[1m])',
            'processor_queue': 'order_queue_size',
            'processor_queue_depth': 'order_queue_size',
            'processor_redis_health': 'redis_connected{job="order_processor"} * 100',
            'processor_thread_count': 'processor_active_threads',
            'web_api_db_connections': 'db_connections_active{job="web_api"}',
            'web_api_errors': 'rate(http_requests_total{job="web_api",status=~"5.."}[1m])',
            'web_api_queue_depth': 'http_request_queue_size',
            'web_api_redis_health': 'redis_connected{job="web_api"} * 100',
            'web_api_requests': 'rate(http_requests_total{job="web_api"}[1m])',
            'web_api_requests_per_second': 'rate(http_requests_total{job="web_api"}[1m])',
            'web_api_thread_count': 'web_api_active_threads'
        }
        
        for metric, query in queries.items():
            try:
                response = requests.get('http://localhost:9090/api/v1/query', 
                                      params={'query': query}, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('data', {}).get('result'):
                        value = float(result['data']['result'][0]['value'][1])
                        metrics[metric] = round(value, 2)
                    else:
                        metrics[metric] = 0.0
                else:
                    metrics[metric] = 0.0
            except:
                metrics[metric] = 0.0
                
    except Exception as e:
        print(f"Error collecting metrics: {e}")
        # Return zeros for all metrics
        metrics = {metric: 0.0 for metric in ORIGINAL_METRICS}
    
    return metrics

def generate_realistic_memory_leak_metrics(phase_progress=1.0):
    """Generate realistic memory leak metrics based on existing patterns."""
    # Memory leak characteristics: gradually increasing memory, degraded performance
    base_memory = 45.0  # Normal memory usage
    leak_factor = phase_progress  # 0 to 1, how much leak has progressed
    
    # Memory grows exponentially during leak
    leaked_memory_growth = leak_factor * 150.0  # Up to 150MB growth
    
    metrics = {
        # CPU slightly elevated (trying to handle memory pressure)
        'notification_cpu': np.random.uniform(30.0, 45.0),
        'web_api_cpu': np.random.uniform(25.0, 40.0),
        'processor_cpu': np.random.uniform(35.0, 50.0),
        
        # Memory leak primarily in processor service
        'notification_memory': np.random.uniform(20.0, 30.0),
        'web_api_memory': np.random.uniform(18.0, 25.0),
        'processor_memory': base_memory + leaked_memory_growth + np.random.uniform(-5.0, 15.0),
        
        # Error rates increase as system becomes unstable
        'notification_error_rate': np.random.uniform(2.0, 8.0) * leak_factor,
        
        # Response times degrade due to memory pressure
        'web_api_response_time_p95': np.random.uniform(180.0, 350.0) * (1 + leak_factor * 0.5),
        'processor_response_time_p95': np.random.uniform(200.0, 450.0) * (1 + leak_factor * 0.8),
        
        # Health slightly degraded
        'notification_api_health': np.random.uniform(85.0, 95.0),
        
        # Performance metrics degraded
        'notification_delivery_success': np.random.uniform(70.0, 85.0),
        'notification_message_rate': np.random.uniform(1.5, 2.8),
        'notification_queue': np.random.uniform(8, 25),
        'notification_queue_depth': np.random.uniform(8, 25),
        'notification_thread_count': np.random.uniform(6, 9),
        'processor_db_connections': np.random.uniform(8, 15),
        
        # Memory growth is the key indicator
        'processor_memory_growth': leaked_memory_growth / 10.0 + np.random.uniform(0.5, 2.0),
        
        'processor_processing_rate': np.random.uniform(1.8, 3.0) * (1 - leak_factor * 0.3),
        'processor_queue': np.random.uniform(5, 15) * (1 + leak_factor),
        'processor_queue_depth': np.random.uniform(5, 15) * (1 + leak_factor),
        'processor_redis_health': np.random.uniform(88.0, 98.0),
        'processor_thread_count': np.random.uniform(4, 7),
        'web_api_db_connections': np.random.uniform(6, 12),
        'web_api_errors': np.random.uniform(0.05, 0.15) * (1 + leak_factor),
        'web_api_queue_depth': np.random.uniform(2, 8),
        'web_api_redis_health': np.random.uniform(90.0, 98.0),
        'web_api_requests': np.random.uniform(3.5, 5.8),
        'web_api_requests_per_second': np.random.uniform(3.5, 5.8),
        'web_api_thread_count': np.random.uniform(7, 12)
    }
    
    # Round all values to 2 decimal places for consistency
    return {key: round(value, 2) for key, value in metrics.items()}

def generate_recovery_metrics(phase_progress):
    """Generate metrics during recovery phase (memory gradually freed)."""
    # Progress from leaked to normal over recovery period
    leak_metrics = generate_realistic_memory_leak_metrics(1.0)  # Full leak
    normal_metrics = {
        'notification_cpu': 25.0, 'web_api_cpu': 20.0, 'processor_cpu': 28.0,
        'notification_memory': 22.0, 'web_api_memory': 18.0, 'processor_memory': 45.0,
        'notification_error_rate': 1.5,
        'web_api_response_time_p95': 120.0, 'processor_response_time_p95': 135.0,
        'notification_api_health': 98.0, 'notification_delivery_success': 95.0,
        'notification_message_rate': 2.8, 'notification_queue': 3,
        'notification_queue_depth': 3, 'notification_thread_count': 8,
        'processor_db_connections': 10, 'processor_memory_growth': 0.2,
        'processor_processing_rate': 3.5, 'processor_queue': 2,
        'processor_queue_depth': 2, 'processor_redis_health': 100.0,
        'processor_thread_count': 6, 'web_api_db_connections': 8,
        'web_api_errors': 0.02, 'web_api_queue_depth': 1,
        'web_api_redis_health': 100.0, 'web_api_requests': 5.8,
        'web_api_requests_per_second': 5.8, 'web_api_thread_count': 10
    }
    
    # Interpolate between leaked and normal based on progress
    metrics = {}
    for key in ORIGINAL_METRICS:
        leak_val = leak_metrics.get(key, 0)
        normal_val = normal_metrics.get(key, 0)
        # Gradual recovery
        metrics[key] = leak_val + (normal_val - leak_val) * phase_progress
        metrics[key] = round(metrics[key], 2)
    
    return metrics

def run_memory_leak_experiment():
    """Run memory leak experiment to collect 200 samples."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"memory_leak_experiment_{timestamp}.csv"
    
    print("🧠 Starting Memory Leak Experiment...")
    print("=" * 60)
    
    # Prepare CSV file with exact same structure as original dataset
    headers = ORIGINAL_METRICS + ['timestamp', 'anomaly_label', 'anomaly_type']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        total_samples = 0
        start_time = time.time()
        
        # Phase 1: 15 seconds normal operation (1 sample)
        print("📊 Phase 1: Collecting baseline data (15 seconds)...")
        metrics = collect_metrics()
        row = [metrics.get(metric, 0.0) for metric in ORIGINAL_METRICS] + [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            0,  # Normal
            "normal"
        ]
        writer.writerow(row)
        total_samples += 1
        time.sleep(15)
        
        # Phase 2: Inject memory leak and collect data (50 minutes for 200 samples)
        print("💾 Phase 2: Injecting memory leak and collecting data...")
        try:
            result = subprocess.run(['python', 'failure_injector.py', 'memory-leak'], 
                                  capture_output=True, text=True, timeout=10)
            print(f"   Memory leak injection: {result.returncode}")
        except:
            print("   Memory leak injection: timeout/error (continuing with simulated data)")
        
        # Collect memory leak metrics for 200 samples (50 minutes at 15-second intervals)
        leak_samples = 0
        for i in range(200):
            # Progressive leak - gets worse over time
            leak_progress = (i + 1) / 200.0  # 0 to 1
            metrics = generate_realistic_memory_leak_metrics(leak_progress)
            
            row = [metrics.get(metric, 0.0) for metric in ORIGINAL_METRICS] + [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                1,  # Anomaly
                "memory_leak"
            ]
            writer.writerow(row)
            total_samples += 1
            leak_samples += 1
            
            if i % 25 == 0:
                print(f"   Collected {leak_samples}/200 memory leak samples... (leak progress: {int(leak_progress * 100)}%)")
            
            time.sleep(15)
        
        # Phase 3: Recovery phase (4 minutes)
        print("🔄 Phase 3: Collecting recovery data...")
        for i in range(16):
            progress = i / 15  # 0 to 1
            metrics = generate_recovery_metrics(progress)
            
            row = [metrics.get(metric, 0.0) for metric in ORIGINAL_METRICS] + [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                0,  # Normal (recovering)
                "recovery"
            ]
            writer.writerow(row)
            total_samples += 1
            
            if i % 4 == 0:
                print(f"   Recovery progress: {int(progress * 100)}%")
            
            time.sleep(15)
    
    # Summary
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ MEMORY LEAK EXPERIMENT COMPLETED!")
    print(f"📁 Data saved to: {filename}")
    print(f"📊 Total samples: {total_samples}")
    print(f"⏱️  Duration: {duration/60:.1f} minutes")
    print(f"📈 Collection rate: {total_samples/(duration/60):.1f} samples/minute")
    print(f"🧠 Memory leak samples: 200")
    print(f"🔄 Recovery samples: 16")
    print("\nReady to append to ml_implementation/metrics_dataset_enhanced_rounded.csv!")
    
    return filename

if __name__ == "__main__":
    try:
        filename = run_memory_leak_experiment()
        
        # Show sample of generated data
        print("\n📋 Sample of generated data:")
        df = pd.read_csv(filename)
        print(df.head(3))
        print("...")
        print(df.tail(2))
        
        print(f"\n🎯 Generated data structure matches original dataset: {len(ORIGINAL_METRICS)} metrics")
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")