#!/usr/bin/env python3
"""
MAIN EXPERIMENT SCRIPT
Run this file to automatically collect data for anomaly detection.

Usage: python run_experiments.py
"""

import asyncio
import time
import logging
import os
import sys
import subprocess
import requests
from datetime import datetime

class ExperimentOrchestrator:
    def __init__(self):
        self.setup_logging()
        self.experiments_completed = []
        
    def setup_logging(self):
        """Setup logging"""
        os.makedirs('data', exist_ok=True)
        log_file = f"data/experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        print(f"📝 Logging to: {log_file}")
    
    def check_services(self):
        """Check if all services are running"""
        services = {
            'Web API': 'http://localhost:8001/health',
            'Order Processor': 'http://localhost:8002/health',
            'Prometheus': 'http://localhost:9090/-/healthy'
        }
        
        print("🔍 Checking services...")
        for name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ {name}: Running")
                else:
                    print(f"   ❌ {name}: Error {response.status_code}")
                    return False
            except Exception as e:
                print(f"   ❌ {name}: Not accessible - {str(e)}")
                print(f"      Make sure to run: python {name.lower().replace(' ', '_')}.py")
                return False
        
        print("✅ All services are running!")
        return True
    
    async def run_baseline_experiment(self):
        """Run 30 minutes of normal operation"""
        print("\n📊 EXPERIMENT 1: Baseline (Normal Operation)")
        print("Duration: 30 minutes")
        print("Purpose: Collect normal behavior data")
        
        # Import here to avoid circular imports
        from data_collector import FixedDataCollector
        from load_generator import LoadGenerator
        
        collector = FixedDataCollector()
        load_gen = LoadGenerator()
        
        # Start data collection
        collection_task = asyncio.create_task(
            collector.start_collection(duration_minutes=30, experiment_name="baseline")
        )
        
        # Start normal traffic
        load_task = asyncio.create_task(
            load_gen.generate_normal_traffic(duration_minutes=30, rps=2.0)
        )
        
        # Wait for both to complete
        await asyncio.gather(collection_task, load_task)
        
        # Export data
        csv_file = collector.export_to_csv("data/baseline_experiment.csv")
        self.experiments_completed.append(csv_file)
        
        print("✅ Baseline experiment completed!")
        logging.info("Baseline experiment completed")
        
        return csv_file
    
    async def run_cpu_spike_experiment(self):
        """Run CPU spike experiment - FIXED ANOMALY LABELING"""
        print("\n🔥 EXPERIMENT 2: CPU Spike")
        print("Target: Web API service")
        print("Duration: 20 minutes (5 normal + 10 spike + 5 recovery)")
        
        from data_collector import FixedDataCollector
        from load_generator import LoadGenerator
        from failure_injector import FailureInjector
        
        collector = FixedDataCollector()
        load_gen = LoadGenerator()
        injector = FailureInjector()
        
        # Start traffic first
        load_task = asyncio.create_task(
            load_gen.generate_normal_traffic(duration_minutes=20, rps=3.0)
        )
        
        # FIXED: Custom collection with proper anomaly timing
        print("📊 Starting controlled data collection...")
        collector.experiment_name = "cpu_spike_web_api"
        
        # Phase 1: Normal operation (5 minutes)
        print("⏱️  Phase 1: 5 minutes normal operation...")
        end_phase1 = time.time() + 300
        collection_count = 0
        
        while time.time() < end_phase1:
            snapshot = await collector.collect_single_snapshot()
            collector.save_snapshot(snapshot)
            collection_count += 1
            await asyncio.sleep(collector.collection_interval)
        
        # Phase 2: Anomaly injection (10 minutes)
        print("💥 Phase 2: Injecting CPU spike in Web API...")
        collector.start_anomaly_period("cpu_spike", "web-api")
        success = injector.inject_cpu_spike("web-api", duration=600)
        
        if success:
            print("⏱️  10 minutes with CPU spike (collecting anomaly data)...")
            end_phase2 = time.time() + 600
            
            while time.time() < end_phase2:
                snapshot = await collector.collect_single_snapshot()
                collector.save_snapshot(snapshot)
                collection_count += 1
                
                if collection_count % 10 == 0:
                    print(f"   📈 Collected {collection_count} snapshots (anomaly_label={snapshot.get('anomaly_label', 'unknown')})")
                
                await asyncio.sleep(collector.collection_interval)
            
            print("🔧 Stopping CPU spike...")
            injector.stop_cpu_spike("web-api")
            collector.end_anomaly_period()
            
            # Phase 3: Recovery (5 minutes)
            print("⏱️  Phase 3: 5 minutes recovery...")
            end_phase3 = time.time() + 300
            
            while time.time() < end_phase3:
                snapshot = await collector.collect_single_snapshot()
                collector.save_snapshot(snapshot)
                collection_count += 1
                await asyncio.sleep(collector.collection_interval)
        else:
            print("❌ Failed to inject CPU spike")
            collector.end_anomaly_period()
        
        print(f"✅ Controlled collection completed: {collection_count} snapshots")
        
        # Wait for load task to complete
        await load_task
        
        # Export data
        csv_file = collector.export_to_csv("data/cpu_spike_experiment.csv")
        self.experiments_completed.append(csv_file)
        
        print("✅ CPU spike experiment completed!")
        return csv_file
    
    async def run_memory_leak_experiment(self):
        """Run memory leak experiment - FIXED ANOMALY LABELING"""
        print("\n🧠 EXPERIMENT 3: Memory Leak")
        print("Target: Order Processor service")
        print("Duration: 20 minutes (5 normal + 10 leak + 5 recovery)")
        
        from data_collector import FixedDataCollector
        from load_generator import LoadGenerator
        from failure_injector import FailureInjector
        
        collector = FixedDataCollector()
        load_gen = LoadGenerator()
        injector = FailureInjector()
        
        # Start traffic
        load_task = asyncio.create_task(
            load_gen.generate_normal_traffic(duration_minutes=20, rps=3.0)
        )
        
        # FIXED: Custom collection with proper anomaly timing
        print("📊 Starting controlled data collection...")
        collector.experiment_name = "memory_leak_processor"
        collection_count = 0
        
        # Phase 1: Normal operation (5 minutes)
        print("⏱️  Phase 1: 5 minutes normal operation...")
        end_phase1 = time.time() + 300
        
        while time.time() < end_phase1:
            snapshot = await collector.collect_single_snapshot()
            collector.save_snapshot(snapshot)
            collection_count += 1
            await asyncio.sleep(collector.collection_interval)
        
        # Phase 2: Memory leak injection (10 minutes)
        print("💥 Phase 2: Starting memory leak in Order Processor...")
        collector.start_anomaly_period("memory_leak", "order-processor")
        success = injector.inject_memory_leak("order-processor", duration=600)
        
        if success:
            print("⏱️  10 minutes with memory leak (collecting anomaly data)...")
            end_phase2 = time.time() + 600
            
            while time.time() < end_phase2:
                snapshot = await collector.collect_single_snapshot()
                collector.save_snapshot(snapshot)
                collection_count += 1
                
                if collection_count % 10 == 0:
                    print(f"   📈 Collected {collection_count} snapshots (anomaly_label={snapshot.get('anomaly_label', 'unknown')})")
                
                await asyncio.sleep(collector.collection_interval)
            
            print("🔧 Stopping memory leak...")
            injector.stop_memory_leak("order-processor")
            collector.end_anomaly_period()
            
            # Phase 3: Recovery (5 minutes)
            print("⏱️  Phase 3: 5 minutes recovery...")
            end_phase3 = time.time() + 300
            
            while time.time() < end_phase3:
                snapshot = await collector.collect_single_snapshot()
                collector.save_snapshot(snapshot)
                collection_count += 1
                await asyncio.sleep(collector.collection_interval)
        else:
            print("❌ Failed to inject memory leak")
            collector.end_anomaly_period()
        
        print(f"✅ Controlled collection completed: {collection_count} snapshots")
        
        # Wait for load task
        await load_task
        
        if success:
            print("⏱️  10 minutes with memory leak...")
            await asyncio.sleep(600)
            
            print("🔧 Stopping memory leak...")
            injector.stop_memory_leak("order-processor")
            collector.end_anomaly_period()
            
            print("⏱️  5 minutes recovery...")
            await asyncio.sleep(300)
        else:
            print("❌ Failed to inject memory leak")
            collector.end_anomaly_period()
        
        # Wait for load task
        await load_task
        
        csv_file = collector.export_to_csv("data/memory_leak_experiment.csv")
        self.experiments_completed.append(csv_file)
        
        print("✅ Memory leak experiment completed!")
        return csv_file
    
    async def run_service_crash_experiment(self):
        """Run service crash experiment"""
        print("\n💀 EXPERIMENT 4: Service Crash")
        print("Target: Notification Service")
        print("Duration: 15 minutes (5 normal + 5 crash + 5 recovery)")
        print("NOTE: You'll need to manually restart notification_service.py after crash")
        
        from data_collector import FixedDataCollector
        from load_generator import LoadGenerator
        from failure_injector import FailureInjector
        
        collector = FixedDataCollector()
        load_gen = LoadGenerator()
        injector = FailureInjector()
        
        collection_task = asyncio.create_task(
            collector.start_collection(duration_minutes=15, experiment_name="service_crash_test")
        )
        
        load_task = asyncio.create_task(
            load_gen.generate_normal_traffic(duration_minutes=15, rps=2.0)
        )
        
        # 5 minutes normal
        print("⏱️  5 minutes normal operation...")
        await asyncio.sleep(300)
        
        # Crash service
        print("💥 Crashing Notification Service...")
        collector.start_anomaly_period("service_crash", "notification-service")
        success = injector.inject_service_crash("notification-service")
        
        if success:
            print("💀 Service crashed! Messages will queue up...")
            print("⏱️  5 minutes with crashed service...")
            await asyncio.sleep(300)
            
            print("🔧 Please restart notification service in another terminal:")
            print("   python notification_service.py")
            print("   Press Enter when you've restarted it...")
            input()
            
            collector.end_anomaly_period()
            print("⏱️  5 minutes recovery...")
            await asyncio.sleep(300)
        else:
            print("❌ Failed to crash service")
            collector.end_anomaly_period()
        
        await asyncio.gather(collection_task, load_task)
        
        csv_file = collector.export_to_csv("data/service_crash_experiment.csv")
        self.experiments_completed.append(csv_file)
        
        print("✅ Service crash experiment completed!")
        return csv_file
    
    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*60)
        print("🎉 ALL EXPERIMENTS COMPLETED!")
        print("="*60)
        
        print(f"\n📁 Data files created ({len(self.experiments_completed)} files):")
        for i, file_path in enumerate(self.experiments_completed, 1):
            print(f"   {i}. {file_path}")
        
        print(f"\n📊 Dataset Summary:")
        
        # Try to load and analyze data
        try:
            import pandas as pd
            total_points = 0
            anomaly_points = 0
            
            for file_path in self.experiments_completed:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    points = len(df)
                    anomalies = df['anomaly_label'].sum() if 'anomaly_label' in df.columns else 0
                    total_points += points
                    anomaly_points += anomalies
                    
                    filename = os.path.basename(file_path)
                    print(f"   {filename}: {points} data points ({anomalies} anomalies)")
            
            print(f"\n📈 Total: {total_points} data points")
            print(f"   Normal: {total_points - anomaly_points} ({((total_points - anomaly_points)/total_points*100):.1f}%)")
            print(f"   Anomalies: {anomaly_points} ({(anomaly_points/total_points*100):.1f}%)")
            
        except Exception as e:
            print(f"   Could not analyze data: {e}")
        
        print(f"\n🚀 Next Steps:")
        print("   1. Load data: df = pd.read_csv('data/baseline_experiment.csv')")
        print("   2. Feature engineering: rolling averages, correlations")
        print("   3. Train models: use anomaly_label as target")
        print("   4. Build explainable AI: use anomaly_type for explanations")
        
        print("\n💡 Data Format:")
        print("   - timestamp: When data was collected")
        print("   - service_name: Which service the metric is from") 
        print("   - metric_name: CPU, memory, request_rate, etc.")
        print("   - metric_value: The actual measurement")
        print("   - anomaly_label: 0=normal, 1=anomaly")
        print("   - anomaly_type: cpu_spike, memory_leak, service_crash, normal")

async def main():
    """Main function to run all experiments"""
    print("🚀 Microservices Anomaly Detection - Automated Data Collection")
    print("=" * 70)
    
    orchestrator = ExperimentOrchestrator()
    
    # Check if services are running
    if not orchestrator.check_services():
        print("\n❌ Services not running. Please start them first:")
        print("   1. docker-compose up -d")
        print("   2. python web_api.py")
        print("   3. python order_processor.py") 
        print("   4. python notification_service.py")
        print("   5. Then run: python run_experiments.py")
        return
    
    print(f"\n⏰ Total estimated time: ~1.5 hours")
    print("   - Baseline: 30 minutes")
    print("   - CPU Spike: 20 minutes")
    print("   - Memory Leak: 20 minutes") 
    print("   - Service Crash: 15 minutes")
    print("   - Delays: 5 minutes")
    
    # Ask for confirmation
    response = input("\nStart automated data collection? (y/N): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return
    
    start_time = time.time()
    
    try:
        # Run all experiments
        await orchestrator.run_baseline_experiment()
        
        print("\n⏳ Waiting 2 minutes between experiments...")
        await asyncio.sleep(120)
        
        await orchestrator.run_cpu_spike_experiment()
        
        print("\n⏳ Waiting 2 minutes between experiments...")
        await asyncio.sleep(120)
        
        await orchestrator.run_memory_leak_experiment()
        
        print("\n⏳ Waiting 2 minutes between experiments...")
        await asyncio.sleep(120)
        
        await orchestrator.run_service_crash_experiment()
        
        # Print final summary
        total_time = (time.time() - start_time) / 3600  # Convert to hours
        print(f"\n⏱️  Total time: {total_time:.1f} hours")
        
        orchestrator.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        print("Partial data may be available in data/ folder")
        
    except Exception as e:
        print(f"\n\n❌ Experiment failed: {str(e)}")
        logging.error(f"Experiment failed: {str(e)}")

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    try:
        # Run main experiment
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)