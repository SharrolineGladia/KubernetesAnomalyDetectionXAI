#!/usr/bin/env python3
"""
ANOMALY EXPERIMENTS ONLY - Skip baseline since it exists
Run this to collect only the anomaly experiments with fixed labeling.
"""

import asyncio
import time
import logging
import os
import sys
from datetime import datetime

# Import the fixed experiment orchestrator
from run_experiments import ExperimentOrchestrator

async def run_anomaly_experiments_only():
    """Run only the anomaly experiments (skip baseline)"""
    print("🚀 Microservices Anomaly Detection - ANOMALY EXPERIMENTS ONLY")
    print("=" * 70)
    
    orchestrator = ExperimentOrchestrator()
    
    # Check if services are running
    if not orchestrator.check_services():
        print("\n❌ Services not running. Please start them first:")
        print("   1. Services should already be running")
        print("   2. If not: run start_services.bat")
        return
    
    # Check if baseline exists
    baseline_file = "data/baseline_experiment.csv"
    if os.path.exists(baseline_file):
        print(f"✅ Found existing baseline: {baseline_file}")
        orchestrator.experiments_completed.append(baseline_file)
    else:
        print("⚠️  No baseline found - will skip for now")
    
    print(f"\n⏰ Estimated time: ~1 hour (skipping baseline)")
    print("   - CPU Spike: 20 minutes (FIXED labeling)")
    print("   - Memory Leak: 20 minutes (FIXED labeling)") 
    print("   - Service Crash: 15 minutes (FIXED labeling)")
    print("   - Delays: 5 minutes")
    
    # Ask for confirmation
    response = input("\nStart anomaly experiments with FIXED labeling? (y/N): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return
    
    start_time = time.time()
    
    try:
        print("\n🎯 STARTING ANOMALY EXPERIMENTS WITH FIXED LABELING...")
        
        # CPU Spike (fixed)
        await orchestrator.run_cpu_spike_experiment()
        
        print("\n⏳ Waiting 2 minutes between experiments...")
        await asyncio.sleep(120)
        
        # Memory Leak (fixed)
        await orchestrator.run_memory_leak_experiment()
        
        print("\n⏳ Waiting 2 minutes between experiments...")
        await asyncio.sleep(120)
        
        # Service Crash (needs to be manually handled)
        print("\n💀 EXPERIMENT 4: Service Crash")
        print("⚠️  NOTE: This experiment requires manual service restart")
        print("     You'll be prompted when to restart the notification service")
        
        crash_response = input("\nRun service crash experiment? (y/N): ").lower().strip()
        if crash_response == 'y':
            await orchestrator.run_service_crash_experiment()
        else:
            print("⏭️  Skipping service crash experiment")
        
        # Print final summary
        total_time = (time.time() - start_time) / 3600  # Convert to hours
        print(f"\n⏱️  Total time: {total_time:.1f} hours")
        
        orchestrator.print_summary()
        
        print("\n🎉 ANOMALY EXPERIMENTS COMPLETED!")
        print("✅ All experiments now have PROPER anomaly labeling")
        print("✅ Ready for ML training and explainable AI")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        print("Partial data may be available in data/ folder")
        
    except Exception as e:
        print(f"\n\n❌ Experiments failed: {str(e)}")
        logging.error(f"Experiments failed: {str(e)}")

if __name__ == "__main__":
    # Setup logging
    os.makedirs('data', exist_ok=True)
    log_file = f"data/anomaly_experiments_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print(f"📝 Logging to: {log_file}")
    
    try:
        asyncio.run(run_anomaly_experiments_only())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)