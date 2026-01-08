#!/usr/bin/env python3
"""
SELECTIVE ANOMALY EXPERIMENTS
Run specific anomaly types for targeted data generation

Usage: 
  python run_specific_experiment.py cpu_spike
  python run_specific_experiment.py memory_leak  
  python run_specific_experiment.py service_crash
  python run_specific_experiment.py menu
"""

import asyncio
import time
import sys
import os
from datetime import datetime
from run_experiments import ExperimentOrchestrator

class SelectiveExperimentRunner:
    def __init__(self):
        self.orchestrator = ExperimentOrchestrator()
        
    async def run_single_experiment(self, experiment_type: str):
        """Run a single specific experiment"""
        
        # Check services first
        if not self.orchestrator.check_services():
            print("\n❌ Services not running. Please start them first:")
            print("   1. Start Docker: docker-compose up -d")
            print("   2. Start services: ../infrastructure/start_services.bat")
            return False
            
        print(f"\n🎯 RUNNING {experiment_type.upper()} EXPERIMENT")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            if experiment_type == "cpu_spike":
                print("🔥 Starting CPU Spike Experiment...")
                print("   Duration: 20 minutes")
                print("   Target: Web API service")
                await self.orchestrator.run_cpu_spike_experiment()
                
            elif experiment_type == "memory_leak":
                print("🟡 Starting Memory Leak Experiment...")
                print("   Duration: 20 minutes")
                print("   Target: Order Processor service")
                await self.orchestrator.run_memory_leak_experiment()
                
            elif experiment_type == "service_crash":
                print("💥 Starting Service Crash Experiment...")
                print("   Duration: 15 minutes")
                print("   Target: Notification service")
                
                confirm = input("\n⚠️  This will crash the notification service. Continue? (y/N): ").lower().strip()
                if confirm == 'y':
                    await self.orchestrator.run_service_crash_experiment()
                else:
                    print("Cancelled.")
                    return False
                    
            else:
                print(f"❌ Unknown experiment type: {experiment_type}")
                return False
                
            duration = (time.time() - start_time) / 60  # minutes
            print(f"\n✅ {experiment_type.upper()} EXPERIMENT COMPLETED!")
            print(f"   Duration: {duration:.1f} minutes")
            print(f"   Data saved to: data/")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            return False
    
    def show_menu(self):
        """Interactive menu for selecting experiments"""
        print("\n🧪 SELECTIVE EXPERIMENT MENU")
        print("=" * 40)
        print("1. CPU Spike (20 min)")
        print("2. Memory Leak (20 min)")
        print("3. Service Crash (15 min)")
        print("4. Exit")
        
        while True:
            try:
                choice = input("\nSelect experiment (1-4): ").strip()
                
                if choice == "1":
                    return "cpu_spike"
                elif choice == "2":
                    return "memory_leak"
                elif choice == "3":
                    return "service_crash"
                elif choice == "4":
                    print("Goodbye!")
                    return None
                else:
                    print("❌ Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return None

async def main():
    """Main function"""
    runner = SelectiveExperimentRunner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1].lower()
        
        if experiment_type == "menu":
            experiment_type = runner.show_menu()
            if not experiment_type:
                return
                
        valid_experiments = ["cpu_spike", "memory_leak", "service_crash"]
        if experiment_type in valid_experiments:
            await runner.run_single_experiment(experiment_type)
        else:
            print(f"❌ Invalid experiment: {experiment_type}")
            print(f"   Valid options: {', '.join(valid_experiments)}, menu")
            
    else:
        # Show interactive menu
        print("🧪 SELECTIVE ANOMALY EXPERIMENTS")
        print("=" * 40)
        
        experiment_type = runner.show_menu()
        if experiment_type:
            await runner.run_single_experiment(experiment_type)

if __name__ == "__main__":
    asyncio.run(main())