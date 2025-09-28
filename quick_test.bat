REM ==============================================================================
REM quick_test.bat - Quick 2-minute test
REM ==============================================================================

@echo off
echo 🚀 Quick Test - 2 Minute Data Collection
echo This will test the complete system quickly
echo.

REM Check if services are running
call test_services.bat
echo.

set /p "continue=Services look good? Continue with test? (y/N): "
if /i not "%continue%"=="y" (
    echo Cancelled.
    pause
    exit /b 1
)

echo.
echo 🧪 Starting 2-minute test...
echo    This will collect baseline data for 2 minutes
echo    Then inject a CPU spike for 1 minute
echo    Total time: ~5 minutes
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run quick test
python -c "
import asyncio
import sys
sys.path.append('.')

async def quick_test():
    try:
        from data_collector import DataCollector
        from load_generator import LoadGenerator
        from failure_injector import FailureInjector
        
        print('📊 Phase 1: Baseline data collection (2 minutes)')
        collector = DataCollector()
        load_gen = LoadGenerator()
        
        # Start data collection and load generation
        collection_task = asyncio.create_task(
            collector.start_collection(duration_minutes=2, experiment_name='quick_test_baseline')
        )
        load_task = asyncio.create_task(
            load_gen.generate_normal_traffic(duration_minutes=2, rps=3.0)
        )
        
        await asyncio.gather(collection_task, load_task)
        
        print('💥 Phase 2: CPU spike test (1 minute)')
        injector = FailureInjector()
        
        # Start another collection
        collection_task = asyncio.create_task(
            collector.start_collection(duration_minutes=1, experiment_name='quick_test_spike')
        )
        load_task = asyncio.create_task(
            load_gen.generate_normal_traffic(duration_minutes=1, rps=3.0)
        )
        
        # Inject CPU spike
        collector.start_anomaly_period('cpu_spike', 'web-api')
        success = injector.inject_cpu_spike('web-api', duration=60)
        
        if success:
            await asyncio.sleep(60)  # Let it run for 1 minute
            injector.stop_cpu_spike('web-api')
            collector.end_anomaly_period()
        
        await asyncio.gather(collection_task, load_task)
        
        print('📁 Exporting data...')
        csv_file1 = collector.export_to_csv('data/quick_test_baseline.csv')
        csv_file2 = collector.export_to_csv('data/quick_test_spike.csv')
        
        print('✅ Quick test completed!')
        print(f'📁 Files created:')
        print(f'   {csv_file1}')
        print(f'   {csv_file2}')
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        
asyncio.run(quick_test())
"

echo.
echo 🎉 Quick test completed!
echo    Check the data\ folder for CSV files
echo    If successful, run: python run_experiments.py
echo.
pause