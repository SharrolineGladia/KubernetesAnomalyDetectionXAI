# 🚀 Usage Guide - File Organization

## 📁 Directory Structure After Reorganization

Your project is now professionally organized. Here's how to use it:

## ⚡ Quick Start Commands

### From Project Root Directory:

```bash
# Start all services
infrastructure/start_services.bat

# Stop all services
infrastructure/stop_services.bat

# Run main experiments
python experiments/run_experiments.py
python experiments/run_anomaly_experiments.py

# Run ML system
cd ml_implementation
python integrated_demo.py

# Launch UI
cd ui
python launch_demo.py
```

### From Infrastructure Directory:

```bash
cd infrastructure
start_services.bat    # Start services
stop_services.bat     # Stop services
docker-compose up     # Start with Docker
```

### From Experiments Directory:

```bash
cd experiments
python run_experiments.py           # Main experiment suite
python run_anomaly_experiments.py   # Anomaly detection tests
python failure_injector.py          # Manual anomaly injection
```

### From Tools Directory:

```bash
cd tools
python dataset_report.py    # Generate dataset analysis
python load_generator.py    # Generate test traffic
```

## 🎯 Key Path Changes Made:

1. **Service Scripts**: Now reference `../services/` for service files
2. **Infrastructure Scripts**: Updated to change directory before running services
3. **Documentation**: Updated file references in README.md
4. **Experiment Scripts**: Path references corrected for new structure

## ✅ Verification:

All path references have been updated to work with the new organization. The system maintains full functionality while providing better project structure!
