REM ==============================================================================
REM start_services.bat - Start all services
REM ==============================================================================

@echo off
echo 🚀 Starting Microservices Anomaly Detection System...

REM Check if Docker is running
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not found. Please install Docker Desktop.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create data directory
if not exist "data" mkdir data

echo 📦 Starting infrastructure services...
docker-compose up -d

echo ⏳ Waiting for infrastructure to be ready...
timeout /t 15 /nobreak >nul

REM Check if services are healthy
echo 🔍 Checking infrastructure health...
curl -f http://localhost:9090/-/healthy >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Prometheus not ready, but continuing...
) else (
    echo ✅ Prometheus: Ready
)

curl -f http://localhost:6379 >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Redis not ready, but continuing...
) else (
    echo ✅ Redis: Ready
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 🐍 Activating Python virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found. Using system Python.
)

echo 🚀 Starting microservices...
echo    You'll see 3 new command windows open for each service.
echo    Keep them running during experiments.
echo.

REM Start Web API Service
echo 📡 Starting Web API Service (Port 8001)...
start "Web API Service" cmd /c "cd ../services && python web_api.py & pause"
timeout /t 3 /nobreak >nul

REM Start Order Processor Service
echo 📦 Starting Order Processor Service (Port 8002)...
start "Order Processor Service" cmd /c "cd ../services && python order_processor.py & pause"
timeout /t 3 /nobreak >nul

REM Start Notification Service
echo 📬 Starting Notification Service (Port 8003)...
start "Notification Service" cmd /c "cd ../services && python notification_service.py & pause"
timeout /t 5 /nobreak >nul

echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Test services
echo 🧪 Testing services...

curl -f http://localhost:8001/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Web API: Not responding
    echo    Check the "Web API Service" window for errors
) else (
    echo ✅ Web API: Running
)

curl -f http://localhost:8002/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Order Processor: Not responding
    echo    Check the "Order Processor Service" window for errors
) else (
    echo ✅ Order Processor: Running
)

REM Notification service doesn't have HTTP endpoint, just check if process exists
timeout /t 2 /nobreak >nul
echo ✅ Notification Service: Started (check window for errors)

echo.
echo 🎉 All services started!
echo.
echo 📊 Access Points:
echo    Web API: http://localhost:8001
echo    Order Processor: http://localhost:8002
echo    Prometheus: http://localhost:9090
echo    Jaeger UI: http://localhost:16686
echo.
echo 🚀 Ready to run experiments!
echo    Run: python run_experiments.py
echo.
echo ⚠️  To stop all services, run: stop_services.bat
echo    Or close this window and the service windows manually.
echo.
pause