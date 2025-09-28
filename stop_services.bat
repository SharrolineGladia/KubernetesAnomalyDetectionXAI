REM ==============================================================================
REM stop_services.bat - Stop all services
REM ==============================================================================

@echo off
echo 🛑 Stopping Microservices System...

echo 📦 Stopping Docker services...
docker-compose down

echo 🔍 Finding and stopping Python services...

REM Kill processes on specific ports
echo 📡 Stopping Web API (port 8001)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    echo    Killing PID %%a
    taskkill /PID %%a /F >nul 2>&1
)

echo 📦 Stopping Order Processor (port 8002)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8002 ^| findstr LISTENING') do (
    echo    Killing PID %%a
    taskkill /PID %%a /F >nul 2>&1
)

echo 📬 Stopping Notification Service (port 8003)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8003 ^| findstr LISTENING') do (
    echo    Killing PID %%a
    taskkill /PID %%a /F >nul 2>&1
)

REM Kill any remaining Python processes with our service names in command line
echo 🧹 Cleaning up remaining processes...
wmic process where "name='python.exe' and (commandline like '%%web_api.py%%' or commandline like '%%order_processor.py%%' or commandline like '%%notification_service.py%%')" delete >nul 2>&1

REM Close service windows
echo 🪟 Closing service windows...
taskkill /FI "WindowTitle eq Web API Service*" /F >nul 2>&1
taskkill /FI "WindowTitle eq Order Processor Service*" /F >nul 2>&1
taskkill /FI "WindowTitle eq Notification Service*" /F >nul 2>&1

echo ✅ All services stopped!
echo.
echo 📁 Data files preserved in data\ folder
echo 🔄 To restart: run start_services.bat
echo.
pause
