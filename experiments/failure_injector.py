"""
Failure Injector - Simple failure injection for Windows/Linux
Injects CPU spikes, memory leaks, and service crashes
"""

import requests
import subprocess
import time
import os
import signal
import threading
from typing import Optional

class FailureInjector:
    def __init__(self):
        self.active_failures = {}
        self.service_ports = {
            'web-api': 8001,
            'order-processor': 8002,
            'notification-service': 8003
        }
        
        print("💥 Failure Injector initialized")
    
    def find_service_pid(self, service_name: str) -> Optional[int]:
        """Find process ID of service by port"""
        port = self.service_ports.get(service_name)
        if not port:
            return None
        
        try:
            if os.name == 'nt':  # Windows
                # Use netstat on Windows
                result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if f':{port} ' in line and 'LISTENING' in line:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            return int(parts[-1])
            else:  # Linux/macOS
                # Use lsof on Unix systems
                result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip())
        except:
            pass
        
        return None
    
    def inject_cpu_spike(self, service_name: str, duration: int = 600) -> bool:
        """Inject CPU spike into service"""
        print(f"💥 Injecting CPU spike into {service_name} for {duration} seconds...")
        
        pid = self.find_service_pid(service_name)
        if not pid:
            print(f"❌ Service {service_name} not found")
            return False
        
        try:
            if os.name == 'nt':  # Windows
                # Create PowerShell script for CPU consumption
                ps_script = f"""
$targetPID = {pid}
$endTime = (Get-Date).AddSeconds({duration})

Write-Host "Starting CPU spike for PID $targetPID"
while ((Get-Date) -lt $endTime) {{
    # CPU-intensive loop
    for ($i = 0; $i -lt 100000; $i++) {{
        $dummy = [Math]::Sqrt($i)
    }}
    Start-Sleep -Milliseconds 1
}}
Write-Host "CPU spike completed"
"""
                
                # Save script to temp file
                temp_dir = os.environ.get('TEMP', 'C:\\temp')
                script_path = os.path.join(temp_dir, f'cpu_spike_{service_name}_{int(time.time())}.ps1')
                
                with open(script_path, 'w') as f:
                    f.write(ps_script)
                
                # Run PowerShell script in background
                process = subprocess.Popen([
                    'powershell.exe', 
                    '-ExecutionPolicy', 'Bypass',
                    '-WindowStyle', 'Hidden',
                    '-File', script_path
                ], creationflags=subprocess.CREATE_NO_WINDOW)
                
            else:  # Linux/macOS
                # Use stress command or Python CPU burner
                process = subprocess.Popen([
                    'python', '-c', 
                    f"""
import time
import multiprocessing

def cpu_burn():
    end_time = time.time() + {duration}
    while time.time() < end_time:
        pass

# Use multiple processes for more CPU load
processes = []
for _ in range(min(4, multiprocessing.cpu_count())):
    p = multiprocessing.Process(target=cpu_burn)
    p.start()
    processes.append(p)

for p in processes:
    p.join()
"""
                ])
            
            self.active_failures[f'{service_name}_cpu'] = {
                'type': 'cpu_spike',
                'process': process,
                'start_time': time.time(),
                'duration': duration
            }
            
            print(f"✅ CPU spike started for {service_name} (PID: {pid})")
            
            # Schedule automatic cleanup
            def cleanup():
                time.sleep(duration + 10)  # Wait a bit extra
                self.stop_cpu_spike(service_name)
            
            threading.Thread(target=cleanup, daemon=True).start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to inject CPU spike: {e}")
            return False
    
    def stop_cpu_spike(self, service_name: str) -> bool:
        """Stop CPU spike injection"""
        failure_key = f'{service_name}_cpu'
        
        if failure_key in self.active_failures:
            try:
                failure = self.active_failures[failure_key]
                process = failure['process']
                
                # Terminate the process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                del self.active_failures[failure_key]
                print(f"✅ CPU spike stopped for {service_name}")
                return True
                
            except Exception as e:
                print(f"⚠️  Error stopping CPU spike: {e}")
        
        return False
    
    def inject_memory_leak(self, service_name: str, duration: int = 600) -> bool:
        """Inject memory leak (only works for order-processor)"""
        if service_name != 'order-processor':
            print(f"❌ Memory leak only supported for order-processor service")
            return False
        
        print(f"💥 Starting memory leak in {service_name} for {duration} seconds...")
        
        try:
            # Call the service's memory leak endpoint
            response = requests.post('http://localhost:8002/simulate-memory-leak', timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Memory leak started in {service_name}")
                
                # Schedule automatic cleanup
                def cleanup():
                    time.sleep(duration)
                    self.stop_memory_leak(service_name)
                
                threading.Thread(target=cleanup, daemon=True).start()
                
                return True
            else:
                print(f"❌ Failed to start memory leak: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to inject memory leak: {e}")
            return False
    
    def stop_memory_leak(self, service_name: str) -> bool:
        """Stop memory leak"""
        if service_name != 'order-processor':
            return False
        
        try:
            response = requests.post('http://localhost:8002/stop-memory-leak', timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Memory leak stopped in {service_name}")
                return True
            else:
                print(f"⚠️  Failed to stop memory leak: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"⚠️  Error stopping memory leak: {e}")
            return False
    
    def inject_service_crash(self, service_name: str) -> bool:
        """Crash a service by killing its process"""
        print(f"💥 Crashing {service_name} service...")
        
        pid = self.find_service_pid(service_name)
        if not pid:
            print(f"❌ Service {service_name} not found")
            return False
        
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], check=True)
            else:  # Linux/macOS
                os.kill(pid, signal.SIGKILL)
            
            print(f"💀 Service {service_name} crashed (PID: {pid})")
            print(f"   Queue messages will back up until service is restarted")
            return True
            
        except Exception as e:
            print(f"❌ Failed to crash service: {e}")
            return False
    
    def cleanup_all(self):
        """Clean up all active failures"""
        print("🧹 Cleaning up all active failures...")
        
        # Stop all CPU spikes
        for service in ['web-api', 'order-processor', 'notification-service']:
            self.stop_cpu_spike(service)
            self.stop_memory_leak(service)
        
        self.active_failures.clear()
        print("✅ All failures cleaned up")

# Test function
def test_injector():
    """Test the failure injector"""
    injector = FailureInjector()
    
    print("🧪 Testing failure injector...")
    
    # Test finding services
    for service_name in ['web-api', 'order-processor', 'notification-service']:
        pid = injector.find_service_pid(service_name)
        if pid:
            print(f"✅ Found {service_name}: PID {pid}")
        else:
            print(f"❌ {service_name} not running")
    
    print("🧪 Test CPU spike for 10 seconds...")
    success = injector.inject_cpu_spike('web-api', duration=10)
    
    if success:
        print("⏱️  Waiting 15 seconds...")
        time.sleep(15)
        injector.stop_cpu_spike('web-api')
        print("✅ CPU spike test completed")
    else:
        print("❌ CPU spike test failed")
    
    print("🧪 Test memory leak for 10 seconds...")
    success = injector.inject_memory_leak('order-processor', duration=10)
    
    if success:
        print("⏱️  Waiting 15 seconds...")
        time.sleep(15)
        print("✅ Memory leak test completed")
    else:
        print("❌ Memory leak test failed")

if __name__ == "__main__":
    test_injector()