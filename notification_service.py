import time
import json
import threading
import psutil
import os
import sys
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simple logging function
def log_message(level, message, **kwargs):
    """Simple logging without external dependencies"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data = {
        'timestamp': timestamp,
        'service': 'notification-service',
        'level': level,
        'message': message,
        'pid': os.getpid(),
        **kwargs
    }
    print(json.dumps(log_data))

# Prometheus metrics
NOTIFICATIONS_PROCESSED = Counter('notification_service_processed_total', 'Notifications processed', ['type', 'status'])
PROCESSING_DURATION = Histogram('notification_service_duration_seconds', 'Processing duration')
QUEUE_SIZE = Gauge('notification_service_queue_size', 'Queue size')
WORKER_THREADS = Gauge('notification_service_workers', 'Active worker threads')
CPU_USAGE = Gauge('notification_service_cpu_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('notification_service_memory_mb', 'Memory usage in MB')

# ROOT CAUSE ANALYSIS METRICS
MESSAGE_RATE = Gauge('notification_service_message_rate_per_sec', 'Messages processed per second')
DELIVERY_SUCCESS_RATE = Gauge('notification_service_delivery_success_rate', 'Successful delivery rate (0-1)')
THREAD_COUNT = Gauge('notification_service_thread_count', 'Number of active threads')
EXTERNAL_API_HEALTH = Gauge('notification_service_external_api_health', 'External API health (1=ok, 0=fail)')
QUEUE_DEPTH = Gauge('notification_service_internal_queue_depth', 'Internal queue depth')
ERROR_RATE = Gauge('notification_service_error_rate', 'Current error rate')
RESPONSE_TIME_P95 = Gauge('notification_service_response_time_p95_ms', '95th percentile response time')

# Global variables
try:
    process = psutil.Process()
except:
    process = None

running = True
cpu_spike_active = False

# Simulate message queue (in-memory)
message_queue = []
queue_lock = threading.Lock()

def update_system_metrics():
    """Update system metrics"""
    try:
        if process:
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            CPU_USAGE.set(cpu_percent)
            MEMORY_USAGE.set(memory_mb)
            
            # RCA METRICS: Thread count
            try:
                THREAD_COUNT.set(process.num_threads())
            except:
                THREAD_COUNT.set(0)
        
        with queue_lock:
            queue_size = len(message_queue)
        QUEUE_SIZE.set(queue_size)
        
        # RCA METRICS: Calculate notification metrics
        current_time = time.time()
        
        # Message processing rate
        global processed_messages_timestamps
        if 'processed_messages_timestamps' not in globals():
            processed_messages_timestamps = []
        
        recent_messages = [t for t in processed_messages_timestamps if current_time - t < 60]
        message_rate = len(recent_messages) / 60.0
        MESSAGE_RATE.set(message_rate)
        
        # Delivery success rate simulation
        success_rate = 0.99 if queue_size < 50 else max(0.8, 0.99 - (queue_size / 1000))
        DELIVERY_SUCCESS_RATE.set(success_rate)
        
        # External API health simulation
        api_health = 1 if queue_size < 100 and success_rate > 0.95 else 0
        EXTERNAL_API_HEALTH.set(api_health)
        
        # Queue depth (backlog)
        queue_depth = max(0, queue_size - 10)  # Backlog beyond normal capacity
        QUEUE_DEPTH.set(queue_depth)
        
        # Error rate
        error_rate = 1.0 - success_rate
        ERROR_RATE.set(error_rate)
        
        # Response time P95 simulation
        base_time = 30 + (queue_size * 2)  # Base 30ms + 2ms per queue item
        p95_time = base_time * 1.3  # P95 is ~30% higher
        RESPONSE_TIME_P95.set(p95_time)
        
    except Exception as e:
        log_message('ERROR', 'Failed to update metrics', error=str(e))

def simulate_cpu_spike():
    """CPU spike simulation thread"""
    global cpu_spike_active
    while running:
        try:
            if cpu_spike_active:
                # CPU-intensive operations
                for i in range(100000):
                    _ = i * i * i
            else:
                time.sleep(0.1)
        except Exception as e:
            log_message('ERROR', 'CPU spike simulation error', error=str(e))
            time.sleep(1)

def process_notification(notification_data):
    """Process a single notification"""
    start_time = time.time()
    
    try:
        notification = notification_data
        if isinstance(notification_data, str):
            notification = json.loads(notification_data)
        
        notification_type = notification.get('type', 'unknown')
        
        # Simulate processing time
        time.sleep(0.2)
        
        # Simulate external API call for order confirmations
        if notification_type == 'order_confirmation':
            time.sleep(0.3)
        
        duration = time.time() - start_time
        PROCESSING_DURATION.observe(duration)
        NOTIFICATIONS_PROCESSED.labels(type=notification_type, status='success').inc()
        
        log_message('INFO', 'Notification processed',
                   notification_id=notification.get('order_id'),
                   type=notification_type,
                   duration=duration)
        
        return True
        
    except Exception as e:
        NOTIFICATIONS_PROCESSED.labels(type='unknown', status='error').inc()
        log_message('ERROR', 'Notification processing failed', error=str(e))
        return False

def notification_worker(worker_id):
    """Worker thread to process notifications"""
    log_message('INFO', 'Worker started', worker_id=worker_id)
    WORKER_THREADS.inc()
    
    try:
        while running:
            try:
                notification_data = None
                
                # Get message from queue
                with queue_lock:
                    if message_queue:
                        notification_data = message_queue.pop(0)
                
                if notification_data:
                    process_notification(notification_data)
                else:
                    # Simulate getting messages from order processor
                    try:
                        import requests
                        response = requests.get('http://localhost:8002/queue-status', timeout=1)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('queue_size', 0) > 0:
                                # Simulate processing a message
                                fake_notification = {
                                    'order_id': f'simulated_{int(time.time())}',
                                    'type': 'order_confirmation',
                                    'timestamp': datetime.now().isoformat()
                                }
                                process_notification(fake_notification)
                    except:
                        pass  # Ignore connection errors
                    
                    time.sleep(1)  # Wait if no messages
                
                update_system_metrics()
                
            except Exception as e:
                log_message('ERROR', 'Worker error', worker_id=worker_id, error=str(e))
                time.sleep(1)
                
    finally:
        WORKER_THREADS.dec()
        log_message('INFO', 'Worker stopped', worker_id=worker_id)

def main():
    """Main function"""
    global running
    
    log_message('INFO', 'Starting Notification Service')
    print("🚀 Starting Notification Service...")
    print("   Metrics: http://localhost:8003/metrics")
    print("   Press Ctrl+C to stop")
    
    # Start Prometheus metrics server
    try:
        start_http_server(8003)
        log_message('INFO', 'Metrics server started on port 8003')
        print("   ✅ Metrics server running on port 8003")
    except Exception as e:
        log_message('ERROR', 'Failed to start metrics server', error=str(e))
        print(f"   ❌ Failed to start metrics server: {e}")
    
    # Start CPU spike simulation thread
    cpu_thread = threading.Thread(target=simulate_cpu_spike, daemon=True)
    cpu_thread.start()
    
    # Start worker threads
    num_workers = 2
    workers = []
    
    for i in range(num_workers):
        worker = threading.Thread(target=notification_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)
    
    try:
        print("   ✅ Service started with 2 workers")
        print("   ✅ CPU spike simulation ready")
        print("   ✅ Processing queue messages")
        
        # Keep main thread alive
        while True:
            time.sleep(10)
            update_system_metrics()
            
            # Log status periodically
            with queue_lock:
                queue_size = len(message_queue)
            
            if queue_size > 0:
                log_message('INFO', 'Queue status', queue_size=queue_size)
                
    except KeyboardInterrupt:
        running = False
        log_message('INFO', 'Shutting down notification service')
        print("\n⏹️  Notification Service stopped")
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        log_message('ERROR', 'Service startup failed', error=str(e))
        input("Press Enter to exit...")