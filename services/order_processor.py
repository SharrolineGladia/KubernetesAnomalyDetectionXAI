from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
import json
import sqlite3
import threading
import psutil
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simple logging function
def log_message(level, message, **kwargs):
    """Simple logging without external dependencies"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data = {
        'timestamp': timestamp,
        'service': 'order-processor',
        'level': level,
        'message': message,
        'pid': os.getpid(),
        **kwargs
    }
    print(json.dumps(log_data))

# Prometheus metrics
ORDERS_PROCESSED = Counter('order_processor_orders_total', 'Total orders processed', ['status'])
PROCESSING_DURATION = Histogram('order_processor_duration_seconds', 'Order processing duration')
QUEUE_SIZE = Gauge('order_processor_queue_size', 'Queue size')
CPU_USAGE = Gauge('order_processor_cpu_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('order_processor_memory_mb', 'Memory usage in MB')

# ROOT CAUSE ANALYSIS METRICS
PROCESSING_RATE = Gauge('order_processor_processing_rate_per_sec', 'Orders processed per second')
RESPONSE_TIME_P95 = Gauge('order_processor_response_time_p95_ms', '95th percentile processing time in ms')
DB_CONNECTIONS = Gauge('order_processor_db_connections_active', 'Active database connections')
REDIS_HEALTH = Gauge('order_processor_redis_connection_health', 'Redis connection health (1=ok, 0=fail)')
THREAD_COUNT = Gauge('order_processor_thread_count', 'Number of active threads')
INTERNAL_QUEUE_DEPTH = Gauge('order_processor_internal_queue_depth', 'Internal processing queue depth')
ERROR_RATE = Gauge('order_processor_error_rate', 'Current error rate')
MEMORY_GROWTH_RATE = Gauge('order_processor_memory_growth_mb_per_min', 'Memory growth rate MB/minute')

app = FastAPI(title="Order Processor Service")

# Global variables
try:
    process = psutil.Process()
except:
    process = None

# Redis simulation (in-memory queue)
message_queue = []
queue_lock = threading.Lock()

# Memory leak simulation
memory_hog = []
leak_active = False

# RCA metrics tracking
processed_orders_timestamps = []
last_memory = 0
last_time = 0

def init_database():
    """Initialize SQLite database"""
    try:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect('data/orders.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                amount REAL,
                status TEXT,
                created_at TIMESTAMP,
                processed_at TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        log_message('INFO', 'Database initialized')
    except Exception as e:
        log_message('ERROR', 'Database init failed', error=str(e))

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
        
        # Simulate queue size
        with queue_lock:
            queue_size = len(message_queue)
        QUEUE_SIZE.set(queue_size)
        
        # RCA METRICS: Calculate processing metrics
        current_time = time.time()
        
        # Processing rate (orders per second)
        with queue_lock:
            recent_orders = [t for t in processed_orders_timestamps if current_time - t < 60]
            processing_rate = len(recent_orders) / 60.0
            PROCESSING_RATE.set(processing_rate)
        
        # Response time P95 simulation
        if processing_rate > 0:
            # Simulate response time based on load
            base_time = 50 + (queue_size * 10)  # Base 50ms + 10ms per queue item
            p95_time = base_time * 1.2  # P95 is ~20% higher than average
            RESPONSE_TIME_P95.set(p95_time)
        else:
            RESPONSE_TIME_P95.set(50)  # Default low latency
        
        # Database connections simulation
        db_connections = min(10, max(1, queue_size // 5 + 1))
        DB_CONNECTIONS.set(db_connections)
        
        # Redis health simulation
        redis_health = 1 if queue_size < 100 else 0  # Fail if overloaded
        REDIS_HEALTH.set(redis_health)
        
        # Internal queue depth (processing backlog)
        internal_queue = max(0, queue_size - 20)  # Backlog beyond normal capacity
        INTERNAL_QUEUE_DEPTH.set(internal_queue)
        
        # Error rate simulation
        error_rate = min(0.1, queue_size / 1000.0)  # Higher error rate with more load
        ERROR_RATE.set(error_rate)
        
        # Memory growth rate (MB/minute)
        global last_memory, last_time
        if last_memory > 0 and last_time > 0:
            time_diff = current_time - last_time
            if time_diff > 0:
                memory_diff = memory_mb - last_memory if process else 0
                growth_rate = (memory_diff / time_diff) * 60  # MB per minute
                MEMORY_GROWTH_RATE.set(growth_rate)
        
        # Store for next calculation
        if process:
            last_memory = memory_mb
            last_time = current_time
        
    except Exception as e:
        log_message('ERROR', 'Failed to update metrics', error=str(e))

def simulate_memory_leak():
    """Memory leak simulation thread"""
    global memory_hog, leak_active
    while True:
        try:
            if leak_active:
                # Allocate 1MB chunks
                chunk = 'x' * (1024 * 1024)
                memory_hog.append(chunk)
                if len(memory_hog) % 10 == 0:  # Log every 10MB
                    log_message('DEBUG', 'Memory leak progress', total_mb=len(memory_hog))
            time.sleep(1)
        except Exception as e:
            log_message('ERROR', 'Memory leak simulation error', error=str(e))
            time.sleep(5)

# Start background threads
threading.Thread(target=simulate_memory_leak, daemon=True).start()

@app.middleware("http")
async def monitoring_middleware(request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        log_message('INFO', 'Request processed',
                   method=request.method,
                   path=request.url.path,
                   status=response.status_code,
                   duration=duration)
        
        update_system_metrics()
        return response
        
    except Exception as e:
        log_message('ERROR', 'Request failed',
                   method=request.method,
                   path=request.url.path,
                   error=str(e))
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "order-processor"}

@app.post("/process-order")
async def process_order(order_data: dict):
    """Process order"""
    start_time = time.time()
    
    try:
        order_id = order_data.get("id", f"order_{int(time.time())}")
        
        # CPU-intensive simulation if requested
        if order_data.get("cpu_intensive", False):
            # Burn CPU for a short time
            result = sum(i * i for i in range(50000))
            log_message('DEBUG', 'CPU intensive processing', result=result)
        
        # Database operation
        try:
            conn = sqlite3.connect('data/orders.db')
            conn.execute('''
                INSERT OR REPLACE INTO orders (id, customer_id, amount, status, created_at, processed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                order_id,
                order_data.get("customer_id", "unknown"),
                order_data.get("total_amount", 0),
                "processed",
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as db_error:
            log_message('ERROR', 'Database error', error=str(db_error))
        
        # Add to message queue (simulate Redis)
        notification_data = {
            "order_id": order_id,
            "customer_id": order_data.get("customer_id"),
            "type": "order_confirmation",
            "timestamp": datetime.now().isoformat()
        }
        
        with queue_lock:
            message_queue.append(notification_data)
        
        duration = time.time() - start_time
        PROCESSING_DURATION.observe(duration)
        ORDERS_PROCESSED.labels(status="success").inc()
        
        log_message('INFO', 'Order processed successfully',
                   order_id=order_id,
                   processing_time=duration)
        
        return {
            "order_id": order_id,
            "status": "processed",
            "processing_time": duration
        }
        
    except Exception as e:
        ORDERS_PROCESSED.labels(status="error").inc()
        log_message('ERROR', 'Order processing failed', error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate-memory-leak")
async def start_memory_leak():
    """Start memory leak simulation"""
    global leak_active
    leak_active = True
    log_message('WARNING', 'Memory leak simulation started')
    return {"status": "memory_leak_started"}

@app.post("/stop-memory-leak")
async def stop_memory_leak():
    """Stop memory leak and clean up"""
    global leak_active, memory_hog
    leak_active = False
    memory_hog.clear()
    log_message('INFO', 'Memory leak simulation stopped')
    return {"status": "memory_leak_stopped"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/queue-status")
async def queue_status():
    """Get queue status"""
    with queue_lock:
        size = len(message_queue)
    return {"queue_size": size, "leak_active": leak_active}

if __name__ == "__main__":
    import uvicorn
    
    # Initialize database
    init_database()
    
    log_message('INFO', 'Starting Order Processor service on port 8002')
    print("🚀 Starting Order Processor Service...")
    print("   URL: http://localhost:8002")
    print("   Health: http://localhost:8002/health")
    print("   Metrics: http://localhost:8002/metrics")
    print("   Queue Status: http://localhost:8002/queue-status")
    print("   Press Ctrl+C to stop")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")
    except KeyboardInterrupt:
        print("\n⏹️  Order Processor Service stopped")
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        input("Press Enter to exit...")
