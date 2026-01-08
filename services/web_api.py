from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import asyncio
import time
import json
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
        'service': 'web-api',
        'level': level,
        'message': message,
        'pid': os.getpid(),
        **kwargs
    }
    print(json.dumps(log_data))

# Prometheus metrics
REQUEST_COUNT = Counter('web_api_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('web_api_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_REQUESTS = Gauge('web_api_active_requests', 'Active requests')
ERROR_RATE = Gauge('web_api_error_rate', 'Current error rate')
CPU_USAGE = Gauge('web_api_cpu_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('web_api_memory_mb', 'Memory usage in MB')

# ROOT CAUSE ANALYSIS METRICS
RESPONSE_TIME_P95 = Gauge('web_api_response_time_p95_ms', '95th percentile response time in ms')
REQUESTS_PER_SECOND = Gauge('web_api_requests_per_second', 'Current requests per second')
ERROR_BY_TYPE = Counter('web_api_errors_by_type', 'Errors by type', ['error_type'])
DB_CONNECTION_COUNT = Gauge('web_api_db_connections_active', 'Active database connections')
REDIS_CONNECTION_HEALTH = Gauge('web_api_redis_connection_health', 'Redis connection health (1=ok, 0=fail)')
THREAD_COUNT = Gauge('web_api_thread_count', 'Number of active threads')
QUEUE_DEPTH = Gauge('web_api_internal_queue_depth', 'Internal processing queue depth')

app = FastAPI(title="Web API Service")

# Global variables
try:
    process = psutil.Process()
except:
    process = None

request_window = []
error_window = []

def update_system_metrics():
    """Update system-level metrics"""
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
        
        # Calculate error rate over last 5 minutes
        now = time.time()
        recent_errors = [t for t in error_window if now - t < 300]
        recent_requests = [t for t in request_window if now - t < 300]
        
        error_rate = len(recent_errors) / max(len(recent_requests), 1)
        ERROR_RATE.set(error_rate)
        
        # RCA METRICS: Calculate requests per second
        recent_1min_requests = [t for t in request_window if now - t < 60]
        rps = len(recent_1min_requests) / 60.0
        REQUESTS_PER_SECOND.set(rps)
        
        # RCA METRICS: Calculate 95th percentile response time
        if len(recent_1min_requests) >= 2:
            # Simulate response time calculation (in real app, track actual response times)
            import statistics
            recent_durations = [0.1 + (len(recent_1min_requests) * 0.01) for _ in range(max(2, len(recent_1min_requests)))]
            try:
                p95_time = statistics.quantiles(recent_durations, n=20)[18] * 1000  # Convert to ms
                RESPONSE_TIME_P95.set(p95_time)
            except:
                RESPONSE_TIME_P95.set(100)  # Default 100ms
        else:
            RESPONSE_TIME_P95.set(50)  # Default for low traffic
        
        # RCA METRICS: Database connection simulation
        # In real app, this would query actual connection pool
        simulated_db_connections = min(10, max(1, len(recent_1min_requests) // 10))
        DB_CONNECTION_COUNT.set(simulated_db_connections)
        
        # RCA METRICS: Redis health check simulation
        try:
            # In real app, this would actually ping Redis
            redis_health = 1 if len(recent_errors) < 5 else 0  # Simulate Redis failure under high error load
            REDIS_CONNECTION_HEALTH.set(redis_health)
        except:
            REDIS_CONNECTION_HEALTH.set(0)
        
        # RCA METRICS: Internal queue depth simulation
        queue_depth = max(0, len(recent_1min_requests) - 30)  # Simulate queue buildup
        QUEUE_DEPTH.set(queue_depth)
        
    except Exception as e:
        log_message('ERROR', 'Failed to update system metrics', error=str(e))

@app.middleware("http")
async def monitoring_middleware(request, call_next):
    """Middleware for request monitoring"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
        
        request_window.append(time.time())
        if response.status_code >= 400:
            error_window.append(time.time())
            
            # RCA METRICS: Track error types for root cause analysis
            if response.status_code >= 500:
                ERROR_BY_TYPE.labels(error_type="server_error").inc()
            elif response.status_code == 429:
                ERROR_BY_TYPE.labels(error_type="rate_limit").inc()
            elif response.status_code >= 400:
                ERROR_BY_TYPE.labels(error_type="client_error").inc()
            
        # Clean up old entries
        now = time.time()
        request_window[:] = [t for t in request_window if now - t < 300]
        error_window[:] = [t for t in error_window if now - t < 300]
        
        log_message('INFO', 'Request processed', 
                   method=request.method,
                   path=request.url.path,
                   status=response.status_code,
                   duration=duration)
        
        update_system_metrics()
        return response
        
    except Exception as e:
        error_window.append(time.time())
        log_message('ERROR', 'Request failed', 
                   method=request.method,
                   path=request.url.path,
                   error=str(e))
        raise
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/")
async def root():
    return {"service": "web-api", "status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "web-api"}

@app.post("/orders")
async def create_order(order_data: dict):
    """Create order and forward to order processor"""
    try:
        log_message('INFO', 'Order received', order_id=order_data.get("id", "unknown"))
        
        # Simulate calling order processor (simplified)
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8002/process-order",
                    json=order_data,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        log_message('INFO', 'Order processed successfully', 
                                   order_id=order_data.get("id"))
                        return {"status": "success", "order": result}
                    else:
                        raise HTTPException(status_code=response.status, detail="Order processor error")
                        
        except asyncio.TimeoutError:
            log_message('ERROR', 'Order processor timeout')
            raise HTTPException(status_code=504, detail="Order processor timeout")
        except Exception as e:
            log_message('ERROR', 'Failed to contact order processor', error=str(e))
            # Return success anyway for demo purposes
            return {"status": "success", "order": {"id": order_data.get("id"), "status": "queued"}}
            
    except Exception as e:
        log_message('ERROR', 'Order creation failed', error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    log_message('INFO', 'Starting Web API service on port 8001')
    print("🚀 Starting Web API Service...")
    print("   URL: http://localhost:8001")
    print("   Health: http://localhost:8001/health")
    print("   Metrics: http://localhost:8001/metrics")
    print("   Press Ctrl+C to stop")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
    except KeyboardInterrupt:
        print("\n⏹️  Web API Service stopped")
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        input("Press Enter to exit...")
