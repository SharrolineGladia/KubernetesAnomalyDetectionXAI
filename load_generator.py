"""
Load Generator - Generates realistic traffic to microservices
Creates consistent load for data collection
"""

import asyncio
import aiohttp
import json
import random
import time
from datetime import datetime

class LoadGenerator:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.running = False
        
        # Statistics
        self.stats = {
            'requests_sent': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'total_response_time': 0.0
        }
        
        print("🚛 Load Generator initialized")
        print(f"   Target: {self.base_url}")
    
    def generate_realistic_order(self) -> dict:
        """Generate realistic order data"""
        order_id = f"order_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Realistic customer data
        customers = ["alice_smith", "bob_jones", "charlie_brown", "diana_prince", "eve_wilson"]
        items = ["laptop", "phone", "tablet", "headphones", "camera", "watch", "keyboard"]
        
        return {
            "id": order_id,
            "customer_id": random.choice(customers),
            "items": [
                {
                    "name": random.choice(items),
                    "quantity": random.randint(1, 3),
                    "price": round(random.uniform(50, 500), 2)
                }
                for _ in range(random.randint(1, 4))
            ],
            "total_amount": round(random.uniform(100, 2000), 2),
            "priority": random.choice(["normal", "high", "urgent"]),
            "cpu_intensive": random.random() < 0.05,  # 5% CPU-intensive orders
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def send_single_request(self, session: aiohttp.ClientSession) -> bool:
        """Send one request and track stats"""
        start_time = time.time()
        
        try:
            order_data = self.generate_realistic_order()
            
            async with session.post(
                f"{self.base_url}/orders",
                json=order_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                # Read response
                result = await response.json()
                duration = time.time() - start_time
                
                # Update stats
                self.stats['requests_sent'] += 1
                self.stats['total_response_time'] += duration
                
                if response.status == 200:
                    self.stats['requests_successful'] += 1
                    return True
                else:
                    self.stats['requests_failed'] += 1
                    print(f"⚠️  Request failed: HTTP {response.status}")
                    return False
                
        except asyncio.TimeoutError:
            self.stats['requests_failed'] += 1
            print(f"⚠️  Request timeout after {time.time() - start_time:.1f}s")
            return False
            
        except Exception as e:
            self.stats['requests_failed'] += 1
            print(f"⚠️  Request error: {e}")
            return False
    
    def print_stats(self):
        """Print current statistics"""
        total = self.stats['requests_sent']
        success = self.stats['requests_successful']
        failed = self.stats['requests_failed']
        
        if total > 0:
            success_rate = (success / total) * 100
            avg_response_time = self.stats['total_response_time'] / total
            
            print(f"📊 Load Stats: {total} requests, "
                  f"{success_rate:.1f}% success, "
                  f"{avg_response_time:.2f}s avg response")
        else:
            print("📊 Load Stats: No requests sent yet")
    
    async def generate_normal_traffic(self, duration_minutes: int, rps: float = 2.0):
        """Generate steady traffic at specified rate"""
        self.running = True
        end_time = time.time() + (duration_minutes * 60)
        
        print(f"🚛 Starting normal traffic generation:")
        print(f"   Rate: {rps} requests/second")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Expected total: {int(duration_minutes * 60 * rps)} requests")
        
        async with aiohttp.ClientSession() as session:
            last_stats_time = time.time()
            
            while time.time() < end_time and self.running:
                batch_start = time.time()
                
                # Send requests in batches to maintain rate
                batch_size = max(1, int(rps))
                tasks = []
                
                for _ in range(batch_size):
                    if time.time() >= end_time or not self.running:
                        break
                    
                    task = self.send_single_request(session)
                    tasks.append(task)
                
                # Wait for batch to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Print stats every 2 minutes
                if time.time() - last_stats_time > 120:
                    self.print_stats()
                    last_stats_time = time.time()
                
                # Sleep to maintain rate
                batch_duration = time.time() - batch_start
                sleep_time = max(0, (1.0 / rps) - batch_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        self.running = False
        
        # Final stats
        print(f"✅ Traffic generation completed")
        self.print_stats()
        
        return self.stats
    
    async def generate_burst_traffic(self, duration_minutes: int = 5, peak_rps: int = 10):
        """Generate burst traffic with higher request rate"""
        print(f"🚀 Starting burst traffic generation:")
        print(f"   Peak rate: {peak_rps} requests/second")
        print(f"   Duration: {duration_minutes} minutes")
        
        return await self.generate_normal_traffic(duration_minutes, peak_rps)
    
    async def generate_mixed_traffic(self, duration_minutes: int = 30):
        """Generate mixed traffic with varying patterns"""
        print(f"🌊 Starting mixed traffic generation:")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Pattern: Variable load (1-8 RPS)")
        
        self.running = True
        end_time = time.time() + (duration_minutes * 60)
        
        # Traffic patterns: (rate, duration_minutes)
        patterns = [
            (1.0, 3),   # Low
            (2.0, 5),   # Normal
            (4.0, 2),   # Medium
            (8.0, 1),   # High burst
            (1.5, 4),   # Cool down
        ]
        
        async with aiohttp.ClientSession() as session:
            while time.time() < end_time and self.running:
                # Pick random pattern
                rate, pattern_duration = random.choice(patterns)
                
                # Don't exceed remaining time
                remaining_time = (end_time - time.time()) / 60
                actual_duration = min(pattern_duration, remaining_time)
                
                if actual_duration <= 0:
                    break
                
                print(f"🔄 Pattern: {rate} RPS for {actual_duration:.1f} minutes")
                
                # Generate traffic for this pattern
                pattern_end = time.time() + (actual_duration * 60)
                last_stats_time = time.time()
                
                while time.time() < pattern_end and time.time() < end_time and self.running:
                    batch_start = time.time()
                    
                    # Send requests
                    batch_size = max(1, int(rate))
                    tasks = []
                    
                    for _ in range(batch_size):
                        if time.time() >= pattern_end or not self.running:
                            break
                        task = self.send_single_request(session)
                        tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Stats update
                    if time.time() - last_stats_time > 60:  # Every minute
                        self.print_stats()
                        last_stats_time = time.time()
                    
                    # Rate limiting
                    batch_duration = time.time() - batch_start
                    sleep_time = max(0, (1.0 / rate) - batch_duration)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
        
        self.running = False
        print(f"✅ Mixed traffic generation completed")
        self.print_stats()
        
        return self.stats
    
    def stop(self):
        """Stop traffic generation"""
        self.running = False
        print("⏹️  Load generation stopped")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'requests_sent': 0,
            'requests_successful': 0, 
            'requests_failed': 0,
            'total_response_time': 0.0
        }
        print("📊 Stats reset")

# Test function
async def test_load_generator():
    """Test the load generator"""
    generator = LoadGenerator()
    
    print("🧪 Testing load generator for 1 minute...")
    
    # Test health endpoint first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{generator.base_url}/health", timeout=5) as response:
                if response.status == 200:
                    print("✅ Web API is accessible")
                else:
                    print(f"⚠️  Web API returned {response.status}")
                    
    except Exception as e:
        print(f"❌ Web API not accessible: {e}")
        print("   Make sure to start: python web_api.py")
        return
    
    # Generate some test traffic
    await generator.generate_normal_traffic(duration_minutes=1, rps=3.0)
    
    if generator.stats['requests_successful'] > 0:
        print("✅ Load generator test completed successfully!")
    else:
        print("❌ Load generator test failed - no successful requests")

if __name__ == "__main__":
    asyncio.run(test_load_generator())