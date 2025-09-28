"""
FIXED Data Collector - Actually collects and saves data
This fixes the Prometheus query issues
"""

import asyncio
import aiohttp
import pandas as pd
import sqlite3
import json
import time
import os
from datetime import datetime
from typing import Dict, Optional

class FixedDataCollector:
    def __init__(self):
        self.prometheus_url = "http://localhost:9090"
        self.collection_interval = 15  # seconds
        self.running = False
        self.data_points = []
        
        # Create database
        self.db_path = os.path.join('data', 'metrics.db')
        self.init_database()
        
        # Current experiment context
        self.current_anomaly = None
        self.experiment_name = "unknown"
        
        # FIXED: Direct metric names that actually exist
        self.metrics = {
            # Web API - Basic metrics
            'web_api_cpu': 'web_api_cpu_percent',
            'web_api_memory': 'web_api_memory_mb', 
            'web_api_requests': 'web_api_active_requests',
            'web_api_errors': 'web_api_error_rate',
            
            # Web API - ROOT CAUSE ANALYSIS metrics
            'web_api_response_time_p95': 'web_api_response_time_p95_ms',
            'web_api_requests_per_second': 'web_api_requests_per_second',
            'web_api_db_connections': 'web_api_db_connections_active',
            'web_api_redis_health': 'web_api_redis_connection_health',
            'web_api_thread_count': 'web_api_thread_count',
            'web_api_queue_depth': 'web_api_internal_queue_depth',
            
            # Order Processor - Basic metrics
            'processor_cpu': 'order_processor_cpu_percent',
            'processor_memory': 'order_processor_memory_mb',
            'processor_queue': 'order_processor_queue_size',
            
            # Order Processor - ROOT CAUSE ANALYSIS metrics
            'processor_processing_rate': 'order_processor_processing_rate_per_sec',
            'processor_response_time_p95': 'order_processor_response_time_p95_ms',
            'processor_db_connections': 'order_processor_db_connections_active',
            'processor_redis_health': 'order_processor_redis_connection_health',
            'processor_thread_count': 'order_processor_thread_count',
            'processor_queue_depth': 'order_processor_internal_queue_depth',
            'processor_error_rate': 'order_processor_error_rate',
            'processor_memory_growth': 'order_processor_memory_growth_mb_per_min',
            
            # Notification Service - Basic metrics
            'notification_cpu': 'notification_service_cpu_percent',
            'notification_memory': 'notification_service_memory_mb',
            'notification_queue': 'notification_service_queue_size',
            
            # Notification Service - ROOT CAUSE ANALYSIS metrics
            'notification_message_rate': 'notification_service_message_rate_per_sec',
            'notification_delivery_success': 'notification_service_delivery_success_rate',
            'notification_thread_count': 'notification_service_thread_count',
            'notification_api_health': 'notification_service_external_api_health',
            'notification_queue_depth': 'notification_service_internal_queue_depth',
            'notification_error_rate': 'notification_service_error_rate',
            'notification_response_time_p95': 'notification_service_response_time_p95_ms',
        }
        
        print(f"🔧 FIXED Data Collector initialized")
        print(f"   Database: {self.db_path}")
        print(f"   Collection interval: {self.collection_interval} seconds")
        print(f"   Metrics to collect: {len(self.metrics)}")
    
    def init_database(self):
        """Initialize SQLite database"""
        os.makedirs('data', exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('DROP TABLE IF EXISTS metrics')  # Fresh start
        conn.execute('''
            CREATE TABLE metrics (
                timestamp TEXT,
                service_name TEXT,
                metric_name TEXT,
                metric_value REAL,
                anomaly_label INTEGER DEFAULT 0,
                anomaly_type TEXT DEFAULT 'normal',
                experiment_name TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print(f"✅ Database reset and ready")
    
    async def query_prometheus(self, query: str) -> Optional[float]:
        """Get metric value from Prometheus - FIXED VERSION"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.prometheus_url}/api/v1/query"
                params = {'query': query}
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if (data.get('status') == 'success' and 
                            data.get('data', {}).get('result') and 
                            len(data['data']['result']) > 0):
                            
                            # Get the first result's value
                            result = data['data']['result'][0]
                            if 'value' in result and len(result['value']) >= 2:
                                value = float(result['value'][1])
                                return value
                    
                    # Debug failed queries
                    print(f"   ⚠️  Query failed: {query} -> {response.status}")
                    return None
                    
        except Exception as e:
            print(f"   ❌ Query error: {query} -> {e}")
            return None
    
    async def test_all_metrics(self):
        """Test all metrics to see what works"""
        print("🧪 Testing all metric queries...")
        
        working_metrics = {}
        
        for metric_name, prometheus_query in self.metrics.items():
            value = await self.query_prometheus(prometheus_query)
            if value is not None:
                print(f"   ✅ {metric_name}: {value}")
                working_metrics[metric_name] = prometheus_query
            else:
                print(f"   ❌ {metric_name}: No data")
        
        print(f"\n📊 Working metrics: {len(working_metrics)}/{len(self.metrics)}")
        
        # Update metrics to only use working ones
        if working_metrics:
            self.metrics = working_metrics
            print("   Updated to use only working metrics")
        
        return len(working_metrics) > 0
    
    async def collect_single_snapshot(self) -> Dict:
        """Collect one snapshot of all metrics"""
        timestamp = datetime.utcnow().isoformat()
        snapshot = {'timestamp': timestamp}
        
        successful_collections = 0
        
        # Collect all metrics
        for metric_name, prometheus_query in self.metrics.items():
            value = await self.query_prometheus(prometheus_query)
            snapshot[metric_name] = value
            if value is not None:
                successful_collections += 1
        
        # Add anomaly context
        if self.current_anomaly:
            snapshot['anomaly_label'] = 1
            snapshot['anomaly_type'] = self.current_anomaly['type']
        else:
            snapshot['anomaly_label'] = 0
            snapshot['anomaly_type'] = 'normal'
        
        snapshot['experiment_name'] = self.experiment_name
        snapshot['successful_collections'] = successful_collections
        
        return snapshot
    
    def save_snapshot(self, snapshot: Dict):
        """Save snapshot to database - FIXED VERSION"""
        conn = sqlite3.connect(self.db_path)
        
        timestamp = snapshot['timestamp']
        anomaly_label = snapshot['anomaly_label'] 
        anomaly_type = snapshot['anomaly_type']
        experiment_name = snapshot['experiment_name']
        
        rows_inserted = 0
        
        # Save each metric as separate row
        for key, value in snapshot.items():
            if key in ['timestamp', 'anomaly_label', 'anomaly_type', 'experiment_name', 'successful_collections']:
                continue
            
            if value is not None:
                # Determine service name
                if 'web_api' in key:
                    service_name = 'web-api'
                elif 'processor' in key:
                    service_name = 'order-processor'  
                elif 'notification' in key:
                    service_name = 'notification-service'
                else:
                    service_name = 'system'
                
                try:
                    conn.execute('''
                        INSERT INTO metrics 
                        (timestamp, service_name, metric_name, metric_value, 
                         anomaly_label, anomaly_type, experiment_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (timestamp, service_name, key, value, 
                          anomaly_label, anomaly_type, experiment_name))
                    rows_inserted += 1
                except Exception as e:
                    print(f"   ❌ Insert error: {e}")
        
        conn.commit()
        conn.close()
        
        return rows_inserted
    
    async def start_collection(self, duration_minutes: int, experiment_name: str):
        """Start collecting data for specified duration"""
        self.running = True
        self.experiment_name = experiment_name
        
        print(f"📊 Starting FIXED data collection: {experiment_name}")
        print(f"   Duration: {duration_minutes} minutes")
        
        # First, test which metrics work
        has_working_metrics = await self.test_all_metrics()
        if not has_working_metrics:
            print("❌ No working metrics found! Check services and Prometheus.")
            return 0
        
        end_time = time.time() + (duration_minutes * 60)
        collection_count = 0
        total_rows_saved = 0
        
        print(f"   Starting collection loop...")
        
        while time.time() < end_time and self.running:
            try:
                # Collect metrics snapshot
                snapshot = await self.collect_single_snapshot()
                
                # Save to database
                rows_saved = self.save_snapshot(snapshot)
                total_rows_saved += rows_saved
                
                # Keep in memory for export
                self.data_points.append(snapshot)
                
                collection_count += 1
                
                # Progress update every 5 minutes
                if collection_count % 20 == 0:
                    elapsed = (time.time() - (end_time - duration_minutes * 60)) / 60
                    remaining = duration_minutes - elapsed
                    successful = snapshot.get('successful_collections', 0)
                    print(f"   📈 Snapshot {collection_count}: {successful}/{len(self.metrics)} metrics, {rows_saved} rows saved")
                    print(f"      {elapsed:.1f}min elapsed, {remaining:.1f}min remaining, {total_rows_saved} total rows")
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"   ⚠️  Collection error: {e}")
                await asyncio.sleep(self.collection_interval)
        
        self.running = False
        print(f"✅ Data collection completed: {collection_count} snapshots, {total_rows_saved} rows saved")
        
        return collection_count
    
    def start_anomaly_period(self, anomaly_type: str, affected_service: str):
        """Mark start of anomaly period"""
        self.current_anomaly = {
            'type': anomaly_type,
            'service': affected_service,
            'start_time': datetime.utcnow()
        }
        
        print(f"🚨 Anomaly period started: {anomaly_type} in {affected_service}")
    
    def end_anomaly_period(self):
        """Mark end of anomaly period"""
        if self.current_anomaly:
            duration = datetime.utcnow() - self.current_anomaly['start_time']
            print(f"✅ Anomaly period ended: {self.current_anomaly['type']} (duration: {duration.total_seconds():.0f}s)")
            self.current_anomaly = None
        else:
            print("⚠️  No active anomaly period to end")
    
    def get_current_anomaly_state(self):
        """Get current anomaly state - used by experiments"""
        if self.current_anomaly:
            return True, self.current_anomaly['type']
        else:
            return False, 'normal'
    
    def export_to_csv(self, filename: str) -> str:
        """Export collected data to CSV - FIXED VERSION"""
        try:
            # Query database for this experiment
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM metrics WHERE experiment_name = '{self.experiment_name}' ORDER BY timestamp"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) == 0:
                print(f"❌ No data found for experiment: {self.experiment_name}")
                # Try to get any data
                conn = sqlite3.connect(self.db_path)
                df_all = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp", conn)
                conn.close()
                
                if len(df_all) > 0:
                    print(f"   Found {len(df_all)} rows in total, using all data")
                    df = df_all
                else:
                    print(f"   No data in database at all!")
                    return None
            
            # Save to CSV
            full_path = os.path.abspath(filename)
            df.to_csv(full_path, index=False)
            
            # Print summary
            total_points = len(df)
            
            if total_points > 0:
                anomaly_points = df['anomaly_label'].sum() if 'anomaly_label' in df.columns else 0
                normal_points = total_points - anomaly_points
                
                print(f"✅ Data exported: {filename}")
                print(f"   Total points: {total_points}")
                print(f"   Normal: {normal_points} ({(normal_points/total_points*100):.1f}%)")
                print(f"   Anomalies: {anomaly_points} ({(anomaly_points/total_points*100):.1f}%)")
                
                # Show sample data
                print(f"   Services: {df['service_name'].unique().tolist()}")
                print(f"   Metrics: {df['metric_name'].nunique()} different metrics")
                print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return full_path
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        print("⏹️  Data collection stopped")

# Test function
async def test_fixed_collector():
    """Test the fixed data collector"""
    collector = FixedDataCollector()
    
    print("🧪 Testing FIXED data collector for 2 minutes...")
    
    # Test collection
    count = await collector.start_collection(duration_minutes=2, experiment_name="test_fixed")
    
    # Export test data
    csv_file = collector.export_to_csv("data/test_fixed_collection.csv")
    
    if csv_file:
        print(f"✅ Test completed successfully!")
        print(f"📁 Test data saved to: {csv_file}")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    # Run test
    print("🚀 Testing FIXED Data Collector")
    asyncio.run(test_fixed_collector())