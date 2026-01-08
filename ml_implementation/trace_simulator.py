#!/usr/bin/env python3
"""
Trace Simulator for Causal Root Cause Analysis
Generates realistic distributed trace data for microservice interactions
"""

import random
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

class TraceSimulator:
    """
    Simulates distributed traces for microservice interactions
    """
    
    def __init__(self):
        """Initialize the trace simulator"""
        self.services = {
            'web_api': {'port': 8001, 'base_latency': 50},
            'order_processor': {'port': 8002, 'base_latency': 120},
            'notification_service': {'port': 8003, 'base_latency': 80}
        }
        
        # Service call patterns (who calls who)
        self.service_interactions = {
            'web_api': ['order_processor', 'notification_service'],
            'order_processor': ['notification_service'],
            'notification_service': []
        }
        
        print("✅ Trace simulator initialized")
    
    def generate_trace_id(self) -> str:
        """Generate a unique trace ID"""
        return str(uuid.uuid4())[:16]
    
    def generate_span_id(self) -> str:
        """Generate a unique span ID"""
        return str(uuid.uuid4())[:8]
    
    def simulate_normal_trace(self, trace_id: str = None, start_time: datetime = None) -> Dict:
        """
        Generate a normal trace showing healthy service interactions
        web_api -> order_processor -> notification_service
        """
        if trace_id is None:
            trace_id = self.generate_trace_id()
        if start_time is None:
            start_time = datetime.now()
        
        spans = []
        current_time = start_time
        
        # Root span: web_api request
        web_api_duration = self.services['web_api']['base_latency'] + random.randint(-10, 15)
        web_api_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': None,
            'service_name': 'web_api',
            'operation_name': 'process_request',
            'start_time': current_time.isoformat(),
            'duration_ms': web_api_duration,
            'end_time': (current_time + timedelta(milliseconds=web_api_duration)).isoformat(),
            'status': 'success',
            'tags': {
                'http.method': 'POST',
                'http.url': '/api/orders',
                'http.status_code': 200
            }
        }
        spans.append(web_api_span)
        
        # Child span: order_processor call
        order_start = current_time + timedelta(milliseconds=5)
        order_duration = self.services['order_processor']['base_latency'] + random.randint(-15, 25)
        order_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': web_api_span['span_id'],
            'service_name': 'order_processor',
            'operation_name': 'process_order',
            'start_time': order_start.isoformat(),
            'duration_ms': order_duration,
            'end_time': (order_start + timedelta(milliseconds=order_duration)).isoformat(),
            'status': 'success',
            'tags': {
                'order.id': f'order-{random.randint(1000, 9999)}',
                'order.amount': f'{random.randint(10, 500)}.00'
            }
        }
        spans.append(order_span)
        
        # Child span: notification_service call
        notification_start = order_start + timedelta(milliseconds=order_duration - 20)
        notification_duration = self.services['notification_service']['base_latency'] + random.randint(-10, 20)
        notification_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': order_span['span_id'],
            'service_name': 'notification_service',
            'operation_name': 'send_notification',
            'start_time': notification_start.isoformat(),
            'duration_ms': notification_duration,
            'end_time': (notification_start + timedelta(milliseconds=notification_duration)).isoformat(),
            'status': 'success',
            'tags': {
                'notification.type': 'order_confirmation',
                'notification.channel': random.choice(['email', 'sms', 'push'])
            }
        }
        spans.append(notification_span)
        
        return {
            'trace_id': trace_id,
            'spans': spans,
            'total_duration_ms': web_api_duration,
            'service_count': 3,
            'span_count': len(spans),
            'status': 'normal'
        }
    
    def simulate_cpu_spike_trace(self, trace_id: str = None, start_time: datetime = None, 
                                affected_service: str = 'order_processor') -> Dict:
        """
        Generate a trace showing CPU spike impact on service latencies
        """
        if trace_id is None:
            trace_id = self.generate_trace_id()
        if start_time is None:
            start_time = datetime.now()
        
        spans = []
        current_time = start_time
        
        # Latency multipliers for CPU spike
        latency_multipliers = {
            'web_api': 1.2 if affected_service != 'web_api' else 3.5,
            'order_processor': 1.1 if affected_service != 'order_processor' else 4.2,
            'notification_service': 1.0 if affected_service != 'notification_service' else 2.8
        }
        
        # Root span: web_api (potentially affected by downstream issues)
        web_api_duration = int(self.services['web_api']['base_latency'] * latency_multipliers['web_api']) + random.randint(10, 40)
        web_api_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': None,
            'service_name': 'web_api',
            'operation_name': 'process_request',
            'start_time': current_time.isoformat(),
            'duration_ms': web_api_duration,
            'end_time': (current_time + timedelta(milliseconds=web_api_duration)).isoformat(),
            'status': 'success' if affected_service != 'web_api' else 'degraded',
            'tags': {
                'http.method': 'POST',
                'http.url': '/api/orders',
                'http.status_code': 200 if affected_service != 'web_api' else 503
            }
        }
        spans.append(web_api_span)
        
        # Order processor span (potentially the source of CPU spike)
        order_start = current_time + timedelta(milliseconds=8)
        order_duration = int(self.services['order_processor']['base_latency'] * latency_multipliers['order_processor']) + random.randint(20, 80)
        order_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': web_api_span['span_id'],
            'service_name': 'order_processor',
            'operation_name': 'process_order',
            'start_time': order_start.isoformat(),
            'duration_ms': order_duration,
            'end_time': (order_start + timedelta(milliseconds=order_duration)).isoformat(),
            'status': 'success' if affected_service != 'order_processor' else 'degraded',
            'tags': {
                'order.id': f'order-{random.randint(1000, 9999)}',
                'order.amount': f'{random.randint(10, 500)}.00',
                'cpu.usage_percent': 85.5 if affected_service == 'order_processor' else 23.1
            }
        }
        spans.append(order_span)
        
        # Notification service span (may be affected by upstream issues)
        notification_start = order_start + timedelta(milliseconds=order_duration - 30)
        notification_duration = int(self.services['notification_service']['base_latency'] * latency_multipliers['notification_service']) + random.randint(5, 35)
        notification_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': order_span['span_id'],
            'service_name': 'notification_service',
            'operation_name': 'send_notification',
            'start_time': notification_start.isoformat(),
            'duration_ms': notification_duration,
            'end_time': (notification_start + timedelta(milliseconds=notification_duration)).isoformat(),
            'status': 'success' if affected_service != 'notification_service' else 'degraded',
            'tags': {
                'notification.type': 'order_confirmation',
                'notification.channel': random.choice(['email', 'sms', 'push']),
                'cpu.usage_percent': 78.2 if affected_service == 'notification_service' else 19.8
            }
        }
        spans.append(notification_span)
        
        return {
            'trace_id': trace_id,
            'spans': spans,
            'total_duration_ms': web_api_duration,
            'service_count': 3,
            'span_count': len(spans),
            'status': 'cpu_spike',
            'affected_service': affected_service,
            'anomaly_indicators': {
                'high_latency_services': [s['service_name'] for s in spans if s['duration_ms'] > 200],
                'degraded_services': [s['service_name'] for s in spans if s['status'] == 'degraded'],
                'max_cpu_usage': max([float(s['tags'].get('cpu.usage_percent', 0)) for s in spans])
            }
        }
    
    def simulate_memory_leak_trace(self, trace_id: str = None, start_time: datetime = None,
                                  affected_service: str = 'order_processor') -> Dict:
        """
        Generate a trace showing memory leak impact (gradual performance degradation)
        """
        if trace_id is None:
            trace_id = self.generate_trace_id()
        if start_time is None:
            start_time = datetime.now()
        
        spans = []
        current_time = start_time
        
        # Memory leak causes gradual latency increase and occasional failures
        memory_pressure_multipliers = {
            'web_api': 1.3 if affected_service != 'web_api' else 2.8,
            'order_processor': 1.1 if affected_service != 'order_processor' else 3.5,
            'notification_service': 1.0 if affected_service != 'notification_service' else 2.2
        }
        
        # Simulate occasional GC pauses for affected service
        gc_pause = random.choice([True, False]) if affected_service != 'web_api' else False
        
        # Web API span
        web_api_duration = int(self.services['web_api']['base_latency'] * memory_pressure_multipliers['web_api']) + random.randint(15, 50)
        web_api_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': None,
            'service_name': 'web_api',
            'operation_name': 'process_request',
            'start_time': current_time.isoformat(),
            'duration_ms': web_api_duration,
            'end_time': (current_time + timedelta(milliseconds=web_api_duration)).isoformat(),
            'status': 'success' if affected_service != 'web_api' else 'degraded',
            'tags': {
                'http.method': 'POST',
                'http.url': '/api/orders',
                'http.status_code': 200,
                'memory.heap_used_mb': 145.2 if affected_service == 'web_api' else 89.3
            }
        }
        spans.append(web_api_span)
        
        # Order processor span (with potential GC pause)
        order_start = current_time + timedelta(milliseconds=12)
        order_base_duration = int(self.services['order_processor']['base_latency'] * memory_pressure_multipliers['order_processor'])
        gc_pause_duration = random.randint(200, 800) if (gc_pause and affected_service == 'order_processor') else 0
        order_duration = order_base_duration + gc_pause_duration + random.randint(25, 75)
        
        order_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': web_api_span['span_id'],
            'service_name': 'order_processor',
            'operation_name': 'process_order',
            'start_time': order_start.isoformat(),
            'duration_ms': order_duration,
            'end_time': (order_start + timedelta(milliseconds=order_duration)).isoformat(),
            'status': 'success' if affected_service != 'order_processor' else ('timeout' if gc_pause else 'degraded'),
            'tags': {
                'order.id': f'order-{random.randint(1000, 9999)}',
                'order.amount': f'{random.randint(10, 500)}.00',
                'memory.heap_used_mb': 387.8 if affected_service == 'order_processor' else 156.2,
                'gc.pause_ms': gc_pause_duration if gc_pause else 0
            }
        }
        spans.append(order_span)
        
        # Notification service span
        notification_start = order_start + timedelta(milliseconds=order_duration - 25)
        notification_duration = int(self.services['notification_service']['base_latency'] * memory_pressure_multipliers['notification_service']) + random.randint(10, 45)
        notification_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': order_span['span_id'],
            'service_name': 'notification_service',
            'operation_name': 'send_notification',
            'start_time': notification_start.isoformat(),
            'duration_ms': notification_duration,
            'end_time': (notification_start + timedelta(milliseconds=notification_duration)).isoformat(),
            'status': 'success' if affected_service != 'notification_service' else 'degraded',
            'tags': {
                'notification.type': 'order_confirmation',
                'notification.channel': random.choice(['email', 'sms', 'push']),
                'memory.heap_used_mb': 298.5 if affected_service == 'notification_service' else 78.1
            }
        }
        spans.append(notification_span)
        
        return {
            'trace_id': trace_id,
            'spans': spans,
            'total_duration_ms': web_api_duration,
            'service_count': 3,
            'span_count': len(spans),
            'status': 'memory_leak',
            'affected_service': affected_service,
            'anomaly_indicators': {
                'high_memory_services': [s['service_name'] for s in spans 
                                       if float(s['tags'].get('memory.heap_used_mb', 0)) > 200],
                'gc_pauses': [s for s in spans if s['tags'].get('gc.pause_ms', 0) > 0],
                'max_memory_usage': max([float(s['tags'].get('memory.heap_used_mb', 0)) for s in spans])
            }
        }
    
    def simulate_service_crash_trace(self, trace_id: str = None, start_time: datetime = None,
                                    crashed_service: str = 'notification_service') -> Dict:
        """
        Generate a trace showing service crash and error propagation
        """
        if trace_id is None:
            trace_id = self.generate_trace_id()
        if start_time is None:
            start_time = datetime.now()
        
        spans = []
        current_time = start_time
        
        # Web API span (may fail if downstream crashes)
        web_api_duration = self.services['web_api']['base_latency'] + random.randint(10, 30)
        if crashed_service in ['order_processor', 'notification_service']:
            web_api_duration += random.randint(50, 200)  # Timeout waiting for downstream
        
        web_api_status = 'error' if crashed_service == 'order_processor' else 'degraded'
        web_api_span = {
            'trace_id': trace_id,
            'span_id': self.generate_span_id(),
            'parent_span_id': None,
            'service_name': 'web_api',
            'operation_name': 'process_request',
            'start_time': current_time.isoformat(),
            'duration_ms': web_api_duration,
            'end_time': (current_time + timedelta(milliseconds=web_api_duration)).isoformat(),
            'status': web_api_status if crashed_service != 'web_api' else 'crash',
            'tags': {
                'http.method': 'POST',
                'http.url': '/api/orders',
                'http.status_code': 500 if crashed_service in ['web_api', 'order_processor'] else 502,
                'error.message': f'Service {crashed_service} unavailable' if crashed_service != 'web_api' else 'Service crashed'
            }
        }
        spans.append(web_api_span)
        
        # Order processor span (may be the crashed service)
        if crashed_service != 'order_processor':
            order_start = current_time + timedelta(milliseconds=15)
            order_duration = self.services['order_processor']['base_latency'] + random.randint(20, 60)
            if crashed_service == 'notification_service':
                order_duration += random.randint(100, 300)  # Timeout waiting for notification
            
            order_span = {
                'trace_id': trace_id,
                'span_id': self.generate_span_id(),
                'parent_span_id': web_api_span['span_id'],
                'service_name': 'order_processor',
                'operation_name': 'process_order',
                'start_time': order_start.isoformat(),
                'duration_ms': order_duration,
                'end_time': (order_start + timedelta(milliseconds=order_duration)).isoformat(),
                'status': 'error' if crashed_service == 'notification_service' else 'success',
                'tags': {
                    'order.id': f'order-{random.randint(1000, 9999)}',
                    'order.amount': f'{random.randint(10, 500)}.00',
                    'error.message': 'Notification service unavailable' if crashed_service == 'notification_service' else None
                }
            }
            spans.append(order_span)
            
            # Notification service span (may be the crashed service)
            if crashed_service != 'notification_service':
                notification_start = order_start + timedelta(milliseconds=order_duration - 40)
                notification_duration = self.services['notification_service']['base_latency'] + random.randint(5, 25)
                notification_span = {
                    'trace_id': trace_id,
                    'span_id': self.generate_span_id(),
                    'parent_span_id': order_span['span_id'],
                    'service_name': 'notification_service',
                    'operation_name': 'send_notification',
                    'start_time': notification_start.isoformat(),
                    'duration_ms': notification_duration,
                    'end_time': (notification_start + timedelta(milliseconds=notification_duration)).isoformat(),
                    'status': 'success',
                    'tags': {
                        'notification.type': 'order_confirmation',
                        'notification.channel': random.choice(['email', 'sms', 'push'])
                    }
                }
                spans.append(notification_span)
        
        return {
            'trace_id': trace_id,
            'spans': spans,
            'total_duration_ms': web_api_duration,
            'service_count': len([s for s in spans]),
            'span_count': len(spans),
            'status': 'service_crash',
            'crashed_service': crashed_service,
            'anomaly_indicators': {
                'error_services': [s['service_name'] for s in spans if s['status'] in ['error', 'crash']],
                'high_latency_services': [s['service_name'] for s in spans if s['duration_ms'] > 300],
                'failed_operations': len([s for s in spans if s['status'] in ['error', 'crash', 'timeout']])
            }
        }
    
    def extract_trace_features(self, trace: Dict) -> Dict:
        """
        Extract numerical features from trace data for ML analysis
        """
        spans = trace['spans']
        
        # Basic trace metrics
        total_spans = len(spans)
        total_duration = trace['total_duration_ms']
        service_count = len(set(s['service_name'] for s in spans))
        
        # Latency features
        latencies = [s['duration_ms'] for s in spans]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        
        # Service-specific features
        service_latencies = {}
        service_statuses = {}
        for span in spans:
            service = span['service_name']
            service_latencies[f'{service}_latency_ms'] = span['duration_ms']
            service_statuses[f'{service}_status'] = 1 if span['status'] == 'success' else 0
        
        # Error features
        error_count = len([s for s in spans if s['status'] in ['error', 'crash', 'timeout', 'degraded']])
        error_rate = error_count / total_spans if total_spans > 0 else 0
        
        # Memory features (if available)
        memory_features = {}
        for span in spans:
            if 'memory.heap_used_mb' in span['tags']:
                service = span['service_name']
                memory_features[f'{service}_memory_mb'] = float(span['tags']['memory.heap_used_mb'])
        
        # CPU features (if available)
        cpu_features = {}
        for span in spans:
            if 'cpu.usage_percent' in span['tags']:
                service = span['service_name']
                cpu_features[f'{service}_cpu_percent'] = float(span['tags']['cpu.usage_percent'])
        
        # Combine all features
        features = {
            'trace_total_duration_ms': total_duration,
            'trace_span_count': total_spans,
            'trace_service_count': service_count,
            'trace_avg_latency_ms': avg_latency,
            'trace_max_latency_ms': max_latency,
            'trace_min_latency_ms': min_latency,
            'trace_error_count': error_count,
            'trace_error_rate': error_rate,
            **service_latencies,
            **service_statuses,
            **memory_features,
            **cpu_features
        }
        
        return features

def demo_trace_simulator():
    """Demonstration of trace simulation functionality"""
    print("🔍 Trace Simulator Demo")
    print("=" * 50)
    
    simulator = TraceSimulator()
    
    # Generate different types of traces
    print("\n📊 Generating Sample Traces:")
    
    # Normal trace
    normal_trace = simulator.simulate_normal_trace()
    print(f"✅ Normal trace: {normal_trace['trace_id']} ({normal_trace['total_duration_ms']}ms)")
    
    # CPU spike trace
    cpu_trace = simulator.simulate_cpu_spike_trace(affected_service='order_processor')
    print(f"🔥 CPU spike trace: {cpu_trace['trace_id']} ({cpu_trace['total_duration_ms']}ms)")
    
    # Memory leak trace
    memory_trace = simulator.simulate_memory_leak_trace(affected_service='order_processor')
    print(f"📈 Memory leak trace: {memory_trace['trace_id']} ({memory_trace['total_duration_ms']}ms)")
    
    # Service crash trace
    crash_trace = simulator.simulate_service_crash_trace(crashed_service='notification_service')
    print(f"💥 Service crash trace: {crash_trace['trace_id']} ({crash_trace['total_duration_ms']}ms)")
    
    # Extract features from traces
    print(f"\n🔧 Trace Feature Extraction:")
    for trace_type, trace in [('Normal', normal_trace), ('CPU Spike', cpu_trace), 
                             ('Memory Leak', memory_trace), ('Crash', crash_trace)]:
        features = simulator.extract_trace_features(trace)
        print(f"   {trace_type}: {len(features)} features extracted")
        print(f"      Duration: {features['trace_total_duration_ms']}ms")
        print(f"      Error Rate: {features['trace_error_rate']:.2f}")
        if 'order_processor_memory_mb' in features:
            print(f"      Order Processor Memory: {features['order_processor_memory_mb']:.1f}MB")
    
    return simulator

if __name__ == "__main__":
    # Run demonstration
    simulator = demo_trace_simulator()
    
    print(f"\n🚀 Trace simulator ready for causal analysis integration!")