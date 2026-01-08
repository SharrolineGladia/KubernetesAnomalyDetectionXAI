#!/usr/bin/env python3
"""
Service Dependency Graph for Causal Root Cause Analysis
Creates and manages service dependency relationships for microservices
"""

import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class ServiceDependencyGraph:
    """
    Manages service dependency relationships and provides causal analysis capabilities
    """
    
    def __init__(self):
        """Initialize the service dependency graph"""
        self.graph = nx.DiGraph()  # Directed graph for dependencies
        self.service_dependencies = {
            'web_api': ['order_processor', 'notification_service'],
            'order_processor': ['notification_service'],
            'notification_service': []  # Leaf service with no dependencies
        }
        self.service_ports = {
            'web_api': 8001,
            'order_processor': 8002,
            'notification_service': 8003
        }
        self._build_graph()
    
    def _build_graph(self):
        """Build the NetworkX graph from service dependencies"""
        # Add all services as nodes
        for service in self.service_dependencies:
            self.graph.add_node(service, port=self.service_ports[service])
        
        # Add dependency edges (A depends on B -> edge from A to B)
        for service, dependencies in self.service_dependencies.items():
            for dependency in dependencies:
                self.graph.add_edge(service, dependency, 
                                  relationship='depends_on',
                                  weight=1.0)
        
        print(f"✅ Service dependency graph built with {len(self.graph.nodes)} services")
    
    def get_dependencies(self, service: str) -> List[str]:
        """Get list of services that the given service depends on"""
        return self.service_dependencies.get(service, [])
    
    def get_dependents(self, service: str) -> List[str]:
        """Get list of services that depend on the given service"""
        dependents = []
        for svc, deps in self.service_dependencies.items():
            if service in deps:
                dependents.append(svc)
        return dependents
    
    def get_all_downstream_services(self, service: str) -> List[str]:
        """Get all services that could be affected by failures in the given service"""
        try:
            # Get all nodes reachable from this service (reverse direction)
            # In dependency graph, if A depends on B, failure in B affects A
            # So we need to find who depends on this service (directly or indirectly)
            downstream = []
            for node in self.graph.nodes():
                if node != service:
                    try:
                        # Check if there's a path from node to service (node depends on service)
                        if nx.has_path(self.graph, node, service):
                            downstream.append(node)
                    except nx.NetworkXNoPath:
                        continue
            return downstream
        except nx.NetworkXError:
            return []
    
    def get_all_upstream_services(self, service: str) -> List[str]:
        """Get all services that could affect the given service"""
        try:
            # Get all nodes that this service depends on (directly or indirectly)
            if service in self.graph:
                upstream = []
                try:
                    # Find all nodes reachable from this service (what it depends on)
                    for node in self.graph.nodes():
                        if node != service:
                            try:
                                if nx.has_path(self.graph, service, node):
                                    upstream.append(node)
                            except nx.NetworkXNoPath:
                                continue
                    return upstream
                except nx.NetworkXError:
                    return []
            return []
        except nx.NetworkXError:
            return []
    
    def find_causal_path(self, root_service: str, affected_service: str) -> Optional[List[str]]:
        """Find the dependency path from root cause to affected service"""
        try:
            # If affected service depends on root service, there's a causal path
            if nx.has_path(self.graph, affected_service, root_service):
                # Path from affected to root (dependency direction)
                path = nx.shortest_path(self.graph, affected_service, root_service)
                return path
            return None
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return None
    
    def analyze_failure_propagation(self, failed_service: str) -> Dict:
        """Analyze potential failure propagation from a failed service"""
        directly_affected = self.get_dependents(failed_service)
        all_affected = self.get_all_downstream_services(failed_service)
        
        propagation_paths = []
        for service in all_affected:
            path = self.find_causal_path(failed_service, service)
            if path:
                propagation_paths.append({
                    'target': service,
                    'path': path,
                    'hops': len(path) - 1
                })
        
        return {
            'failed_service': failed_service,
            'directly_affected': directly_affected,
            'all_affected': all_affected,
            'propagation_paths': propagation_paths
        }
    
    def determine_likely_root_cause(self, anomalous_services: List[str]) -> Dict:
        """
        Determine most likely root cause from a list of anomalous services
        based on dependency relationships
        """
        if not anomalous_services:
            return {'root_cause': None, 'confidence': 0.0, 'reasoning': 'No anomalous services provided'}
        
        if len(anomalous_services) == 1:
            return {
                'root_cause': anomalous_services[0],
                'confidence': 0.7,  # Medium confidence for single service
                'reasoning': f'Only one anomalous service detected: {anomalous_services[0]}'
            }
        
        # Score each service based on how many other anomalous services it could affect
        root_cause_scores = {}
        
        for candidate in anomalous_services:
            # Services that this candidate could affect
            affected_by_candidate = set()
            
            for other_service in anomalous_services:
                if other_service != candidate:
                    path = self.find_causal_path(candidate, other_service)
                    if path:
                        affected_by_candidate.add(other_service)
            
            # Score based on how many anomalous services could be explained
            score = len(affected_by_candidate) / (len(anomalous_services) - 1) if len(anomalous_services) > 1 else 0
            root_cause_scores[candidate] = {
                'score': score,
                'affected_services': list(affected_by_candidate)
            }
        
        # Find the service with highest score
        if not root_cause_scores or all(score['score'] == 0 for score in root_cause_scores.values()):
            return {
                'root_cause': anomalous_services[0],  # Fallback
                'confidence': 0.3,
                'reasoning': 'No clear dependency relationship found'
            }
        
        best_candidate = max(root_cause_scores.keys(), 
                           key=lambda x: root_cause_scores[x]['score'])
        best_score = root_cause_scores[best_candidate]['score']
        affected = root_cause_scores[best_candidate]['affected_services']
        
        # Calculate confidence based on score
        confidence = min(0.95, 0.5 + (best_score * 0.45))  # Scale to 0.5-0.95 range
        
        reasoning = f"{best_candidate} can explain {len(affected)} out of {len(anomalous_services)-1} other anomalous services"
        
        return {
            'root_cause': best_candidate,
            'confidence': confidence,
            'reasoning': reasoning,
            'affected_services': affected,
            'all_scores': root_cause_scores
        }
    
    def visualize_graph(self, save_path: str = None, highlight_services: List[str] = None):
        """Visualize the service dependency graph"""
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Default colors
        node_colors = []
        for node in self.graph.nodes():
            if highlight_services and node in highlight_services:
                node_colors.append('red')  # Highlight anomalous services
            else:
                node_colors.append('lightblue')
        
        # Draw the graph
        nx.draw(self.graph, pos, 
                with_labels=True,
                node_color=node_colors,
                node_size=3000,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                arrowstyle='->')
        
        # Add port labels
        port_labels = {node: f":{self.service_ports[node]}" 
                      for node in self.graph.nodes()}
        pos_below = {node: (x, y-0.15) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(self.graph, pos_below, port_labels, 
                               font_size=8, font_color='gray')
        
        plt.title("Microservice Dependency Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Dependency graph saved to: {save_path}")
        
        return plt
    
    def export_graph_data(self) -> Dict:
        """Export graph data for external analysis"""
        return {
            'services': list(self.graph.nodes()),
            'dependencies': dict(self.service_dependencies),
            'ports': dict(self.service_ports),
            'graph_info': {
                'nodes': len(self.graph.nodes()),
                'edges': len(self.graph.edges()),
                'is_dag': nx.is_directed_acyclic_graph(self.graph)
            }
        }

def demo_service_graph():
    """Demonstration of service dependency graph functionality"""
    print("🔍 Service Dependency Graph Demo")
    print("=" * 50)
    
    # Create graph
    graph = ServiceDependencyGraph()
    
    # Show basic information
    print(f"📊 Services: {list(graph.service_dependencies.keys())}")
    
    # Analyze dependencies
    for service in graph.service_dependencies:
        deps = graph.get_dependencies(service)
        dependents = graph.get_dependents(service)
        print(f"🔧 {service}:")
        print(f"   └─ Depends on: {deps if deps else 'None'}")
        print(f"   └─ Depended by: {dependents if dependents else 'None'}")
    
    # Simulate failure analysis
    print(f"\n🚨 Failure Propagation Analysis:")
    for service in ['order_processor', 'notification_service']:
        analysis = graph.analyze_failure_propagation(service)
        print(f"   If {service} fails → affects: {analysis['all_affected']}")
    
    # Test root cause analysis
    print(f"\n🎯 Root Cause Analysis Examples:")
    
    # Scenario 1: Single service anomaly
    anomalous_services = ['order_processor']
    result = graph.determine_likely_root_cause(anomalous_services)
    print(f"   Anomalous: {anomalous_services}")
    print(f"   Root Cause: {result['root_cause']} (confidence: {result['confidence']:.2f})")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Scenario 2: Multiple service anomalies  
    anomalous_services = ['web_api', 'order_processor']
    result = graph.determine_likely_root_cause(anomalous_services)
    print(f"\n   Anomalous: {anomalous_services}")
    print(f"   Root Cause: {result['root_cause']} (confidence: {result['confidence']:.2f})")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Scenario 3: Cascading failure
    anomalous_services = ['web_api', 'order_processor', 'notification_service']
    result = graph.determine_likely_root_cause(anomalous_services)
    print(f"\n   Anomalous: {anomalous_services}")
    print(f"   Root Cause: {result['root_cause']} (confidence: {result['confidence']:.2f})")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Export data
    graph_data = graph.export_graph_data()
    print(f"\n📈 Graph Statistics:")
    print(f"   Services: {graph_data['graph_info']['nodes']}")
    print(f"   Dependencies: {graph_data['graph_info']['edges']}")
    print(f"   Is DAG: {graph_data['graph_info']['is_dag']}")
    
    return graph

if __name__ == "__main__":
    # Run demonstration
    graph = demo_service_graph()
    
    # Try to create visualization
    try:
        matplotlib_backend = plt.get_backend()
        if matplotlib_backend.lower() != 'agg':
            plt.switch_backend('Agg')  # Use non-interactive backend
        
        plt = graph.visualize_graph('service_dependency_graph.png')
        print(f"\n📊 Graph visualization saved as 'service_dependency_graph.png'")
        plt.close()  # Close the plot to free memory
    except Exception as e:
        print(f"\n📊 Could not create graph visualization: {e}")
        print("   Graph analysis functionality still works without visualization")