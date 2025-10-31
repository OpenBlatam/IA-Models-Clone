"""
Edge Computing & IoT Testing Framework for HeyGen AI Testing System.
Advanced edge computing and IoT testing including device simulation,
network testing, and distributed system validation.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import random
import math
import threading
import queue
import socket
import struct
from collections import defaultdict, deque
import sqlite3
import requests
import websocket
import paho.mqtt.client as mqtt
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

@dataclass
class IoTDevice:
    """Represents an IoT device."""
    device_id: str
    name: str
    device_type: str  # "sensor", "actuator", "gateway", "edge_compute"
    capabilities: List[str]
    location: Tuple[float, float, float]  # lat, lon, alt
    status: str = "offline"  # "online", "offline", "error"
    last_seen: Optional[datetime] = None
    data_rate: float = 1.0  # messages per second
    battery_level: float = 100.0
    signal_strength: float = -50.0  # dBm
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class EdgeNode:
    """Represents an edge computing node."""
    node_id: str
    name: str
    location: Tuple[float, float, float]
    compute_capacity: Dict[str, float]  # CPU, memory, storage
    network_bandwidth: float  # Mbps
    connected_devices: List[str] = field(default_factory=list)
    status: str = "offline"
    last_heartbeat: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class NetworkMessage:
    """Represents a network message."""
    message_id: str
    source_device: str
    target_device: str
    message_type: str  # "data", "command", "heartbeat", "alert"
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class EdgeTestResult:
    """Represents an edge/IoT test result."""
    result_id: str
    test_name: str
    test_type: str  # "connectivity", "latency", "throughput", "reliability", "security"
    success: bool
    metrics: Dict[str, float]
    devices_involved: List[str]
    network_conditions: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class IoTDeviceSimulator:
    """Simulates IoT devices for testing."""
    
    def __init__(self):
        self.devices = {}
        self.device_threads = {}
        self.running = False
        self.message_queue = queue.Queue()
        self.data_generators = {
            "temperature": self._generate_temperature_data,
            "humidity": self._generate_humidity_data,
            "pressure": self._generate_pressure_data,
            "motion": self._generate_motion_data,
            "light": self._generate_light_data,
            "sound": self._generate_sound_data
        }
    
    def create_device(self, name: str, device_type: str, 
                     capabilities: List[str], location: Tuple[float, float, float]) -> IoTDevice:
        """Create an IoT device."""
        device = IoTDevice(
            device_id=f"device_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            device_type=device_type,
            capabilities=capabilities,
            location=location,
            status="offline"
        )
        
        self.devices[device.device_id] = device
        return device
    
    def start_device_simulation(self, device_id: str):
        """Start simulating a device."""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        device = self.devices[device_id]
        device.status = "online"
        device.last_seen = datetime.now()
        
        # Start device thread
        thread = threading.Thread(target=self._simulate_device, args=(device_id,), daemon=True)
        thread.start()
        self.device_threads[device_id] = thread
    
    def stop_device_simulation(self, device_id: str):
        """Stop simulating a device."""
        if device_id in self.devices:
            self.devices[device_id].status = "offline"
        
        if device_id in self.device_threads:
            # Thread will stop when device status changes
            pass
    
    def _simulate_device(self, device_id: str):
        """Simulate device behavior."""
        device = self.devices[device_id]
        
        while device.status == "online":
            try:
                # Generate data based on capabilities
                data = {}
                for capability in device.capabilities:
                    if capability in self.data_generators:
                        data[capability] = self.data_generators[capability]()
                
                # Create message
                message = NetworkMessage(
                    message_id=f"msg_{int(time.time())}_{random.randint(1000, 9999)}",
                    source_device=device_id,
                    target_device="gateway",
                    message_type="data",
                    payload=data,
                    timestamp=datetime.now(),
                    priority=random.choice([1, 2, 3])
                )
                
                # Send message
                self.message_queue.put(message)
                
                # Update device status
                device.last_seen = datetime.now()
                device.battery_level = max(0, device.battery_level - random.uniform(0.1, 0.5))
                device.signal_strength = random.uniform(-80, -30)
                
                # Sleep based on data rate
                time.sleep(1.0 / device.data_rate)
                
            except Exception as e:
                logging.error(f"Error simulating device {device_id}: {e}")
                device.status = "error"
                break
    
    def _generate_temperature_data(self) -> float:
        """Generate temperature data."""
        base_temp = 20.0
        variation = 10.0 * math.sin(time.time() / 3600)  # Daily variation
        noise = random.uniform(-2, 2)
        return base_temp + variation + noise
    
    def _generate_humidity_data(self) -> float:
        """Generate humidity data."""
        base_humidity = 50.0
        variation = 20.0 * math.sin(time.time() / 7200)  # 2-hour variation
        noise = random.uniform(-5, 5)
        return max(0, min(100, base_humidity + variation + noise))
    
    def _generate_pressure_data(self) -> float:
        """Generate pressure data."""
        base_pressure = 1013.25  # Standard atmospheric pressure
        variation = 10.0 * math.sin(time.time() / 1800)  # 30-minute variation
        noise = random.uniform(-2, 2)
        return base_pressure + variation + noise
    
    def _generate_motion_data(self) -> bool:
        """Generate motion data."""
        return random.random() < 0.1  # 10% chance of motion
    
    def _generate_light_data(self) -> float:
        """Generate light data."""
        base_light = 100.0
        variation = 80.0 * math.sin(time.time() / 14400)  # 4-hour variation
        noise = random.uniform(-10, 10)
        return max(0, base_light + variation + noise)
    
    def _generate_sound_data(self) -> float:
        """Generate sound data."""
        base_sound = 40.0  # dB
        variation = 20.0 * random.random()
        noise = random.uniform(-5, 5)
        return max(0, base_sound + variation + noise)
    
    def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """Get device status."""
        if device_id not in self.devices:
            return {}
        
        device = self.devices[device_id]
        return {
            "device_id": device.device_id,
            "name": device.name,
            "status": device.status,
            "last_seen": device.last_seen.isoformat() if device.last_seen else None,
            "battery_level": device.battery_level,
            "signal_strength": device.signal_strength,
            "data_rate": device.data_rate
        }

class EdgeComputingSimulator:
    """Simulates edge computing nodes."""
    
    def __init__(self):
        self.nodes = {}
        self.compute_tasks = queue.Queue()
        self.running = False
        self.worker_threads = []
    
    def create_edge_node(self, name: str, location: Tuple[float, float, float],
                        compute_capacity: Dict[str, float], network_bandwidth: float) -> EdgeNode:
        """Create an edge computing node."""
        node = EdgeNode(
            node_id=f"node_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            location=location,
            compute_capacity=compute_capacity,
            network_bandwidth=network_bandwidth,
            status="offline"
        )
        
        self.nodes[node.node_id] = node
        return node
    
    def start_edge_node(self, node_id: str):
        """Start an edge node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        node.status = "online"
        node.last_heartbeat = datetime.now()
        
        # Start worker threads
        num_workers = int(node.compute_capacity.get("cpu", 4))
        for i in range(num_workers):
            thread = threading.Thread(target=self._edge_worker, args=(node_id,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
    
    def stop_edge_node(self, node_id: str):
        """Stop an edge node."""
        if node_id in self.nodes:
            self.nodes[node_id].status = "offline"
    
    def submit_compute_task(self, task: Dict[str, Any]):
        """Submit a compute task to edge nodes."""
        self.compute_tasks.put(task)
    
    def _edge_worker(self, node_id: str):
        """Edge computing worker thread."""
        node = self.nodes[node_id]
        
        while node.status == "online":
            try:
                # Get task from queue
                task = self.compute_tasks.get(timeout=1)
                
                # Process task
                result = self._process_compute_task(task, node)
                
                # Update node status
                node.last_heartbeat = datetime.now()
                
                self.compute_tasks.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in edge worker {node_id}: {e}")
    
    def _process_compute_task(self, task: Dict[str, Any], node: EdgeNode) -> Dict[str, Any]:
        """Process a compute task."""
        task_type = task.get("type", "data_processing")
        
        # Simulate processing time based on task complexity
        processing_time = random.uniform(0.1, 2.0)
        time.sleep(processing_time)
        
        # Simulate processing result
        result = {
            "task_id": task.get("task_id", "unknown"),
            "node_id": node.node_id,
            "processing_time": processing_time,
            "result": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_node_status(self, node_id: str) -> Dict[str, Any]:
        """Get edge node status."""
        if node_id not in self.nodes:
            return {}
        
        node = self.nodes[node_id]
        return {
            "node_id": node.node_id,
            "name": node.name,
            "status": node.status,
            "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None,
            "compute_capacity": node.compute_capacity,
            "network_bandwidth": node.network_bandwidth,
            "connected_devices": len(node.connected_devices)
        }

class NetworkSimulator:
    """Simulates network conditions for IoT/Edge testing."""
    
    def __init__(self):
        self.network_conditions = {
            "latency": 0.0,  # ms
            "bandwidth": 1000.0,  # Mbps
            "packet_loss": 0.0,  # percentage
            "jitter": 0.0,  # ms
            "reliability": 1.0  # 0-1
        }
        self.message_history = []
        self.connection_pool = {}
    
    def set_network_conditions(self, conditions: Dict[str, float]):
        """Set network conditions."""
        self.network_conditions.update(conditions)
    
    def simulate_message_transmission(self, message: NetworkMessage) -> bool:
        """Simulate message transmission with network conditions."""
        # Simulate latency
        if self.network_conditions["latency"] > 0:
            time.sleep(self.network_conditions["latency"] / 1000.0)
        
        # Simulate packet loss
        if random.random() < self.network_conditions["packet_loss"] / 100.0:
            return False
        
        # Simulate jitter
        jitter = random.uniform(-self.network_conditions["jitter"], 
                              self.network_conditions["jitter"]) / 1000.0
        if jitter > 0:
            time.sleep(jitter)
        
        # Record message
        self.message_history.append({
            "message_id": message.message_id,
            "timestamp": datetime.now(),
            "success": True,
            "latency": self.network_conditions["latency"],
            "jitter": jitter
        })
        
        return True
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network performance metrics."""
        if not self.message_history:
            return {}
        
        recent_messages = self.message_history[-100:]  # Last 100 messages
        
        success_rate = sum(1 for m in recent_messages if m["success"]) / len(recent_messages)
        avg_latency = np.mean([m["latency"] for m in recent_messages])
        avg_jitter = np.mean([m["jitter"] for m in recent_messages])
        
        return {
            "success_rate": success_rate,
            "average_latency": avg_latency,
            "average_jitter": avg_jitter,
            "total_messages": len(self.message_history),
            "current_conditions": self.network_conditions
        }

class EdgeIoTTestFramework:
    """Main Edge/IoT testing framework."""
    
    def __init__(self):
        self.device_simulator = IoTDeviceSimulator()
        self.edge_simulator = EdgeComputingSimulator()
        self.network_simulator = NetworkSimulator()
        self.test_results = []
        self.devices = {}
        self.edge_nodes = {}
    
    def create_iot_network(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Create an IoT network based on configuration."""
        created_devices = {}
        created_nodes = {}
        
        # Create devices
        for device_config in config.get("devices", []):
            device = self.device_simulator.create_device(
                name=device_config["name"],
                device_type=device_config["type"],
                capabilities=device_config["capabilities"],
                location=device_config["location"]
            )
            created_devices[device.device_id] = device
            self.devices[device.device_id] = device
        
        # Create edge nodes
        for node_config in config.get("edge_nodes", []):
            node = self.edge_simulator.create_edge_node(
                name=node_config["name"],
                location=node_config["location"],
                compute_capacity=node_config["compute_capacity"],
                network_bandwidth=node_config["network_bandwidth"]
            )
            created_nodes[node.node_id] = node
            self.edge_nodes[node.node_id] = node
        
        return {
            "devices": list(created_devices.keys()),
            "edge_nodes": list(created_nodes.keys())
        }
    
    def test_connectivity(self, device_ids: List[str], duration: float = 30.0) -> EdgeTestResult:
        """Test connectivity between devices."""
        # Start devices
        for device_id in device_ids:
            if device_id in self.devices:
                self.device_simulator.start_device_simulation(device_id)
        
        # Monitor connectivity
        start_time = time.time()
        connectivity_events = []
        
        while time.time() - start_time < duration:
            for device_id in device_ids:
                if device_id in self.devices:
                    device = self.devices[device_id]
                    connectivity_events.append({
                        "device_id": device_id,
                        "status": device.status,
                        "timestamp": datetime.now()
                    })
            time.sleep(1)
        
        # Stop devices
        for device_id in device_ids:
            if device_id in self.devices:
                self.device_simulator.stop_device_simulation(device_id)
        
        # Calculate metrics
        online_devices = sum(1 for event in connectivity_events 
                           if event["status"] == "online")
        total_events = len(connectivity_events)
        connectivity_rate = online_devices / total_events if total_events > 0 else 0
        
        metrics = {
            "connectivity_rate": connectivity_rate,
            "online_devices": online_devices,
            "total_devices": len(device_ids),
            "test_duration": duration
        }
        
        result = EdgeTestResult(
            result_id=f"connectivity_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Connectivity Test",
            test_type="connectivity",
            success=connectivity_rate > 0.8,
            metrics=metrics,
            devices_involved=device_ids,
            network_conditions=self.network_simulator.network_conditions
        )
        
        self.test_results.append(result)
        return result
    
    def test_latency(self, source_device: str, target_device: str, 
                    num_messages: int = 100) -> EdgeTestResult:
        """Test network latency between devices."""
        latencies = []
        
        for i in range(num_messages):
            # Create test message
            message = NetworkMessage(
                message_id=f"latency_test_{i}",
                source_device=source_device,
                target_device=target_device,
                message_type="data",
                payload={"test_data": f"message_{i}"},
                timestamp=datetime.now()
            )
            
            # Measure transmission time
            start_time = time.time()
            success = self.network_simulator.simulate_message_transmission(message)
            end_time = time.time()
            
            if success:
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
        
        # Calculate metrics
        if latencies:
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            latency_std = np.std(latencies)
        else:
            avg_latency = min_latency = max_latency = latency_std = 0
        
        metrics = {
            "average_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "latency_std": latency_std,
            "successful_messages": len(latencies),
            "total_messages": num_messages
        }
        
        result = EdgeTestResult(
            result_id=f"latency_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Latency Test",
            test_type="latency",
            success=avg_latency < 100,  # Threshold: 100ms
            metrics=metrics,
            devices_involved=[source_device, target_device],
            network_conditions=self.network_simulator.network_conditions
        )
        
        self.test_results.append(result)
        return result
    
    def test_throughput(self, device_ids: List[str], duration: float = 60.0) -> EdgeTestResult:
        """Test network throughput."""
        # Start devices
        for device_id in device_ids:
            if device_id in self.devices:
                self.device_simulator.start_device_simulation(device_id)
        
        # Monitor throughput
        start_time = time.time()
        message_count = 0
        total_bytes = 0
        
        while time.time() - start_time < duration:
            # Count messages from queue
            try:
                message = self.device_simulator.message_queue.get(timeout=0.1)
                message_count += 1
                # Estimate message size
                message_size = len(json.dumps(message.payload).encode())
                total_bytes += message_size
            except queue.Empty:
                continue
        
        # Stop devices
        for device_id in device_ids:
            if device_id in self.devices:
                self.device_simulator.stop_device_simulation(device_id)
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        messages_per_second = message_count / actual_duration
        bytes_per_second = total_bytes / actual_duration
        mbps = (bytes_per_second * 8) / (1024 * 1024)  # Convert to Mbps
        
        metrics = {
            "messages_per_second": messages_per_second,
            "bytes_per_second": bytes_per_second,
            "mbps": mbps,
            "total_messages": message_count,
            "total_bytes": total_bytes,
            "test_duration": actual_duration
        }
        
        result = EdgeTestResult(
            result_id=f"throughput_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Throughput Test",
            test_type="throughput",
            success=mbps > 1.0,  # Threshold: 1 Mbps
            metrics=metrics,
            devices_involved=device_ids,
            network_conditions=self.network_simulator.network_conditions
        )
        
        self.test_results.append(result)
        return result
    
    def test_reliability(self, device_ids: List[str], duration: float = 300.0) -> EdgeTestResult:
        """Test system reliability over time."""
        # Start devices
        for device_id in device_ids:
            if device_id in self.devices:
                self.device_simulator.start_device_simulation(device_id)
        
        # Monitor reliability
        start_time = time.time()
        reliability_events = []
        
        while time.time() - start_time < duration:
            for device_id in device_ids:
                if device_id in self.devices:
                    device = self.devices[device_id]
                    reliability_events.append({
                        "device_id": device_id,
                        "status": device.status,
                        "battery_level": device.battery_level,
                        "signal_strength": device.signal_strength,
                        "timestamp": datetime.now()
                    })
            time.sleep(10)  # Check every 10 seconds
        
        # Stop devices
        for device_id in device_ids:
            if device_id in self.devices:
                self.device_simulator.stop_device_simulation(device_id)
        
        # Calculate metrics
        online_events = sum(1 for event in reliability_events 
                          if event["status"] == "online")
        total_events = len(reliability_events)
        uptime_percentage = (online_events / total_events * 100) if total_events > 0 else 0
        
        # Calculate battery degradation
        battery_levels = [event["battery_level"] for event in reliability_events]
        battery_degradation = battery_levels[0] - battery_levels[-1] if len(battery_levels) > 1 else 0
        
        metrics = {
            "uptime_percentage": uptime_percentage,
            "total_events": total_events,
            "battery_degradation": battery_degradation,
            "test_duration": duration
        }
        
        result = EdgeTestResult(
            result_id=f"reliability_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Reliability Test",
            test_type="reliability",
            success=uptime_percentage > 95,  # Threshold: 95% uptime
            metrics=metrics,
            devices_involved=device_ids,
            network_conditions=self.network_simulator.network_conditions
        )
        
        self.test_results.append(result)
        return result
    
    def test_edge_computing(self, node_ids: List[str], task_count: int = 50) -> EdgeTestResult:
        """Test edge computing performance."""
        # Start edge nodes
        for node_id in node_ids:
            if node_id in self.edge_nodes:
                self.edge_simulator.start_edge_node(node_id)
        
        # Submit compute tasks
        task_results = []
        start_time = time.time()
        
        for i in range(task_count):
            task = {
                "task_id": f"task_{i}",
                "type": "data_processing",
                "complexity": random.uniform(0.1, 1.0),
                "priority": random.choice([1, 2, 3, 4])
            }
            
            self.edge_simulator.submit_compute_task(task)
        
        # Wait for tasks to complete
        time.sleep(2)  # Allow time for processing
        
        # Stop edge nodes
        for node_id in node_ids:
            if node_id in self.edge_nodes:
                self.edge_simulator.stop_edge_node(node_id)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tasks_per_second = task_count / total_time
        
        metrics = {
            "tasks_per_second": tasks_per_second,
            "total_tasks": task_count,
            "total_time": total_time,
            "active_nodes": len(node_ids)
        }
        
        result = EdgeTestResult(
            result_id=f"edge_computing_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Edge Computing Test",
            test_type="edge_computing",
            success=tasks_per_second > 10,  # Threshold: 10 tasks/second
            metrics=metrics,
            devices_involved=node_ids,
            network_conditions=self.network_simulator.network_conditions
        )
        
        self.test_results.append(result)
        return result
    
    def generate_edge_iot_report(self) -> Dict[str, Any]:
        """Generate comprehensive Edge/IoT test report."""
        if not self.test_results:
            return {"message": "No test results available"}
        
        # Analyze results by type
        test_types = {}
        for result in self.test_results:
            if result.test_type not in test_types:
                test_types[result.test_type] = []
            test_types[result.test_type].append(result)
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        # Performance analysis
        performance_analysis = self._analyze_performance_metrics()
        
        # Generate recommendations
        recommendations = self._generate_edge_iot_recommendations()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "devices_tested": len(self.devices),
                "edge_nodes_tested": len(self.edge_nodes)
            },
            "by_test_type": {test_type: len(results) for test_type, results in test_types.items()},
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "detailed_results": [r.__dict__ for r in self.test_results]
        }
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics across all tests."""
        all_metrics = [r.metrics for r in self.test_results]
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {}
        for result_metrics in all_metrics:
            for metric_name, value in result_metrics.items():
                if metric_name not in aggregated:
                    aggregated[metric_name] = []
                aggregated[metric_name].append(value)
        
        # Calculate statistics
        performance_stats = {}
        for metric_name, values in aggregated.items():
            performance_stats[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return performance_stats
    
    def _generate_edge_iot_recommendations(self) -> List[str]:
        """Generate Edge/IoT specific recommendations."""
        recommendations = []
        
        # Analyze connectivity results
        connectivity_results = [r for r in self.test_results if r.test_type == "connectivity"]
        if connectivity_results:
            avg_connectivity = np.mean([r.metrics.get("connectivity_rate", 0) for r in connectivity_results])
            if avg_connectivity < 0.9:
                recommendations.append("Improve device connectivity and network stability")
        
        # Analyze latency results
        latency_results = [r for r in self.test_results if r.test_type == "latency"]
        if latency_results:
            avg_latency = np.mean([r.metrics.get("average_latency", 0) for r in latency_results])
            if avg_latency > 100:
                recommendations.append("Optimize network latency for real-time applications")
        
        # Analyze throughput results
        throughput_results = [r for r in self.test_results if r.test_type == "throughput"]
        if throughput_results:
            avg_throughput = np.mean([r.metrics.get("mbps", 0) for r in throughput_results])
            if avg_throughput < 10:
                recommendations.append("Increase network bandwidth for better throughput")
        
        # Analyze reliability results
        reliability_results = [r for r in self.test_results if r.test_type == "reliability"]
        if reliability_results:
            avg_uptime = np.mean([r.metrics.get("uptime_percentage", 0) for r in reliability_results])
            if avg_uptime < 95:
                recommendations.append("Improve device reliability and fault tolerance")
        
        return recommendations

# Example usage and demo
def demo_edge_iot_testing():
    """Demonstrate Edge/IoT testing capabilities."""
    print("🌐 Edge Computing & IoT Testing Framework Demo")
    print("=" * 50)
    
    # Create Edge/IoT testing framework
    framework = EdgeIoTTestFramework()
    
    # Create IoT network configuration
    network_config = {
        "devices": [
            {
                "name": "Temperature Sensor 1",
                "type": "sensor",
                "capabilities": ["temperature", "humidity"],
                "location": (40.7128, -74.0060, 10.0)  # NYC coordinates
            },
            {
                "name": "Motion Sensor 1",
                "type": "sensor",
                "capabilities": ["motion", "light"],
                "location": (40.7128, -74.0060, 15.0)
            },
            {
                "name": "Smart Actuator 1",
                "type": "actuator",
                "capabilities": ["control"],
                "location": (40.7128, -74.0060, 5.0)
            }
        ],
        "edge_nodes": [
            {
                "name": "Edge Node 1",
                "location": (40.7128, -74.0060, 20.0),
                "compute_capacity": {"cpu": 4, "memory": 8, "storage": 100},
                "network_bandwidth": 1000.0
            }
        ]
    }
    
    # Create IoT network
    print("🏗️ Creating IoT network...")
    network = framework.create_iot_network(network_config)
    print(f"✅ Created {len(network['devices'])} devices and {len(network['edge_nodes'])} edge nodes")
    
    # Set network conditions
    framework.network_simulator.set_network_conditions({
        "latency": 50.0,  # 50ms
        "bandwidth": 100.0,  # 100 Mbps
        "packet_loss": 1.0,  # 1%
        "jitter": 10.0,  # 10ms
        "reliability": 0.99
    })
    
    # Run comprehensive tests
    print("\n🧪 Running comprehensive Edge/IoT tests...")
    
    # Test connectivity
    connectivity_result = framework.test_connectivity(network['devices'], duration=10.0)
    print(f"📡 Connectivity Test: {'✅' if connectivity_result.success else '❌'}")
    print(f"   Connectivity Rate: {connectivity_result.metrics['connectivity_rate']:.1%}")
    
    # Test latency
    if len(network['devices']) >= 2:
        latency_result = framework.test_latency(
            network['devices'][0], network['devices'][1], num_messages=50
        )
        print(f"⏱️ Latency Test: {'✅' if latency_result.success else '❌'}")
        print(f"   Average Latency: {latency_result.metrics['average_latency']:.2f}ms")
    
    # Test throughput
    throughput_result = framework.test_throughput(network['devices'], duration=15.0)
    print(f"📊 Throughput Test: {'✅' if throughput_result.success else '❌'}")
    print(f"   Throughput: {throughput_result.metrics['mbps']:.2f} Mbps")
    
    # Test reliability
    reliability_result = framework.test_reliability(network['devices'], duration=20.0)
    print(f"🔧 Reliability Test: {'✅' if reliability_result.success else '❌'}")
    print(f"   Uptime: {reliability_result.metrics['uptime_percentage']:.1f}%")
    
    # Test edge computing
    if network['edge_nodes']:
        edge_result = framework.test_edge_computing(network['edge_nodes'], task_count=30)
        print(f"💻 Edge Computing Test: {'✅' if edge_result.success else '❌'}")
        print(f"   Tasks/Second: {edge_result.metrics['tasks_per_second']:.2f}")
    
    # Generate comprehensive report
    print("\n📈 Generating comprehensive Edge/IoT report...")
    report = framework.generate_edge_iot_report()
    
    print(f"\n📊 Comprehensive Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"  Devices Tested: {report['summary']['devices_tested']}")
    print(f"  Edge Nodes Tested: {report['summary']['edge_nodes_tested']}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    # Run demo
    demo_edge_iot_testing()
