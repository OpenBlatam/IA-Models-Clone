"""
DNA Computing Testing Framework for HeyGen AI Testing System.
Advanced DNA computing testing including molecular operations,
biochemical reactions, and biological computation validation.
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
from collections import defaultdict, deque
import sqlite3
from itertools import product
import re

@dataclass
class DNAStrand:
    """Represents a DNA strand."""
    strand_id: str
    sequence: str
    length: int
    gc_content: float
    melting_temperature: float
    secondary_structure: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DNAOperation:
    """Represents a DNA operation."""
    operation_id: str
    operation_type: str  # "hybridization", "denaturation", "ligation", "pcr", "gel_electrophoresis"
    input_strands: List[str]
    output_strands: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 1.0
    duration: float = 0.0

@dataclass
class DNACircuit:
    """Represents a DNA circuit."""
    circuit_id: str
    name: str
    strands: List[DNAStrand]
    operations: List[DNAOperation]
    input_gates: List[str] = field(default_factory=list)
    output_gates: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DNATestResult:
    """Represents a DNA computing test result."""
    result_id: str
    test_name: str
    test_type: str
    success: bool
    dna_metrics: Dict[str, float]
    molecular_metrics: Dict[str, float]
    biochemical_metrics: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class DNAStrandGenerator:
    """Generates DNA strands for testing."""
    
    def __init__(self):
        self.bases = ['A', 'T', 'G', 'C']
        self.complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    def generate_strand(self, length: int, gc_content: float = 0.5) -> DNAStrand:
        """Generate a random DNA strand."""
        # Calculate number of GC pairs
        gc_pairs = int(length * gc_content)
        at_pairs = length - gc_pairs
        
        # Generate sequence
        sequence = ""
        for _ in range(gc_pairs):
            sequence += random.choice(['G', 'C'])
        for _ in range(at_pairs):
            sequence += random.choice(['A', 'T'])
        
        # Shuffle sequence
        sequence = ''.join(random.sample(sequence, len(sequence)))
        
        # Calculate properties
        actual_gc_content = (sequence.count('G') + sequence.count('C')) / length
        melting_temperature = self._calculate_melting_temperature(sequence)
        
        strand = DNAStrand(
            strand_id=f"strand_{int(time.time())}_{random.randint(1000, 9999)}",
            sequence=sequence,
            length=length,
            gc_content=actual_gc_content,
            melting_temperature=melting_temperature
        )
        
        return strand
    
    def generate_complementary_strand(self, template: DNAStrand) -> DNAStrand:
        """Generate complementary strand."""
        complement_sequence = ""
        for base in template.sequence:
            complement_sequence += self.complement_map[base]
        
        complement = DNAStrand(
            strand_id=f"complement_{template.strand_id}",
            sequence=complement_sequence,
            length=template.length,
            gc_content=template.gc_content,
            melting_temperature=template.melting_temperature
        )
        
        return complement
    
    def _calculate_melting_temperature(self, sequence: str) -> float:
        """Calculate melting temperature using simple formula."""
        # Simplified melting temperature calculation
        gc_count = sequence.count('G') + sequence.count('C')
        at_count = sequence.count('A') + sequence.count('T')
        
        # Basic formula: Tm = 4*(G+C) + 2*(A+T)
        tm = 4 * gc_count + 2 * at_count
        
        return tm

class DNAOperationEngine:
    """Engine for DNA operations."""
    
    def __init__(self):
        self.operation_history = []
        self.strand_generator = DNAStrandGenerator()
    
    def hybridize(self, strand1: DNAStrand, strand2: DNAStrand, 
                 temperature: float = 37.0) -> DNAOperation:
        """Perform DNA hybridization."""
        # Check if strands are complementary
        is_complementary = self._check_complementarity(strand1, strand2)
        
        # Calculate hybridization success rate
        success_rate = self._calculate_hybridization_success(strand1, strand2, temperature)
        
        # Generate output strands
        output_strands = []
        if success_rate > 0.5:  # Successful hybridization
            # Create double-stranded DNA
            ds_dna = f"{strand1.sequence}-{strand2.sequence}"
            output_strands.append(ds_dna)
        
        operation = DNAOperation(
            operation_id=f"hybridization_{int(time.time())}_{random.randint(1000, 9999)}",
            operation_type="hybridization",
            input_strands=[strand1.strand_id, strand2.strand_id],
            output_strands=output_strands,
            parameters={"temperature": temperature, "complementary": is_complementary},
            success_rate=success_rate,
            duration=random.uniform(0.1, 1.0)
        )
        
        self.operation_history.append(operation)
        return operation
    
    def denature(self, ds_dna: str, temperature: float = 95.0) -> DNAOperation:
        """Perform DNA denaturation."""
        # Calculate denaturation success rate
        success_rate = self._calculate_denaturation_success(temperature)
        
        # Generate output strands
        output_strands = []
        if success_rate > 0.8:  # Successful denaturation
            strands = ds_dna.split('-')
            output_strands.extend(strands)
        
        operation = DNAOperation(
            operation_id=f"denaturation_{int(time.time())}_{random.randint(1000, 9999)}",
            operation_type="denaturation",
            input_strands=[ds_dna],
            output_strands=output_strands,
            parameters={"temperature": temperature},
            success_rate=success_rate,
            duration=random.uniform(0.5, 2.0)
        )
        
        self.operation_history.append(operation)
        return operation
    
    def pcr_amplify(self, template: DNAStrand, primers: Tuple[DNAStrand, DNAStrand],
                   cycles: int = 30) -> DNAOperation:
        """Perform PCR amplification."""
        # Calculate PCR success rate
        success_rate = self._calculate_pcr_success(template, primers, cycles)
        
        # Generate output strands
        output_strands = []
        if success_rate > 0.7:  # Successful PCR
            # Simulate exponential amplification
            copies = 2 ** cycles
            for _ in range(min(copies, 1000)):  # Limit to 1000 copies
                output_strands.append(template.sequence)
        
        operation = DNAOperation(
            operation_id=f"pcr_{int(time.time())}_{random.randint(1000, 9999)}",
            operation_type="pcr",
            input_strands=[template.strand_id, primers[0].strand_id, primers[1].strand_id],
            output_strands=output_strands,
            parameters={"cycles": cycles, "template_length": template.length},
            success_rate=success_rate,
            duration=random.uniform(1.0, 5.0)
        )
        
        self.operation_history.append(operation)
        return operation
    
    def gel_electrophoresis(self, strands: List[DNAStrand], 
                          voltage: float = 100.0) -> DNAOperation:
        """Perform gel electrophoresis."""
        # Calculate migration distances
        migration_distances = []
        for strand in strands:
            distance = self._calculate_migration_distance(strand, voltage)
            migration_distances.append(distance)
        
        # Sort strands by migration distance
        sorted_strands = sorted(zip(strands, migration_distances), 
                              key=lambda x: x[1], reverse=True)
        
        output_strands = [strand.strand_id for strand, _ in sorted_strands]
        
        operation = DNAOperation(
            operation_id=f"gel_{int(time.time())}_{random.randint(1000, 9999)}",
            operation_type="gel_electrophoresis",
            input_strands=[strand.strand_id for strand in strands],
            output_strands=output_strands,
            parameters={"voltage": voltage, "migration_distances": migration_distances},
            success_rate=1.0,  # Gel electrophoresis is usually successful
            duration=random.uniform(2.0, 4.0)
        )
        
        self.operation_history.append(operation)
        return operation
    
    def _check_complementarity(self, strand1: DNAStrand, strand2: DNAStrand) -> bool:
        """Check if two strands are complementary."""
        if strand1.length != strand2.length:
            return False
        
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        for i in range(strand1.length):
            if complement_map.get(strand1.sequence[i], '') != strand2.sequence[i]:
                return False
        
        return True
    
    def _calculate_hybridization_success(self, strand1: DNAStrand, strand2: DNAStrand, 
                                       temperature: float) -> float:
        """Calculate hybridization success rate."""
        # Base success rate
        base_success = 0.8
        
        # Temperature effect
        temp_factor = 1.0 - abs(temperature - 37.0) / 100.0
        temp_factor = max(0.0, min(1.0, temp_factor))
        
        # Length effect
        length_factor = min(1.0, strand1.length / 20.0)
        
        # GC content effect
        gc_factor = 1.0 - abs(strand1.gc_content - 0.5) * 2
        
        success_rate = base_success * temp_factor * length_factor * gc_factor
        return max(0.0, min(1.0, success_rate))
    
    def _calculate_denaturation_success(self, temperature: float) -> float:
        """Calculate denaturation success rate."""
        if temperature >= 95.0:
            return 1.0
        elif temperature >= 80.0:
            return 0.8
        elif temperature >= 60.0:
            return 0.5
        else:
            return 0.1
    
    def _calculate_pcr_success(self, template: DNAStrand, primers: Tuple[DNAStrand, DNAStrand],
                             cycles: int) -> float:
        """Calculate PCR success rate."""
        # Base success rate
        base_success = 0.9
        
        # Cycle effect
        cycle_factor = 1.0 - (cycles - 25) / 100.0
        cycle_factor = max(0.5, min(1.0, cycle_factor))
        
        # Primer length effect
        primer_factor = min(1.0, (primers[0].length + primers[1].length) / 40.0)
        
        # Template length effect
        template_factor = 1.0 - (template.length - 100) / 1000.0
        template_factor = max(0.5, min(1.0, template_factor))
        
        success_rate = base_success * cycle_factor * primer_factor * template_factor
        return max(0.0, min(1.0, success_rate))
    
    def _calculate_migration_distance(self, strand: DNAStrand, voltage: float) -> float:
        """Calculate gel electrophoresis migration distance."""
        # Migration distance is inversely proportional to length
        base_distance = 100.0  # Maximum distance
        length_factor = 1.0 / (strand.length / 10.0)
        voltage_factor = voltage / 100.0
        
        distance = base_distance * length_factor * voltage_factor
        return max(0.0, distance)

class DNACircuitBuilder:
    """Builder for DNA circuits."""
    
    def __init__(self):
        self.strand_generator = DNAStrandGenerator()
        self.operation_engine = DNAOperationEngine()
    
    def create_and_gate(self, input1: DNAStrand, input2: DNAStrand) -> DNACircuit:
        """Create a DNA AND gate."""
        circuit = DNACircuit(
            circuit_id=f"and_gate_{int(time.time())}_{random.randint(1000, 9999)}",
            name="AND Gate",
            strands=[],
            operations=[]
        )
        
        # Add input strands
        circuit.strands.extend([input1, input2])
        circuit.input_gates.extend([input1.strand_id, input2.strand_id])
        
        # Create output strand
        output_strand = self.strand_generator.generate_strand(20, 0.5)
        circuit.strands.append(output_strand)
        circuit.output_gates.append(output_strand.strand_id)
        
        # Add hybridization operation
        hybridization = self.operation_engine.hybridize(input1, input2)
        circuit.operations.append(hybridization)
        
        return circuit
    
    def create_or_gate(self, input1: DNAStrand, input2: DNAStrand) -> DNACircuit:
        """Create a DNA OR gate."""
        circuit = DNACircuit(
            circuit_id=f"or_gate_{int(time.time())}_{random.randint(1000, 9999)}",
            name="OR Gate",
            strands=[],
            operations=[]
        )
        
        # Add input strands
        circuit.strands.extend([input1, input2])
        circuit.input_gates.extend([input1.strand_id, input2.strand_id])
        
        # Create output strand
        output_strand = self.strand_generator.generate_strand(20, 0.5)
        circuit.strands.append(output_strand)
        circuit.output_gates.append(output_strand.strand_id)
        
        # Add hybridization operations for both inputs
        hybridization1 = self.operation_engine.hybridize(input1, output_strand)
        hybridization2 = self.operation_engine.hybridize(input2, output_strand)
        circuit.operations.extend([hybridization1, hybridization2])
        
        return circuit
    
    def create_not_gate(self, input_strand: DNAStrand) -> DNACircuit:
        """Create a DNA NOT gate."""
        circuit = DNACircuit(
            circuit_id=f"not_gate_{int(time.time())}_{random.randint(1000, 9999)}",
            name="NOT Gate",
            strands=[],
            operations=[]
        )
        
        # Add input strand
        circuit.strands.append(input_strand)
        circuit.input_gates.append(input_strand.strand_id)
        
        # Create complementary strand
        complement = self.strand_generator.generate_complementary_strand(input_strand)
        circuit.strands.append(complement)
        circuit.output_gates.append(complement.strand_id)
        
        # Add hybridization operation
        hybridization = self.operation_engine.hybridize(input_strand, complement)
        circuit.operations.append(hybridization)
        
        return circuit

class DNATestFramework:
    """Main DNA computing test framework."""
    
    def __init__(self):
        self.strand_generator = DNAStrandGenerator()
        self.operation_engine = DNAOperationEngine()
        self.circuit_builder = DNACircuitBuilder()
        self.test_results = []
    
    def test_dna_hybridization(self, num_tests: int = 100) -> DNATestResult:
        """Test DNA hybridization performance."""
        success_count = 0
        total_duration = 0.0
        success_rates = []
        
        for _ in range(num_tests):
            # Generate random strands
            strand1 = self.strand_generator.generate_strand(random.randint(10, 30))
            strand2 = self.strand_generator.generate_strand(strand1.length)
            
            # Perform hybridization
            operation = self.operation_engine.hybridize(strand1, strand2)
            
            if operation.success_rate > 0.5:
                success_count += 1
            
            total_duration += operation.duration
            success_rates.append(operation.success_rate)
        
        # Calculate metrics
        success_rate = success_count / num_tests
        avg_duration = total_duration / num_tests
        avg_success_rate = np.mean(success_rates)
        
        dna_metrics = {
            "total_tests": num_tests,
            "successful_tests": success_count,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "average_success_rate": avg_success_rate
        }
        
        result = DNATestResult(
            result_id=f"hybridization_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="DNA Hybridization Test",
            test_type="hybridization",
            success=success_rate > 0.7,
            dna_metrics=dna_metrics,
            molecular_metrics={},
            biochemical_metrics={}
        )
        
        self.test_results.append(result)
        return result
    
    def test_pcr_amplification(self, num_tests: int = 50) -> DNATestResult:
        """Test PCR amplification performance."""
        success_count = 0
        total_duration = 0.0
        amplification_factors = []
        
        for _ in range(num_tests):
            # Generate template and primers
            template = self.strand_generator.generate_strand(random.randint(100, 500))
            primer1 = self.strand_generator.generate_strand(random.randint(15, 25))
            primer2 = self.strand_generator.generate_strand(random.randint(15, 25))
            
            # Perform PCR
            operation = self.operation_engine.pcr_amplify(template, (primer1, primer2))
            
            if operation.success_rate > 0.7:
                success_count += 1
            
            total_duration += operation.duration
            amplification_factors.append(len(operation.output_strands))
        
        # Calculate metrics
        success_rate = success_count / num_tests
        avg_duration = total_duration / num_tests
        avg_amplification = np.mean(amplification_factors)
        
        molecular_metrics = {
            "total_tests": num_tests,
            "successful_tests": success_count,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "average_amplification": avg_amplification
        }
        
        result = DNATestResult(
            result_id=f"pcr_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="PCR Amplification Test",
            test_type="pcr",
            success=success_rate > 0.8,
            dna_metrics={},
            molecular_metrics=molecular_metrics,
            biochemical_metrics={}
        )
        
        self.test_results.append(result)
        return result
    
    def test_dna_circuit(self, circuit_type: str = "and") -> DNATestResult:
        """Test DNA circuit performance."""
        # Generate input strands
        input1 = self.strand_generator.generate_strand(20)
        input2 = self.strand_generator.generate_strand(20)
        
        # Create circuit
        if circuit_type == "and":
            circuit = self.circuit_builder.create_and_gate(input1, input2)
        elif circuit_type == "or":
            circuit = self.circuit_builder.create_or_gate(input1, input2)
        elif circuit_type == "not":
            circuit = self.circuit_builder.create_not_gate(input1)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Test circuit
        successful_operations = 0
        total_duration = 0.0
        
        for operation in circuit.operations:
            if operation.success_rate > 0.5:
                successful_operations += 1
            total_duration += operation.duration
        
        # Calculate metrics
        operation_success_rate = successful_operations / len(circuit.operations)
        avg_duration = total_duration / len(circuit.operations)
        
        biochemical_metrics = {
            "circuit_type": circuit_type,
            "total_operations": len(circuit.operations),
            "successful_operations": successful_operations,
            "operation_success_rate": operation_success_rate,
            "average_duration": avg_duration,
            "total_strands": len(circuit.strands)
        }
        
        result = DNATestResult(
            result_id=f"circuit_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name=f"DNA {circuit_type.upper()} Circuit Test",
            test_type="circuit",
            success=operation_success_rate > 0.7,
            dna_metrics={},
            molecular_metrics={},
            biochemical_metrics=biochemical_metrics
        )
        
        self.test_results.append(result)
        return result
    
    def test_gel_electrophoresis(self, num_strands: int = 10) -> DNATestResult:
        """Test gel electrophoresis performance."""
        # Generate strands of different lengths
        strands = []
        for i in range(num_strands):
            length = random.randint(10, 100)
            strand = self.strand_generator.generate_strand(length)
            strands.append(strand)
        
        # Perform gel electrophoresis
        operation = self.operation_engine.gel_electrophoresis(strands)
        
        # Calculate metrics
        separation_quality = len(operation.output_strands) / len(strands)
        migration_efficiency = operation.duration / len(strands)
        
        molecular_metrics = {
            "total_strands": len(strands),
            "separated_strands": len(operation.output_strands),
            "separation_quality": separation_quality,
            "migration_efficiency": migration_efficiency,
            "duration": operation.duration
        }
        
        result = DNATestResult(
            result_id=f"gel_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Gel Electrophoresis Test",
            test_type="gel_electrophoresis",
            success=separation_quality > 0.9,
            dna_metrics={},
            molecular_metrics=molecular_metrics,
            biochemical_metrics={}
        )
        
        self.test_results.append(result)
        return result
    
    def generate_dna_report(self) -> Dict[str, Any]:
        """Generate comprehensive DNA computing test report."""
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
        performance_analysis = self._analyze_dna_performance()
        
        # Generate recommendations
        recommendations = self._generate_dna_recommendations()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0
            },
            "by_test_type": {test_type: len(results) for test_type, results in test_types.items()},
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "detailed_results": [r.__dict__ for r in self.test_results]
        }
    
    def _analyze_dna_performance(self) -> Dict[str, Any]:
        """Analyze DNA computing performance."""
        all_metrics = []
        
        for result in self.test_results:
            all_metrics.extend(result.dna_metrics.values())
            all_metrics.extend(result.molecular_metrics.values())
            all_metrics.extend(result.biochemical_metrics.values())
        
        if not all_metrics:
            return {}
        
        return {
            "average_metric": np.mean(all_metrics),
            "metric_std": np.std(all_metrics),
            "min_metric": np.min(all_metrics),
            "max_metric": np.max(all_metrics)
        }
    
    def _generate_dna_recommendations(self) -> List[str]:
        """Generate DNA computing specific recommendations."""
        recommendations = []
        
        # Analyze hybridization results
        hybridization_results = [r for r in self.test_results if r.test_type == "hybridization"]
        if hybridization_results:
            avg_success = np.mean([r.dna_metrics.get('success_rate', 0) for r in hybridization_results])
            if avg_success < 0.8:
                recommendations.append("Optimize hybridization conditions for better success rates")
        
        # Analyze PCR results
        pcr_results = [r for r in self.test_results if r.test_type == "pcr"]
        if pcr_results:
            avg_success = np.mean([r.molecular_metrics.get('success_rate', 0) for r in pcr_results])
            if avg_success < 0.9:
                recommendations.append("Improve PCR conditions and primer design")
        
        # Analyze circuit results
        circuit_results = [r for r in self.test_results if r.test_type == "circuit"]
        if circuit_results:
            avg_success = np.mean([r.biochemical_metrics.get('operation_success_rate', 0) for r in circuit_results])
            if avg_success < 0.8:
                recommendations.append("Enhance DNA circuit design for better reliability")
        
        return recommendations

# Example usage and demo
def demo_dna_computing_testing():
    """Demonstrate DNA computing testing capabilities."""
    print("🧬 DNA Computing Testing Framework Demo")
    print("=" * 50)
    
    # Create DNA computing test framework
    framework = DNATestFramework()
    
    # Run comprehensive tests
    print("🧪 Running DNA computing tests...")
    
    # Test DNA hybridization
    print("\n🔗 Testing DNA hybridization...")
    hybridization_result = framework.test_dna_hybridization(num_tests=50)
    print(f"Hybridization Test: {'✅' if hybridization_result.success else '❌'}")
    print(f"  Success Rate: {hybridization_result.dna_metrics.get('success_rate', 0):.1%}")
    print(f"  Average Duration: {hybridization_result.dna_metrics.get('average_duration', 0):.3f}s")
    
    # Test PCR amplification
    print("\n🔄 Testing PCR amplification...")
    pcr_result = framework.test_pcr_amplification(num_tests=25)
    print(f"PCR Test: {'✅' if pcr_result.success else '❌'}")
    print(f"  Success Rate: {pcr_result.molecular_metrics.get('success_rate', 0):.1%}")
    print(f"  Average Amplification: {pcr_result.molecular_metrics.get('average_amplification', 0):.0f}x")
    
    # Test DNA circuits
    print("\n⚡ Testing DNA circuits...")
    for circuit_type in ["and", "or", "not"]:
        circuit_result = framework.test_dna_circuit(circuit_type)
        print(f"{circuit_type.upper()} Circuit: {'✅' if circuit_result.success else '❌'}")
        print(f"  Operation Success: {circuit_result.biochemical_metrics.get('operation_success_rate', 0):.1%}")
    
    # Test gel electrophoresis
    print("\n📊 Testing gel electrophoresis...")
    gel_result = framework.test_gel_electrophoresis(num_strands=8)
    print(f"Gel Electrophoresis: {'✅' if gel_result.success else '❌'}")
    print(f"  Separation Quality: {gel_result.molecular_metrics.get('separation_quality', 0):.1%}")
    print(f"  Duration: {gel_result.molecular_metrics.get('duration', 0):.2f}s")
    
    # Generate comprehensive report
    print("\n📈 Generating DNA computing report...")
    report = framework.generate_dna_report()
    
    print(f"\n📊 DNA Computing Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    # Run demo
    demo_dna_computing_testing()
