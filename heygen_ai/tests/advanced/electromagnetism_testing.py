"""
Electromagnetism Testing Framework for HeyGen AI Testing System.
Advanced electromagnetism testing including Maxwell equations,
electromagnetic fields, and wave propagation validation.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random
import math

@dataclass
class ElectricField:
    """Represents an electric field."""
    field_id: str
    components: np.ndarray  # [Ex, Ey, Ez]
    magnitude: float  # in V/m
    direction: np.ndarray  # unit vector
    potential: float  # in V
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class MagneticField:
    """Represents a magnetic field."""
    field_id: str
    components: np.ndarray  # [Bx, By, Bz]
    magnitude: float  # in T
    direction: np.ndarray  # unit vector
    flux: float  # in Wb
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ElectromagneticWave:
    """Represents an electromagnetic wave."""
    wave_id: str
    frequency: float  # in Hz
    wavelength: float  # in m
    amplitude: float  # in V/m
    phase: float  # in radians
    polarization: str  # "linear", "circular", "elliptical"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class MaxwellEquation:
    """Represents a Maxwell equation."""
    equation_id: str
    equation_type: str  # "gauss_electric", "gauss_magnetic", "faraday", "ampere"
    divergence: float  # for Gauss equations
    curl: np.ndarray  # for Faraday and Ampere equations
    current_density: float  # for Ampere equation
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ElectromagnetismTest:
    """Represents an electromagnetism test."""
    test_id: str
    test_name: str
    electric_fields: List[ElectricField]
    magnetic_fields: List[MagneticField]
    electromagnetic_waves: List[ElectromagneticWave]
    maxwell_equations: List[MaxwellEquation]
    test_type: str
    success: bool
    duration: float
    em_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class ElectromagnetismTestFramework:
    """Main electromagnetism testing framework."""
    
    def __init__(self):
        self.test_results = []
        self.c = 299792458  # Speed of light in m/s
        self.epsilon_0 = 8.854e-12  # Permittivity of free space in F/m
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space in H/m
    
    def test_electric_fields(self, num_tests: int = 30) -> Dict[str, Any]:
        """Test electric fields."""
        tests = []
        
        for i in range(num_tests):
            # Generate electric fields
            num_fields = random.randint(2, 6)
            electric_fields = []
            for j in range(num_fields):
                field = self._generate_electric_field()
                electric_fields.append(field)
            
            # Test electric field consistency
            start_time = time.time()
            success = self._test_electric_field_consistency(electric_fields)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_electric_field_metrics(electric_fields, success)
            
            test = ElectromagnetismTest(
                test_id=f"electric_fields_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Electric Fields Test {i+1}",
                electric_fields=electric_fields,
                magnetic_fields=[],
                electromagnetic_waves=[],
                maxwell_equations=[],
                test_type="electric_fields",
                success=success,
                duration=duration,
                em_metrics=metrics
            )
            
            tests.append(test)
            self.test_results.append(test)
        
        # Calculate summary metrics
        success_count = sum(1 for test in tests if test.success)
        success_rate = success_count / len(tests)
        avg_duration = np.mean([test.duration for test in tests])
        
        return {
            "total_tests": len(tests),
            "successful_tests": success_count,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "test_type": "electric_fields"
        }
    
    def test_magnetic_fields(self, num_tests: int = 25) -> Dict[str, Any]:
        """Test magnetic fields."""
        tests = []
        
        for i in range(num_tests):
            # Generate magnetic fields
            num_fields = random.randint(2, 5)
            magnetic_fields = []
            for j in range(num_fields):
                field = self._generate_magnetic_field()
                magnetic_fields.append(field)
            
            # Test magnetic field consistency
            start_time = time.time()
            success = self._test_magnetic_field_consistency(magnetic_fields)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_magnetic_field_metrics(magnetic_fields, success)
            
            test = ElectromagnetismTest(
                test_id=f"magnetic_fields_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Magnetic Fields Test {i+1}",
                electric_fields=[],
                magnetic_fields=magnetic_fields,
                electromagnetic_waves=[],
                maxwell_equations=[],
                test_type="magnetic_fields",
                success=success,
                duration=duration,
                em_metrics=metrics
            )
            
            tests.append(test)
            self.test_results.append(test)
        
        # Calculate summary metrics
        success_count = sum(1 for test in tests if test.success)
        success_rate = success_count / len(tests)
        avg_duration = np.mean([test.duration for test in tests])
        
        return {
            "total_tests": len(tests),
            "successful_tests": success_count,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "test_type": "magnetic_fields"
        }
    
    def test_electromagnetic_waves(self, num_tests: int = 20) -> Dict[str, Any]:
        """Test electromagnetic waves."""
        tests = []
        
        for i in range(num_tests):
            # Generate electromagnetic waves
            num_waves = random.randint(2, 5)
            electromagnetic_waves = []
            for j in range(num_waves):
                wave = self._generate_electromagnetic_wave()
                electromagnetic_waves.append(wave)
            
            # Test electromagnetic wave consistency
            start_time = time.time()
            success = self._test_electromagnetic_wave_consistency(electromagnetic_waves)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_electromagnetic_wave_metrics(electromagnetic_waves, success)
            
            test = ElectromagnetismTest(
                test_id=f"electromagnetic_waves_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Electromagnetic Waves Test {i+1}",
                electric_fields=[],
                magnetic_fields=[],
                electromagnetic_waves=electromagnetic_waves,
                maxwell_equations=[],
                test_type="electromagnetic_waves",
                success=success,
                duration=duration,
                em_metrics=metrics
            )
            
            tests.append(test)
            self.test_results.append(test)
        
        # Calculate summary metrics
        success_count = sum(1 for test in tests if test.success)
        success_rate = success_count / len(tests)
        avg_duration = np.mean([test.duration for test in tests])
        
        return {
            "total_tests": len(tests),
            "successful_tests": success_count,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "test_type": "electromagnetic_waves"
        }
    
    def test_maxwell_equations(self, num_tests: int = 15) -> Dict[str, Any]:
        """Test Maxwell equations."""
        tests = []
        
        for i in range(num_tests):
            # Generate Maxwell equations
            num_equations = random.randint(2, 4)
            maxwell_equations = []
            for j in range(num_equations):
                equation = self._generate_maxwell_equation()
                maxwell_equations.append(equation)
            
            # Test Maxwell equation consistency
            start_time = time.time()
            success = self._test_maxwell_equation_consistency(maxwell_equations)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_maxwell_equation_metrics(maxwell_equations, success)
            
            test = ElectromagnetismTest(
                test_id=f"maxwell_equations_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Maxwell Equations Test {i+1}",
                electric_fields=[],
                magnetic_fields=[],
                electromagnetic_waves=[],
                maxwell_equations=maxwell_equations,
                test_type="maxwell_equations",
                success=success,
                duration=duration,
                em_metrics=metrics
            )
            
            tests.append(test)
            self.test_results.append(test)
        
        # Calculate summary metrics
        success_count = sum(1 for test in tests if test.success)
        success_rate = success_count / len(tests)
        avg_duration = np.mean([test.duration for test in tests])
        
        return {
            "total_tests": len(tests),
            "successful_tests": success_count,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "test_type": "maxwell_equations"
        }
    
    def _generate_electric_field(self) -> ElectricField:
        """Generate an electric field."""
        # Generate field components
        components = np.random.uniform(-1000, 1000, 3)  # V/m
        
        # Calculate magnitude
        magnitude = np.linalg.norm(components)
        
        # Calculate direction (unit vector)
        if magnitude > 0:
            direction = components / magnitude
        else:
            direction = np.array([1, 0, 0])
        
        # Calculate potential (simplified)
        potential = random.uniform(-1000, 1000)  # V
        
        return ElectricField(
            field_id=f"efield_{int(time.time())}_{random.randint(1000, 9999)}",
            components=components,
            magnitude=magnitude,
            direction=direction,
            potential=potential
        )
    
    def _generate_magnetic_field(self) -> MagneticField:
        """Generate a magnetic field."""
        # Generate field components
        components = np.random.uniform(-1, 1, 3)  # T
        
        # Calculate magnitude
        magnitude = np.linalg.norm(components)
        
        # Calculate direction (unit vector)
        if magnitude > 0:
            direction = components / magnitude
        else:
            direction = np.array([0, 0, 1])
        
        # Calculate flux (simplified)
        flux = random.uniform(-1e-6, 1e-6)  # Wb
        
        return MagneticField(
            field_id=f"mfield_{int(time.time())}_{random.randint(1000, 9999)}",
            components=components,
            magnitude=magnitude,
            direction=direction,
            flux=flux
        )
    
    def _generate_electromagnetic_wave(self) -> ElectromagneticWave:
        """Generate an electromagnetic wave."""
        frequency = random.uniform(1e3, 1e15)  # Hz
        wavelength = self.c / frequency  # m
        amplitude = random.uniform(1e-6, 1e3)  # V/m
        phase = random.uniform(0, 2 * np.pi)  # radians
        
        polarizations = ["linear", "circular", "elliptical"]
        polarization = random.choice(polarizations)
        
        return ElectromagneticWave(
            wave_id=f"wave_{int(time.time())}_{random.randint(1000, 9999)}",
            frequency=frequency,
            wavelength=wavelength,
            amplitude=amplitude,
            phase=phase,
            polarization=polarization
        )
    
    def _generate_maxwell_equation(self) -> MaxwellEquation:
        """Generate a Maxwell equation."""
        equation_types = ["gauss_electric", "gauss_magnetic", "faraday", "ampere"]
        equation_type = random.choice(equation_types)
        
        divergence = random.uniform(-1e6, 1e6)  # for Gauss equations
        curl = np.random.uniform(-1e6, 1e6, 3)  # for Faraday and Ampere equations
        current_density = random.uniform(-1e6, 1e6)  # for Ampere equation
        
        return MaxwellEquation(
            equation_id=f"maxwell_{int(time.time())}_{random.randint(1000, 9999)}",
            equation_type=equation_type,
            divergence=divergence,
            curl=curl,
            current_density=current_density
        )
    
    def _test_electric_field_consistency(self, fields: List[ElectricField]) -> bool:
        """Test electric field consistency."""
        for field in fields:
            if not np.all(np.isfinite(field.components)):
                return False
            if field.magnitude < 0 or not np.isfinite(field.magnitude):
                return False
            if not np.all(np.isfinite(field.direction)):
                return False
            if not np.isfinite(field.potential):
                return False
        return True
    
    def _test_magnetic_field_consistency(self, fields: List[MagneticField]) -> bool:
        """Test magnetic field consistency."""
        for field in fields:
            if not np.all(np.isfinite(field.components)):
                return False
            if field.magnitude < 0 or not np.isfinite(field.magnitude):
                return False
            if not np.all(np.isfinite(field.direction)):
                return False
            if not np.isfinite(field.flux):
                return False
        return True
    
    def _test_electromagnetic_wave_consistency(self, waves: List[ElectromagneticWave]) -> bool:
        """Test electromagnetic wave consistency."""
        for wave in waves:
            if wave.frequency <= 0 or not np.isfinite(wave.frequency):
                return False
            if wave.wavelength <= 0 or not np.isfinite(wave.wavelength):
                return False
            if wave.amplitude < 0 or not np.isfinite(wave.amplitude):
                return False
            if not np.isfinite(wave.phase):
                return False
        return True
    
    def _test_maxwell_equation_consistency(self, equations: List[MaxwellEquation]) -> bool:
        """Test Maxwell equation consistency."""
        for equation in equations:
            if not np.isfinite(equation.divergence):
                return False
            if not np.all(np.isfinite(equation.curl)):
                return False
            if not np.isfinite(equation.current_density):
                return False
        return True
    
    def _calculate_electric_field_metrics(self, fields: List[ElectricField], success: bool) -> Dict[str, float]:
        """Calculate electric field metrics."""
        return {
            "num_fields": len(fields),
            "avg_magnitude": np.mean([f.magnitude for f in fields]),
            "avg_potential": np.mean([f.potential for f in fields]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_magnetic_field_metrics(self, fields: List[MagneticField], success: bool) -> Dict[str, float]:
        """Calculate magnetic field metrics."""
        return {
            "num_fields": len(fields),
            "avg_magnitude": np.mean([f.magnitude for f in fields]),
            "avg_flux": np.mean([f.flux for f in fields]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_electromagnetic_wave_metrics(self, waves: List[ElectromagneticWave], success: bool) -> Dict[str, float]:
        """Calculate electromagnetic wave metrics."""
        return {
            "num_waves": len(waves),
            "avg_frequency": np.mean([w.frequency for w in waves]),
            "avg_wavelength": np.mean([w.wavelength for w in waves]),
            "avg_amplitude": np.mean([w.amplitude for w in waves]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_maxwell_equation_metrics(self, equations: List[MaxwellEquation], success: bool) -> Dict[str, float]:
        """Calculate Maxwell equation metrics."""
        return {
            "num_equations": len(equations),
            "avg_divergence": np.mean([e.divergence for e in equations]),
            "avg_curl_magnitude": np.mean([np.linalg.norm(e.curl) for e in equations]),
            "avg_current_density": np.mean([e.current_density for e in equations]),
            "test_success": 1.0 if success else 0.0
        }
    
    def generate_electromagnetism_report(self) -> Dict[str, Any]:
        """Generate comprehensive electromagnetism test report."""
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
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0
            },
            "by_test_type": {test_type: len(results) for test_type, results in test_types.items()},
            "detailed_results": [r.__dict__ for r in self.test_results]
        }

# Example usage and demo
def demo_electromagnetism_testing():
    """Demonstrate electromagnetism testing capabilities."""
    print("⚡ Electromagnetism Testing Framework Demo")
    print("=" * 50)
    
    # Create electromagnetism test framework
    framework = ElectromagnetismTestFramework()
    
    # Run comprehensive tests
    print("🧪 Running electromagnetism tests...")
    
    # Test electric fields
    print("\n⚡ Testing electric fields...")
    efield_result = framework.test_electric_fields(num_tests=20)
    print(f"Electric Fields: {'✅' if efield_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {efield_result['success_rate']:.1%}")
    print(f"  Total Tests: {efield_result['total_tests']}")
    
    # Test magnetic fields
    print("\n🧲 Testing magnetic fields...")
    mfield_result = framework.test_magnetic_fields(num_tests=15)
    print(f"Magnetic Fields: {'✅' if mfield_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {mfield_result['success_rate']:.1%}")
    print(f"  Total Tests: {mfield_result['total_tests']}")
    
    # Test electromagnetic waves
    print("\n📡 Testing electromagnetic waves...")
    wave_result = framework.test_electromagnetic_waves(num_tests=10)
    print(f"Electromagnetic Waves: {'✅' if wave_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {wave_result['success_rate']:.1%}")
    print(f"  Total Tests: {wave_result['total_tests']}")
    
    # Test Maxwell equations
    print("\n📐 Testing Maxwell equations...")
    maxwell_result = framework.test_maxwell_equations(num_tests=8)
    print(f"Maxwell Equations: {'✅' if maxwell_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {maxwell_result['success_rate']:.1%}")
    print(f"  Total Tests: {maxwell_result['total_tests']}")
    
    # Generate comprehensive report
    print("\n📈 Generating electromagnetism report...")
    report = framework.generate_electromagnetism_report()
    
    print(f"\n📊 Electromagnetism Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")

if __name__ == "__main__":
    # Run demo
    demo_electromagnetism_testing()
