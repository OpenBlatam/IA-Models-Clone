"""
Optics Testing Framework for HeyGen AI Testing System.
Advanced optics testing including wave propagation, interference,
and optical systems validation.
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
class LightWave:
    """Represents a light wave."""
    wave_id: str
    wavelength: float  # in nm
    frequency: float  # in Hz
    amplitude: float  # in V/m
    phase: float  # in radians
    polarization: str  # "linear", "circular", "elliptical"
    intensity: float  # in W/m²
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OpticalElement:
    """Represents an optical element."""
    element_id: str
    element_type: str  # "lens", "mirror", "prism", "grating", "filter"
    focal_length: float  # in m
    refractive_index: float
    transmission: float  # 0-1
    reflection: float  # 0-1
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class InterferencePattern:
    """Represents an interference pattern."""
    pattern_id: str
    wave1: LightWave
    wave2: LightWave
    path_difference: float  # in m
    phase_difference: float  # in radians
    visibility: float  # 0-1
    fringe_spacing: float  # in m
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OpticalSystem:
    """Represents an optical system."""
    system_id: str
    elements: List[OpticalElement]
    total_transmission: float  # 0-1
    total_magnification: float
    numerical_aperture: float
    resolution: float  # in m
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OpticsTest:
    """Represents an optics test."""
    test_id: str
    test_name: str
    light_waves: List[LightWave]
    optical_elements: List[OpticalElement]
    interference_patterns: List[InterferencePattern]
    optical_systems: List[OpticalSystem]
    test_type: str
    success: bool
    duration: float
    optics_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class OpticsTestFramework:
    """Main optics testing framework."""
    
    def __init__(self):
        self.test_results = []
        self.c = 299792458  # Speed of light in m/s
        self.h = 6.626e-34  # Planck constant in J·s
        self.epsilon_0 = 8.854e-12  # Permittivity of free space in F/m
    
    def test_light_waves(self, num_tests: int = 30) -> Dict[str, Any]:
        """Test light waves."""
        tests = []
        
        for i in range(num_tests):
            # Generate light waves
            num_waves = random.randint(3, 8)
            light_waves = []
            for j in range(num_waves):
                wave = self._generate_light_wave()
                light_waves.append(wave)
            
            # Test light wave consistency
            start_time = time.time()
            success = self._test_light_wave_consistency(light_waves)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_light_wave_metrics(light_waves, success)
            
            test = OpticsTest(
                test_id=f"light_waves_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Light Waves Test {i+1}",
                light_waves=light_waves,
                optical_elements=[],
                interference_patterns=[],
                optical_systems=[],
                test_type="light_waves",
                success=success,
                duration=duration,
                optics_metrics=metrics
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
            "test_type": "light_waves"
        }
    
    def test_optical_elements(self, num_tests: int = 25) -> Dict[str, Any]:
        """Test optical elements."""
        tests = []
        
        for i in range(num_tests):
            # Generate optical elements
            num_elements = random.randint(2, 6)
            optical_elements = []
            for j in range(num_elements):
                element = self._generate_optical_element()
                optical_elements.append(element)
            
            # Test optical element consistency
            start_time = time.time()
            success = self._test_optical_element_consistency(optical_elements)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_optical_element_metrics(optical_elements, success)
            
            test = OpticsTest(
                test_id=f"optical_elements_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Optical Elements Test {i+1}",
                light_waves=[],
                optical_elements=optical_elements,
                interference_patterns=[],
                optical_systems=[],
                test_type="optical_elements",
                success=success,
                duration=duration,
                optics_metrics=metrics
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
            "test_type": "optical_elements"
        }
    
    def test_interference_patterns(self, num_tests: int = 20) -> Dict[str, Any]:
        """Test interference patterns."""
        tests = []
        
        for i in range(num_tests):
            # Generate interference patterns
            num_patterns = random.randint(2, 5)
            interference_patterns = []
            for j in range(num_patterns):
                pattern = self._generate_interference_pattern()
                interference_patterns.append(pattern)
            
            # Test interference pattern consistency
            start_time = time.time()
            success = self._test_interference_pattern_consistency(interference_patterns)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_interference_pattern_metrics(interference_patterns, success)
            
            test = OpticsTest(
                test_id=f"interference_patterns_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Interference Patterns Test {i+1}",
                light_waves=[],
                optical_elements=[],
                interference_patterns=interference_patterns,
                optical_systems=[],
                test_type="interference_patterns",
                success=success,
                duration=duration,
                optics_metrics=metrics
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
            "test_type": "interference_patterns"
        }
    
    def test_optical_systems(self, num_tests: int = 15) -> Dict[str, Any]:
        """Test optical systems."""
        tests = []
        
        for i in range(num_tests):
            # Generate optical systems
            num_systems = random.randint(2, 4)
            optical_systems = []
            for j in range(num_systems):
                system = self._generate_optical_system()
                optical_systems.append(system)
            
            # Test optical system consistency
            start_time = time.time()
            success = self._test_optical_system_consistency(optical_systems)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_optical_system_metrics(optical_systems, success)
            
            test = OpticsTest(
                test_id=f"optical_systems_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Optical Systems Test {i+1}",
                light_waves=[],
                optical_elements=[],
                interference_patterns=[],
                optical_systems=optical_systems,
                test_type="optical_systems",
                success=success,
                duration=duration,
                optics_metrics=metrics
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
            "test_type": "optical_systems"
        }
    
    def _generate_light_wave(self) -> LightWave:
        """Generate a light wave."""
        wavelength = random.uniform(400, 700)  # nm (visible spectrum)
        frequency = self.c / (wavelength * 1e-9)  # Hz
        amplitude = random.uniform(1e-6, 1e-3)  # V/m
        phase = random.uniform(0, 2 * np.pi)  # radians
        
        polarizations = ["linear", "circular", "elliptical"]
        polarization = random.choice(polarizations)
        
        # Calculate intensity
        intensity = 0.5 * self.epsilon_0 * self.c * amplitude**2  # W/m²
        
        return LightWave(
            wave_id=f"wave_{int(time.time())}_{random.randint(1000, 9999)}",
            wavelength=wavelength,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            polarization=polarization,
            intensity=intensity
        )
    
    def _generate_optical_element(self) -> OpticalElement:
        """Generate an optical element."""
        element_types = ["lens", "mirror", "prism", "grating", "filter"]
        element_type = random.choice(element_types)
        
        focal_length = random.uniform(0.01, 1.0)  # m
        refractive_index = random.uniform(1.0, 2.5)
        transmission = random.uniform(0.1, 0.95)
        reflection = random.uniform(0.05, 0.9)
        
        # Ensure transmission + reflection <= 1
        if transmission + reflection > 1:
            total = transmission + reflection
            transmission = transmission / total * 0.95
            reflection = reflection / total * 0.95
        
        return OpticalElement(
            element_id=f"element_{int(time.time())}_{random.randint(1000, 9999)}",
            element_type=element_type,
            focal_length=focal_length,
            refractive_index=refractive_index,
            transmission=transmission,
            reflection=reflection
        )
    
    def _generate_interference_pattern(self) -> InterferencePattern:
        """Generate an interference pattern."""
        wave1 = self._generate_light_wave()
        wave2 = self._generate_light_wave()
        
        path_difference = random.uniform(0, 1e-6)  # m
        phase_difference = 2 * np.pi * path_difference / wave1.wavelength
        
        # Calculate visibility
        visibility = abs(np.cos(phase_difference / 2))
        
        # Calculate fringe spacing (simplified)
        fringe_spacing = wave1.wavelength * 1e-9 / (2 * np.sin(phase_difference / 2)) if phase_difference != 0 else 1e-6
        
        return InterferencePattern(
            pattern_id=f"pattern_{int(time.time())}_{random.randint(1000, 9999)}",
            wave1=wave1,
            wave2=wave2,
            path_difference=path_difference,
            phase_difference=phase_difference,
            visibility=visibility,
            fringe_spacing=fringe_spacing
        )
    
    def _generate_optical_system(self) -> OpticalSystem:
        """Generate an optical system."""
        num_elements = random.randint(2, 5)
        elements = [self._generate_optical_element() for _ in range(num_elements)]
        
        # Calculate total transmission
        total_transmission = np.prod([e.transmission for e in elements])
        
        # Calculate total magnification (simplified)
        total_magnification = random.uniform(0.1, 10.0)
        
        # Calculate numerical aperture
        numerical_aperture = random.uniform(0.1, 1.4)
        
        # Calculate resolution (simplified)
        resolution = 0.61 * 500e-9 / numerical_aperture  # m (using 500 nm wavelength)
        
        return OpticalSystem(
            system_id=f"system_{int(time.time())}_{random.randint(1000, 9999)}",
            elements=elements,
            total_transmission=total_transmission,
            total_magnification=total_magnification,
            numerical_aperture=numerical_aperture,
            resolution=resolution
        )
    
    def _test_light_wave_consistency(self, waves: List[LightWave]) -> bool:
        """Test light wave consistency."""
        for wave in waves:
            if wave.wavelength <= 0 or not np.isfinite(wave.wavelength):
                return False
            if wave.frequency <= 0 or not np.isfinite(wave.frequency):
                return False
            if wave.amplitude < 0 or not np.isfinite(wave.amplitude):
                return False
            if not np.isfinite(wave.phase):
                return False
            if wave.intensity < 0 or not np.isfinite(wave.intensity):
                return False
        return True
    
    def _test_optical_element_consistency(self, elements: List[OpticalElement]) -> bool:
        """Test optical element consistency."""
        for element in elements:
            if element.focal_length <= 0 or not np.isfinite(element.focal_length):
                return False
            if element.refractive_index < 1.0 or not np.isfinite(element.refractive_index):
                return False
            if not (0 <= element.transmission <= 1) or not np.isfinite(element.transmission):
                return False
            if not (0 <= element.reflection <= 1) or not np.isfinite(element.reflection):
                return False
        return True
    
    def _test_interference_pattern_consistency(self, patterns: List[InterferencePattern]) -> bool:
        """Test interference pattern consistency."""
        for pattern in patterns:
            if not self._test_light_wave_consistency([pattern.wave1, pattern.wave2]):
                return False
            if pattern.path_difference < 0 or not np.isfinite(pattern.path_difference):
                return False
            if not np.isfinite(pattern.phase_difference):
                return False
            if not (0 <= pattern.visibility <= 1) or not np.isfinite(pattern.visibility):
                return False
            if pattern.fringe_spacing <= 0 or not np.isfinite(pattern.fringe_spacing):
                return False
        return True
    
    def _test_optical_system_consistency(self, systems: List[OpticalSystem]) -> bool:
        """Test optical system consistency."""
        for system in systems:
            if not self._test_optical_element_consistency(system.elements):
                return False
            if not (0 <= system.total_transmission <= 1) or not np.isfinite(system.total_transmission):
                return False
            if system.total_magnification <= 0 or not np.isfinite(system.total_magnification):
                return False
            if system.numerical_aperture <= 0 or not np.isfinite(system.numerical_aperture):
                return False
            if system.resolution <= 0 or not np.isfinite(system.resolution):
                return False
        return True
    
    def _calculate_light_wave_metrics(self, waves: List[LightWave], success: bool) -> Dict[str, float]:
        """Calculate light wave metrics."""
        return {
            "num_waves": len(waves),
            "avg_wavelength": np.mean([w.wavelength for w in waves]),
            "avg_frequency": np.mean([w.frequency for w in waves]),
            "avg_amplitude": np.mean([w.amplitude for w in waves]),
            "avg_intensity": np.mean([w.intensity for w in waves]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_optical_element_metrics(self, elements: List[OpticalElement], success: bool) -> Dict[str, float]:
        """Calculate optical element metrics."""
        return {
            "num_elements": len(elements),
            "avg_focal_length": np.mean([e.focal_length for e in elements]),
            "avg_refractive_index": np.mean([e.refractive_index for e in elements]),
            "avg_transmission": np.mean([e.transmission for e in elements]),
            "avg_reflection": np.mean([e.reflection for e in elements]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_interference_pattern_metrics(self, patterns: List[InterferencePattern], success: bool) -> Dict[str, float]:
        """Calculate interference pattern metrics."""
        return {
            "num_patterns": len(patterns),
            "avg_visibility": np.mean([p.visibility for p in patterns]),
            "avg_fringe_spacing": np.mean([p.fringe_spacing for p in patterns]),
            "avg_path_difference": np.mean([p.path_difference for p in patterns]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_optical_system_metrics(self, systems: List[OpticalSystem], success: bool) -> Dict[str, float]:
        """Calculate optical system metrics."""
        return {
            "num_systems": len(systems),
            "avg_total_transmission": np.mean([s.total_transmission for s in systems]),
            "avg_total_magnification": np.mean([s.total_magnification for s in systems]),
            "avg_numerical_aperture": np.mean([s.numerical_aperture for s in systems]),
            "avg_resolution": np.mean([s.resolution for s in systems]),
            "test_success": 1.0 if success else 0.0
        }
    
    def generate_optics_report(self) -> Dict[str, Any]:
        """Generate comprehensive optics test report."""
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
def demo_optics_testing():
    """Demonstrate optics testing capabilities."""
    print("🔬 Optics Testing Framework Demo")
    print("=" * 50)
    
    # Create optics test framework
    framework = OpticsTestFramework()
    
    # Run comprehensive tests
    print("🧪 Running optics tests...")
    
    # Test light waves
    print("\n💡 Testing light waves...")
    wave_result = framework.test_light_waves(num_tests=20)
    print(f"Light Waves: {'✅' if wave_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {wave_result['success_rate']:.1%}")
    print(f"  Total Tests: {wave_result['total_tests']}")
    
    # Test optical elements
    print("\n🔍 Testing optical elements...")
    element_result = framework.test_optical_elements(num_tests=15)
    print(f"Optical Elements: {'✅' if element_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {element_result['success_rate']:.1%}")
    print(f"  Total Tests: {element_result['total_tests']}")
    
    # Test interference patterns
    print("\n🌈 Testing interference patterns...")
    pattern_result = framework.test_interference_patterns(num_tests=10)
    print(f"Interference Patterns: {'✅' if pattern_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {pattern_result['success_rate']:.1%}")
    print(f"  Total Tests: {pattern_result['total_tests']}")
    
    # Test optical systems
    print("\n🔬 Testing optical systems...")
    system_result = framework.test_optical_systems(num_tests=8)
    print(f"Optical Systems: {'✅' if system_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {system_result['success_rate']:.1%}")
    print(f"  Total Tests: {system_result['total_tests']}")
    
    # Generate comprehensive report
    print("\n📈 Generating optics report...")
    report = framework.generate_optics_report()
    
    print(f"\n📊 Optics Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")

if __name__ == "__main__":
    # Run demo
    demo_optics_testing()
