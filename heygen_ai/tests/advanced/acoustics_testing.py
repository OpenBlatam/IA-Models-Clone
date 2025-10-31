"""
Acoustics Testing Framework for HeyGen AI Testing System.
Advanced acoustics testing including sound propagation, wave analysis,
and acoustic systems validation.
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
class SoundWave:
    """Represents a sound wave."""
    wave_id: str
    frequency: float  # in Hz
    wavelength: float  # in m
    amplitude: float  # in Pa
    phase: float  # in radians
    sound_pressure_level: float  # in dB
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AcousticMedium:
    """Represents an acoustic medium."""
    medium_id: str
    density: float  # in kg/m³
    sound_speed: float  # in m/s
    impedance: float  # in Pa·s/m
    absorption_coefficient: float  # 0-1
    temperature: float  # in K
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AcousticSystem:
    """Represents an acoustic system."""
    system_id: str
    medium: AcousticMedium
    waves: List[SoundWave]
    resonance_frequencies: List[float]  # in Hz
    bandwidth: float  # in Hz
    quality_factor: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AcousticTest:
    """Represents an acoustics test."""
    test_id: str
    test_name: str
    sound_waves: List[SoundWave]
    acoustic_mediums: List[AcousticMedium]
    acoustic_systems: List[AcousticSystem]
    test_type: str
    success: bool
    duration: float
    acoustics_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class AcousticsTestFramework:
    """Main acoustics testing framework."""
    
    def __init__(self):
        self.test_results = []
        self.c_air = 343  # Speed of sound in air at 20°C in m/s
        self.rho_air = 1.225  # Air density at sea level in kg/m³
        self.p_ref = 20e-6  # Reference sound pressure in Pa
    
    def test_sound_waves(self, num_tests: int = 30) -> Dict[str, Any]:
        """Test sound waves."""
        tests = []
        
        for i in range(num_tests):
            # Generate sound waves
            num_waves = random.randint(3, 8)
            sound_waves = []
            for j in range(num_waves):
                wave = self._generate_sound_wave()
                sound_waves.append(wave)
            
            # Test sound wave consistency
            start_time = time.time()
            success = self._test_sound_wave_consistency(sound_waves)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_sound_wave_metrics(sound_waves, success)
            
            test = AcousticTest(
                test_id=f"sound_waves_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Sound Waves Test {i+1}",
                sound_waves=sound_waves,
                acoustic_mediums=[],
                acoustic_systems=[],
                test_type="sound_waves",
                success=success,
                duration=duration,
                acoustics_metrics=metrics
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
            "test_type": "sound_waves"
        }
    
    def test_acoustic_mediums(self, num_tests: int = 25) -> Dict[str, Any]:
        """Test acoustic mediums."""
        tests = []
        
        for i in range(num_tests):
            # Generate acoustic mediums
            num_mediums = random.randint(2, 5)
            acoustic_mediums = []
            for j in range(num_mediums):
                medium = self._generate_acoustic_medium()
                acoustic_mediums.append(medium)
            
            # Test acoustic medium consistency
            start_time = time.time()
            success = self._test_acoustic_medium_consistency(acoustic_mediums)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_acoustic_medium_metrics(acoustic_mediums, success)
            
            test = AcousticTest(
                test_id=f"acoustic_mediums_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Acoustic Mediums Test {i+1}",
                sound_waves=[],
                acoustic_mediums=acoustic_mediums,
                acoustic_systems=[],
                test_type="acoustic_mediums",
                success=success,
                duration=duration,
                acoustics_metrics=metrics
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
            "test_type": "acoustic_mediums"
        }
    
    def test_acoustic_systems(self, num_tests: int = 20) -> Dict[str, Any]:
        """Test acoustic systems."""
        tests = []
        
        for i in range(num_tests):
            # Generate acoustic systems
            num_systems = random.randint(2, 4)
            acoustic_systems = []
            for j in range(num_systems):
                system = self._generate_acoustic_system()
                acoustic_systems.append(system)
            
            # Test acoustic system consistency
            start_time = time.time()
            success = self._test_acoustic_system_consistency(acoustic_systems)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_acoustic_system_metrics(acoustic_systems, success)
            
            test = AcousticTest(
                test_id=f"acoustic_systems_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Acoustic Systems Test {i+1}",
                sound_waves=[],
                acoustic_mediums=[],
                acoustic_systems=acoustic_systems,
                test_type="acoustic_systems",
                success=success,
                duration=duration,
                acoustics_metrics=metrics
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
            "test_type": "acoustic_systems"
        }
    
    def _generate_sound_wave(self) -> SoundWave:
        """Generate a sound wave."""
        frequency = random.uniform(20, 20000)  # Hz (audible range)
        wavelength = self.c_air / frequency  # m
        amplitude = random.uniform(1e-6, 100)  # Pa
        phase = random.uniform(0, 2 * np.pi)  # radians
        
        # Calculate sound pressure level
        spl = 20 * np.log10(amplitude / self.p_ref)  # dB
        
        return SoundWave(
            wave_id=f"wave_{int(time.time())}_{random.randint(1000, 9999)}",
            frequency=frequency,
            wavelength=wavelength,
            amplitude=amplitude,
            phase=phase,
            sound_pressure_level=spl
        )
    
    def _generate_acoustic_medium(self) -> AcousticMedium:
        """Generate an acoustic medium."""
        density = random.uniform(0.1, 10000)  # kg/m³
        sound_speed = random.uniform(100, 6000)  # m/s
        impedance = density * sound_speed  # Pa·s/m
        absorption_coefficient = random.uniform(0, 1)
        temperature = random.uniform(200, 3000)  # K
        
        return AcousticMedium(
            medium_id=f"medium_{int(time.time())}_{random.randint(1000, 9999)}",
            density=density,
            sound_speed=sound_speed,
            impedance=impedance,
            absorption_coefficient=absorption_coefficient,
            temperature=temperature
        )
    
    def _generate_acoustic_system(self) -> AcousticSystem:
        """Generate an acoustic system."""
        # Generate medium
        medium = self._generate_acoustic_medium()
        
        # Generate waves
        num_waves = random.randint(2, 5)
        waves = [self._generate_sound_wave() for _ in range(num_waves)]
        
        # Calculate resonance frequencies
        resonance_frequencies = [random.uniform(100, 5000) for _ in range(random.randint(1, 5))]
        
        # Calculate bandwidth
        bandwidth = random.uniform(10, 1000)  # Hz
        
        # Calculate quality factor
        quality_factor = random.uniform(1, 100)
        
        return AcousticSystem(
            system_id=f"system_{int(time.time())}_{random.randint(1000, 9999)}",
            medium=medium,
            waves=waves,
            resonance_frequencies=resonance_frequencies,
            bandwidth=bandwidth,
            quality_factor=quality_factor
        )
    
    def _test_sound_wave_consistency(self, waves: List[SoundWave]) -> bool:
        """Test sound wave consistency."""
        for wave in waves:
            if wave.frequency <= 0 or not np.isfinite(wave.frequency):
                return False
            if wave.wavelength <= 0 or not np.isfinite(wave.wavelength):
                return False
            if wave.amplitude <= 0 or not np.isfinite(wave.amplitude):
                return False
            if not np.isfinite(wave.phase):
                return False
            if not np.isfinite(wave.sound_pressure_level):
                return False
        return True
    
    def _test_acoustic_medium_consistency(self, mediums: List[AcousticMedium]) -> bool:
        """Test acoustic medium consistency."""
        for medium in mediums:
            if medium.density <= 0 or not np.isfinite(medium.density):
                return False
            if medium.sound_speed <= 0 or not np.isfinite(medium.sound_speed):
                return False
            if medium.impedance <= 0 or not np.isfinite(medium.impedance):
                return False
            if not (0 <= medium.absorption_coefficient <= 1) or not np.isfinite(medium.absorption_coefficient):
                return False
            if medium.temperature <= 0 or not np.isfinite(medium.temperature):
                return False
        return True
    
    def _test_acoustic_system_consistency(self, systems: List[AcousticSystem]) -> bool:
        """Test acoustic system consistency."""
        for system in systems:
            if not self._test_acoustic_medium_consistency([system.medium]):
                return False
            if not self._test_sound_wave_consistency(system.waves):
                return False
            if system.bandwidth <= 0 or not np.isfinite(system.bandwidth):
                return False
            if system.quality_factor <= 0 or not np.isfinite(system.quality_factor):
                return False
        return True
    
    def _calculate_sound_wave_metrics(self, waves: List[SoundWave], success: bool) -> Dict[str, float]:
        """Calculate sound wave metrics."""
        return {
            "num_waves": len(waves),
            "avg_frequency": np.mean([w.frequency for w in waves]),
            "avg_wavelength": np.mean([w.wavelength for w in waves]),
            "avg_amplitude": np.mean([w.amplitude for w in waves]),
            "avg_spl": np.mean([w.sound_pressure_level for w in waves]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_acoustic_medium_metrics(self, mediums: List[AcousticMedium], success: bool) -> Dict[str, float]:
        """Calculate acoustic medium metrics."""
        return {
            "num_mediums": len(mediums),
            "avg_density": np.mean([m.density for m in mediums]),
            "avg_sound_speed": np.mean([m.sound_speed for m in mediums]),
            "avg_impedance": np.mean([m.impedance for m in mediums]),
            "avg_absorption": np.mean([m.absorption_coefficient for m in mediums]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_acoustic_system_metrics(self, systems: List[AcousticSystem], success: bool) -> Dict[str, float]:
        """Calculate acoustic system metrics."""
        return {
            "num_systems": len(systems),
            "avg_bandwidth": np.mean([s.bandwidth for s in systems]),
            "avg_quality_factor": np.mean([s.quality_factor for s in systems]),
            "avg_num_resonances": np.mean([len(s.resonance_frequencies) for s in systems]),
            "test_success": 1.0 if success else 0.0
        }
    
    def generate_acoustics_report(self) -> Dict[str, Any]:
        """Generate comprehensive acoustics test report."""
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
def demo_acoustics_testing():
    """Demonstrate acoustics testing capabilities."""
    print("🔊 Acoustics Testing Framework Demo")
    print("=" * 50)
    
    # Create acoustics test framework
    framework = AcousticsTestFramework()
    
    # Run comprehensive tests
    print("🧪 Running acoustics tests...")
    
    # Test sound waves
    print("\n🌊 Testing sound waves...")
    wave_result = framework.test_sound_waves(num_tests=20)
    print(f"Sound Waves: {'✅' if wave_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {wave_result['success_rate']:.1%}")
    print(f"  Total Tests: {wave_result['total_tests']}")
    
    # Test acoustic mediums
    print("\n🌬️ Testing acoustic mediums...")
    medium_result = framework.test_acoustic_mediums(num_tests=15)
    print(f"Acoustic Mediums: {'✅' if medium_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {medium_result['success_rate']:.1%}")
    print(f"  Total Tests: {medium_result['total_tests']}")
    
    # Test acoustic systems
    print("\n🎵 Testing acoustic systems...")
    system_result = framework.test_acoustic_systems(num_tests=10)
    print(f"Acoustic Systems: {'✅' if system_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {system_result['success_rate']:.1%}")
    print(f"  Total Tests: {system_result['total_tests']}")
    
    # Generate comprehensive report
    print("\n📈 Generating acoustics report...")
    report = framework.generate_acoustics_report()
    
    print(f"\n📊 Acoustics Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")

if __name__ == "__main__":
    # Run demo
    demo_acoustics_testing()
