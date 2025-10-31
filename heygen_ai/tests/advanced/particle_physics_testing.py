"""
Particle Physics Testing Framework for HeyGen AI Testing System.
Advanced particle physics testing including particle interactions,
decay processes, and cross-section calculations.
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
class Particle:
    """Represents a fundamental particle."""
    particle_id: str
    name: str
    mass: float  # in GeV/c²
    charge: float  # in elementary charge units
    spin: float  # in units of ħ
    lifetime: float  # in seconds
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ParticleInteraction:
    """Represents a particle interaction."""
    interaction_id: str
    initial_particles: List[Particle]
    final_particles: List[Particle]
    cross_section: float  # in barns
    energy: float  # in GeV
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DecayProcess:
    """Represents a particle decay process."""
    decay_id: str
    parent_particle: Particle
    daughter_particles: List[Particle]
    branching_ratio: float
    decay_width: float  # in GeV
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ParticlePhysicsTest:
    """Represents a particle physics test."""
    test_id: str
    test_name: str
    particles: List[Particle]
    interactions: List[ParticleInteraction]
    decays: List[DecayProcess]
    test_type: str
    success: bool
    duration: float
    pp_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class ParticlePhysicsTestFramework:
    """Main particle physics testing framework."""
    
    def __init__(self):
        self.test_results = []
        self.particle_database = self._initialize_particle_database()
    
    def _initialize_particle_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize particle database with known particles."""
        return {
            "electron": {"mass": 0.000511, "charge": -1, "spin": 0.5, "lifetime": float('inf')},
            "proton": {"mass": 0.938, "charge": 1, "spin": 0.5, "lifetime": float('inf')},
            "neutron": {"mass": 0.940, "charge": 0, "spin": 0.5, "lifetime": 881.5},
            "photon": {"mass": 0, "charge": 0, "spin": 1, "lifetime": float('inf')},
            "muon": {"mass": 0.106, "charge": -1, "spin": 0.5, "lifetime": 2.2e-6},
            "pion": {"mass": 0.140, "charge": 1, "spin": 0, "lifetime": 2.6e-8},
            "kaon": {"mass": 0.494, "charge": 1, "spin": 0, "lifetime": 1.2e-8},
            "tau": {"mass": 1.777, "charge": -1, "spin": 0.5, "lifetime": 2.9e-13}
        }
    
    def test_particle_properties(self, num_tests: int = 30) -> Dict[str, Any]:
        """Test particle properties."""
        tests = []
        
        for i in range(num_tests):
            # Generate particles
            num_particles = random.randint(3, 8)
            particles = []
            for j in range(num_particles):
                particle = self._generate_particle()
                particles.append(particle)
            
            # Test particle properties consistency
            start_time = time.time()
            success = self._test_particle_properties_consistency(particles)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_particle_properties_metrics(particles, success)
            
            test = ParticlePhysicsTest(
                test_id=f"particle_properties_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Particle Properties Test {i+1}",
                particles=particles,
                interactions=[],
                decays=[],
                test_type="particle_properties",
                success=success,
                duration=duration,
                pp_metrics=metrics
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
            "test_type": "particle_properties"
        }
    
    def test_particle_interactions(self, num_tests: int = 25) -> Dict[str, Any]:
        """Test particle interactions."""
        tests = []
        
        for i in range(num_tests):
            # Generate interactions
            num_interactions = random.randint(2, 5)
            interactions = []
            for j in range(num_interactions):
                interaction = self._generate_particle_interaction()
                interactions.append(interaction)
            
            # Test interaction consistency
            start_time = time.time()
            success = self._test_particle_interaction_consistency(interactions)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_particle_interaction_metrics(interactions, success)
            
            test = ParticlePhysicsTest(
                test_id=f"particle_interactions_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Particle Interactions Test {i+1}",
                particles=[],
                interactions=interactions,
                decays=[],
                test_type="particle_interactions",
                success=success,
                duration=duration,
                pp_metrics=metrics
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
            "test_type": "particle_interactions"
        }
    
    def test_decay_processes(self, num_tests: int = 20) -> Dict[str, Any]:
        """Test particle decay processes."""
        tests = []
        
        for i in range(num_tests):
            # Generate decay processes
            num_decays = random.randint(2, 4)
            decays = []
            for j in range(num_decays):
                decay = self._generate_decay_process()
                decays.append(decay)
            
            # Test decay consistency
            start_time = time.time()
            success = self._test_decay_process_consistency(decays)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_decay_process_metrics(decays, success)
            
            test = ParticlePhysicsTest(
                test_id=f"decay_processes_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Decay Processes Test {i+1}",
                particles=[],
                interactions=[],
                decays=decays,
                test_type="decay_processes",
                success=success,
                duration=duration,
                pp_metrics=metrics
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
            "test_type": "decay_processes"
        }
    
    def _generate_particle(self) -> Particle:
        """Generate a particle."""
        particle_names = list(self.particle_database.keys())
        name = random.choice(particle_names)
        properties = self.particle_database[name]
        
        # Add some randomness to properties
        mass = properties["mass"] * random.uniform(0.9, 1.1)
        charge = properties["charge"]
        spin = properties["spin"]
        lifetime = properties["lifetime"] * random.uniform(0.8, 1.2)
        
        return Particle(
            particle_id=f"particle_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            mass=mass,
            charge=charge,
            spin=spin,
            lifetime=lifetime
        )
    
    def _generate_particle_interaction(self) -> ParticleInteraction:
        """Generate a particle interaction."""
        # Generate initial particles
        num_initial = random.randint(1, 3)
        initial_particles = [self._generate_particle() for _ in range(num_initial)]
        
        # Generate final particles
        num_final = random.randint(1, 4)
        final_particles = [self._generate_particle() for _ in range(num_final)]
        
        # Calculate cross section (simplified)
        cross_section = random.uniform(1e-6, 1e-2)  # in barns
        
        # Calculate energy
        energy = random.uniform(1, 1000)  # in GeV
        
        return ParticleInteraction(
            interaction_id=f"interaction_{int(time.time())}_{random.randint(1000, 9999)}",
            initial_particles=initial_particles,
            final_particles=final_particles,
            cross_section=cross_section,
            energy=energy
        )
    
    def _generate_decay_process(self) -> DecayProcess:
        """Generate a decay process."""
        # Generate parent particle
        parent_particle = self._generate_particle()
        
        # Generate daughter particles
        num_daughters = random.randint(2, 4)
        daughter_particles = [self._generate_particle() for _ in range(num_daughters)]
        
        # Calculate branching ratio
        branching_ratio = random.uniform(0.1, 1.0)
        
        # Calculate decay width
        decay_width = random.uniform(1e-6, 1e-2)  # in GeV
        
        return DecayProcess(
            decay_id=f"decay_{int(time.time())}_{random.randint(1000, 9999)}",
            parent_particle=parent_particle,
            daughter_particles=daughter_particles,
            branching_ratio=branching_ratio,
            decay_width=decay_width
        )
    
    def _test_particle_properties_consistency(self, particles: List[Particle]) -> bool:
        """Test particle properties consistency."""
        for particle in particles:
            # Check mass is non-negative
            if particle.mass < 0 or not np.isfinite(particle.mass):
                return False
            
            # Check charge is integer
            if not np.isclose(particle.charge, round(particle.charge)):
                return False
            
            # Check spin is non-negative
            if particle.spin < 0 or not np.isfinite(particle.spin):
                return False
            
            # Check lifetime is positive
            if particle.lifetime <= 0 or not np.isfinite(particle.lifetime):
                return False
        
        return True
    
    def _test_particle_interaction_consistency(self, interactions: List[ParticleInteraction]) -> bool:
        """Test particle interaction consistency."""
        for interaction in interactions:
            # Check cross section is positive
            if interaction.cross_section <= 0 or not np.isfinite(interaction.cross_section):
                return False
            
            # Check energy is positive
            if interaction.energy <= 0 or not np.isfinite(interaction.energy):
                return False
            
            # Check particles are valid
            all_particles = interaction.initial_particles + interaction.final_particles
            if not self._test_particle_properties_consistency(all_particles):
                return False
        
        return True
    
    def _test_decay_process_consistency(self, decays: List[DecayProcess]) -> bool:
        """Test decay process consistency."""
        for decay in decays:
            # Check branching ratio is between 0 and 1
            if not (0 <= decay.branching_ratio <= 1) or not np.isfinite(decay.branching_ratio):
                return False
            
            # Check decay width is positive
            if decay.decay_width <= 0 or not np.isfinite(decay.decay_width):
                return False
            
            # Check parent particle is valid
            if not self._test_particle_properties_consistency([decay.parent_particle]):
                return False
            
            # Check daughter particles are valid
            if not self._test_particle_properties_consistency(decay.daughter_particles):
                return False
        
        return True
    
    def _calculate_particle_properties_metrics(self, particles: List[Particle], success: bool) -> Dict[str, float]:
        """Calculate particle properties metrics."""
        return {
            "num_particles": len(particles),
            "avg_mass": np.mean([p.mass for p in particles]),
            "avg_charge": np.mean([p.charge for p in particles]),
            "avg_spin": np.mean([p.spin for p in particles]),
            "avg_lifetime": np.mean([p.lifetime for p in particles if p.lifetime != float('inf')]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_particle_interaction_metrics(self, interactions: List[ParticleInteraction], success: bool) -> Dict[str, float]:
        """Calculate particle interaction metrics."""
        return {
            "num_interactions": len(interactions),
            "avg_cross_section": np.mean([i.cross_section for i in interactions]),
            "avg_energy": np.mean([i.energy for i in interactions]),
            "avg_initial_particles": np.mean([len(i.initial_particles) for i in interactions]),
            "avg_final_particles": np.mean([len(i.final_particles) for i in interactions]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_decay_process_metrics(self, decays: List[DecayProcess], success: bool) -> Dict[str, float]:
        """Calculate decay process metrics."""
        return {
            "num_decays": len(decays),
            "avg_branching_ratio": np.mean([d.branching_ratio for d in decays]),
            "avg_decay_width": np.mean([d.decay_width for d in decays]),
            "avg_daughter_particles": np.mean([len(d.daughter_particles) for d in decays]),
            "test_success": 1.0 if success else 0.0
        }
    
    def generate_particle_physics_report(self) -> Dict[str, Any]:
        """Generate comprehensive particle physics test report."""
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
def demo_particle_physics_testing():
    """Demonstrate particle physics testing capabilities."""
    print("🔬 Particle Physics Testing Framework Demo")
    print("=" * 50)
    
    # Create particle physics test framework
    framework = ParticlePhysicsTestFramework()
    
    # Run comprehensive tests
    print("🧪 Running particle physics tests...")
    
    # Test particle properties
    print("\n⚛️ Testing particle properties...")
    particle_result = framework.test_particle_properties(num_tests=20)
    print(f"Particle Properties: {'✅' if particle_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {particle_result['success_rate']:.1%}")
    print(f"  Total Tests: {particle_result['total_tests']}")
    
    # Test particle interactions
    print("\n💥 Testing particle interactions...")
    interaction_result = framework.test_particle_interactions(num_tests=15)
    print(f"Particle Interactions: {'✅' if interaction_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {interaction_result['success_rate']:.1%}")
    print(f"  Total Tests: {interaction_result['total_tests']}")
    
    # Test decay processes
    print("\n🔄 Testing decay processes...")
    decay_result = framework.test_decay_processes(num_tests=10)
    print(f"Decay Processes: {'✅' if decay_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {decay_result['success_rate']:.1%}")
    print(f"  Total Tests: {decay_result['total_tests']}")
    
    # Generate comprehensive report
    print("\n📈 Generating particle physics report...")
    report = framework.generate_particle_physics_report()
    
    print(f"\n📊 Particle Physics Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")

if __name__ == "__main__":
    # Run demo
    demo_particle_physics_testing()
