"""
Solid Mechanics Testing Framework for HeyGen AI Testing System.
Advanced solid mechanics testing including stress analysis, strain,
and material behavior validation.
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
class StressTensor:
    """Represents a stress tensor."""
    tensor_id: str
    components: np.ndarray  # 3x3 stress tensor
    principal_stresses: np.ndarray  # [σ1, σ2, σ3]
    von_mises_stress: float  # in Pa
    hydrostatic_stress: float  # in Pa
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class StrainTensor:
    """Represents a strain tensor."""
    tensor_id: str
    components: np.ndarray  # 3x3 strain tensor
    principal_strains: np.ndarray  # [ε1, ε2, ε3]
    volumetric_strain: float
    shear_strain: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class MaterialProperty:
    """Represents material properties."""
    material_id: str
    young_modulus: float  # in Pa
    poisson_ratio: float
    shear_modulus: float  # in Pa
    bulk_modulus: float  # in Pa
    yield_strength: float  # in Pa
    ultimate_strength: float  # in Pa
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class FiniteElement:
    """Represents a finite element."""
    element_id: str
    element_type: str  # "tetrahedral", "hexahedral", "triangular", "quadrilateral"
    nodes: List[int]
    shape_functions: np.ndarray
    stiffness_matrix: np.ndarray
    mass_matrix: np.ndarray
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SolidMechanicsTest:
    """Represents a solid mechanics test."""
    test_id: str
    test_name: str
    stress_tensors: List[StressTensor]
    strain_tensors: List[StrainTensor]
    material_properties: List[MaterialProperty]
    finite_elements: List[FiniteElement]
    test_type: str
    success: bool
    duration: float
    sm_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class SolidMechanicsTestFramework:
    """Main solid mechanics testing framework."""
    
    def __init__(self):
        self.test_results = []
        self.E_steel = 200e9  # Steel Young's modulus in Pa
        self.nu_steel = 0.3  # Steel Poisson's ratio
        self.sigma_y_steel = 250e6  # Steel yield strength in Pa
    
    def test_stress_analysis(self, num_tests: int = 30) -> Dict[str, Any]:
        """Test stress analysis."""
        tests = []
        
        for i in range(num_tests):
            # Generate stress tensors
            num_tensors = random.randint(2, 6)
            stress_tensors = []
            for j in range(num_tensors):
                tensor = self._generate_stress_tensor()
                stress_tensors.append(tensor)
            
            # Test stress analysis consistency
            start_time = time.time()
            success = self._test_stress_analysis_consistency(stress_tensors)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_stress_analysis_metrics(stress_tensors, success)
            
            test = SolidMechanicsTest(
                test_id=f"stress_analysis_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Stress Analysis Test {i+1}",
                stress_tensors=stress_tensors,
                strain_tensors=[],
                material_properties=[],
                finite_elements=[],
                test_type="stress_analysis",
                success=success,
                duration=duration,
                sm_metrics=metrics
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
            "test_type": "stress_analysis"
        }
    
    def test_strain_analysis(self, num_tests: int = 25) -> Dict[str, Any]:
        """Test strain analysis."""
        tests = []
        
        for i in range(num_tests):
            # Generate strain tensors
            num_tensors = random.randint(2, 5)
            strain_tensors = []
            for j in range(num_tensors):
                tensor = self._generate_strain_tensor()
                strain_tensors.append(tensor)
            
            # Test strain analysis consistency
            start_time = time.time()
            success = self._test_strain_analysis_consistency(strain_tensors)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_strain_analysis_metrics(strain_tensors, success)
            
            test = SolidMechanicsTest(
                test_id=f"strain_analysis_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Strain Analysis Test {i+1}",
                stress_tensors=[],
                strain_tensors=strain_tensors,
                material_properties=[],
                finite_elements=[],
                test_type="strain_analysis",
                success=success,
                duration=duration,
                sm_metrics=metrics
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
            "test_type": "strain_analysis"
        }
    
    def test_material_properties(self, num_tests: int = 20) -> Dict[str, Any]:
        """Test material properties."""
        tests = []
        
        for i in range(num_tests):
            # Generate material properties
            num_materials = random.randint(2, 5)
            material_properties = []
            for j in range(num_materials):
                material = self._generate_material_property()
                material_properties.append(material)
            
            # Test material property consistency
            start_time = time.time()
            success = self._test_material_property_consistency(material_properties)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_material_property_metrics(material_properties, success)
            
            test = SolidMechanicsTest(
                test_id=f"material_properties_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Material Properties Test {i+1}",
                stress_tensors=[],
                strain_tensors=[],
                material_properties=material_properties,
                finite_elements=[],
                test_type="material_properties",
                success=success,
                duration=duration,
                sm_metrics=metrics
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
            "test_type": "material_properties"
        }
    
    def test_finite_elements(self, num_tests: int = 15) -> Dict[str, Any]:
        """Test finite elements."""
        tests = []
        
        for i in range(num_tests):
            # Generate finite elements
            num_elements = random.randint(3, 8)
            finite_elements = []
            for j in range(num_elements):
                element = self._generate_finite_element()
                finite_elements.append(element)
            
            # Test finite element consistency
            start_time = time.time()
            success = self._test_finite_element_consistency(finite_elements)
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_finite_element_metrics(finite_elements, success)
            
            test = SolidMechanicsTest(
                test_id=f"finite_elements_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}",
                test_name=f"Finite Elements Test {i+1}",
                stress_tensors=[],
                strain_tensors=[],
                material_properties=[],
                finite_elements=finite_elements,
                test_type="finite_elements",
                success=success,
                duration=duration,
                sm_metrics=metrics
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
            "test_type": "finite_elements"
        }
    
    def _generate_stress_tensor(self) -> StressTensor:
        """Generate a stress tensor."""
        # Generate 3x3 stress tensor
        components = np.random.uniform(-1e6, 1e6, (3, 3))
        
        # Make it symmetric
        components = (components + components.T) / 2
        
        # Calculate principal stresses
        eigenvalues = np.linalg.eigvals(components)
        principal_stresses = np.sort(eigenvalues)[::-1]  # Sort in descending order
        
        # Calculate von Mises stress
        von_mises_stress = np.sqrt(0.5 * ((principal_stresses[0] - principal_stresses[1])**2 + 
                                         (principal_stresses[1] - principal_stresses[2])**2 + 
                                         (principal_stresses[2] - principal_stresses[0])**2))
        
        # Calculate hydrostatic stress
        hydrostatic_stress = np.trace(components) / 3
        
        return StressTensor(
            tensor_id=f"stress_{int(time.time())}_{random.randint(1000, 9999)}",
            components=components,
            principal_stresses=principal_stresses,
            von_mises_stress=von_mises_stress,
            hydrostatic_stress=hydrostatic_stress
        )
    
    def _generate_strain_tensor(self) -> StrainTensor:
        """Generate a strain tensor."""
        # Generate 3x3 strain tensor
        components = np.random.uniform(-0.1, 0.1, (3, 3))
        
        # Make it symmetric
        components = (components + components.T) / 2
        
        # Calculate principal strains
        eigenvalues = np.linalg.eigvals(components)
        principal_strains = np.sort(eigenvalues)[::-1]  # Sort in descending order
        
        # Calculate volumetric strain
        volumetric_strain = np.trace(components)
        
        # Calculate shear strain (simplified)
        shear_strain = np.sqrt(0.5 * ((principal_strains[0] - principal_strains[1])**2 + 
                                     (principal_strains[1] - principal_strains[2])**2 + 
                                     (principal_strains[2] - principal_strains[0])**2))
        
        return StrainTensor(
            tensor_id=f"strain_{int(time.time())}_{random.randint(1000, 9999)}",
            components=components,
            principal_strains=principal_strains,
            volumetric_strain=volumetric_strain,
            shear_strain=shear_strain
        )
    
    def _generate_material_property(self) -> MaterialProperty:
        """Generate material properties."""
        # Generate properties based on steel (with some variation)
        young_modulus = self.E_steel * random.uniform(0.5, 2.0)
        poisson_ratio = random.uniform(0.2, 0.4)
        
        # Calculate derived properties
        shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
        bulk_modulus = young_modulus / (3 * (1 - 2 * poisson_ratio))
        
        yield_strength = self.sigma_y_steel * random.uniform(0.5, 2.0)
        ultimate_strength = yield_strength * random.uniform(1.2, 2.0)
        
        return MaterialProperty(
            material_id=f"material_{int(time.time())}_{random.randint(1000, 9999)}",
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            shear_modulus=shear_modulus,
            bulk_modulus=bulk_modulus,
            yield_strength=yield_strength,
            ultimate_strength=ultimate_strength
        )
    
    def _generate_finite_element(self) -> FiniteElement:
        """Generate a finite element."""
        element_types = ["tetrahedral", "hexahedral", "triangular", "quadrilateral"]
        element_type = random.choice(element_types)
        
        # Generate nodes based on element type
        if element_type == "tetrahedral":
            num_nodes = 4
        elif element_type == "hexahedral":
            num_nodes = 8
        elif element_type == "triangular":
            num_nodes = 3
        else:  # quadrilateral
            num_nodes = 4
        
        nodes = [random.randint(1, 1000) for _ in range(num_nodes)]
        
        # Generate shape functions (simplified)
        shape_functions = np.random.uniform(0, 1, (num_nodes, num_nodes))
        
        # Generate stiffness matrix
        size = num_nodes * 3  # 3 DOF per node
        stiffness_matrix = np.random.uniform(-1e6, 1e6, (size, size))
        stiffness_matrix = (stiffness_matrix + stiffness_matrix.T) / 2  # Make symmetric
        
        # Generate mass matrix
        mass_matrix = np.random.uniform(0, 1e3, (size, size))
        mass_matrix = (mass_matrix + mass_matrix.T) / 2  # Make symmetric
        
        return FiniteElement(
            element_id=f"element_{int(time.time())}_{random.randint(1000, 9999)}",
            element_type=element_type,
            nodes=nodes,
            shape_functions=shape_functions,
            stiffness_matrix=stiffness_matrix,
            mass_matrix=mass_matrix
        )
    
    def _test_stress_analysis_consistency(self, tensors: List[StressTensor]) -> bool:
        """Test stress analysis consistency."""
        for tensor in tensors:
            if not np.all(np.isfinite(tensor.components)):
                return False
            if not np.all(np.isfinite(tensor.principal_stresses)):
                return False
            if tensor.von_mises_stress < 0 or not np.isfinite(tensor.von_mises_stress):
                return False
            if not np.isfinite(tensor.hydrostatic_stress):
                return False
        return True
    
    def _test_strain_analysis_consistency(self, tensors: List[StrainTensor]) -> bool:
        """Test strain analysis consistency."""
        for tensor in tensors:
            if not np.all(np.isfinite(tensor.components)):
                return False
            if not np.all(np.isfinite(tensor.principal_strains)):
                return False
            if not np.isfinite(tensor.volumetric_strain):
                return False
            if tensor.shear_strain < 0 or not np.isfinite(tensor.shear_strain):
                return False
        return True
    
    def _test_material_property_consistency(self, materials: List[MaterialProperty]) -> bool:
        """Test material property consistency."""
        for material in materials:
            if material.young_modulus <= 0 or not np.isfinite(material.young_modulus):
                return False
            if not (0 <= material.poisson_ratio < 0.5) or not np.isfinite(material.poisson_ratio):
                return False
            if material.shear_modulus <= 0 or not np.isfinite(material.shear_modulus):
                return False
            if material.bulk_modulus <= 0 or not np.isfinite(material.bulk_modulus):
                return False
            if material.yield_strength <= 0 or not np.isfinite(material.yield_strength):
                return False
            if material.ultimate_strength <= 0 or not np.isfinite(material.ultimate_strength):
                return False
        return True
    
    def _test_finite_element_consistency(self, elements: List[FiniteElement]) -> bool:
        """Test finite element consistency."""
        for element in elements:
            if not np.all(np.isfinite(element.shape_functions)):
                return False
            if not np.all(np.isfinite(element.stiffness_matrix)):
                return False
            if not np.all(np.isfinite(element.mass_matrix)):
                return False
            if len(element.nodes) == 0:
                return False
        return True
    
    def _calculate_stress_analysis_metrics(self, tensors: List[StressTensor], success: bool) -> Dict[str, float]:
        """Calculate stress analysis metrics."""
        return {
            "num_tensors": len(tensors),
            "avg_von_mises_stress": np.mean([t.von_mises_stress for t in tensors]),
            "avg_hydrostatic_stress": np.mean([t.hydrostatic_stress for t in tensors]),
            "avg_principal_stress": np.mean([np.mean(t.principal_stresses) for t in tensors]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_strain_analysis_metrics(self, tensors: List[StrainTensor], success: bool) -> Dict[str, float]:
        """Calculate strain analysis metrics."""
        return {
            "num_tensors": len(tensors),
            "avg_volumetric_strain": np.mean([t.volumetric_strain for t in tensors]),
            "avg_shear_strain": np.mean([t.shear_strain for t in tensors]),
            "avg_principal_strain": np.mean([np.mean(t.principal_strains) for t in tensors]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_material_property_metrics(self, materials: List[MaterialProperty], success: bool) -> Dict[str, float]:
        """Calculate material property metrics."""
        return {
            "num_materials": len(materials),
            "avg_young_modulus": np.mean([m.young_modulus for m in materials]),
            "avg_poisson_ratio": np.mean([m.poisson_ratio for m in materials]),
            "avg_yield_strength": np.mean([m.yield_strength for m in materials]),
            "avg_ultimate_strength": np.mean([m.ultimate_strength for m in materials]),
            "test_success": 1.0 if success else 0.0
        }
    
    def _calculate_finite_element_metrics(self, elements: List[FiniteElement], success: bool) -> Dict[str, float]:
        """Calculate finite element metrics."""
        return {
            "num_elements": len(elements),
            "avg_nodes_per_element": np.mean([len(e.nodes) for e in elements]),
            "avg_stiffness_matrix_size": np.mean([e.stiffness_matrix.shape[0] for e in elements]),
            "avg_mass_matrix_size": np.mean([e.mass_matrix.shape[0] for e in elements]),
            "test_success": 1.0 if success else 0.0
        }
    
    def generate_solid_mechanics_report(self) -> Dict[str, Any]:
        """Generate comprehensive solid mechanics test report."""
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
def demo_solid_mechanics_testing():
    """Demonstrate solid mechanics testing capabilities."""
    print("🔧 Solid Mechanics Testing Framework Demo")
    print("=" * 50)
    
    # Create solid mechanics test framework
    framework = SolidMechanicsTestFramework()
    
    # Run comprehensive tests
    print("🧪 Running solid mechanics tests...")
    
    # Test stress analysis
    print("\n⚡ Testing stress analysis...")
    stress_result = framework.test_stress_analysis(num_tests=20)
    print(f"Stress Analysis: {'✅' if stress_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {stress_result['success_rate']:.1%}")
    print(f"  Total Tests: {stress_result['total_tests']}")
    
    # Test strain analysis
    print("\n📏 Testing strain analysis...")
    strain_result = framework.test_strain_analysis(num_tests=15)
    print(f"Strain Analysis: {'✅' if strain_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {strain_result['success_rate']:.1%}")
    print(f"  Total Tests: {strain_result['total_tests']}")
    
    # Test material properties
    print("\n🧱 Testing material properties...")
    material_result = framework.test_material_properties(num_tests=10)
    print(f"Material Properties: {'✅' if material_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {material_result['success_rate']:.1%}")
    print(f"  Total Tests: {material_result['total_tests']}")
    
    # Test finite elements
    print("\n🔺 Testing finite elements...")
    element_result = framework.test_finite_elements(num_tests=8)
    print(f"Finite Elements: {'✅' if element_result['success_rate'] > 0.7 else '❌'}")
    print(f"  Success Rate: {element_result['success_rate']:.1%}")
    print(f"  Total Tests: {element_result['total_tests']}")
    
    # Generate comprehensive report
    print("\n📈 Generating solid mechanics report...")
    report = framework.generate_solid_mechanics_report()
    
    print(f"\n📊 Solid Mechanics Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")

if __name__ == "__main__":
    # Run demo
    demo_solid_mechanics_testing()
