"""
Test Suite - Sistema de testing automatizado
=============================================
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.prototype_generator import PrototypeGenerator
from models.schemas import PrototypeRequest, ProductType


class TestPrototypeGenerator:
    """Tests para el generador de prototipos"""
    
    @pytest.fixture
    def generator(self):
        """Fixture para el generador"""
        return PrototypeGenerator(output_dir="tests/output")
    
    @pytest.mark.asyncio
    async def test_generate_licuadora(self, generator):
        """Test: Generar prototipo de licuadora"""
        request = PrototypeRequest(
            product_description="Quiero hacer una licuadora",
            product_type=ProductType.LICUADORA,
            budget=150.0
        )
        
        response = await generator.generate_prototype(request)
        
        assert response.product_name is not None
        assert len(response.materials) > 0
        assert len(response.cad_parts) > 0
        assert response.total_cost_estimate > 0
        assert response.difficulty_level in ["Fácil", "Media", "Difícil"]
    
    @pytest.mark.asyncio
    async def test_generate_estufa(self, generator):
        """Test: Generar prototipo de estufa"""
        request = PrototypeRequest(
            product_description="Necesito una estufa de gas",
            product_type=ProductType.ESTUFA
        )
        
        response = await generator.generate_prototype(request)
        
        assert response.product_name is not None
        assert "estufa" in response.product_name.lower() or "stove" in response.product_name.lower()
        assert len(response.materials) > 0
    
    @pytest.mark.asyncio
    async def test_budget_options(self, generator):
        """Test: Verificar opciones de presupuesto"""
        request = PrototypeRequest(
            product_description="Licuadora básica",
            budget=100.0
        )
        
        response = await generator.generate_prototype(request)
        
        assert len(response.budget_options) > 0
        assert all(opt.total_cost > 0 for opt in response.budget_options)
    
    @pytest.mark.asyncio
    async def test_materials_have_sources(self, generator):
        """Test: Verificar que los materiales tengan fuentes"""
        request = PrototypeRequest(
            product_description="Licuadora",
            product_type=ProductType.LICUADORA
        )
        
        response = await generator.generate_prototype(request)
        
        for material in response.materials:
            assert len(material.sources) > 0, f"Material {material.name} no tiene fuentes"
    
    @pytest.mark.asyncio
    async def test_assembly_instructions(self, generator):
        """Test: Verificar instrucciones de ensamblaje"""
        request = PrototypeRequest(
            product_description="Licuadora",
            product_type=ProductType.LICUADORA
        )
        
        response = await generator.generate_prototype(request)
        
        assert len(response.assembly_instructions) > 0
        assert all(step.step_number > 0 for step in response.assembly_instructions)
        assert all(step.description for step in response.assembly_instructions)


class TestMaterialDatabase:
    """Tests para la base de datos de materiales"""
    
    def test_material_info(self):
        """Test: Obtener información de materiales"""
        from core.prototype_generator import MaterialDatabase
        
        db = MaterialDatabase()
        info = db.get_material_info("acero_inoxidable", 1.0, "kg")
        
        assert "price_per_unit" in info
        assert "sources" in info
        assert info["price_per_unit"] > 0


class TestSchemas:
    """Tests para los schemas"""
    
    def test_prototype_request(self):
        """Test: Validar PrototypeRequest"""
        request = PrototypeRequest(
            product_description="Test product",
            product_type=ProductType.LICUADORA,
            budget=100.0
        )
        
        assert request.product_description == "Test product"
        assert request.product_type == ProductType.LICUADORA
        assert request.budget == 100.0
    
    def test_material_schema(self):
        """Test: Validar Material schema"""
        from models.schemas import Material, MaterialSource
        
        source = MaterialSource(
            name="Test Store",
            location="Test Location",
            availability="Available"
        )
        
        material = Material(
            name="Test Material",
            quantity=1.0,
            unit="kg",
            price_per_unit=10.0,
            total_price=10.0,
            category="Test",
            sources=[source]
        )
        
        assert material.name == "Test Material"
        assert material.total_price == 10.0
        assert len(material.sources) == 1




