"""
Salary Benchmarking Service - Comparación salarial
===================================================

Sistema para comparar y analizar salarios del mercado.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SalaryData:
    """Datos salariales"""
    role: str
    location: str
    experience_years: int
    company_size: str
    industry: str
    salary_min: float
    salary_median: float
    salary_max: float
    currency: str = "USD"
    data_source: str = "market"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SalaryComparison:
    """Comparación salarial"""
    user_salary: float
    market_data: SalaryData
    percentile: float
    recommendation: str
    negotiation_power: float
    factors: Dict[str, Any]


class SalaryBenchmarkingService:
    """Servicio de comparación salarial"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.salary_database: Dict[str, List[SalaryData]] = {}
        logger.info("SalaryBenchmarkingService initialized")
    
    def add_salary_data(
        self,
        role: str,
        location: str,
        experience_years: int,
        salary_min: float,
        salary_median: float,
        salary_max: float,
        company_size: str = "medium",
        industry: str = "Technology"
    ) -> SalaryData:
        """Agregar datos salariales"""
        salary_data = SalaryData(
            role=role,
            location=location,
            experience_years=experience_years,
            company_size=company_size,
            industry=industry,
            salary_min=salary_min,
            salary_median=salary_median,
            salary_max=salary_max,
        )
        
        key = f"{role}_{location}_{experience_years}"
        if key not in self.salary_database:
            self.salary_database[key] = []
        
        self.salary_database[key].append(salary_data)
        
        logger.info(f"Salary data added: {key}")
        return salary_data
    
    def benchmark_salary(
        self,
        role: str,
        location: str,
        experience_years: int,
        current_salary: float,
        company_size: Optional[str] = None,
        industry: Optional[str] = None
    ) -> SalaryComparison:
        """Comparar salario con mercado"""
        # Buscar datos de mercado
        market_data = self._find_market_data(role, location, experience_years, company_size, industry)
        
        if not market_data:
            # Datos simulados si no hay datos reales
            market_data = SalaryData(
                role=role,
                location=location,
                experience_years=experience_years,
                company_size=company_size or "medium",
                industry=industry or "Technology",
                salary_min=current_salary * 0.7,
                salary_median=current_salary * 1.1,
                salary_max=current_salary * 1.5,
            )
        
        # Calcular percentil
        percentile = self._calculate_percentile(current_salary, market_data)
        
        # Calcular poder de negociación
        negotiation_power = self._calculate_negotiation_power(current_salary, market_data, percentile)
        
        # Generar recomendación
        recommendation = self._generate_recommendation(current_salary, market_data, percentile)
        
        # Factores de influencia
        factors = {
            "location_impact": self._calculate_location_impact(location),
            "experience_impact": self._calculate_experience_impact(experience_years),
            "company_size_impact": self._calculate_company_size_impact(company_size),
            "industry_impact": self._calculate_industry_impact(industry),
        }
        
        return SalaryComparison(
            user_salary=current_salary,
            market_data=market_data,
            percentile=percentile,
            recommendation=recommendation,
            negotiation_power=negotiation_power,
            factors=factors,
        )
    
    def _find_market_data(
        self,
        role: str,
        location: str,
        experience_years: int,
        company_size: Optional[str],
        industry: Optional[str]
    ) -> Optional[SalaryData]:
        """Buscar datos de mercado"""
        key = f"{role}_{location}_{experience_years}"
        candidates = self.salary_database.get(key, [])
        
        if not candidates:
            return None
        
        # Filtrar por company_size e industry si se proporcionan
        if company_size:
            candidates = [c for c in candidates if c.company_size == company_size]
        if industry:
            candidates = [c for c in candidates if c.industry == industry]
        
        if not candidates:
            return None
        
        # Retornar el más reciente
        return max(candidates, key=lambda c: c.timestamp)
    
    def _calculate_percentile(self, salary: float, market_data: SalaryData) -> float:
        """Calcular percentil del salario"""
        if salary <= market_data.salary_min:
            return 0.0
        elif salary >= market_data.salary_max:
            return 100.0
        
        # Interpolación lineal
        range_size = market_data.salary_max - market_data.salary_min
        position = (salary - market_data.salary_min) / range_size if range_size > 0 else 0.5
        
        return position * 100
    
    def _calculate_negotiation_power(
        self,
        current_salary: float,
        market_data: SalaryData,
        percentile: float
    ) -> float:
        """Calcular poder de negociación"""
        # Si está por debajo del percentil 50, tiene más poder
        if percentile < 50:
            power = 0.8
        elif percentile < 75:
            power = 0.6
        else:
            power = 0.3
        
        # Ajustar según diferencia con mediana
        median_diff = (current_salary - market_data.salary_median) / market_data.salary_median
        if median_diff < -0.2:  # 20% por debajo de la mediana
            power = min(1.0, power + 0.2)
        
        return round(power, 2)
    
    def _generate_recommendation(
        self,
        current_salary: float,
        market_data: SalaryData,
        percentile: float
    ) -> str:
        """Generar recomendación"""
        if percentile < 25:
            return f"Tu salario está en el percentil {percentile:.0f}. Estás significativamente por debajo del mercado. Considera negociar un aumento o buscar nuevas oportunidades."
        elif percentile < 50:
            return f"Tu salario está en el percentil {percentile:.0f}. Estás por debajo de la mediana del mercado. Tienes buen poder de negociación."
        elif percentile < 75:
            return f"Tu salario está en el percentil {percentile:.0f}. Estás en un rango competitivo. Puedes negociar beneficios adicionales."
        else:
            return f"Tu salario está en el percentil {percentile:.0f}. Estás por encima de la mediana del mercado. Excelente posición."
    
    def _calculate_location_impact(self, location: str) -> str:
        """Calcular impacto de ubicación"""
        high_cost_locations = ["San Francisco", "New York", "London", "Tokyo"]
        if location in high_cost_locations:
            return "high"  # Salarios más altos esperados
        return "medium"
    
    def _calculate_experience_impact(self, years: int) -> str:
        """Calcular impacto de experiencia"""
        if years < 2:
            return "low"
        elif years < 5:
            return "medium"
        else:
            return "high"
    
    def _calculate_company_size_impact(self, size: Optional[str]) -> str:
        """Calcular impacto de tamaño de empresa"""
        if size == "large":
            return "high"  # Grandes empresas suelen pagar más
        return "medium"
    
    def _calculate_industry_impact(self, industry: Optional[str]) -> str:
        """Calcular impacto de industria"""
        high_paying = ["Technology", "Finance", "Consulting"]
        if industry in high_paying:
            return "high"
        return "medium"
    
    def compare_multiple_roles(
        self,
        roles: List[str],
        location: str,
        experience_years: int
    ) -> Dict[str, Any]:
        """Comparar múltiples roles"""
        comparisons = []
        
        for role in roles:
            market_data = self._find_market_data(role, location, experience_years, None, None)
            if market_data:
                comparisons.append({
                    "role": role,
                    "median_salary": market_data.salary_median,
                    "salary_range": {
                        "min": market_data.salary_min,
                        "max": market_data.salary_max,
                    },
                })
        
        # Ordenar por salario mediano
        comparisons.sort(key=lambda x: x["median_salary"], reverse=True)
        
        return {
            "location": location,
            "experience_years": experience_years,
            "comparisons": comparisons,
            "recommendation": self._recommend_best_role(comparisons),
        }
    
    def _recommend_best_role(self, comparisons: List[Dict[str, Any]]) -> str:
        """Recomendar mejor rol"""
        if not comparisons:
            return "No hay datos suficientes para recomendar"
        
        top_role = comparisons[0]
        return f"Basado en el mercado, {top_role['role']} tiene el salario mediano más alto: ${top_role['median_salary']:,.0f}"

