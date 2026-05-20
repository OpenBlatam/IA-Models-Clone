"""Services module for Contabilidad Mexicana AI."""

from .calculadora_impuestos import CalculadoraImpuestos
from .comparador_regimenes import ComparadorRegimenes
from .exportador_resultados import ExportadorResultados
from .analizador_deducciones import AnalizadorDeducciones
from .calculadora_plataformas import CalculadoraPlataformas

__all__ = [
    "CalculadoraImpuestos",
    "ComparadorRegimenes",
    "ExportadorResultados",
    "AnalizadorDeducciones",
    "CalculadoraPlataformas"
]
