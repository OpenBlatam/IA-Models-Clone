"""
Formatters for Contabilidad Mexicana AI
========================================

Refactored with:
- Formatter registry pattern
- BaseFormatter abstract class
- Dataclass for format options
- Factory for formatter creation
"""

from typing import Dict, Any, List, Optional, Callable, Protocol
from decimal import Decimal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class FormatType(Enum):
    """Types of formatting."""
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    TAX_CALCULATION = "tax_calculation"
    DATE = "date"
    NUMBER = "number"


@dataclass
class FormatOptions:
    """Options for formatting."""
    currency: str = "MXN"
    decimals: int = 2
    locale: str = "es_MX"
    thousand_separator: str = ","
    decimal_separator: str = "."
    show_currency_symbol: bool = True


class Formatter(Protocol):
    """Protocol for formatters."""
    
    def format(self, value: Any, options: Optional[FormatOptions] = None) -> str:
        """Format a value."""
        ...


class BaseFormatter(ABC):
    """Abstract base class for formatters."""
    
    @property
    @abstractmethod
    def format_type(self) -> FormatType:
        """The type of formatting this formatter handles."""
        pass
    
    @abstractmethod
    def format(self, value: Any, options: Optional[FormatOptions] = None) -> str:
        """Format a value."""
        pass
    
    def _get_options(self, options: Optional[FormatOptions]) -> FormatOptions:
        """Get options with defaults."""
        return options or FormatOptions()


class CurrencyFormatter(BaseFormatter):
    """Formats monetary values."""
    
    format_type = FormatType.CURRENCY
    
    def format(self, value: Any, options: Optional[FormatOptions] = None) -> str:
        """Format amount as currency."""
        opts = self._get_options(options)
        amount = float(value) if not isinstance(value, float) else value
        
        formatted = f"{amount:,.{opts.decimals}f}"
        
        if opts.show_currency_symbol:
            return f"${formatted} {opts.currency}"
        return formatted


class PercentageFormatter(BaseFormatter):
    """Formats percentage values."""
    
    format_type = FormatType.PERCENTAGE
    
    def format(self, value: Any, options: Optional[FormatOptions] = None) -> str:
        """Format value as percentage."""
        opts = self._get_options(options)
        pct_value = float(value) * 100
        return f"{pct_value:.{opts.decimals}f}%"


class NumberFormatter(BaseFormatter):
    """Formats numeric values."""
    
    format_type = FormatType.NUMBER
    
    def format(self, value: Any, options: Optional[FormatOptions] = None) -> str:
        """Format number with thousand separators."""
        opts = self._get_options(options)
        num_value = float(value) if not isinstance(value, (int, float)) else value
        return f"{num_value:,.{opts.decimals}f}"


class FormatterRegistry:
    """Registry for formatters."""
    
    def __init__(self):
        self._formatters: Dict[FormatType, BaseFormatter] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default formatters."""
        self.register(CurrencyFormatter())
        self.register(PercentageFormatter())
        self.register(NumberFormatter())
    
    def register(self, formatter: BaseFormatter):
        """Register a formatter."""
        self._formatters[formatter.format_type] = formatter
    
    def get(self, format_type: FormatType) -> Optional[BaseFormatter]:
        """Get formatter by type."""
        return self._formatters.get(format_type)
    
    def format(
        self, 
        format_type: FormatType, 
        value: Any, 
        options: Optional[FormatOptions] = None
    ) -> str:
        """Format value using registered formatter."""
        formatter = self.get(format_type)
        if not formatter:
            raise ValueError(f"No formatter registered for {format_type}")
        return formatter.format(value, options)


# Global registry instance
_registry = FormatterRegistry()


@dataclass
class TaxCalculationResult:
    """Represents a tax calculation result."""
    regimen: Optional[str] = None
    tipo_impuesto: Optional[str] = None
    ingresos_anuales: Optional[float] = None
    impuesto_total: Optional[float] = None
    tasa_efectiva: Optional[float] = None
    desglose: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaxCalculationResult":
        """Create from dictionary."""
        return cls(
            regimen=data.get("regimen"),
            tipo_impuesto=data.get("tipo_impuesto"),
            ingresos_anuales=data.get("ingresos_anuales"),
            impuesto_total=data.get("impuesto_total"),
            tasa_efectiva=data.get("tasa_efectiva"),
            desglose=data.get("desglose", []),
        )


class TaxCalculationFormatter:
    """Formats tax calculation results."""
    
    def __init__(self, registry: Optional[FormatterRegistry] = None):
        self.registry = registry or _registry
    
    def format(
        self, 
        result: TaxCalculationResult, 
        options: Optional[FormatOptions] = None
    ) -> str:
        """Format tax calculation result for display."""
        lines = []
        opts = options or FormatOptions()
        
        if result.regimen:
            lines.append(f"Régimen: {result.regimen}")
        
        if result.tipo_impuesto:
            lines.append(f"Tipo de Impuesto: {result.tipo_impuesto}")
        
        if result.ingresos_anuales is not None:
            formatted = self.registry.format(
                FormatType.CURRENCY, result.ingresos_anuales, opts
            )
            lines.append(f"Ingresos Anuales: {formatted}")
        
        if result.impuesto_total is not None:
            formatted = self.registry.format(
                FormatType.CURRENCY, result.impuesto_total, opts
            )
            lines.append(f"Impuesto Total: {formatted}")
        
        if result.tasa_efectiva is not None:
            formatted = self.registry.format(
                FormatType.PERCENTAGE, result.tasa_efectiva, opts
            )
            lines.append(f"Tasa Efectiva: {formatted}")
        
        if result.desglose:
            lines.append("\nDesglose por Tramos:")
            for tramo in result.desglose:
                lines.append(self._format_tramo(tramo, opts))
        
        return "\n".join(lines)
    
    def _format_tramo(self, tramo: Dict[str, Any], opts: FormatOptions) -> str:
        """Format a single tramo."""
        base = self.registry.format(FormatType.CURRENCY, tramo.get("base", 0), opts)
        tasa = self.registry.format(FormatType.PERCENTAGE, tramo.get("tasa", 0), opts)
        impuesto = self.registry.format(FormatType.CURRENCY, tramo.get("impuesto", 0), opts)
        
        return f"  {tramo.get('tramo', '')}: Base {base}, Tasa {tasa}, Impuesto {impuesto}"


# === Convenience Functions (Backward Compatible) ===

def format_currency(amount: float, currency: str = "MXN") -> str:
    """Format amount as currency."""
    return _registry.format(
        FormatType.CURRENCY, 
        amount, 
        FormatOptions(currency=currency)
    )


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return _registry.format(
        FormatType.PERCENTAGE, 
        value, 
        FormatOptions(decimals=decimals)
    )


def format_tax_calculation(result: Dict[str, Any]) -> str:
    """Format tax calculation result for display."""
    tax_result = TaxCalculationResult.from_dict(result)
    formatter = TaxCalculationFormatter()
    return formatter.format(tax_result)


def get_formatter_registry() -> FormatterRegistry:
    """Get the global formatter registry."""
    return _registry
