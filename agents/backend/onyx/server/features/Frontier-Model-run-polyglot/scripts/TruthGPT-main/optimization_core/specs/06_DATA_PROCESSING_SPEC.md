# 📊 Especificación de Procesamiento de Datos - Optimization Core

## 📋 Resumen

Este documento especifica el sistema de procesamiento de datos de alto rendimiento usando Polars, que proporciona 10-100x mejor rendimiento que pandas.

## 🎯 Objetivos

1. **Alto Rendimiento**: 10-100x más rápido que pandas
2. **Eficiencia de Memoria**: Uso eficiente de memoria con lazy evaluation
3. **Escalabilidad**: Procesamiento de datasets grandes (streaming)
4. **Flexibilidad**: Múltiples formatos de entrada/salida
5. **Optimización Automática**: Query optimization automático

## 🏗️ Arquitectura

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────┐
│              IDataProcessor (Interface)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              BaseDataProcessor (Abstract)                │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              PolarsProcessor (Implementation)            │
└──────────────────────────────────────────────────────────┘
```

## 📦 Componentes

### IDataProcessor

**Propósito**: Interfaz base para procesadores de datos.

**Especificación**:

```python
class IDataProcessor(IComponent):
    """Interface for data processors."""
    
    @abstractmethod
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process data.
        
        Args:
            data: Data to process
            **kwargs: Processing parameters
        
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def read(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> Any:
        """
        Read data from file.
        
        Args:
            path: Path to data file
            **kwargs: Reading parameters
        
        Returns:
            Data object
        """
        pass
    
    @abstractmethod
    def write(
        self,
        data: Any,
        path: Union[str, Path],
        **kwargs
    ) -> bool:
        """
        Write data to file.
        
        Args:
            data: Data to write
            path: Output path
            **kwargs: Writing parameters
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate data.
        
        Args:
            data: Data to validate
        
        Returns:
            True if valid
        """
        pass
```

### BaseDataProcessor

**Propósito**: Clase base abstracta con funcionalidad común.

**Especificación**:

```python
class BaseDataProcessor(ABC):
    """
    Abstract base class for data processors.
    
    Provides common functionality for all processors.
    """
    
    def __init__(self, lazy: bool = True, **kwargs):
        """
        Initialize base processor.
        
        Args:
            lazy: Enable lazy evaluation
            **kwargs: Processor-specific parameters
        """
        self.lazy = lazy
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def read(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Read data from file.
        
        Args:
            path: Path to file
            format: File format (auto-detect if None)
            **kwargs: Format-specific parameters
        
        Returns:
            Data object (lazy or eager)
        """
        path = Path(path)
        
        if format is None:
            format = self._detect_format(path)
        
        return self._read_impl(path, format, **kwargs)
    
    @abstractmethod
    def _read_impl(
        self,
        path: Path,
        format: str,
        **kwargs
    ) -> Any:
        """Implementation of reading."""
        pass
    
    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        ext = path.suffix.lower()
        format_map = {
            ".parquet": "parquet",
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".arrow": "arrow",
        }
        return format_map.get(ext, "unknown")
```

### PolarsProcessor

**Propósito**: Procesador de datos usando Polars.

**Especificación**:

```python
import polars as pl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class PolarsProcessor(BaseDataProcessor):
    """
    Polars-based data processor.
    
    Features:
    - 10-100x faster than pandas
    - Lazy evaluation with automatic optimization
    - Multi-threaded processing
    - Streaming for large datasets
    """
    
    def __init__(
        self,
        lazy: bool = True,
        streaming: bool = False,
        **kwargs
    ):
        """
        Initialize Polars processor.
        
        Args:
            lazy: Enable lazy evaluation
            streaming: Enable streaming for large datasets
        """
        super().__init__(lazy=lazy, **kwargs)
        self.streaming = streaming
    
    def _read_impl(
        self,
        path: Path,
        format: str,
        **kwargs
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Read data using Polars.
        
        Args:
            path: Path to file
            format: File format
            **kwargs: Format-specific parameters
        
        Returns:
            DataFrame (eager) or LazyFrame (lazy)
        """
        if format == "parquet":
            if self.lazy:
                return pl.scan_parquet(str(path), **kwargs)
            else:
                return pl.read_parquet(str(path), **kwargs)
        
        elif format == "csv":
            if self.lazy:
                return pl.scan_csv(str(path), **kwargs)
            else:
                return pl.read_csv(str(path), **kwargs)
        
        elif format == "json":
            if self.lazy:
                # JSON doesn't support lazy, read then convert
                df = pl.read_json(str(path), **kwargs)
                return df.lazy()
            else:
                return pl.read_json(str(path), **kwargs)
        
        elif format == "jsonl":
            if self.lazy:
                df = pl.read_ndjson(str(path), **kwargs)
                return df.lazy()
            else:
                return pl.read_ndjson(str(path), **kwargs)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def process(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame],
        operations: List[Dict[str, Any]],
        **kwargs
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Process data with a series of operations.
        
        Args:
            data: Input data (DataFrame or LazyFrame)
            operations: List of operations to apply
            **kwargs: Additional parameters
        
        Returns:
            Processed data
        """
        # Convert to LazyFrame if needed
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        
        # Apply operations
        for op in operations:
            op_type = op.get("type")
            op_params = op.get("params", {})
            
            if op_type == "filter":
                data = data.filter(pl.col(op_params["column"]) > op_params["value"])
            
            elif op_type == "select":
                data = data.select(op_params["columns"])
            
            elif op_type == "group_by":
                data = data.group_by(op_params["by"]).agg(op_params["aggs"])
            
            elif op_type == "join":
                other = op_params["other"]
                data = data.join(other, on=op_params["on"], how=op_params.get("how", "inner"))
            
            elif op_type == "sort":
                data = data.sort(op_params["by"])
            
            else:
                raise ValueError(f"Unknown operation: {op_type}")
        
        # Collect if not lazy
        if not self.lazy:
            return data.collect()
        
        return data
    
    def write(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame],
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Write data to file.
        
        Args:
            data: Data to write
            path: Output path
            format: Output format (auto-detect if None)
            **kwargs: Format-specific parameters
        
        Returns:
            True if successful
        """
        path = Path(path)
        
        if format is None:
            format = self._detect_format(path)
        
        # Collect if lazy
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        
        if format == "parquet":
            data.write_parquet(str(path), **kwargs)
        elif format == "csv":
            data.write_csv(str(path), **kwargs)
        elif format == "json":
            data.write_json(str(path), **kwargs)
        elif format == "jsonl":
            data.write_ndjson(str(path), **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return True
    
    def validate(self, data: Any) -> bool:
        """
        Validate data.
        
        Args:
            data: Data to validate
        
        Returns:
            True if valid
        """
        if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return False
        
        # Additional validation
        if isinstance(data, pl.DataFrame):
            if data.is_empty():
                return False
        
        return True
    
    def process_training_data(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        min_tokens: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Process training data with common transformations.
        
        Args:
            input_path: Input data path
            output_path: Output data path
            min_tokens: Minimum tokens per sample
            filters: Additional filters
            **kwargs: Additional parameters
        
        Returns:
            True if successful
        """
        # Read data
        df = self.read(input_path, lazy=True)
        
        # Apply filters
        if filters:
            for col, value in filters.items():
                df = df.filter(pl.col(col) > value)
        
        # Filter by token count
        if "tokens" in df.columns:
            df = df.filter(pl.col("tokens") >= min_tokens)
        
        # Write output
        self.write(df, output_path)
        
        return True
```

## 🏭 ProcessorFactory

**Propósito**: Factory para crear procesadores de datos.

**Especificación**:

```python
class ProcessorType(Enum):
    """Processor type enumeration."""
    AUTO = "auto"  # Auto-select best available
    POLARS = "polars"
    PANDAS = "pandas"  # Fallback

class ProcessorFactory:
    """Factory for creating data processors."""
    
    @staticmethod
    def create_processor(
        processor_type: ProcessorType = ProcessorType.AUTO,
        **kwargs
    ) -> IDataProcessor:
        """
        Create data processor.
        
        Args:
            processor_type: Processor type
            **kwargs: Processor-specific parameters
        
        Returns:
            Data processor instance
        """
        if processor_type == ProcessorType.AUTO:
            processor_type = ProcessorFactory._select_best_processor()
        
        if processor_type == ProcessorType.POLARS:
            return PolarsProcessor(**kwargs)
        elif processor_type == ProcessorType.PANDAS:
            from data.pandas_processor import PandasProcessor
            return PandasProcessor(**kwargs)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
    
    @staticmethod
    def _select_best_processor() -> ProcessorType:
        """Select best available processor."""
        try:
            import polars
            return ProcessorType.POLARS
        except ImportError:
            return ProcessorType.PANDAS
```

## 📊 Rendimiento

### Benchmarks Esperados

| Operación | Polars | pandas | Mejora |
|-----------|--------|--------|--------|
| Read Parquet (1GB) | 2.1s | 8.5s | 4x |
| Filter (100M rows) | 0.8s | 12.3s | 15x |
| Group By | 1.2s | 18.7s | 15x |
| Join (large) | 3.4s | 45.2s | 13x |
| Write Parquet | 2.5s | 9.8s | 4x |

### Optimizaciones

1. **Lazy Evaluation**: Query optimization automático
2. **Multi-threading**: Procesamiento paralelo nativo
3. **SIMD**: Operaciones vectorizadas
4. **Streaming**: Procesamiento de datasets grandes sin cargar en memoria

## 🧪 Testing

### Tests Requeridos

1. **Unit Tests**: Cada método individual
2. **Integration Tests**: Flujos completos
3. **Performance Tests**: Benchmarks de rendimiento
4. **Memory Tests**: Uso de memoria con datasets grandes

### Ejemplo de Test

```python
def test_polars_processor_read_write():
    """Test read and write operations."""
    processor = PolarsProcessor(lazy=False)
    
    # Create test data
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    })
    
    # Write
    processor.write(df, "test.parquet")
    
    # Read
    df_read = processor.read("test.parquet")
    
    assert df.equals(df_read)
```

## 📝 Ejemplos de Uso

### Uso Básico

```python
from optimization_core.data import ProcessorFactory

# Crear processor
processor = ProcessorFactory.create_processor()

# Leer datos
df = processor.read("data.parquet", lazy=True)

# Procesar
df = df.filter(pl.col("tokens") > 1000)
df = df.group_by("category").agg([pl.mean("loss"), pl.count()])

# Escribir
processor.write(df, "output.parquet")
```

### Procesamiento de Training Data

```python
processor = PolarsProcessor(lazy=True)

# Procesar datos de entrenamiento
processor.process_training_data(
    input_path="raw_data.parquet",
    output_path="processed_data.parquet",
    min_tokens=1000,
    filters={"quality_score": 0.8}
)
```

### Streaming para Datasets Grandes

```python
processor = PolarsProcessor(streaming=True)

# Leer y procesar en streaming
df = processor.read("large_dataset.parquet", streaming=True)

# Aplicar transformaciones (lazy)
df = df.filter(pl.col("tokens") > 1000)

# Escribir en streaming
processor.write(df, "output.parquet", streaming=True)
```

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2025




