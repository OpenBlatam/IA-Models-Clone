"""
Data IO Generator - Generador de utilidades de I/O de datos
===========================================================

Genera utilidades para importación/exportación de datos:
- Data export/import
- Format conversion
- Data backup/restore
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DataIOGenerator:
    """Generador de utilidades de I/O de datos"""
    
    def __init__(self):
        """Inicializa el generador de I/O de datos"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de I/O de datos.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        data_io_dir = utils_dir / "data_io"
        data_io_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_data_exporter(data_io_dir, keywords, project_info)
        self._generate_data_importer(data_io_dir, keywords, project_info)
        self._generate_data_io_init(data_io_dir, keywords)
    
    def _generate_data_io_init(
        self,
        data_io_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de I/O de datos"""
        
        init_content = '''"""
Data IO Utilities Module
==========================

Utilidades para importación y exportación de datos.
"""

from .data_exporter import (
    DataExporter,
    export_to_json,
    export_to_csv,
    export_to_pickle,
    export_to_hdf5,
)
from .data_importer import (
    DataImporter,
    import_from_json,
    import_from_csv,
    import_from_pickle,
    import_from_hdf5,
)

__all__ = [
    "DataExporter",
    "export_to_json",
    "export_to_csv",
    "export_to_pickle",
    "export_to_hdf5",
    "DataImporter",
    "import_from_json",
    "import_from_csv",
    "import_from_pickle",
    "import_from_hdf5",
]
'''
        
        (data_io_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_data_exporter(
        self,
        data_io_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera exportador de datos"""
        
        exporter_content = '''"""
Data Exporter - Exportador de datos
====================================

Utilidades para exportar datos a diferentes formatos.
"""

import json
import pickle
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Exportador de datos a diferentes formatos.
    
    Soporta JSON, CSV, Pickle, HDF5.
    """
    
    def __init__(self):
        """Inicializa el exportador"""
        pass
    
    def export_to_json(
        self,
        data: Any,
        file_path: Path,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> Path:
        """
        Exporta datos a JSON.
        
        Args:
            data: Datos a exportar
            file_path: Ruta del archivo
            indent: Indentación (opcional)
            ensure_ascii: Si asegurar ASCII
        
        Returns:
            Ruta al archivo exportado
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir numpy arrays a listas
        if NUMPY_AVAILABLE:
            data = self._convert_numpy_types(data)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        logger.info(f"Datos exportados a JSON: {file_path}")
        return file_path
    
    def export_to_csv(
        self,
        data: Any,
        file_path: Path,
        **kwargs,
    ) -> Path:
        """
        Exporta datos a CSV.
        
        Args:
            data: Datos a exportar (DataFrame o lista de dicts)
            file_path: Ruta del archivo
            **kwargs: Argumentos adicionales para pandas.to_csv
        
        Returns:
            Ruta al archivo exportado
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas no disponible. Instala con: pip install pandas")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, (list, dict)):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Tipo de datos no soportado: {type(data)}")
        
        df.to_csv(file_path, index=False, **kwargs)
        logger.info(f"Datos exportados a CSV: {file_path}")
        return file_path
    
    def export_to_pickle(
        self,
        data: Any,
        file_path: Path,
        protocol: int = 4,
    ) -> Path:
        """
        Exporta datos a Pickle.
        
        Args:
            data: Datos a exportar
            file_path: Ruta del archivo
            protocol: Protocolo de pickle (opcional)
        
        Returns:
            Ruta al archivo exportado
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
        
        logger.info(f"Datos exportados a Pickle: {file_path}")
        return file_path
    
    def export_to_hdf5(
        self,
        data: Dict[str, Any],
        file_path: Path,
        compression: Optional[str] = "gzip",
    ) -> Path:
        """
        Exporta datos a HDF5.
        
        Args:
            data: Diccionario con datos a exportar
            file_path: Ruta del archivo
            compression: Compresión a usar (opcional)
        
        Returns:
            Ruta al archivo exportado
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py no disponible. Instala con: pip install h5py")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(file_path, "w") as f:
            for key, value in data.items():
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    f.create_dataset(
                        key,
                        data=value,
                        compression=compression,
                    )
                else:
                    # Convertir a numpy array
                    if NUMPY_AVAILABLE:
                        arr = np.array(value)
                        f.create_dataset(
                            key,
                            data=arr,
                            compression=compression,
                        )
                    else:
                        raise ValueError(f"No se puede exportar {type(value)} a HDF5 sin numpy")
        
        logger.info(f"Datos exportados a HDF5: {file_path}")
        return file_path
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Convierte tipos numpy a tipos nativos de Python.
        
        Args:
            obj: Objeto a convertir
        
        Returns:
            Objeto convertido
        """
        if NUMPY_AVAILABLE:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: self._convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._convert_numpy_types(item) for item in obj]
        
        return obj


def export_to_json(
    data: Any,
    file_path: Path,
    **kwargs,
) -> Path:
    """
    Función helper para exportar a JSON.
    
    Args:
        data: Datos a exportar
        file_path: Ruta del archivo
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al archivo exportado
    """
    exporter = DataExporter()
    return exporter.export_to_json(data, file_path, **kwargs)


def export_to_csv(
    data: Any,
    file_path: Path,
    **kwargs,
) -> Path:
    """
    Función helper para exportar a CSV.
    
    Args:
        data: Datos a exportar
        file_path: Ruta del archivo
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al archivo exportado
    """
    exporter = DataExporter()
    return exporter.export_to_csv(data, file_path, **kwargs)


def export_to_pickle(
    data: Any,
    file_path: Path,
    **kwargs,
) -> Path:
    """
    Función helper para exportar a Pickle.
    
    Args:
        data: Datos a exportar
        file_path: Ruta del archivo
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al archivo exportado
    """
    exporter = DataExporter()
    return exporter.export_to_pickle(data, file_path, **kwargs)


def export_to_hdf5(
    data: Dict[str, Any],
    file_path: Path,
    **kwargs,
) -> Path:
    """
    Función helper para exportar a HDF5.
    
    Args:
        data: Datos a exportar
        file_path: Ruta del archivo
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al archivo exportado
    """
    exporter = DataExporter()
    return exporter.export_to_hdf5(data, file_path, **kwargs)
'''
        
        (data_io_dir / "data_exporter.py").write_text(exporter_content, encoding="utf-8")
    
    def _generate_data_importer(
        self,
        data_io_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera importador de datos"""
        
        importer_content = '''"""
Data Importer - Importador de datos
====================================

Utilidades para importar datos desde diferentes formatos.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import logging

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataImporter:
    """
    Importador de datos desde diferentes formatos.
    
    Soporta JSON, CSV, Pickle, HDF5.
    """
    
    def __init__(self):
        """Inicializa el importador"""
        pass
    
    def import_from_json(
        self,
        file_path: Path,
    ) -> Any:
        """
        Importa datos desde JSON.
        
        Args:
            file_path: Ruta del archivo
        
        Returns:
            Datos importados
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Datos importados desde JSON: {file_path}")
        return data
    
    def import_from_csv(
        self,
        file_path: Path,
        **kwargs,
    ) -> Any:
        """
        Importa datos desde CSV.
        
        Args:
            file_path: Ruta del archivo
            **kwargs: Argumentos adicionales para pandas.read_csv
        
        Returns:
            DataFrame con datos importados
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas no disponible. Instala con: pip install pandas")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Datos importados desde CSV: {file_path}")
        return df
    
    def import_from_pickle(
        self,
        file_path: Path,
    ) -> Any:
        """
        Importa datos desde Pickle.
        
        Args:
            file_path: Ruta del archivo
        
        Returns:
            Datos importados
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        logger.info(f"Datos importados desde Pickle: {file_path}")
        return data
    
    def import_from_hdf5(
        self,
        file_path: Path,
        keys: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Importa datos desde HDF5.
        
        Args:
            file_path: Ruta del archivo
            keys: Claves a importar (opcional, todas si None)
        
        Returns:
            Diccionario con datos importados
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py no disponible. Instala con: pip install h5py")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        data = {}
        with h5py.File(file_path, "r") as f:
            if keys is None:
                keys = list(f.keys())
            
            for key in keys:
                if key in f:
                    data[key] = f[key][:]
        
        logger.info(f"Datos importados desde HDF5: {file_path}")
        return data


def import_from_json(
    file_path: Path,
) -> Any:
    """
    Función helper para importar desde JSON.
    
    Args:
        file_path: Ruta del archivo
    
    Returns:
        Datos importados
    """
    importer = DataImporter()
    return importer.import_from_json(file_path)


def import_from_csv(
    file_path: Path,
    **kwargs,
) -> Any:
    """
    Función helper para importar desde CSV.
    
    Args:
        file_path: Ruta del archivo
        **kwargs: Argumentos adicionales
    
    Returns:
        Datos importados
    """
    importer = DataImporter()
    return importer.import_from_csv(file_path, **kwargs)


def import_from_pickle(
    file_path: Path,
) -> Any:
    """
    Función helper para importar desde Pickle.
    
    Args:
        file_path: Ruta del archivo
    
    Returns:
        Datos importados
    """
    importer = DataImporter()
    return importer.import_from_pickle(file_path)


def import_from_hdf5(
    file_path: Path,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para importar desde HDF5.
    
    Args:
        file_path: Ruta del archivo
        **kwargs: Argumentos adicionales
    
    Returns:
        Datos importados
    """
    importer = DataImporter()
    return importer.import_from_hdf5(file_path, **kwargs)
'''
        
        (data_io_dir / "data_importer.py").write_text(importer_content, encoding="utf-8")

