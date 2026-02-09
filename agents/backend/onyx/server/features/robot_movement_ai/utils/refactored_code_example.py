"""
Código Refactorizado - Ejemplo Completo
========================================

Este archivo muestra el código ANTES (problemático) y DESPUÉS (refactorizado)
de manera clara y directa.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


# ============================================================================
# ❌ CÓDIGO ANTES - CON PROBLEMAS
# ============================================================================

class RecordStorage_BEFORE:
    """
    VERSIÓN PROBLEMÁTICA - NO USAR EN PRODUCCIÓN
    
    Problemas:
    1. No usa context managers - archivos pueden no cerrarse
    2. Errores de indentación en read y update
    3. Manejo incorrecto de registros en update
    4. Sin validación de entrada
    5. Sin manejo de errores apropiado
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read(self):
        # ❌ PROBLEMA 1: No usa context manager
        f = open(self.file_path, 'r')
        data = json.load(f)
        f.close()
        # ❌ PROBLEMA 2: Indentación incorrecta - falta validación
        if 'records' in data:
        return data['records']
        return []
    
    def write(self, records):
        # ❌ PROBLEMA 1: No usa context manager
        # ❌ PROBLEMA 2: Sin validación de entrada
        f = open(self.file_path, 'w')
        json.dump({"records": records}, f)
        f.close()
    
    def update(self, record_id, updates):
        # ❌ PROBLEMA 1: Indentación incorrecta
        records = self.read()
        for record in records:
            if record['id'] == record_id:
                # ❌ PROBLEMA 2: Reemplaza todo el registro en lugar de fusionar
                record = updates
                break
        # ❌ PROBLEMA 3: Indentación incorrecta - write fuera del contexto correcto
        self.write(records)


# ============================================================================
# ✅ CÓDIGO DESPUÉS - REFACTORIZADO CORRECTAMENTE
# ============================================================================

class RecordStorage_AFTER:
    """
    VERSIÓN REFACTORIZADA - LISTA PARA PRODUCCIÓN
    
    Mejoras:
    1. ✅ Usa context managers para todas las operaciones de archivo
    2. ✅ Indentación correcta en todos los métodos
    3. ✅ Manejo correcto de registros (fusiona actualizaciones)
    4. ✅ Validación completa de entrada
    5. ✅ Manejo robusto de errores
    """
    
    def __init__(self, file_path: str):
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path debe ser una cadena no vacía")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        """Inicializar archivo con estructura vacía."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"No se puede inicializar el archivo: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Leer todos los registros del archivo.
        
        ✅ FIXED: Context manager
        ✅ FIXED: Indentación correcta
        ✅ FIXED: Validación y manejo de errores
        """
        if not self.file_path.exists():
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'records' not in data:
                return []
            
            records = data.get('records', [])
            if not isinstance(records, list):
                return []
            
            return records
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON inválido: {e}") from e
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error al leer archivo: {e}") from e
    
    def write(self, records: List[Dict[str, Any]]) -> bool:
        """
        Escribir registros al archivo.
        
        ✅ FIXED: Context manager
        ✅ FIXED: Validación de entrada
        ✅ FIXED: Manejo de errores
        """
        if not isinstance(records, list):
            raise ValueError("records debe ser una lista")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Elemento en índice {i} no es un diccionario")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error al escribir archivo: {e}") from e
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Error al serializar: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualizar un registro específico.
        
        ✅ FIXED: Indentación correcta
        ✅ FIXED: Fusiona actualizaciones (no reemplaza)
        ✅ FIXED: Guarda correctamente en archivo
        ✅ FIXED: Validación y manejo de errores
        """
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id debe ser una cadena no vacía")
        
        if not isinstance(updates, dict):
            raise ValueError("updates debe ser un diccionario")
        
        if not updates:
            return False
        
        try:
            records = self.read()
            
            record_found = False
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                
                if record.get('id') == record_id:
                    # ✅ CORRECTO: Fusiona actualizaciones
                    original_id = record.get('id')
                    records[i].update(updates)
                    # Preserva el ID original
                    if 'id' not in records[i] or records[i].get('id') != original_id:
                        records[i]['id'] = original_id
                    record_found = True
                    break
            
            if not record_found:
                return False
            
            # ✅ CORRECTO: Indentación correcta - write después del loop
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": records}, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error al actualizar: {e}") from e
        except (ValueError, TypeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Error inesperado: {e}") from e


# ============================================================================
# COMPARACIÓN LADO A LADO
# ============================================================================

"""
COMPARACIÓN DE CAMBIOS CLAVE:

1. CONTEXT MANAGERS:
   ❌ ANTES: f = open(...); f.close()
   ✅ DESPUÉS: with open(...) as f:

2. INDENTACIÓN EN read():
   ❌ ANTES: if 'records' in data:\nreturn data['records']  # Indentación incorrecta
   ✅ DESPUÉS: if 'records' in data:\n    return data['records']  # Correcta

3. MANEJO DE REGISTROS EN update():
   ❌ ANTES: record = updates  # Reemplaza todo
   ✅ DESPUÉS: records[i].update(updates)  # Fusiona

4. VALIDACIÓN:
   ❌ ANTES: Sin validación
   ✅ DESPUÉS: Validación completa de tipos y valores

5. MANEJO DE ERRORES:
   ❌ ANTES: Sin try/except
   ✅ DESPUÉS: Try/except con mensajes claros
"""


