"""
Ejemplo de uso de RecordStorage
================================

Este archivo demuestra cómo usar la clase RecordStorage refactorizada
con todas las mejoras de manejo de archivos y errores.
"""

import logging
from record_storage import RecordStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Ejemplo completo de uso de RecordStorage."""
    
    print("=" * 70)
    print("Ejemplo de uso de RecordStorage")
    print("=" * 70)
    
    storage = RecordStorage("data/example_records.json")
    
    print("\n1. Agregando registros iniciales...")
    try:
        records = [
            {"id": "user_001", "name": "Juan Pérez", "email": "juan@example.com", "age": 30},
            {"id": "user_002", "name": "María García", "email": "maria@example.com", "age": 25},
            {"id": "user_003", "name": "Carlos López", "email": "carlos@example.com", "age": 28}
        ]
        
        if storage.write(records):
            print(f"✅ {len(records)} registros escritos exitosamente")
        else:
            print("❌ Error al escribir registros")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n2. Leyendo todos los registros...")
    try:
        all_records = storage.read()
        print(f"✅ Leídos {len(all_records)} registros")
        for record in all_records:
            print(f"   - {record.get('id')}: {record.get('name')}")
    except Exception as e:
        print(f"❌ Error al leer: {e}")
        return
    
    print("\n3. Obteniendo un registro específico...")
    try:
        record = storage.get("user_001")
        if record:
            print(f"✅ Registro encontrado: {record}")
        else:
            print("❌ Registro no encontrado")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n4. Actualizando un registro...")
    try:
        if storage.update("user_001", {"age": 31, "city": "Madrid"}):
            print("✅ Registro actualizado exitosamente")
            updated = storage.get("user_001")
            if updated:
                print(f"   Registro actualizado: {updated}")
        else:
            print("❌ No se pudo actualizar el registro")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n5. Agregando un nuevo registro...")
    try:
        new_record = {
            "id": "user_004",
            "name": "Ana Martínez",
            "email": "ana@example.com",
            "age": 32
        }
        if storage.add(new_record):
            print("✅ Nuevo registro agregado")
        else:
            print("❌ No se pudo agregar el registro")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n6. Listando todos los registros después de las operaciones...")
    try:
        all_records = storage.read()
        print(f"✅ Total de registros: {len(all_records)}")
        for record in all_records:
            print(f"   - {record.get('id')}: {record.get('name')} (edad: {record.get('age')})")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n7. Eliminando un registro...")
    try:
        if storage.delete("user_002"):
            print("✅ Registro eliminado exitosamente")
        else:
            print("❌ No se pudo eliminar el registro")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n8. Verificando estado final...")
    try:
        final_records = storage.read()
        print(f"✅ Registros restantes: {len(final_records)}")
        for record in final_records:
            print(f"   - {record.get('id')}: {record.get('name')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n9. Probando validación de errores...")
    print("   - Intentando actualizar con ID inválido...")
    try:
        result = storage.update("", {"test": "value"})
        print(f"   Resultado: {result}")
    except (ValueError, TypeError) as e:
        print(f"   ✅ Error capturado correctamente: {e}")
    
    print("   - Intentando actualizar con tipo incorrecto...")
    try:
        result = storage.update(123, {"test": "value"})
        print(f"   Resultado: {result}")
    except TypeError as e:
        print(f"   ✅ Error de tipo capturado correctamente: {e}")
    
    print("\n" + "=" * 70)
    print("Ejemplo completado")
    print("=" * 70)


if __name__ == "__main__":
    main()
