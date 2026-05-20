"""
Ejemplo de uso de DataStorage
==============================

Este archivo demuestra cómo usar la clase DataStorage refactorizada.
"""

from data_storage import DataStorage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Ejemplo de uso de DataStorage."""
    
    storage = DataStorage("data/example.json")
    
    # Escribir datos iniciales
    initial_data = {
        "records": {
            "user_001": {
                "name": "Juan Pérez",
                "email": "juan@example.com",
                "age": 30
            },
            "user_002": {
                "name": "María García",
                "email": "maria@example.com",
                "age": 25
            }
        }
    }
    
    print("📝 Escribiendo datos iniciales...")
    if storage.write(initial_data):
        print("✅ Datos escritos exitosamente")
    else:
        print("❌ Error al escribir datos")
        return
    
    # Leer datos
    print("\n📖 Leyendo datos...")
    data = storage.read()
    if data:
        print(f"✅ Datos leídos: {len(data.get('records', {}))} registros")
        for record_id, record in data.get('records', {}).items():
            print(f"   - {record_id}: {record.get('name')}")
    else:
        print("❌ Error al leer datos")
        return
    
    # Actualizar un registro
    print("\n🔄 Actualizando registro 'user_001'...")
    if storage.update("user_001", {"age": 31, "city": "Madrid"}):
        print("✅ Registro actualizado exitosamente")
        
        # Verificar la actualización
        updated_record = storage.get_record("user_001")
        if updated_record:
            print(f"   Registro actualizado: {updated_record}")
    else:
        print("❌ Error al actualizar registro")
    
    # Agregar un nuevo registro
    print("\n➕ Agregando nuevo registro...")
    if storage.add_record("user_003", {
        "name": "Carlos López",
        "email": "carlos@example.com",
        "age": 28
    }):
        print("✅ Nuevo registro agregado")
    else:
        print("❌ Error al agregar registro")
    
    # Listar todos los registros
    print("\n📋 Listando todos los registros...")
    record_ids = storage.list_records()
    print(f"✅ Total de registros: {len(record_ids)}")
    for record_id in record_ids:
        record = storage.get_record(record_id)
        if record:
            print(f"   - {record_id}: {record.get('name')}")
    
    # Eliminar un registro
    print("\n🗑️  Eliminando registro 'user_002'...")
    if storage.delete_record("user_002"):
        print("✅ Registro eliminado exitosamente")
    else:
        print("❌ Error al eliminar registro")
    
    # Verificar estado final
    print("\n📊 Estado final:")
    final_data = storage.read()
    if final_data:
        print(f"✅ Total de registros restantes: {len(final_data.get('records', {}))}")


if __name__ == "__main__":
    main()


