#!/usr/bin/env python3
"""
Interactive Demo - FileStorage
Demonstrates all features with interactive examples
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_storage import FileStorage
from utils.file_storage_variants import (
    ThreadSafeFileStorage,
    CachedFileStorage,
    CompressedFileStorage,
    BackupFileStorage
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_operations():
    """Demonstrate basic CRUD operations"""
    print_section("Demo 1: Operaciones Básicas CRUD")
    
    storage = FileStorage("demo_data/basic.json")
    
    print("📝 Escribiendo datos iniciales...")
    initial_data = [
        {"id": "1", "name": "Alice", "role": "admin", "active": True},
        {"id": "2", "name": "Bob", "role": "user", "active": True},
        {"id": "3", "name": "Charlie", "role": "user", "active": False}
    ]
    storage.write(initial_data)
    print(f"   ✓ Escritos {len(initial_data)} registros")
    
    print("\n📖 Leyendo datos...")
    records = storage.read()
    print(f"   ✓ Leídos {len(records)} registros")
    for record in records:
        print(f"      - {record['name']} ({record['role']})")
    
    print("\n✏️  Actualizando registro '1'...")
    success = storage.update("1", {"role": "super_admin", "last_updated": "2024-01-01"})
    if success:
        updated = storage.get("1")
        print(f"   ✓ Actualizado: {updated['name']} ahora es {updated['role']}")
    
    print("\n➕ Agregando nuevo registro...")
    storage.add({"id": "4", "name": "Diana", "role": "user", "active": True})
    print(f"   ✓ Agregado registro '4'")
    
    print("\n🔍 Obteniendo registro específico...")
    record = storage.get("2")
    if record:
        print(f"   ✓ Encontrado: {record['name']} - {record['role']}")
    
    print("\n🗑️  Eliminando registro '3'...")
    success = storage.delete("3")
    if success:
        print(f"   ✓ Eliminado registro '3'")
    
    print("\n📊 Estado final:")
    final_records = storage.read()
    print(f"   Total registros: {len(final_records)}")
    for record in final_records:
        status = "✓" if record.get('active', False) else "✗"
        print(f"   {status} {record['name']} ({record['role']})")
    
    # Cleanup
    if os.path.exists("demo_data/basic.json"):
        os.remove("demo_data/basic.json")


def demo_error_handling():
    """Demonstrate error handling"""
    print_section("Demo 2: Manejo de Errores")
    
    storage = FileStorage("demo_data/error_demo.json")
    
    print("❌ Intentando escribir tipo incorrecto...")
    try:
        storage.write("not a list")
    except TypeError as e:
        print(f"   ✓ Capturado TypeError: {e}")
    
    print("\n❌ Intentando escribir lista con items inválidos...")
    try:
        storage.write([1, 2, 3])
    except ValueError as e:
        print(f"   ✓ Capturado ValueError: {e}")
    
    print("\n❌ Intentando actualizar con ID de tipo incorrecto...")
    try:
        storage.update(123, {"key": "value"})
    except TypeError as e:
        print(f"   ✓ Capturado TypeError: {e}")
    
    print("\n❌ Intentando actualizar con ID vacío...")
    try:
        storage.update("", {"key": "value"})
    except ValueError as e:
        print(f"   ✓ Capturado ValueError: {e}")
    
    print("\n❌ Intentando actualizar registro inexistente...")
    storage.write([{"id": "1", "name": "Test"}])
    success = storage.update("999", {"name": "Updated"})
    if not success:
        print(f"   ✓ Retornado False (registro no encontrado)")
    
    print("\n❌ Intentando leer archivo inexistente...")
    os.remove("demo_data/error_demo.json")
    records = storage.read()
    print(f"   ✓ Retornado lista vacía: {records}")


def demo_thread_safe():
    """Demonstrate thread-safe operations"""
    print_section("Demo 3: Operaciones Thread-Safe")
    
    import threading
    import time
    
    storage = ThreadSafeFileStorage("demo_data/thread_safe.json")
    storage.write([{"id": "counter", "value": 0}])
    
    def increment_counter(thread_id: int, iterations: int):
        for _ in range(iterations):
            records = storage.read()
            for i, record in enumerate(records):
                if record.get('id') == 'counter':
                    records[i]['value'] = records[i].get('value', 0) + 1
                    storage.write(records)
                    break
            time.sleep(0.001)
    
    print("🔄 Ejecutando 3 threads incrementando contador...")
    threads = []
    for i in range(3):
        t = threading.Thread(target=increment_counter, args=(i, 10))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    final = storage.get("counter")
    print(f"   ✓ Valor final del contador: {final['value']} (esperado: 30)")
    
    # Cleanup
    if os.path.exists("demo_data/thread_safe.json"):
        os.remove("demo_data/thread_safe.json")


def demo_cached():
    """Demonstrate caching"""
    print_section("Demo 4: Cache en Memoria")
    
    storage = CachedFileStorage("demo_data/cached.json", cache_ttl=5.0)
    
    print("📝 Escribiendo datos...")
    storage.write([{"id": "1", "name": "Cached Item"}])
    
    print("\n📖 Primera lectura (sin cache)...")
    import time
    start = time.time()
    data1 = storage.read()
    time1 = time.time() - start
    print(f"   ✓ Tiempo: {time1:.4f}s")
    
    print("\n📖 Segunda lectura (con cache)...")
    start = time.time()
    data2 = storage.read()
    time2 = time.time() - start
    print(f"   ✓ Tiempo: {time2:.4f}s (más rápido)")
    
    print("\n🔄 Invalidando cache y leyendo de nuevo...")
    storage.invalidate_cache()
    start = time.time()
    data3 = storage.read()
    time3 = time.time() - start
    print(f"   ✓ Tiempo: {time3:.4f}s")
    
    # Cleanup
    if os.path.exists("demo_data/cached.json"):
        os.remove("demo_data/cached.json")


def demo_backup():
    """Demonstrate automatic backups"""
    print_section("Demo 5: Backups Automáticos")
    
    storage = BackupFileStorage("demo_data/backup_demo.json", max_backups=3)
    
    print("📝 Escribiendo datos iniciales...")
    storage.write([{"id": "1", "version": 1}])
    
    print("\n✏️  Actualizando (crea backup automático)...")
    storage.update("1", {"version": 2})
    
    print("\n✏️  Actualizando de nuevo...")
    storage.update("1", {"version": 3})
    
    print("\n📁 Verificando backups creados...")
    backup_dir = "backups"
    if os.path.exists(backup_dir):
        backups = [f for f in os.listdir(backup_dir) if f.startswith("backup_demo")]
        print(f"   ✓ Encontrados {len(backups)} backups")
        for backup in sorted(backups)[-3:]:
            print(f"      - {backup}")
    
    # Cleanup
    if os.path.exists("demo_data/backup_demo.json"):
        os.remove("demo_data/backup_demo.json")
    if os.path.exists(backup_dir):
        import shutil
        shutil.rmtree(backup_dir)


def demo_compressed():
    """Demonstrate compression"""
    print_section("Demo 6: Compresión")
    
    normal_storage = FileStorage("demo_data/normal.json")
    compressed_storage = CompressedFileStorage("demo_data/compressed.json.gz")
    
    # Crear datos grandes
    large_data = [{"id": str(i), "data": "x" * 1000} for i in range(100)]
    
    print("📝 Escribiendo datos grandes (normal)...")
    normal_storage.write(large_data)
    normal_size = os.path.getsize("demo_data/normal.json")
    print(f"   ✓ Tamaño: {normal_size:,} bytes")
    
    print("\n📝 Escribiendo datos grandes (comprimido)...")
    compressed_storage.write(large_data)
    compressed_size = os.path.getsize("demo_data/compressed.json.gz")
    print(f"   ✓ Tamaño: {compressed_size:,} bytes")
    
    ratio = (1 - compressed_size / normal_size) * 100
    print(f"\n💾 Compresión: {ratio:.1f}% más pequeño")
    
    print("\n📖 Leyendo datos comprimidos...")
    data = compressed_storage.read()
    print(f"   ✓ Leídos {len(data)} registros correctamente")
    
    # Cleanup
    for file in ["demo_data/normal.json", "demo_data/compressed.json.gz"]:
        if os.path.exists(file):
            os.remove(file)


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  🚀 FileStorage - Demostración Interactiva")
    print("=" * 70)
    
    # Crear directorio de demo
    os.makedirs("demo_data", exist_ok=True)
    
    try:
        demo_basic_operations()
        demo_error_handling()
        demo_thread_safe()
        demo_cached()
        demo_backup()
        demo_compressed()
        
        print_section("✅ Demostración Completada")
        print("Todos los ejemplos se ejecutaron correctamente.")
        print("\nArchivos de demo limpiados.")
        
    except Exception as e:
        print(f"\n❌ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpiar directorio de demo
        if os.path.exists("demo_data"):
            import shutil
            shutil.rmtree("demo_data", ignore_errors=True)


if __name__ == "__main__":
    main()


