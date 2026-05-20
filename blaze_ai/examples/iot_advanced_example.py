"""
Blaze AI Advanced IoT Module Example

This example demonstrates how to use the Advanced IoT module
for device management, data collection, and IoT operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta

from blaze_ai.modules import create_iot_advanced_module
from blaze_ai.modules.iot_advanced import (
    DeviceType, ProtocolType, DeviceStatus, DataType, SecurityLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def iot_basic_example():
    """Basic IoT operations example."""
    print("🌐 Iniciando ejemplo básico de IoT Avanzado...")
    
    # Crear módulo de IoT avanzado
    iot = create_iot_advanced_module(
        network_name="blaze-ai-iot-demo",
        max_devices=1000,
        supported_protocols=[ProtocolType.MQTT, ProtocolType.HTTP, ProtocolType.WEBSOCKET],
        security_level=SecurityLevel.STANDARD,
        health_check_interval=30.0,
        data_sync_interval=120.0
    )
    
    # Inicializar módulo
    success = await iot.initialize()
    if not success:
        print("❌ Error al inicializar el módulo de IoT avanzado")
        return
    
    print("✅ Módulo de IoT avanzado inicializado correctamente")
    
    # Cerrar módulo
    await iot.shutdown()
    print("🔒 Módulo de IoT avanzado cerrado")

async def device_management_example():
    """Device management example."""
    print("\n📱 Ejemplo de gestión de dispositivos...")
    
    iot = create_iot_advanced_module(
        network_name="blaze-ai-device-management",
        auto_discovery=True,
        security_level=SecurityLevel.STANDARD
    )
    
    await iot.initialize()
    
    # Registrar sensor de temperatura
    temp_sensor_id = await iot.register_device({
        "name": "Temperature_Sensor_001",
        "type": "sensor",
        "protocol": "mqtt",
        "ip_address": "192.168.1.100",
        "mac_address": "AA:BB:CC:DD:EE:01",
        "firmware_version": "2.1.0",
        "hardware_version": "1.0.0",
        "capabilities": ["temperature_reading", "humidity_reading"],
        "data_types": ["temperature", "humidity"],
        "location": {"lat": 40.7128, "lon": -74.0060, "alt": 10.0},
        "metadata": {"room": "living_room", "floor": "1"}
    })
    
    if temp_sensor_id:
        print(f"✅ Sensor de temperatura registrado: {temp_sensor_id}")
    else:
        print("❌ Error al registrar sensor de temperatura")
    
    # Registrar actuador de luz
    light_actuator_id = await iot.register_device({
        "name": "Smart_Light_001",
        "type": "actuator",
        "protocol": "http",
        "ip_address": "192.168.1.101",
        "mac_address": "AA:BB:CC:DD:EE:02",
        "firmware_version": "1.5.2",
        "hardware_version": "2.0.0",
        "capabilities": ["light_control", "dimmer", "color_control"],
        "data_types": ["binary", "numeric"],
        "location": {"lat": 40.7128, "lon": -74.0060, "alt": 10.0},
        "metadata": {"room": "living_room", "type": "rgb_bulb"}
    })
    
    if light_actuator_id:
        print(f"✅ Actuador de luz registrado: {light_actuator_id}")
    else:
        print("❌ Error al registrar actuador de luz")
    
    # Registrar cámara de seguridad
    camera_id = await iot.register_device({
        "name": "Security_Camera_001",
        "type": "camera",
        "protocol": "websocket",
        "ip_address": "192.168.1.102",
        "mac_address": "AA:BB:CC:DD:EE:03",
        "firmware_version": "3.0.1",
        "hardware_version": "2.1.0",
        "capabilities": ["video_streaming", "motion_detection", "night_vision"],
        "data_types": ["image", "video", "motion"],
        "location": {"lat": 40.7128, "lon": -74.0060, "alt": 15.0},
        "metadata": {"area": "entrance", "resolution": "1080p"}
    })
    
    if camera_id:
        print(f"✅ Cámara de seguridad registrada: {camera_id}")
    else:
        print("❌ Error al registrar cámara de seguridad")
    
    # Obtener todos los dispositivos
    all_devices = await iot.get_all_devices()
    print(f"\n📊 Total de dispositivos registrados: {len(all_devices)}")
    
    # Obtener dispositivos por tipo
    sensors = await iot.get_devices_by_type("sensor")
    actuators = await iot.get_devices_by_type("actuator")
    cameras = await iot.get_devices_by_type("camera")
    
    print(f"   Sensores: {len(sensors)}")
    print(f"   Actuadores: {len(actuators)}")
    print(f"   Cámaras: {len(cameras)}")
    
    await iot.shutdown()

async def data_collection_example():
    """Data collection and storage example."""
    print("\n📊 Ejemplo de recolección de datos...")
    
    iot = create_iot_advanced_module(
        network_name="blaze-ai-data-collection",
        data_retention_days=7,
        compression_enabled=True
    )
    
    await iot.initialize()
    
    # Registrar dispositivo de prueba
    device_id = await iot.register_device({
        "name": "Test_Device_001",
        "type": "sensor",
        "protocol": "mqtt",
        "ip_address": "192.168.1.200",
        "mac_address": "AA:BB:CC:DD:EE:99",
        "firmware_version": "1.0.0",
        "hardware_version": "1.0.0",
        "capabilities": ["data_generation"],
        "data_types": ["temperature", "humidity", "pressure"],
        "location": {"lat": 40.7128, "lon": -74.0060, "alt": 5.0},
        "metadata": {"purpose": "testing"}
    })
    
    if not device_id:
        print("❌ Error al registrar dispositivo de prueba")
        await iot.shutdown()
        return
    
    print(f"✅ Dispositivo de prueba registrado: {device_id}")
    
    # Simular recolección de datos de temperatura
    for i in range(10):
        temp_data = {
            "device_id": device_id,
            "type": "temperature",
            "value": 20.0 + (i * 0.5) + (i % 2 * 0.3),  # Simular variación
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            "quality": 0.95 - (i * 0.01),
            "unit": "celsius",
            "metadata": {"reading_number": i + 1}
        }
        
        success = await iot.store_device_data(temp_data)
        if success:
            print(f"   📈 Datos de temperatura {i+1} almacenados")
        else:
            print(f"   ❌ Error al almacenar datos de temperatura {i+1}")
    
    # Simular recolección de datos de humedad
    for i in range(8):
        humidity_data = {
            "device_id": device_id,
            "type": "humidity",
            "value": 60.0 + (i * 0.8) + (i % 3 * 0.5),
            "timestamp": (datetime.now() - timedelta(minutes=i*2)).isoformat(),
            "quality": 0.92 - (i * 0.005),
            "unit": "percent",
            "metadata": {"reading_number": i + 1}
        }
        
        success = await iot.store_device_data(humidity_data)
        if success:
            print(f"   💧 Datos de humedad {i+1} almacenados")
        else:
            print(f"   ❌ Error al almacenar datos de humedad {i+1}")
    
    # Obtener datos del dispositivo
    print(f"\n📋 Recuperando datos del dispositivo {device_id}...")
    
    # Datos de temperatura de la última hora
    temp_data = await iot.get_device_data(
        device_id,
        start_time=datetime.now() - timedelta(hours=1),
        data_type=DataType.TEMPERATURE
    )
    
    print(f"   📊 Datos de temperatura (última hora): {len(temp_data)} puntos")
    
    # Datos de humedad de la última hora
    humidity_data = await iot.get_device_data(
        device_id,
        start_time=datetime.now() - timedelta(hours=1),
        data_type=DataType.HUMIDITY
    )
    
    print(f"   📊 Datos de humedad (última hora): {len(humidity_data)} puntos")
    
    # Obtener datos agregados
    print(f"\n📈 Datos agregados del dispositivo {device_id}...")
    
    # Promedio de temperatura de la última hora
    temp_avg = await iot.get_device_aggregated_data(device_id, "average", 60)
    if "error" not in temp_avg:
        print(f"   🌡️ Temperatura promedio (última hora): {temp_avg['value']:.2f}°C")
    
    # Promedio de humedad de la última hora
    humidity_avg = await iot.get_device_aggregated_data(device_id, "average", 60)
    if "error" not in humidity_avg:
        print(f"   💧 Humedad promedio (última hora): {humidity_avg['value']:.2f}%")
    
    await iot.shutdown()

async def device_control_example():
    """Device control and command example."""
    print("\n🎮 Ejemplo de control de dispositivos...")
    
    iot = create_iot_advanced_module(
        network_name="blaze-ai-device-control",
        security_level=SecurityLevel.STANDARD
    )
    
    await iot.initialize()
    
    # Registrar actuador de luz
    light_id = await iot.register_device({
        "name": "Smart_Light_Control",
        "type": "actuator",
        "protocol": "mqtt",
        "ip_address": "192.168.1.150",
        "mac_address": "AA:BB:CC:DD:EE:50",
        "firmware_version": "2.0.0",
        "hardware_version": "2.0.0",
        "capabilities": ["light_control", "dimmer", "color_control", "schedule"],
        "data_types": ["binary", "numeric", "json"],
        "location": {"lat": 40.7128, "lon": -74.0060, "alt": 8.0},
        "metadata": {"room": "bedroom", "type": "smart_bulb"}
    })
    
    if not light_id:
        print("❌ Error al registrar actuador de luz")
        await iot.shutdown()
        return
    
    print(f"✅ Actuador de luz registrado: {light_id}")
    
    # Enviar comando para encender la luz
    turn_on_command = await iot.send_device_command(light_id, {
        "type": "turn_on",
        "parameters": {"brightness": 100, "color": "white"},
        "priority": 3,
        "timeout": 10.0
    })
    
    if turn_on_command:
        print(f"   💡 Comando de encendido enviado: {turn_on_command}")
    else:
        print("   ❌ Error al enviar comando de encendido")
    
    # Enviar comando para ajustar brillo
    dim_command = await iot.send_device_command(light_id, {
        "type": "set_brightness",
        "parameters": {"brightness": 50},
        "priority": 2,
        "timeout": 5.0
    })
    
    if dim_command:
        print(f"   🌟 Comando de ajuste de brillo enviado: {dim_command}")
    else:
        print("   ❌ Error al enviar comando de ajuste de brillo")
    
    # Enviar comando para cambiar color
    color_command = await iot.send_device_command(light_id, {
        "type": "set_color",
        "parameters": {"color": "blue", "intensity": 80},
        "priority": 2,
        "timeout": 5.0
    })
    
    if color_command:
        print(f"   🎨 Comando de cambio de color enviado: {color_command}")
    else:
        print("   ❌ Error al enviar comando de cambio de color")
    
    # Enviar comando para apagar la luz
    turn_off_command = await iot.send_device_command(light_id, {
        "type": "turn_off",
        "parameters": {},
        "priority": 3,
        "timeout": 10.0
    })
    
    if turn_off_command:
        print(f"   🔌 Comando de apagado enviado: {turn_off_command}")
    else:
        print("   ❌ Error al enviar comando de apagado")
    
    await iot.shutdown()

async def security_example():
    """Security and authentication example."""
    print("\n🔐 Ejemplo de seguridad y autenticación...")
    
    # Módulo con alta seguridad
    iot_secure = create_iot_advanced_module(
        network_name="blaze-ai-secure-iot",
        security_level=SecurityLevel.HIGH,
        enable_encryption=True,
        enable_authentication=True,
        certificate_path="./certificates"
    )
    
    await iot_secure.initialize()
    
    # Registrar dispositivo con alta seguridad
    secure_device_id = await iot_secure.register_device({
        "name": "Secure_Sensor_001",
        "type": "sensor",
        "protocol": "mqtt",
        "ip_address": "192.168.1.250",
        "mac_address": "AA:BB:CC:DD:EE:99",
        "firmware_version": "3.0.0",
        "hardware_version": "2.0.0",
        "capabilities": ["encrypted_data", "secure_communication"],
        "data_types": ["numeric", "binary"],
        "location": {"lat": 40.7128, "lon": -74.0060, "alt": 12.0},
        "metadata": {"security_level": "high", "encryption": "AES-256"}
    })
    
    if secure_device_id:
        print(f"✅ Dispositivo seguro registrado: {secure_device_id}")
    else:
        print("❌ Error al registrar dispositivo seguro")
    
    # Módulo con seguridad estándar
    iot_standard = create_iot_advanced_module(
        network_name="blaze-ai-standard-iot",
        security_level=SecurityLevel.STANDARD,
        enable_encryption=True,
        enable_authentication=True
    )
    
    await iot_standard.initialize()
    
    # Registrar dispositivo con seguridad estándar
    standard_device_id = await iot_standard.register_device({
        "name": "Standard_Sensor_001",
        "type": "sensor",
        "protocol": "http",
        "ip_address": "192.168.1.251",
        "mac_address": "AA:BB:CC:DD:EE:98",
        "firmware_version": "2.0.0",
        "hardware_version": "1.5.0",
        "capabilities": ["basic_security"],
        "data_types": ["numeric"],
        "location": {"lat": 40.7128, "lon": -74.0060, "alt": 10.0},
        "metadata": {"security_level": "standard"}
    })
    
    if standard_device_id:
        print(f"✅ Dispositivo estándar registrado: {standard_device_id}")
    else:
        print("❌ Error al registrar dispositivo estándar")
    
    # Cerrar módulos
    await iot_secure.shutdown()
    await iot_standard.shutdown()

async def monitoring_example():
    """System monitoring and metrics example."""
    print("\n📊 Ejemplo de monitoreo y métricas...")
    
    iot = create_iot_advanced_module(
        network_name="blaze-ai-monitoring",
        health_check_interval=15.0,
        data_sync_interval=60.0
    )
    
    await iot.initialize()
    
    # Registrar múltiples dispositivos para monitoreo
    devices = []
    for i in range(5):
        device_id = await iot.register_device({
            "name": f"Monitor_Device_{i+1:03d}",
            "type": "sensor",
            "protocol": "mqtt",
            "ip_address": f"192.168.1.{100+i}",
            "mac_address": f"AA:BB:CC:DD:EE:{50+i:02d}",
            "firmware_version": "1.0.0",
            "hardware_version": "1.0.0",
            "capabilities": ["monitoring"],
            "data_types": ["numeric"],
            "location": {"lat": 40.7128, "lon": -74.0060, "alt": 5.0 + i},
            "metadata": {"zone": f"zone_{i+1}"}
        })
        
        if device_id:
            devices.append(device_id)
            print(f"   📱 Dispositivo {i+1} registrado: {device_id}")
    
    print(f"\n📊 Total de dispositivos registrados: {len(devices)}")
    
    # Simular algunos dispositivos offline
    if len(devices) >= 2:
        await iot.update_device_status(devices[1], "offline")
        print(f"   🔴 Dispositivo {devices[1]} marcado como offline")
    
    if len(devices) >= 4:
        await iot.update_device_status(devices[3], "error")
        print(f"   ⚠️ Dispositivo {devices[3]} marcado como error")
    
    # Esperar un poco para que se ejecuten los health checks
    await asyncio.sleep(20)
    
    # Obtener métricas del sistema
    metrics = await iot.get_metrics()
    print(f"\n📈 Métricas del sistema IoT:")
    print(f"   Total de dispositivos: {metrics.total_devices}")
    print(f"   Dispositivos online: {metrics.online_devices}")
    print(f"   Dispositivos offline: {metrics.offline_devices}")
    print(f"   Dispositivos con error: {metrics.error_devices}")
    print(f"   Total de puntos de datos: {metrics.total_data_points}")
    print(f"   Última actualización: {metrics.last_updated}")
    
    # Verificar estado de salud del módulo
    health = await iot.health_check()
    print(f"\n🏥 Estado de salud del módulo:")
    print(f"   Estado: {health['status']}")
    print(f"   Total de dispositivos: {health['total_devices']}")
    print(f"   Dispositivos online: {health['online_devices']}")
    print(f"   Dispositivos offline: {health['offline_devices']}")
    print(f"   Dispositivos con error: {health['error_devices']}")
    print(f"   Total de puntos de datos: {health['total_data_points']}")
    print(f"   Health check activo: {health['health_check_active']}")
    print(f"   Data sync activo: {health['data_sync_active']}")
    print(f"   Protocolos soportados: {', '.join(health['supported_protocols'])}")
    print(f"   Nivel de seguridad: {health['security_level']}")
    
    await iot.shutdown()

async def main():
    """Main function to run all examples."""
    print("🌐 BLAZE AI ADVANCED IoT MODULE EXAMPLES")
    print("=" * 60)
    
    try:
        # Ejemplo básico
        await iot_basic_example()
        
        # Ejemplo de gestión de dispositivos
        await device_management_example()
        
        # Ejemplo de recolección de datos
        await data_collection_example()
        
        # Ejemplo de control de dispositivos
        await device_control_example()
        
        # Ejemplo de seguridad
        await security_example()
        
        # Ejemplo de monitoreo
        await monitoring_example()
        
        print("\n🎉 ¡Todos los ejemplos completados exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        logger.exception("Error en ejemplo de IoT avanzado")

if __name__ == "__main__":
    asyncio.run(main())

