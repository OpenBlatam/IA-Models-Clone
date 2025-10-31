"""
Blaze AI Federated Learning Advanced Module Example

This example demonstrates how to use the Federated Learning module
for distributed AI training with privacy preservation.
"""

import asyncio
import numpy as np
from datetime import datetime
from blaze_ai.modules import create_federated_learning_module
from blaze_ai.modules.federated_learning import (
    AggregationMethod, PrivacyLevel, CommunicationProtocol
)

async def federated_learning_basic_example():
    """Basic federated learning example with multiple clients."""
    print("🧠 BLAZE AI FEDERATED LEARNING BASIC EXAMPLE")
    print("=" * 50)
    
    # Create federated learning module
    federated_learning = await create_federated_learning_module(
        max_clients=100,
        min_clients_per_round=3,
        max_clients_per_round=5,
        aggregation_method=AggregationMethod.FEDAVG,
        privacy_level=PrivacyLevel.STANDARD,
        communication_protocol=CommunicationProtocol.HTTP,
        enable_encryption=True,
        enable_authentication=True
    )
    
    print("✅ Módulo de Federated Learning inicializado")
    
    # Registrar múltiples clientes
    clients = []
    client_names = ["Hospital_A", "Hospital_B", "Clinic_C", "Research_Lab_D", "Medical_Center_E"]
    
    for i, name in enumerate(client_names):
        client_info = {
            "name": name,
            "capabilities": ["medical_imaging", "patient_data", "ml_training"],
            "data_size": 10000 + i * 5000,  # Diferentes tamaños de datos
            "compute_power": 1.0 + i * 0.2,  # Diferentes capacidades computacionales
            "network_speed": 100 + i * 50,    # Diferentes velocidades de red
            "metadata": {
                "location": f"Region_{i+1}",
                "specialization": "cardiology" if i % 2 == 0 else "neurology",
                "compliance": "HIPAA" if i < 3 else "GDPR"
            }
        }
        
        client_id = await federated_learning.register_client(client_info)
        clients.append(client_id)
        print(f"📱 Cliente registrado: {name} (ID: {client_id})")
    
    print(f"\n📊 Total de clientes registrados: {len(clients)}")
    
    # Iniciar ronda de entrenamiento
    round_config = {
        "num_clients": 4,
        "description": "Entrenamiento de modelo de diagnóstico cardiológico"
    }
    
    round_id = await federated_learning.start_training_round(round_config)
    print(f"\n🚀 Ronda de entrenamiento iniciada: {round_id}")
    
    # Simular envío de actualizaciones de modelo desde los clientes
    for i, client_id in enumerate(clients[:4]):  # Solo los primeros 4 clientes
        # Simular pesos de modelo (en un caso real, estos vendrían del cliente)
        model_weights = {
            "conv1.weight": np.random.randn(32, 3, 3, 3) * 0.1,
            "conv1.bias": np.random.randn(32) * 0.1,
            "conv2.weight": np.random.randn(64, 32, 3, 3) * 0.1,
            "conv2.bias": np.random.randn(64) * 0.1,
            "fc1.weight": np.random.randn(128, 64 * 7 * 7) * 0.1,
            "fc1.bias": np.random.randn(128) * 0.1,
            "fc2.weight": np.random.randn(10, 128) * 0.1,
            "fc2.bias": np.random.randn(10) * 0.1
        }
        
        update_data = {
            "client_id": client_id,
            "model_weights": model_weights,
            "metadata": {
                "training_samples": 1000 + i * 500,
                "validation_accuracy": 0.85 + i * 0.05,
                "training_loss": 0.3 - i * 0.02,
                "epochs_trained": 10 + i * 2
            }
        }
        
        await federated_learning.submit_model_update(round_id, update_data)
        print(f"📤 Actualización enviada desde cliente {client_id}")
    
    # Esperar un momento para que se complete la agregación
    await asyncio.sleep(2)
    
    # Verificar estado de la ronda
    round_status = await federated_learning.get_round_status(round_id)
    print(f"\n📋 Estado de la ronda {round_id}:")
    print(f"   Estado: {round_status['status']}")
    print(f"   Clientes: {round_status['clients']}")
    print(f"   Actualizaciones recibidas: {round_status['updates_received']}")
    print(f"   Total de clientes: {round_status['total_clients']}")
    
    if round_status['aggregation_result']:
        print(f"   Tiempo de agregación: {round_status['aggregation_result']['aggregation_time']:.2f}s")
    
    # Cerrar módulo
    await federated_learning.shutdown()
    print("\n✅ Ejemplo básico completado")

async def secure_aggregation_example():
    """Example using secure aggregation with differential privacy."""
    print("\n🔐 BLAZE AI SECURE AGGREGATION EXAMPLE")
    print("=" * 50)
    
    # Create federated learning module with secure aggregation
    federated_learning = await create_federated_learning_module(
        max_clients=50,
        min_clients_per_round=3,
        max_clients_per_round=5,
        aggregation_method=AggregationMethod.SECURE_AGGREGATION,
        privacy_level=PrivacyLevel.HIGH,
        communication_protocol=CommunicationProtocol.HTTPS,
        enable_encryption=True,
        enable_authentication=True,
        enable_audit_logging=True,
        noise_scale=0.05,
        clipping_norm=0.5,
        epsilon=0.5,
        delta=1e-6
    )
    
    print("✅ Módulo de Federated Learning con agregación segura inicializado")
    
    # Registrar clientes sensibles (hospitales, bancos, etc.)
    sensitive_clients = []
    client_types = [
        {"name": "Hospital_Alpha", "type": "healthcare", "compliance": "HIPAA"},
        {"name": "Bank_Beta", "type": "financial", "compliance": "SOX"},
        {"name": "Research_Gamma", "type": "academic", "compliance": "FERPA"},
        {"name": "Clinic_Delta", "type": "healthcare", "compliance": "GDPR"}
    ]
    
    for client_info in client_types:
        client_data = {
            "name": client_info["name"],
            "capabilities": ["secure_training", "privacy_preserving", "audit_trail"],
            "data_size": 50000,
            "compute_power": 2.0,
            "network_speed": 200,
            "metadata": {
                "industry": client_info["type"],
                "compliance": client_info["compliance"],
                "security_level": "high",
                "encryption": "AES-256"
            }
        }
        
        client_id = await federated_learning.register_client(client_data)
        sensitive_clients.append(client_id)
        print(f"🔒 Cliente sensible registrado: {client_info['name']} (ID: {client_id})")
    
    # Iniciar ronda de entrenamiento seguro
    secure_round_config = {
        "num_clients": 3,
        "description": "Entrenamiento seguro de modelo de detección de fraude",
        "security_requirements": ["encryption", "authentication", "audit"]
    }
    
    secure_round_id = await federated_learning.start_training_round(secure_round_config)
    print(f"\n🛡️ Ronda segura iniciada: {secure_round_id}")
    
    # Simular envío de actualizaciones seguras
    for i, client_id in enumerate(sensitive_clients[:3]):
        # Modelo más simple para demostración
        secure_weights = {
            "input.weight": np.random.randn(64, 10) * 0.1,
            "input.bias": np.random.randn(64) * 0.1,
            "hidden.weight": np.random.randn(32, 64) * 0.1,
            "hidden.bias": np.random.randn(32) * 0.1,
            "output.weight": np.random.randn(2, 32) * 0.1,
            "output.bias": np.random.randn(2) * 0.1
        }
        
        secure_update = {
            "client_id": client_id,
            "model_weights": secure_weights,
            "metadata": {
                "training_samples": 2000 + i * 1000,
                "validation_accuracy": 0.92 + i * 0.03,
                "training_loss": 0.25 - i * 0.03,
                "epochs_trained": 15 + i * 3,
                "privacy_budget_used": 0.1 + i * 0.05,
                "security_checks_passed": True
            },
            "signature": f"secure_signature_{client_id}_{i}"  # En un caso real sería una firma criptográfica real
        }
        
        await federated_learning.submit_model_update(secure_round_id, secure_update)
        print(f"🔐 Actualización segura enviada desde {client_id}")
    
    # Esperar agregación segura
    await asyncio.sleep(3)
    
    # Verificar estado de la ronda segura
    secure_round_status = await federated_learning.get_round_status(secure_round_id)
    print(f"\n📋 Estado de la ronda segura {secure_round_id}:")
    print(f"   Estado: {secure_round_status['status']}")
    print(f"   Clientes: {secure_round_status['clients']}")
    print(f"   Actualizaciones recibidas: {secure_round_status['updates_received']}")
    
    if secure_round_status['aggregation_result']:
        print(f"   Tiempo de agregación segura: {secure_round_status['aggregation_result']['aggregation_time']:.2f}s")
        print(f"   Método de agregación: {federated_learning.config.aggregation_method.value}")
    
    # Cerrar módulo
    await federated_learning.shutdown()
    print("\n✅ Ejemplo de agregación segura completado")

async def differential_privacy_example():
    """Example using differential privacy for maximum privacy protection."""
    print("\n🕵️ BLAZE AI DIFFERENTIAL PRIVACY EXAMPLE")
    print("=" * 50)
    
    # Create federated learning module with differential privacy
    federated_learning = await create_federated_learning_module(
        max_clients=100,
        min_clients_per_round=4,
        max_clients_per_round=6,
        aggregation_method=AggregationMethod.DIFFERENTIAL_PRIVACY,
        privacy_level=PrivacyLevel.MILITARY,
        communication_protocol=CommunicationProtocol.HTTPS,
        enable_encryption=True,
        enable_authentication=True,
        enable_audit_logging=True,
        noise_scale=0.02,  # Muy bajo para máxima privacidad
        clipping_norm=0.3,  # Clipping más agresivo
        epsilon=0.1,        # Epsilon muy bajo para máxima privacidad
        delta=1e-7          # Delta muy bajo
    )
    
    print("✅ Módulo de Federated Learning con privacidad diferencial inicializado")
    
    # Registrar clientes ultra-sensibles
    ultra_sensitive_clients = []
    ultra_client_types = [
        {"name": "Military_Research_A", "type": "defense", "classification": "top_secret"},
        {"name": "Intelligence_Agency_B", "type": "intelligence", "classification": "secret"},
        {"name": "Nuclear_Facility_C", "type": "energy", "classification": "restricted"},
        {"name": "Government_Lab_D", "type": "government", "classification": "confidential"},
        {"name": "Space_Research_E", "type": "aerospace", "classification": "classified"}
    ]
    
    for client_info in ultra_client_types:
        client_data = {
            "name": client_info["name"],
            "capabilities": ["ultra_secure_training", "differential_privacy", "zero_knowledge"],
            "data_size": 100000,
            "compute_power": 5.0,
            "network_speed": 500,
            "metadata": {
                "industry": client_info["type"],
                "classification": client_info["classification"],
                "security_level": "military",
                "encryption": "AES-512",
                "privacy_guarantee": "epsilon=0.1, delta=1e-7"
            }
        }
        
        client_id = await federated_learning.register_client(client_data)
        ultra_sensitive_clients.append(client_id)
        print(f"🕵️ Cliente ultra-sensible registrado: {client_info['name']} (ID: {client_id})")
    
    # Iniciar ronda con privacidad diferencial
    dp_round_config = {
        "num_clients": 4,
        "description": "Entrenamiento ultra-seguro de modelo de reconocimiento de patrones",
        "privacy_requirements": ["differential_privacy", "zero_knowledge", "military_grade"]
    }
    
    dp_round_id = await federated_learning.start_training_round(dp_round_config)
    print(f"\n🕵️ Ronda con privacidad diferencial iniciada: {dp_round_id}")
    
    # Simular envío de actualizaciones ultra-seguras
    for i, client_id in enumerate(ultra_sensitive_clients[:4]):
        # Modelo simple para demostración
        dp_weights = {
            "conv1.weight": np.random.randn(16, 1, 5, 5) * 0.05,
            "conv1.bias": np.random.randn(16) * 0.05,
            "conv2.weight": np.random.randn(32, 16, 5, 5) * 0.05,
            "conv2.bias": np.random.randn(32) * 0.05,
            "fc1.weight": np.random.randn(128, 32 * 5 * 5) * 0.05,
            "fc1.bias": np.random.randn(128) * 0.05,
            "fc2.weight": np.random.randn(10, 128) * 0.05,
            "fc2.bias": np.random.randn(10) * 0.05
        }
        
        dp_update = {
            "client_id": client_id,
            "model_weights": dp_weights,
            "metadata": {
                "training_samples": 5000 + i * 2000,
                "validation_accuracy": 0.95 + i * 0.02,
                "training_loss": 0.15 - i * 0.02,
                "epochs_trained": 20 + i * 5,
                "privacy_budget_used": 0.05 + i * 0.02,
                "differential_privacy_applied": True,
                "noise_added": True,
                "clipping_applied": True,
                "security_audit_passed": True
            },
            "signature": f"ultra_secure_signature_{client_id}_{i}"
        }
        
        await federated_learning.submit_model_update(dp_round_id, dp_update)
        print(f"🕵️ Actualización ultra-segura enviada desde {client_id}")
    
    # Esperar agregación con privacidad diferencial
    await asyncio.sleep(4)
    
    # Verificar estado de la ronda con privacidad diferencial
    dp_round_status = await federated_learning.get_round_status(dp_round_id)
    print(f"\n📋 Estado de la ronda con privacidad diferencial {dp_round_id}:")
    print(f"   Estado: {dp_round_status['status']}")
    print(f"   Clientes: {dp_round_status['clients']}")
    print(f"   Actualizaciones recibidas: {dp_round_status['updates_received']}")
    
    if dp_round_status['aggregation_result']:
        print(f"   Tiempo de agregación con DP: {dp_round_status['aggregation_result']['aggregation_time']:.2f}s")
        print(f"   Método de agregación: {federated_learning.config.aggregation_method.value}")
        print(f"   Nivel de privacidad: {federated_learning.config.privacy_level.value}")
        print(f"   Epsilon: {federated_learning.config.epsilon}")
        print(f"   Delta: {federated_learning.config.delta}")
    
    # Cerrar módulo
    await federated_learning.shutdown()
    print("\n✅ Ejemplo de privacidad diferencial completado")

async def monitoring_and_metrics_example():
    """Example showing monitoring and metrics capabilities."""
    print("\n📊 BLAZE AI MONITORING AND METRICS EXAMPLE")
    print("=" * 50)
    
    # Create federated learning module
    federated_learning = await create_federated_learning_module(
        max_clients=200,
        min_clients_per_round=3,
        max_clients_per_round=8,
        aggregation_method=AggregationMethod.FEDAVG,
        privacy_level=PrivacyLevel.STANDARD
    )
    
    print("✅ Módulo de Federated Learning para monitoreo inicializado")
    
    # Registrar muchos clientes para demostrar métricas
    for i in range(15):
        client_info = {
            "name": f"Client_{i+1:02d}",
            "capabilities": ["ml_training", "data_processing"],
            "data_size": 1000 + i * 1000,
            "compute_power": 1.0 + i * 0.1,
            "network_speed": 50 + i * 10,
            "metadata": {"region": f"Region_{i % 5 + 1}"}
        }
        
        await federated_learning.register_client(client_info)
    
    print(f"📱 Total de clientes registrados: {federated_learning.metrics.total_clients}")
    
    # Iniciar múltiples rondas
    round_ids = []
    for i in range(3):
        round_config = {
            "num_clients": 4 + i,
            "description": f"Ronda de entrenamiento {i+1}"
        }
        
        round_id = await federated_learning.start_training_round(round_config)
        round_ids.append(round_id)
        print(f"🚀 Ronda {i+1} iniciada: {round_id}")
    
    # Simular algunas actualizaciones
    for i, round_id in enumerate(round_ids):
        for j in range(3):  # Solo 3 clientes por ronda para simplicidad
            client_id = f"Client_{(i*3 + j + 1):02d}"
            
            # Verificar si el cliente está en esta ronda
            round_status = await federated_learning.get_round_status(round_id)
            if client_id in round_status['clients']:
                model_weights = {
                    "layer1.weight": np.random.randn(64, 32) * 0.1,
                    "layer1.bias": np.random.randn(64) * 0.1,
                    "layer2.weight": np.random.randn(32, 64) * 0.1,
                    "layer2.bias": np.random.randn(32) * 0.1
                }
                
                update_data = {
                    "client_id": client_id,
                    "model_weights": model_weights,
                    "metadata": {"round": i+1, "client": j+1}
                }
                
                await federated_learning.submit_model_update(round_id, update_data)
    
    # Esperar un momento para que se completen las rondas
    await asyncio.sleep(3)
    
    # Obtener métricas del sistema
    metrics = await federated_learning.get_metrics()
    print(f"\n📊 Métricas del sistema de Federated Learning:")
    print(f"   Total de rondas: {metrics.total_rounds}")
    print(f"   Rondas completadas: {metrics.completed_rounds}")
    print(f"   Rondas fallidas: {metrics.failed_rounds}")
    print(f"   Clientes activos: {metrics.active_clients}")
    print(f"   Total de clientes: {metrics.total_clients}")
    print(f"   Tiempo promedio de ronda: {metrics.average_round_time:.2f}s")
    print(f"   Tiempo promedio de agregación: {metrics.average_aggregation_time:.2f}s")
    print(f"   Violaciones de privacidad: {metrics.privacy_violations}")
    print(f"   Incidentes de seguridad: {metrics.security_incidents}")
    
    # Verificar estado de salud
    health = await federated_learning.health_check()
    print(f"\n🏥 Estado de salud del módulo:")
    print(f"   Estado: {health['status']}")
    print(f"   Clientes activos: {health['active_clients']}")
    print(f"   Total de clientes: {health['total_clients']}")
    print(f"   Rondas activas: {health['active_rounds']}")
    print(f"   Rondas completadas: {health['completed_rounds']}")
    print(f"   Rondas fallidas: {health['failed_rounds']}")
    print(f"   Método de agregación: {health['aggregation_method']}")
    print(f"   Nivel de privacidad: {health['privacy_level']}")
    print(f"   Protocolo de comunicación: {health['communication_protocol']}")
    
    # Obtener información de algunos clientes
    print(f"\n📱 Información de clientes:")
    for i in range(3):
        client_id = f"Client_{(i+1):02d}"
        client_info = await federated_learning.get_client_info(client_id)
        if client_info:
            print(f"   Cliente {client_id}:")
            print(f"     Nombre: {client_info['name']}")
            print(f"     Estado: {client_info['status']}")
            print(f"     Capacidades: {', '.join(client_info['capabilities'])}")
            print(f"     Tamaño de datos: {client_info['data_size']}")
            print(f"     Poder computacional: {client_info['compute_power']}")
            print(f"     Velocidad de red: {client_info['network_speed']}")
            print(f"     Rondas de participación: {client_info['participation_rounds']}")
    
    # Cerrar módulo
    await federated_learning.shutdown()
    print("\n✅ Ejemplo de monitoreo y métricas completado")

async def main():
    """Run all federated learning examples."""
    print("🚀 BLAZE AI FEDERATED LEARNING ADVANCED MODULE EXAMPLES")
    print("=" * 60)
    
    try:
        # Ejemplo básico
        await federated_learning_basic_example()
        
        # Ejemplo de agregación segura
        await secure_aggregation_example()
        
        # Ejemplo de privacidad diferencial
        await differential_privacy_example()
        
        # Ejemplo de monitoreo y métricas
        await monitoring_and_metrics_example()
        
        print("\n🎉 Todos los ejemplos de Federated Learning completados exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error en los ejemplos: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

