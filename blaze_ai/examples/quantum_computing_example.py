"""
🌐 Ejemplos de uso del Módulo de Quantum Computing
Demuestra las capacidades avanzadas de computación cuántica del sistema Blaze AI
"""

import asyncio
import numpy as np
from typing import Dict, List, Any

# Importar el módulo de Quantum Computing
from ..modules.quantum_computing import (
    create_quantum_computing_module,
    QuantumAlgorithmType,
    QuantumBackendType
)


async def ejemplo_basico_quantum():
    """Ejemplo básico de uso del módulo de Quantum Computing"""
    print("🚀 Ejemplo Básico de Quantum Computing")
    print("=" * 50)
    
    # Crear módulo de Quantum Computing
    quantum_module = create_quantum_computing_module(
        max_qubits=16,
        shots=2048,
        hybrid_integration=True
    )
    
    # Inicializar módulo
    await quantum_module.initialize()
    
    # Crear circuito cuántico simple
    circuit_id = await quantum_module.create_circuit(
        name="circuito_basico",
        qubits=4,
        operations=["H(0)", "X(1)", "CX(0,1)", "H(2)"]
    )
    
    print(f"✅ Circuito cuántico creado: {circuit_id}")
    
    # Ejecutar trabajo cuántico con algoritmo QAOA
    job_id = await quantum_module.execute_quantum_job(
        circuit_id=circuit_id,
        algorithm_type=QuantumAlgorithmType.QAOA,
        parameters={"size": 8, "constraints": ["binary"]}
    )
    
    print(f"✅ Trabajo cuántico iniciado: {job_id}")
    
    # Esperar y obtener resultados
    await asyncio.sleep(2)
    
    # Obtener estado del trabajo
    status = await quantum_module.get_job_status(job_id)
    print(f"📊 Estado del trabajo: {status}")
    
    # Obtener resultados
    result = await quantum_module.get_job_result(job_id)
    if result:
        print(f"🎯 Resultado QAOA: {result}")
    
    # Obtener métricas
    metrics = await quantum_module.get_metrics()
    print(f"📈 Métricas del módulo: {metrics}")
    
    # Apagar módulo
    await quantum_module.shutdown()
    print("✅ Módulo apagado correctamente\n")


async def ejemplo_optimizacion_hibrida():
    """Ejemplo de optimización híbrida clásico-cuántica"""
    print("🔄 Ejemplo de Optimización Híbrida Clásico-Cuántica")
    print("=" * 60)
    
    # Crear módulo con configuración híbrida
    quantum_module = create_quantum_computing_module(
        enabled_algorithms=[QuantumAlgorithmType.QAOA, QuantumAlgorithmType.VQE],
        hybrid_integration=True,
        optimization_level=3
    )
    
    await quantum_module.initialize()
    
    # Problema clásico: optimización de portafolio
    classical_problem = {
        "type": "portfolio_optimization",
        "size": 12,
        "constraints": ["budget_limit", "risk_tolerance"],
        "target": "maximize_returns",
        "assets": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"]
    }
    
    # Parámetros cuánticos para QAOA
    quantum_parameters = {
        "size": 12,
        "constraints": ["binary"],
        "optimization_type": "combinatorial",
        "depth": 3
    }
    
    # Ejecutar optimización híbrida
    print("🔄 Ejecutando optimización híbrida...")
    hybrid_result = await quantum_module.hybrid_optimization(
        classical_problem=classical_problem,
        quantum_parameters=quantum_parameters
    )
    
    print("🎯 Resultado de optimización híbrida:")
    print(f"  - Tiempo de ejecución: {hybrid_result['execution_time']:.3f}s")
    print(f"  - Ventaja híbrida: {hybrid_result['hybrid_advantage']:.3f}")
    print(f"  - Solución final: {hybrid_result['final_result']['final_solution']}")
    print(f"  - Costo final: {hybrid_result['final_result']['final_cost']:.4f}")
    
    await quantum_module.shutdown()
    print("✅ Optimización híbrida completada\n")


async def ejemplo_machine_learning_cuantico():
    """Ejemplo de Machine Learning cuántico"""
    print("🧠 Ejemplo de Machine Learning Cuántico")
    print("=" * 50)
    
    # Crear módulo con ML cuántico habilitado
    quantum_module = create_quantum_computing_module(
        quantum_ml_enabled=True,
        max_qubits=20,
        optimization_level=2
    )
    
    await quantum_module.initialize()
    
    # Generar datos de ejemplo
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    
    # Datos de entrenamiento
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    print(f"📊 Datos de entrenamiento: {X_train.shape}")
    print(f"🎯 Etiquetas: {y_train.shape}")
    
    # Entrenar clasificador cuántico
    print("🧠 Entrenando clasificador cuántico...")
    model_id = await quantum_module.quantum_ml.train_quantum_classifier(
        data=X_train,
        labels=y_train
    )
    
    print(f"✅ Modelo cuántico entrenado: {model_id}")
    
    # Obtener información del modelo
    model_info = quantum_module.quantum_ml.models[model_id]
    print(f"📈 Épocas de entrenamiento: {model_info['epochs']}")
    print(f"🎯 Pérdida final: {model_info['final_loss']:.4f}")
    print(f"🔧 Parámetros cuánticos: {model_info['parameters']}")
    
    # Ejecutar trabajo de ML cuántico
    circuit_id = await quantum_module.create_circuit(
        name="ml_circuit",
        qubits=16,
        operations=["H(0)", "RX(1)", "CNOT(0,1)", "RY(2)"]
    )
    
    job_id = await quantum_module.execute_quantum_job(
        circuit_id=circuit_id,
        algorithm_type=QuantumAlgorithmType.QUANTUM_ML,
        parameters={"model_id": model_id, "data_size": n_samples}
    )
    
    # Esperar resultados
    await asyncio.sleep(1.5)
    
    result = await quantum_module.get_job_result(job_id)
    if result:
        print("🎯 Resultado de ML cuántico:")
        print(f"  - Precisión del modelo: {result['model_accuracy']:.3f}")
        print(f"  - Tiempo de entrenamiento: {result['training_time']:.3f}s")
        print(f"  - Ventaja cuántica: {result['quantum_advantage']:.3f}")
    
    await quantum_module.shutdown()
    print("✅ Machine Learning cuántico completado\n")


async def ejemplo_criptografia_post_cuantica():
    """Ejemplo de criptografía post-cuántica"""
    print("🔐 Ejemplo de Criptografía Post-Cuántica")
    print("=" * 50)
    
    # Crear módulo con criptografía post-cuántica
    quantum_module = create_quantum_computing_module(
        post_quantum_crypto=True,
        max_qubits=64
    )
    
    await quantum_module.initialize()
    
    # Generar par de claves post-cuántico
    print("🔑 Generando par de claves post-cuántico...")
    keypair = await quantum_module.post_quantum_crypto.generate_post_quantum_keypair(
        algorithm="lattice"
    )
    
    print(f"✅ Claves generadas:")
    print(f"  - Algoritmo: {keypair['algorithm']}")
    print(f"  - Nivel de seguridad: {keypair['security_level']}")
    print(f"  - Clave pública: {keypair['public_key'][:20]}...")
    print(f"  - Clave privada: {keypair['private_key'][:20]}...")
    
    # Mensaje a firmar
    message = "Mensaje confidencial del sistema Blaze AI"
    print(f"📝 Mensaje: {message}")
    
    # Firmar mensaje
    print("✍️ Firmando mensaje...")
    signature = await quantum_module.post_quantum_crypto.post_quantum_sign(
        message=message,
        private_key=keypair['private_key']
    )
    
    print(f"✅ Firma generada: {signature}")
    
    # Verificar firma
    print("🔍 Verificando firma...")
    is_valid = await quantum_module.post_quantum_crypto.post_quantum_verify(
        message=message,
        signature=signature,
        public_key=keypair['public_key']
    )
    
    print(f"✅ Verificación: {'Exitosa' if is_valid else 'Fallida'}")
    
    # Probar con mensaje modificado
    modified_message = "Mensaje modificado del sistema Blaze AI"
    is_valid_modified = await quantum_module.post_quantum_crypto.post_quantum_verify(
        message=modified_message,
        signature=signature,
        public_key=keypair['public_key']
    )
    
    print(f"🔍 Verificación mensaje modificado: {'Exitosa' if is_valid_modified else 'Fallida'}")
    
    await quantum_module.shutdown()
    print("✅ Criptografía post-cuántica completada\n")


async def ejemplo_algoritmos_cuanticos_avanzados():
    """Ejemplo de algoritmos cuánticos avanzados"""
    print("⚡ Ejemplo de Algoritmos Cuánticos Avanzados")
    print("=" * 55)
    
    # Crear módulo con todos los algoritmos habilitados
    quantum_module = create_quantum_computing_module(
        enabled_algorithms=[
            QuantumAlgorithmType.QAOA,
            QuantumAlgorithmType.VQE,
            QuantumAlgorithmType.GROVER,
            QuantumAlgorithmType.QUANTUM_ML
        ],
        max_qubits=24,
        optimization_level=3
    )
    
    await quantum_module.initialize()
    
    # Crear circuito para múltiples algoritmos
    circuit_id = await quantum_module.create_circuit(
        name="circuito_avanzado",
        qubits=16,
        operations=["H(0)", "X(1)", "CX(0,1)", "H(2)", "RY(3)", "CNOT(2,3)"]
    )
    
    # Ejecutar múltiples algoritmos
    algorithms = [
        (QuantumAlgorithmType.QAOA, "Optimización combinatoria"),
        (QuantumAlgorithmType.VQE, "Eigenvalores variacionales"),
        (QuantumAlgorithmType.GROVER, "Búsqueda cuántica"),
        (QuantumAlgorithmType.QUANTUM_ML, "Machine Learning cuántico")
    ]
    
    jobs = {}
    
    for algorithm_type, description in algorithms:
        print(f"🚀 Ejecutando {description}...")
        
        job_id = await quantum_module.execute_quantum_job(
            circuit_id=circuit_id,
            algorithm_type=algorithm_type,
            parameters={"size": 16, "depth": 2}
        )
        
        jobs[algorithm_type] = job_id
        print(f"✅ Trabajo iniciado: {job_id}")
    
    # Esperar y recopilar resultados
    print("\n⏳ Esperando resultados...")
    await asyncio.sleep(3)
    
    # Mostrar resultados de todos los algoritmos
    print("\n🎯 Resultados de Algoritmos Cuánticos:")
    print("-" * 40)
    
    for algorithm_type, job_id in jobs.items():
        status = await quantum_module.get_job_status(job_id)
        result = await quantum_module.get_job_result(job_id)
        
        print(f"\n🔬 {algorithm_type.value.upper()}:")
        print(f"  - Estado: {status['status']}")
        print(f"  - Tiempo: {status.get('execution_time', 0):.3f}s")
        
        if result and 'error' not in result:
            if algorithm_type == QuantumAlgorithmType.QAOA:
                print(f"  - Solución: {result['best_solution'][:10]}...")
                print(f"  - Costo: {result['best_cost']:.4f}")
            elif algorithm_type == QuantumAlgorithmType.VQE:
                print(f"  - Eigenvalor: {result['eigenvalue']:.4f}")
                print(f"  - Convergido: {result['converged']}")
            elif algorithm_type == QuantumAlgorithmType.GROVER:
                print(f"  - Solución encontrada: {result['solution_found']}")
                print(f"  - Probabilidad: {result['success_probability']:.3f}")
            elif algorithm_type == QuantumAlgorithmType.QUANTUM_ML:
                print(f"  - Precisión: {result['model_accuracy']:.3f}")
                print(f"  - Ventaja cuántica: {result['quantum_advantage']:.3f}")
    
    # Métricas finales
    metrics = await quantum_module.get_metrics()
    print(f"\n📊 Métricas Finales:")
    print(f"  - Total de trabajos: {metrics.total_jobs}")
    print(f"  - Trabajos completados: {metrics.completed_jobs}")
    print(f"  - Trabajos híbridos: {metrics.hybrid_jobs}")
    print(f"  - Trabajos de ML: {metrics.quantum_ml_jobs}")
    print(f"  - Qubits utilizados: {metrics.total_qubits_used}")
    print(f"  - Tiempo promedio: {metrics.average_execution_time:.3f}s")
    
    await quantum_module.shutdown()
    print("\n✅ Algoritmos cuánticos avanzados completados\n")


async def ejemplo_integracion_sistema():
    """Ejemplo de integración con otros módulos del sistema"""
    print("🔗 Ejemplo de Integración con Sistema Blaze AI")
    print("=" * 55)
    
    # Crear módulo con integración completa
    quantum_module = create_quantum_computing_module(
        hybrid_integration=True,
        post_quantum_crypto=True,
        quantum_ml_enabled=True,
        backend_type=QuantumBackendType.HYBRID
    )
    
    await quantum_module.initialize()
    
    # Simular integración con otros módulos
    print("🔗 Simulando integración con módulos del sistema...")
    
    # Integración con ML clásico
    print("  📊 Integrando con módulo de ML...")
    classical_ml_data = {
        "model_type": "neural_network",
        "data_size": 10000,
        "features": 128,
        "target": "classification"
    }
    
    # Integración con blockchain
    print("  ⛓️ Integrando con módulo de blockchain...")
    blockchain_data = {
        "transaction_type": "quantum_optimization",
        "smart_contract": "quantum_optimizer_v1",
        "gas_limit": 5000000
    }
    
    # Integración con edge computing
    print("  🌐 Integrando con módulo de edge computing...")
    edge_data = {
        "node_type": "quantum_edge",
        "processing_capacity": "hybrid",
        "latency_requirement": "ultra_low"
    }
    
    # Crear circuito integrado
    circuit_id = await quantum_module.create_circuit(
        name="circuito_integrado",
        qubits=32,
        operations=["H(0)", "X(1)", "CX(0,1)", "H(2)", "RY(3)", "CNOT(2,3)"]
    )
    
    # Ejecutar optimización integrada
    print("🚀 Ejecutando optimización integrada...")
    
    integrated_result = await quantum_module.hybrid_optimization(
        classical_problem={
            "ml_data": classical_ml_data,
            "blockchain_data": blockchain_data,
            "edge_data": edge_data,
            "size": 32,
            "constraints": ["ml_compatibility", "blockchain_verification", "edge_optimization"]
        },
        quantum_parameters={
            "size": 32,
            "depth": 4,
            "hybrid_mode": True,
            "integration_level": "full"
        }
    )
    
    print("🎯 Resultado de integración:")
    print(f"  - Tiempo total: {integrated_result['execution_time']:.3f}s")
    print(f"  - Ventaja híbrida: {integrated_result['hybrid_advantage']:.3f}")
    print(f"  - Validación clásica: {integrated_result['final_result']['classical_validation']}")
    print(f"  - Ventaja cuántica verificada: {integrated_result['final_result']['quantum_advantage_verified']}")
    
    # Estado de salud del módulo
    health = await quantum_module.get_health_status()
    print(f"\n🏥 Estado de salud del módulo:")
    print(f"  - Estado: {health['status']}")
    print(f"  - Circuitos: {health['circuits_count']}")
    print(f"  - Trabajos activos: {health['active_jobs']}")
    print(f"  - Librerías cuánticas: {'Disponibles' if health['quantum_libraries_available'] else 'Simuladas'}")
    print(f"  - PennyLane: {'Disponible' if health['pennylane_available'] else 'No disponible'}")
    
    await quantum_module.shutdown()
    print("\n✅ Integración con sistema completada\n")


async def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("🌐 SISTEMA BLAZE AI - MÓDULO DE QUANTUM COMPUTING")
    print("=" * 60)
    print("Ejecutando ejemplos de computación cuántica...\n")
    
    try:
        # Ejecutar ejemplos secuencialmente
        await ejemplo_basico_quantum()
        await ejemplo_optimizacion_hibrida()
        await ejemplo_machine_learning_cuantico()
        await ejemplo_criptografia_post_cuantica()
        await ejemplo_algoritmos_cuanticos_avanzados()
        await ejemplo_integracion_sistema()
        
        print("🎉 ¡Todos los ejemplos de Quantum Computing ejecutados exitosamente!")
        print("🚀 El sistema Blaze AI ahora tiene capacidades cuánticas avanzadas")
        
    except Exception as e:
        print(f"❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(main())

