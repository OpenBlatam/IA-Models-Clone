"""
Blaze AI Zero-Knowledge Proofs Module Examples

This file provides comprehensive examples demonstrating how to use the
Zero-Knowledge Proofs Module for privacy-preserving AI operations.
"""

import asyncio
import logging
from datetime import datetime
from blaze_ai.modules.zero_knowledge_proofs import (
    create_zero_knowledge_proofs_module_with_defaults,
    ProofType, CircuitType, ZKProofConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_zk_proofs_example():
    """Basic example of zero-knowledge proofs module usage."""
    print("🔐 Ejemplo Básico de Zero-Knowledge Proofs")
    print("=" * 50)
    
    # Crear módulo de zero-knowledge proofs con configuración básica
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_SNARK, ProofType.ZK_STARK],
        security_level=128,
        circuit_optimization=True,
        parallel_generation=True
    )
    
    print(f"✅ Módulo de zero-knowledge proofs creado")
    print(f"   Tipos de pruebas habilitados: {[pt.value for pt in zk_module.config.enabled_proof_types]}")
    print(f"   Nivel de seguridad: {zk_module.config.security_level} bits")
    print(f"   Optimización de circuitos: {zk_module.config.circuit_optimization}")
    print(f"   Generación paralela: {zk_module.config.parallel_generation}")
    
    # Verificar estado de salud
    health = await zk_module.health_check()
    print(f"\n🏥 Estado de salud del módulo:")
    print(f"   Estado: {health['status']}")
    print(f"   Tipos de pruebas habilitados: {health['enabled_proof_types']}")
    print(f"   Circuitos totales: {health['total_circuits']}")
    print(f"   Pruebas totales: {health['total_proofs']}")
    print(f"   Bibliotecas criptográficas disponibles: {health['crypto_available']}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def arithmetic_circuit_example():
    """Example of creating and using arithmetic circuits."""
    print("\n⚡ Ejemplo de Circuitos Aritméticos")
    print("=" * 50)
    
    # Crear módulo
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_SNARK]
    )
    
    # Crear un circuito aritmético simple: y = a * b + c
    circuit_name = "simple_arithmetic"
    circuit_type = CircuitType.ARITHMETIC
        
    # Definir las puertas del circuito
    gates = [
        {"type": "input", "inputs": ["a"], "output": "a", "operation": "input"},
        {"type": "input", "inputs": ["b"], "output": "b", "operation": "input"},
        {"type": "input", "inputs": ["c"], "output": "c", "operation": "input"},
        {"type": "mul", "inputs": ["a", "b"], "output": "temp1", "operation": "mul"},
        {"type": "add", "inputs": ["temp1", "c"], "output": "y", "operation": "add"},
        {"type": "output", "inputs": ["y"], "output": "y", "operation": "output"}
    ]
    
    inputs = ["a", "b", "c"]
    outputs = ["y"]
    
    try:
        # Crear el circuito
        circuit_id = await zk_module.create_circuit(
            name=circuit_name,
            circuit_type=circuit_type,
            gates=gates,
            inputs=inputs,
            outputs=outputs
        )
        
        print(f"✅ Circuito aritmético creado: {circuit_id}")
        print(f"   Nombre: {circuit_name}")
        print(f"   Tipo: {circuit_type.value}")
        print(f"   Entradas: {inputs}")
        print(f"   Salidas: {outputs}")
        print(f"   Puertas: {len(gates)}")
        
        # Verificar que el circuito se creó correctamente
        circuit = zk_module.circuits[circuit_id]
        print(f"\n📋 Detalles del circuito:")
        print(f"   ID: {circuit.circuit_id}")
        print(f"   Creado: {circuit.created_at}")
        print(f"   Metadatos: {circuit.metadata}")
        
        # Obtener métricas del módulo
        metrics = await zk_module.get_metrics()
        print(f"\n📊 Métricas del módulo:")
        print(f"   Circuitos totales: {metrics.total_circuits}")
        print(f"   Circuitos activos: {metrics.active_circuits}")
        
    except Exception as e:
        print(f"❌ Error creando circuito: {e}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def zk_snark_proof_example():
    """Example of generating and verifying ZK-SNARK proofs."""
    print("\n🔒 Ejemplo de Pruebas ZK-SNARK")
    print("=" * 50)
    
    # Crear módulo
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_SNARK],
        security_level=128
    )
    
    try:
        # Crear un circuito simple para demostración
        circuit_name = "age_verification"
        circuit_type = CircuitType.RANGE_CHECK
        
        gates = [
            {"type": "input", "inputs": ["age"], "output": "age", "operation": "input"},
            {"type": "constant", "inputs": ["18"], "output": "min_age", "operation": "constant"},
            {"type": "constant", "inputs": ["65"], "output": "max_age", "operation": "constant"},
            {"type": "output", "inputs": ["age"], "output": "age", "operation": "output"}
        ]
        
        inputs = ["age"]
        outputs = ["age"]
        
        circuit_id = await zk_module.create_circuit(
            name=circuit_name,
            circuit_type=circuit_type,
            gates=gates,
            inputs=inputs,
            outputs=outputs
        )
        
        print(f"✅ Circuito de verificación de edad creado: {circuit_id}")
        
        # Generar una prueba ZK-SNARK
        print(f"\n🔐 Generando prueba ZK-SNARK...")
        
        public_inputs = [25]  # Edad pública (puede ser verificada)
        private_inputs = [25]  # Edad privada (no se revela)
        
        proof_id = await zk_module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_SNARK,
            public_inputs=public_inputs,
            private_inputs=private_inputs
        )
        
        print(f"✅ Prueba ZK-SNARK generada: {proof_id}")
        
        # Verificar el estado de la prueba
        proof_status = await zk_module.get_proof_status(proof_id)
        if proof_status:
            print(f"\n📋 Estado de la prueba:")
            print(f"   ID: {proof_status['proof_id']}")
            print(f"   Tipo: {proof_status['proof_type']}")
            print(f"   Estado: {proof_status['status']}")
            print(f"   Entradas públicas: {proof_status['public_inputs']}")
            print(f"   Generada: {proof_status['generated_at']}")
        
        # Verificar la prueba
        print(f"\n🔍 Verificando prueba...")
        is_valid = await zk_module.verify_proof(proof_id)
        
        if is_valid:
            print(f"✅ Prueba ZK-SNARK verificada exitosamente")
        else:
            print(f"❌ Verificación de prueba falló")
        
        # Obtener métricas finales
        final_metrics = await zk_module.get_metrics()
        print(f"\n📊 Métricas finales:")
        print(f"   Pruebas totales: {final_metrics.total_proofs}")
        print(f"   Pruebas generadas: {final_metrics.generated_proofs}")
        print(f"   Pruebas verificadas: {final_metrics.verified_proofs}")
        print(f"   Pruebas fallidas: {final_metrics.failed_proofs}")
        
    except Exception as e:
        print(f"❌ Error en ejemplo ZK-SNARK: {e}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def zk_stark_proof_example():
    """Example of generating and verifying ZK-STARK proofs."""
    print("\n🌟 Ejemplo de Pruebas ZK-STARK")
    print("=" * 50)
    
    # Crear módulo con ZK-STARK habilitado
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_STARK],
        security_level=128,
        max_circuit_size=1000000  # STARK es mejor para circuitos grandes
    )
    
    try:
        # Crear un circuito más complejo para STARK
        circuit_name = "ml_model_verification"
        circuit_type = CircuitType.ARITHMETIC
        
        # Circuito que verifica cálculos de ML sin revelar el modelo
        gates = [
            {"type": "input", "inputs": ["input_data"], "output": "input_data", "operation": "input"},
            {"type": "input", "inputs": ["model_hash"], "output": "model_hash", "operation": "input"},
            {"type": "input", "inputs": ["expected_output"], "output": "expected_output", "operation": "input"},
            {"type": "constant", "inputs": ["weight1"], "output": "weight1", "operation": "constant"},
            {"type": "constant", "inputs": ["weight2"], "output": "weight2", "operation": "constant"},
            {"type": "mul", "inputs": ["input_data", "weight1"], "output": "temp1", "operation": "mul"},
            {"type": "add", "inputs": ["temp1", "weight2"], "output": "temp2", "operation": "add"},
            {"type": "output", "inputs": ["temp2"], "output": "temp2", "operation": "output"}
        ]
        
        inputs = ["input_data", "model_hash", "expected_output"]
        outputs = ["temp2"]
        
        circuit_id = await zk_module.create_circuit(
            name=circuit_name,
            circuit_type=circuit_type,
            gates=gates,
            inputs=inputs,
            outputs=outputs
        )
        
        print(f"✅ Circuito de verificación de ML creado: {circuit_id}")
        print(f"   Nombre: {circuit_name}")
        print(f"   Tipo: {circuit_type.value}")
        print(f"   Puertas: {len(gates)}")
        
        # Generar una prueba ZK-STARK
        print(f"\n🌟 Generando prueba ZK-STARK...")
        
        public_inputs = [100, "model_hash_123", 250]  # Datos públicos
        private_inputs = [100, "model_hash_123", 250]  # Datos privados
        
        proof_id = await zk_module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_STARK,
            public_inputs=public_inputs,
            private_inputs=private_inputs
        )
        
        print(f"✅ Prueba ZK-STARK generada: {proof_id}")
        
        # Verificar el estado de la prueba
        proof_status = await zk_module.get_proof_status(proof_id)
        if proof_status:
            print(f"\n📋 Estado de la prueba STARK:")
            print(f"   ID: {proof_status['proof_id']}")
            print(f"   Tipo: {proof_status['proof_type']}")
            print(f"   Estado: {proof_status['status']}")
            print(f"   Entradas públicas: {proof_status['public_inputs']}")
        
        # Verificar la prueba
        print(f"\n🔍 Verificando prueba STARK...")
        is_valid = await zk_module.verify_proof(proof_id)
        
        if is_valid:
            print(f"✅ Prueba ZK-STARK verificada exitosamente")
        else:
            print(f"❌ Verificación de prueba STARK falló")
        
    except Exception as e:
        print(f"❌ Error en ejemplo ZK-STARK: {e}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def range_proof_example():
    """Example of generating and verifying range proofs."""
    print("\n📊 Ejemplo de Pruebas de Rango")
    print("=" * 50)
    
    # Crear módulo
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_SNARK]
    )
    
    try:
        # Ejemplo: Probar que una edad está en un rango sin revelar la edad exacta
        print(f"🔐 Generando prueba de rango para edad...")
        
        age = 30
        min_age = 18
        max_age = 65
        commitment = f"commitment_{age}_{secrets.randbelow(1000)}"
        
        # Generar prueba de rango
        range_proof = await zk_module.generate_range_proof(
            value=age,
            min_value=min_age,
            max_value=max_age,
            commitment=commitment
        )
        
        print(f"✅ Prueba de rango generada")
        print(f"   Valor: {age}")
        print(f"   Rango: [{min_age}, {max_age}]")
        print(f"   Commitment: {commitment}")
        print(f"   Hash de prueba: {range_proof['proof_hash']}")
        
        # Verificar la prueba de rango
        print(f"\n🔍 Verificando prueba de rango...")
        is_valid = await zk_module.range_proof.verify_range_proof(range_proof)
        
        if is_valid:
            print(f"✅ Prueba de rango verificada exitosamente")
            print(f"   Se puede probar que la edad está en [{min_age}, {max_age}]")
            print(f"   Sin revelar la edad exacta: {age}")
        else:
            print(f"❌ Verificación de prueba de rango falló")
        
    except Exception as e:
        print(f"❌ Error en ejemplo de prueba de rango: {e}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def membership_proof_example():
    """Example of generating and verifying membership proofs."""
    print("\n🏛️ Ejemplo de Pruebas de Membresía")
    print("=" * 50)
    
    # Crear módulo
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_SNARK]
    )
    
    try:
        # Ejemplo: Probar que un usuario está en una lista de usuarios autorizados
        print(f"🔐 Generando prueba de membresía...")
        
        authorized_users = ["user1", "user2", "user3", "user4", "user5"]
        user_to_prove = "user3"
        merkle_root = f"merkle_root_{hash(''.join(authorized_users))}"
        
        # Generar prueba de membresía
        membership_proof = await zk_module.generate_membership_proof(
            element=user_to_prove,
            set_elements=authorized_users,
            merkle_root=merkle_root
        )
        
        print(f"✅ Prueba de membresía generada")
        print(f"   Usuario: {user_to_prove}")
        print(f"   Conjunto: {authorized_users}")
        print(f"   Merkle Root: {merkle_root}")
        print(f"   Hash de prueba: {membership_proof['proof_hash']}")
        
        # Verificar la prueba de membresía
        print(f"\n🔍 Verificando prueba de membresía...")
        is_valid = await zk_module.membership_proof.verify_membership_proof(
            membership_proof, authorized_users
        )
        
        if is_valid:
            print(f"✅ Prueba de membresía verificada exitosamente")
            print(f"   Se puede probar que '{user_to_prove}' está en la lista autorizada")
            print(f"   Sin revelar la lista completa de usuarios")
        else:
            print(f"❌ Verificación de prueba de membresía falló")
        
        # Probar con un usuario no autorizado
        unauthorized_user = "user6"
        print(f"\n🔍 Probando con usuario no autorizado: {unauthorized_user}")
        
        fake_proof = await zk_module.generate_membership_proof(
            element=unauthorized_user,
            set_elements=authorized_users,
            merkle_root=merkle_root
        )
        
        is_valid_fake = await zk_module.membership_proof.verify_membership_proof(
            fake_proof, authorized_users
        )
        
        if not is_valid_fake:
            print(f"✅ Correctamente rechazó usuario no autorizado")
        else:
            print(f"❌ Incorrectamente aceptó usuario no autorizado")
        
    except Exception as e:
        print(f"❌ Error en ejemplo de prueba de membresía: {e}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def blockchain_integration_example():
    """Example of blockchain integration with zero-knowledge proofs."""
    print("\n⛓️ Ejemplo de Integración con Blockchain")
    print("=" * 50)
    
    # Crear módulo con integración blockchain
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_SNARK, ProofType.ZK_STARK],
        blockchain_integration=True,
        smart_contract_verification=True,
        gas_optimization=True
    )
    
    try:
        print(f"✅ Módulo con integración blockchain creado")
        print(f"   Integración blockchain: {zk_module.config.blockchain_integration}")
        print(f"   Verificación en smart contracts: {zk_module.config.smart_contract_verification}")
        print(f"   Optimización de gas: {zk_module.config.gas_optimization}")
        
        # Crear un circuito para verificación de identidad
        circuit_name = "identity_verification"
        circuit_type = CircuitType.BOOLEAN
        
        gates = [
            {"type": "input", "inputs": ["age"], "output": "age", "operation": "input"},
            {"type": "input", "inputs": ["citizenship"], "output": "citizenship", "operation": "input"},
            {"type": "input", "inputs": ["document_hash"], "output": "document_hash", "operation": "input"},
            {"type": "constant", "inputs": ["18"], "output": "min_age", "operation": "constant"},
            {"type": "constant", "inputs": ["1"], "output": "valid_citizenship", "operation": "constant"},
            {"type": "output", "inputs": ["age", "citizenship", "document_hash"], "output": "valid_identity", "operation": "output"}
        ]
        
        inputs = ["age", "citizenship", "document_hash"]
        outputs = ["valid_identity"]
        
        circuit_id = await zk_module.create_circuit(
            name=circuit_name,
            circuit_type=circuit_type,
            gates=gates,
            inputs=inputs,
            outputs=outputs
        )
        
        print(f"\n✅ Circuito de verificación de identidad creado: {circuit_id}")
        
        # Generar prueba ZK-SNARK para blockchain
        print(f"\n🔐 Generando prueba para blockchain...")
        
        public_inputs = ["valid_identity_hash", "document_hash"]
        private_inputs = [25, 1, "doc_hash_123"]  # Edad, ciudadanía, hash del documento
        
        proof_id = await zk_module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_SNARK,
            public_inputs=public_inputs,
            private_inputs=private_inputs
        )
        
        print(f"✅ Prueba para blockchain generada: {proof_id}")
        
        # Simular verificación en smart contract
        print(f"\n⛓️ Simulando verificación en smart contract...")
        
        # Obtener la prueba
        proof = zk_module.proofs[proof_id]
        
        # En un smart contract real, esto se haría on-chain
        smart_contract_data = {
            "proof_id": proof_id,
            "circuit_hash": proof.proof_data.get("circuit_hash", ""),
            "public_inputs": proof.public_inputs,
            "verification_key": proof.verification_key,
            "proof_hash": proof.proof_data.get("proof_hash", ""),
            "gas_estimate": 150000,  # Estimación de gas para verificación
            "block_number": 12345,
            "transaction_hash": f"tx_{proof_id[:8]}"
        }
        
        print(f"📋 Datos para smart contract:")
        print(f"   Proof ID: {smart_contract_data['proof_id']}")
        print(f"   Circuit Hash: {smart_contract_data['circuit_hash']}")
        print(f"   Gas estimado: {smart_contract_data['gas_estimate']}")
        print(f"   Block Number: {smart_contract_data['block_number']}")
        print(f"   Transaction Hash: {smart_contract_data['transaction_hash']}")
        
        # Verificar la prueba (simulando verificación on-chain)
        print(f"\n🔍 Verificando prueba (simulando smart contract)...")
        is_valid = await zk_module.verify_proof(proof_id)
        
        if is_valid:
            print(f"✅ Prueba verificada exitosamente en blockchain")
            print(f"   La identidad es válida sin revelar datos personales")
            print(f"   Gas consumido: {smart_contract_data['gas_estimate']}")
        else:
            print(f"❌ Verificación de prueba falló en blockchain")
        
    except Exception as e:
        print(f"❌ Error en ejemplo de integración blockchain: {e}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def privacy_preserving_ai_example():
    """Example of privacy-preserving AI operations with ZK proofs."""
    print("\n🤖 Ejemplo de IA con Preservación de Privacidad")
    print("=" * 50)
    
    # Crear módulo
    zk_module = await create_zero_knowledge_proofs_module_with_defaults(
        enabled_proof_types=[ProofType.ZK_SNARK, ProofType.ZK_STARK],
        security_level=256,  # Mayor seguridad para IA
        max_circuit_size=2000000  # Circuitos más grandes para ML
    )
    
    try:
        print(f"✅ Módulo para IA con preservación de privacidad creado")
        print(f"   Nivel de seguridad: {zk_module.config.security_level} bits")
        print(f"   Tamaño máximo de circuito: {zk_module.config.max_circuit_size:,} puertas")
        
        # Crear un circuito para verificación de modelo de ML
        circuit_name = "ml_model_inference"
        circuit_type = CircuitType.ARITHMETIC
        
        # Circuito que verifica inferencia de ML sin revelar el modelo
        gates = [
            {"type": "input", "inputs": ["input_features"], "output": "input_features", "operation": "input"},
            {"type": "input", "inputs": ["model_commitment"], "output": "model_commitment", "operation": "input"},
            {"type": "input", "inputs": ["expected_prediction"], "output": "expected_prediction", "operation": "input"},
            {"type": "constant", "inputs": ["bias"], "output": "bias", "operation": "constant"},
            {"type": "mul", "inputs": ["input_features", "model_commitment"], "output": "weighted_sum", "operation": "mul"},
            {"type": "add", "inputs": ["weighted_sum", "bias"], "output": "prediction", "operation": "add"},
            {"type": "output", "inputs": ["prediction"], "output": "prediction", "operation": "output"}
        ]
        
        inputs = ["input_features", "model_commitment", "expected_prediction"]
        outputs = ["prediction"]
        
        circuit_id = await zk_module.create_circuit(
            name=circuit_name,
            circuit_type=circuit_type,
            gates=gates,
            inputs=inputs,
            outputs=outputs
        )
        
        print(f"\n✅ Circuito de inferencia ML creado: {circuit_id}")
        
        # Generar prueba para inferencia privada
        print(f"\n🔐 Generando prueba para inferencia privada...")
        
        # Datos públicos (pueden ser verificados)
        public_inputs = [100, "model_commitment_hash", 150]
        
        # Datos privados (no se revelan)
        private_inputs = [100, "model_weights_secret", 150]
        
        proof_id = await zk_module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_STARK,  # STARK para circuitos grandes
            public_inputs=public_inputs,
            private_inputs=private_inputs
        )
        
        print(f"✅ Prueba de inferencia privada generada: {proof_id}")
        
        # Verificar la prueba
        print(f"\n🔍 Verificando prueba de inferencia...")
        is_valid = await zk_module.verify_proof(proof_id)
        
        if is_valid:
            print(f"✅ Prueba de inferencia verificada exitosamente")
            print(f"   Se puede probar que la inferencia es correcta")
            print(f"   Sin revelar los pesos del modelo")
            print(f"   Sin revelar los datos de entrenamiento")
            print(f"   Preservando la privacidad del modelo")
        else:
            print(f"❌ Verificación de prueba de inferencia falló")
        
        # Ejemplo de aplicación en fintech
        print(f"\n💰 Aplicación en Fintech:")
        print(f"   - Verificación de score crediticio sin revelar datos personales")
        print(f"   - Validación de transacciones sospechosas preservando privacidad")
        print(f"   - Auditoría de modelos de ML sin exponer algoritmos propietarios")
        
        # Ejemplo de aplicación en salud
        print(f"\n🏥 Aplicación en Salud:")
        print(f"   - Diagnóstico médico sin revelar historial completo")
        print(f"   - Predicción de enfermedades preservando datos genéticos")
        print(f"   - Investigación médica colaborativa con privacidad")
        
    except Exception as e:
        print(f"❌ Error en ejemplo de IA con preservación de privacidad: {e}")
    
    # Cerrar módulo
    await zk_module.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def main():
    """Main function to run all examples."""
    print("🔐 Blaze AI Zero-Knowledge Proofs Module - Ejemplos Completos")
    print("=" * 70)
    
    try:
        # Ejecutar todos los ejemplos
        await basic_zk_proofs_example()
        await arithmetic_circuit_example()
        await zk_snark_proof_example()
        await zk_stark_proof_example()
        await range_proof_example()
        await membership_proof_example()
        await blockchain_integration_example()
        await privacy_preserving_ai_example()
        
        print(f"\n🎉 Todos los ejemplos completados exitosamente!")
        print(f"🔐 El módulo de Zero-Knowledge Proofs está listo para:")
        print(f"   - Preservar privacidad en operaciones de IA")
        print(f"   - Integrar con blockchain para verificaciones descentralizadas")
        print(f"   - Implementar sistemas de identidad sin revelar datos personales")
        print(f"   - Verificar cálculos complejos sin exponer algoritmos")
        print(f"   - Crear aplicaciones fintech con máxima privacidad")
        
    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())

