"""
Blaze AI Blockchain Module Example

This example demonstrates how to use the Blockchain module
for decentralized AI operations, smart contracts, and consensus.
"""

import asyncio
import json
import logging
from datetime import datetime

from blaze_ai.modules import create_blockchain_module
from blaze_ai.modules.blockchain import (
    ConsensusAlgorithm, TransactionType, SmartContractStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def blockchain_basic_example():
    """Basic blockchain operations example."""
    print("🚀 Iniciando ejemplo básico de Blockchain...")
    
    # Crear módulo de blockchain
    blockchain = create_blockchain_module(
        network_name="blaze-ai-demo",
        consensus_algorithm=ConsensusAlgorithm.PROOF_OF_STAKE,
        block_time=10.0,  # 10 segundos por bloque
        min_validators=2
    )
    
    # Inicializar módulo
    success = await blockchain.initialize()
    if not success:
        print("❌ Error al inicializar el módulo de blockchain")
        return
    
    print("✅ Módulo de blockchain inicializado correctamente")
    
    # Esperar un poco para que se cree el bloque génesis
    await asyncio.sleep(2)
    
    # Obtener estado del blockchain
    status = await blockchain.get_blockchain_status()
    print(f"📊 Estado del blockchain: {json.dumps(status, indent=2, default=str)}")
    
    # Cerrar módulo
    await blockchain.shutdown()
    print("🔒 Módulo de blockchain cerrado")

async def transaction_example():
    """Transaction submission example."""
    print("\n💸 Ejemplo de transacciones...")
    
    blockchain = create_blockchain_module(
        network_name="blaze-ai-transactions",
        block_time=5.0
    )
    
    await blockchain.initialize()
    await asyncio.sleep(2)
    
    # Enviar transacción de entrenamiento de modelo AI
    ai_training_tx = await blockchain.submit_transaction({
        "type": "ai_model_training",
        "sender": "researcher_001",
        "recipient": "ai_training_pool",
        "amount": 100.0,
        "gas_price": 0.00000001,
        "gas_limit": 500000,
        "data": {
            "model_type": "transformer",
            "dataset_size": "1M samples",
            "training_epochs": 100
        }
    })
    
    if ai_training_tx:
        print(f"✅ Transacción de entrenamiento AI enviada: {ai_training_tx}")
    else:
        print("❌ Error al enviar transacción de entrenamiento AI")
    
    # Enviar transacción de compartición de datos
    data_sharing_tx = await blockchain.submit_transaction({
        "type": "data_sharing",
        "sender": "data_provider_001",
        "recipient": "data_marketplace",
        "amount": 50.0,
        "gas_price": 0.00000001,
        "gas_limit": 300000,
        "data": {
            "dataset_name": "medical_images_2024",
            "data_type": "images",
            "privacy_level": "anonymized"
        }
    })
    
    if data_sharing_tx:
        print(f"✅ Transacción de compartición de datos enviada: {data_sharing_tx}")
    else:
        print("❌ Error al enviar transacción de compartición de datos")
    
    # Enviar transacción de alquiler de computación
    computation_tx = await blockchain.submit_transaction({
        "type": "computation_rental",
        "sender": "user_001",
        "recipient": "compute_provider_001",
        "amount": 25.0,
        "gas_price": 0.00000001,
        "gas_limit": 200000,
        "data": {
            "compute_type": "gpu",
            "duration_hours": 4,
            "gpu_model": "RTX 4090"
        }
    })
    
    if computation_tx:
        print(f"✅ Transacción de alquiler de computación enviada: {computation_tx}")
    else:
        print("❌ Error al enviar transacción de alquiler de computación")
    
    # Esperar un poco para que se procesen las transacciones
    await asyncio.sleep(3)
    
    # Obtener información de las transacciones
    for tx_id in [ai_training_tx, data_sharing_tx, computation_tx]:
        if tx_id:
            tx_info = await blockchain.get_transaction_info(tx_id)
            if tx_info:
                print(f"📋 Información de transacción {tx_id[:8]}...")
                print(f"   Tipo: {tx_info['transaction_type']}")
                print(f"   Estado: {tx_info['status']}")
                print(f"   Cantidad: {tx_info['amount']}")
    
    await blockchain.shutdown()

async def smart_contract_example():
    """Smart contract deployment and execution example."""
    print("\n📜 Ejemplo de contratos inteligentes...")
    
    blockchain = create_blockchain_module(
        network_name="blaze-ai-contracts",
        block_time=8.0
    )
    
    await blockchain.initialize()
    await asyncio.sleep(2)
    
    # Contrato inteligente para inferencia AI
    ai_inference_contract = """
def ai_inference_contract(input_data):
    # Simulación de inferencia AI
    if 'image' in input_data:
        return {
            'result': 'cat',
            'confidence': 0.95,
            'processing_time': 0.15
        }
    elif 'text' in input_data:
        return {
            'result': 'positive_sentiment',
            'confidence': 0.87,
            'processing_time': 0.08
        }
    else:
        return {
            'result': 'unknown',
            'confidence': 0.0,
            'processing_time': 0.0
        }
"""
    
    # Desplegar contrato
    contract_id = await blockchain.deploy_smart_contract({
        "name": "AI Inference Contract",
        "code": ai_inference_contract,
        "owner": "ai_developer_001",
        "gas_limit": 1000000,
        "gas_price": 0.00000001
    })
    
    if contract_id:
        print(f"✅ Contrato inteligente desplegado: {contract_id}")
        
        # Ejecutar contrato con datos de imagen
        image_result = await blockchain.execute_smart_contract(contract_id, {
            "operation": "ai_inference",
            "image": "cat_image.jpg",
            "model": "resnet50"
        })
        
        print(f"🖼️ Resultado de inferencia de imagen: {json.dumps(image_result, indent=2)}")
        
        # Ejecutar contrato con datos de texto
        text_result = await blockchain.execute_smart_contract(contract_id, {
            "operation": "ai_inference",
            "text": "I love this product!",
            "model": "bert_sentiment"
        })
        
        print(f"📝 Resultado de análisis de texto: {json.dumps(text_result, indent=2)}")
        
        # Ejecutar contrato con operación matemática
        math_result = await blockchain.execute_smart_contract(contract_id, {
            "operation": "add",
            "a": 15,
            "b": 27
        })
        
        print(f"🧮 Resultado de operación matemática: {json.dumps(math_result, indent=2)}")
        
    else:
        print("❌ Error al desplegar contrato inteligente")
    
    await blockchain.shutdown()

async def consensus_example():
    """Consensus mechanism example."""
    print("\n🤝 Ejemplo de mecanismo de consenso...")
    
    # Crear múltiples nodos para demostrar consenso
    nodes = []
    
    for i in range(3):
        node = create_blockchain_module(
            network_name=f"consensus-demo-node-{i}",
            consensus_algorithm=ConsensusAlgorithm.PROOF_OF_STAKE,
            min_validators=2,
            block_time=12.0
        )
        nodes.append(node)
    
    # Inicializar todos los nodos
    for i, node in enumerate(nodes):
        success = await node.initialize()
        if success:
            print(f"✅ Nodo {i} inicializado correctamente")
        else:
            print(f"❌ Error al inicializar nodo {i}")
    
    await asyncio.sleep(3)
    
    # Simular transacciones en diferentes nodos
    for i, node in enumerate(nodes):
        tx_id = await node.submit_transaction({
            "type": "token_transfer",
            "sender": f"node_{i}_user",
            "recipient": f"node_{(i+1)%3}_user",
            "amount": 10.0 + i,
            "gas_price": 0.00000001,
            "gas_limit": 100000,
            "data": {"message": f"Hello from node {i}"}
        })
        
        if tx_id:
            print(f"✅ Transacción enviada desde nodo {i}: {tx_id[:8]}...")
    
    # Esperar para que se procesen las transacciones
    await asyncio.sleep(5)
    
    # Verificar estado de todos los nodos
    for i, node in enumerate(nodes):
        status = await node.get_blockchain_status()
        print(f"📊 Estado del nodo {i}:")
        print(f"   Altura del bloque: {status['block_height']}")
        print(f"   Transacciones pendientes: {status['pending_transactions']}")
        print(f"   Algoritmo de consenso: {status['consensus_algorithm']}")
    
    # Cerrar todos los nodos
    for i, node in enumerate(nodes):
        await node.shutdown()
        print(f"🔒 Nodo {i} cerrado")
    
    print("✅ Todos los nodos cerrados correctamente")

async def advanced_features_example():
    """Advanced blockchain features example."""
    print("\n🚀 Ejemplo de características avanzadas...")
    
    blockchain = create_blockchain_module(
        network_name="blaze-ai-advanced",
        consensus_algorithm=ConsensusAlgorithm.PROOF_OF_AUTHORITY,
        block_time=6.0,
        enable_encryption=True,
        enable_signature_verification=True
    )
    
    await blockchain.initialize()
    await asyncio.sleep(2)
    
    # Simular múltiples transacciones de diferentes tipos
    transaction_types = [
        ("governance_vote", "Vote for new AI model", {"proposal": "deploy_gpt5", "vote": "yes"}),
        ("ai_model_training", "Train new model", {"model": "vision_transformer", "epochs": 200}),
        ("data_sharing", "Share dataset", {"dataset": "satellite_images", "size": "5GB"}),
        ("computation_rental", "Rent GPU cluster", {"gpus": 8, "hours": 24}),
        ("smart_contract_execution", "Execute contract", {"contract": "ai_oracle", "input": "market_data"})
    ]
    
    submitted_transactions = []
    
    for tx_type, description, data in transaction_types:
        tx_id = await blockchain.submit_transaction({
            "type": tx_type,
            "sender": "advanced_user_001",
            "recipient": "blaze_ai_network",
            "amount": 25.0,
            "gas_price": 0.00000001,
            "gas_limit": 500000,
            "data": data
        })
        
        if tx_id:
            submitted_transactions.append((tx_id, tx_type, description))
            print(f"✅ {description}: {tx_id[:8]}...")
    
    # Esperar para que se procesen
    await asyncio.sleep(4)
    
    # Obtener métricas del blockchain
    metrics = await blockchain.get_metrics()
    print(f"\n📈 Métricas del blockchain:")
    print(f"   Bloques totales: {metrics.total_blocks}")
    print(f"   Transacciones totales: {metrics.total_transactions}")
    print(f"   Contratos totales: {metrics.total_contracts}")
    print(f"   Rondas de consenso: {metrics.consensus_rounds}")
    print(f"   Última actualización: {metrics.last_updated}")
    
    # Verificar estado de salud
    health = await blockchain.health_check()
    print(f"\n🏥 Estado de salud:")
    print(f"   Estado: {health['status']}")
    print(f"   Altura del blockchain: {health['blockchain_height']}")
    print(f"   Minería activa: {health['mining_active']}")
    print(f"   Consenso activo: {health['consensus_active']}")
    
    await blockchain.shutdown()

async def main():
    """Main function to run all examples."""
    print("🔗 BLAZE AI BLOCKCHAIN MODULE EXAMPLES")
    print("=" * 50)
    
    try:
        # Ejemplo básico
        await blockchain_basic_example()
        
        # Ejemplo de transacciones
        await transaction_example()
        
        # Ejemplo de contratos inteligentes
        await smart_contract_example()
        
        # Ejemplo de consenso
        await consensus_example()
        
        # Ejemplo de características avanzadas
        await advanced_features_example()
        
        print("\n🎉 ¡Todos los ejemplos completados exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        logger.exception("Error en ejemplo de blockchain")

if __name__ == "__main__":
    asyncio.run(main())

