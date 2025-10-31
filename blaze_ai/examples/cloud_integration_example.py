"""
Blaze AI Cloud Integration Module Examples

This file provides comprehensive examples demonstrating how to use the
Cloud Integration Module for multi-cloud deployment and management.
"""

import asyncio
import logging
from datetime import datetime
from blaze_ai.modules.cloud_integration import (
    create_cloud_integration_module_with_defaults,
    CloudProvider, ResourceType, ScalingPolicy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_cloud_integration_example():
    """Basic example of cloud integration module usage."""
    print("🚀 Ejemplo Básico de Integración en la Nube")
    print("=" * 50)
    
    # Crear módulo de integración en la nube con configuración básica
    cloud_integration = await create_cloud_integration_module_with_defaults(
        enabled_providers=[CloudProvider.AWS],
        auto_scaling=True,
        load_balancing=True,
        min_instances=2,
        max_instances=10,
        target_cpu_utilization=70.0,
        target_memory_utilization=80.0
    )
    
    print(f"✅ Módulo de integración en la nube creado")
    print(f"   Proveedores habilitados: {[p.value for p in cloud_integration.config.enabled_providers]}")
    print(f"   Auto-scaling: {cloud_integration.config.auto_scaling}")
    print(f"   Load balancing: {cloud_integration.config.load_balancing}")
    print(f"   Instancias mínimas: {cloud_integration.config.min_instances}")
    print(f"   Instancias máximas: {cloud_integration.config.max_instances}")
    
    # Verificar estado de salud
    health = await cloud_integration.health_check()
    print(f"\n🏥 Estado de salud del módulo:")
    print(f"   Estado: {health['status']}")
    print(f"   Proveedores habilitados: {health['enabled_providers']}")
    print(f"   Auto-scaling: {health['auto_scaling']}")
    print(f"   Load balancing: {health['load_balancing']}")
    
    # Cerrar módulo
    await cloud_integration.shutdown()
    print(f"\n✅ Módulo cerrado correctamente")

async def multi_cloud_deployment_example():
    """Example of deploying to multiple cloud providers."""
    print("\n☁️ Ejemplo de Despliegue Multi-Cloud")
    print("=" * 50)
    
    # Crear módulo con múltiples proveedores
    cloud_integration = await create_cloud_integration_module_with_defaults(
        enabled_providers=[CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP],
        auto_scaling=True,
        load_balancing=True,
        min_instances=3,
        max_instances=15
    )
    
    print(f"✅ Módulo multi-cloud creado con proveedores: {[p.value for p in cloud_integration.config.enabled_providers]}")
    
    # Desplegar aplicación web en AWS
    web_app_aws = {
        "name": "web-app-aws",
        "provider": "aws",
        "region": "us-east-1",
        "instance_type": "t3.medium",
        "image_id": "ami-0c55b159cbfafe1f0",
        "min_instances": 2,
        "max_instances": 8,
        "scaling_policy": "cpu_based",
        "load_balancer": True,
        "environment_variables": {
            "NODE_ENV": "production",
            "DB_HOST": "aws-rds-endpoint",
            "REDIS_URL": "aws-elasticache-endpoint"
        },
        "key_name": "web-app-key",
        "security_groups": ["sg-web-app"],
        "subnet_id": "subnet-web-app",
        "launch_template_id": "lt-web-app",
        "subnet_ids": ["subnet-web-app-1", "subnet-web-app-2"]
    }
    
    try:
        aws_deployment_id = await cloud_integration.deploy_to_cloud(web_app_aws)
        print(f"🌐 Aplicación web desplegada en AWS: {aws_deployment_id}")
    except Exception as e:
        print(f"❌ Error desplegando en AWS: {e}")
        aws_deployment_id = None
    
    # Desplegar API en Azure
    api_azure = {
        "name": "api-service-azure",
        "provider": "azure",
        "region": "East US",
        "instance_type": "Standard_D2s_v3",
        "image_id": "Canonical:UbuntuServer:18.04-LTS:latest",
        "min_instances": 3,
        "max_instances": 12,
        "scaling_policy": "memory_based",
        "load_balancer": True,
        "environment_variables": {
            "API_VERSION": "v2.0",
            "DB_CONNECTION": "azure-sql-connection",
            "CACHE_URL": "azure-redis-connection"
        },
        "vm_size": "Standard_D2s_v3",
        "publisher": "Canonical",
        "offer": "UbuntuServer",
        "sku": "18.04-LTS",
        "version": "latest",
        "resource_group": "api-service-rg",
        "network_interface_id": "nic-api-service"
    }
    
    try:
        azure_deployment_id = await cloud_integration.deploy_to_cloud(api_azure)
        print(f"🔵 API desplegada en Azure: {azure_deployment_id}")
    except Exception as e:
        print(f"❌ Error desplegando en Azure: {e}")
        azure_deployment_id = None
    
    # Desplegar servicio de ML en GCP
    ml_gcp = {
        "name": "ml-service-gcp",
        "provider": "gcp",
        "region": "us-central1",
        "instance_type": "n1-standard-4",
        "image_id": "projects/deeplearning-platform-release/global/images/family/tf-latest-gpu",
        "min_instances": 1,
        "max_instances": 5,
        "scaling_policy": "custom_metrics",
        "load_balancer": False,
        "environment_variables": {
            "ML_MODEL_PATH": "/models/bert-large",
            "GPU_ENABLED": "true",
            "BATCH_SIZE": "32"
        },
        "machine_type": "n1-standard-4",
        "source_image": "projects/deeplearning-platform-release/global/images/family/tf-latest-gpu"
    }
    
    try:
        gcp_deployment_id = await cloud_integration.deploy_to_cloud(ml_gcp)
        print(f"🟢 Servicio ML desplegado en GCP: {gcp_deployment_id}")
    except Exception as e:
        print(f"❌ Error desplegando en GCP: {e}")
        gcp_deployment_id = None
    
    # Esperar un momento para que se completen los despliegues
    await asyncio.sleep(3)
    
    # Verificar estado de los despliegues
    print(f"\n📋 Estado de los despliegues:")
    
    for deployment_id, name in [
        (aws_deployment_id, "Web App AWS"),
        (azure_deployment_id, "API Azure"),
        (gcp_deployment_id, "ML Service GCP")
    ]:
        if deployment_id:
            status = await cloud_integration.get_deployment_status(deployment_id)
            if status:
                print(f"   {name}:")
                print(f"     Estado: {status['status']}")
                print(f"     Instancias actuales: {status['current_instances']}")
                print(f"     Instancias objetivo: {status['target_instances']}")
                print(f"     CPU: {status['cpu_utilization']:.1f}%")
                print(f"     Memoria: {status['memory_utilization']:.1f}%")
                print(f"     Estado de salud: {status['health_status']}")
    
    # Obtener métricas del sistema
    metrics = await cloud_integration.get_metrics()
    print(f"\n📊 Métricas del sistema de integración en la nube:")
    print(f"   Total de despliegues: {metrics.total_deployments}")
    print(f"   Despliegues activos: {metrics.active_deployments}")
    print(f"   Total de instancias: {metrics.total_instances}")
    print(f"   Instancias activas: {metrics.active_instances}")
    print(f"   Eventos de escalado: {metrics.scaling_events}")
    print(f"   Despliegues fallidos: {metrics.failed_deployments}")
    
    # Cerrar módulo
    await cloud_integration.shutdown()
    print(f"\n✅ Módulo multi-cloud cerrado correctamente")

async def auto_scaling_example():
    """Example of auto-scaling functionality."""
    print("\n📈 Ejemplo de Auto-Scaling")
    print("=" * 50)
    
    # Crear módulo con auto-scaling habilitado
    cloud_integration = await create_cloud_integration_module_with_defaults(
        enabled_providers=[CloudProvider.AWS],
        auto_scaling=True,
        min_instances=2,
        max_instances=20,
        target_cpu_utilization=60.0,
        target_memory_utilization=75.0,
        monitoring_interval=30.0,  # Monitoreo más frecuente para el ejemplo
        health_check_interval=15.0
    )
    
    print(f"✅ Módulo de auto-scaling creado")
    print(f"   CPU objetivo: {cloud_integration.config.target_cpu_utilization}%")
    print(f"   Memoria objetivo: {cloud_integration.config.target_memory_utilization}%")
    print(f"   Intervalo de monitoreo: {cloud_integration.config.monitoring_interval}s")
    
    # Desplegar aplicación con auto-scaling
    scalable_app = {
        "name": "scalable-web-app",
        "provider": "aws",
        "region": "us-west-2",
        "instance_type": "t3.small",
        "image_id": "ami-0892d3c7ee96c0bf7",
        "min_instances": 2,
        "max_instances": 20,
        "scaling_policy": "cpu_based",
        "load_balancer": True,
        "environment_variables": {
            "APP_ENV": "production",
            "SCALING_ENABLED": "true"
        },
        "key_name": "scalable-app-key",
        "security_groups": ["sg-scalable-app"],
        "subnet_id": "subnet-scalable-app",
        "launch_template_id": "lt-scalable-app",
        "subnet_ids": ["subnet-scalable-app-1", "subnet-scalable-app-2"]
    }
    
    try:
        deployment_id = await cloud_integration.deploy_to_cloud(scalable_app)
        print(f"🚀 Aplicación escalable desplegada: {deployment_id}")
        
        # Simular diferentes cargas para demostrar auto-scaling
        print(f"\n🔄 Simulando diferentes cargas para demostrar auto-scaling:")
        
        for i in range(5):
            await asyncio.sleep(10)  # Esperar 10 segundos entre verificaciones
            
            status = await cloud_integration.get_deployment_status(deployment_id)
            if status:
                print(f"   Verificación {i+1}:")
                print(f"     Instancias actuales: {status['current_instances']}")
                print(f"     CPU: {status['cpu_utilization']:.1f}%")
                print(f"     Memoria: {status['memory_utilization']:.1f}%")
                print(f"     Estado: {status['status']}")
                
                # Simular escalado manual para demostrar la funcionalidad
                if i == 2:
                    print(f"     🔧 Escalando manualmente a 5 instancias...")
                    await cloud_integration.scale_deployment(deployment_id, 5)
                elif i == 4:
                    print(f"     🔧 Escalando manualmente a 3 instancias...")
                    await cloud_integration.scale_deployment(deployment_id, 3)
        
        # Obtener métricas finales
        final_metrics = await cloud_integration.get_metrics()
        print(f"\n📊 Métricas finales:")
        print(f"   Eventos de escalado: {final_metrics.scaling_events}")
        print(f"   Despliegues activos: {final_metrics.active_deployments}")
        print(f"   Instancias activas: {final_metrics.active_instances}")
        
    except Exception as e:
        print(f"❌ Error en ejemplo de auto-scaling: {e}")
    
    # Cerrar módulo
    await cloud_integration.shutdown()
    print(f"\n✅ Módulo de auto-scaling cerrado correctamente")

async def load_balancing_example():
    """Example of load balancing functionality."""
    print("\n⚖️ Ejemplo de Load Balancing")
    print("=" * 50)
    
    # Crear módulo con load balancing
    cloud_integration = await create_cloud_integration_module_with_defaults(
        enabled_providers=[CloudProvider.AWS],
        load_balancing=True,
        auto_scaling=True
    )
    
    print(f"✅ Módulo de load balancing creado")
    
    # Desplegar múltiples instancias para load balancing
    instances = []
    for i in range(3):
        instance_config = {
            "name": f"web-instance-{i+1}",
            "provider": "aws",
            "region": "us-east-1",
            "instance_type": "t3.micro",
            "image_id": "ami-0c55b159cbfafe1f0",
            "min_instances": 1,
            "max_instances": 1,
            "scaling_policy": "manual",
            "load_balancer": True,
            "environment_variables": {
                "INSTANCE_ID": str(i+1),
                "ROLE": "web-server"
            },
            "key_name": "web-instance-key",
            "security_groups": ["sg-web-instance"],
            "subnet_id": "subnet-web-instance"
        }
        
        try:
            deployment_id = await cloud_integration.deploy_to_cloud(instance_config)
            instances.append(deployment_id)
            print(f"🌐 Instancia web {i+1} desplegada: {deployment_id}")
        except Exception as e:
            print(f"❌ Error desplegando instancia {i+1}: {e}")
    
    if instances:
        # Configurar load balancer
        lb_name = "web-app-lb"
        for instance_id in instances:
            await cloud_integration.load_balancer.add_target(lb_name, instance_id)
            print(f"⚖️ Instancia {instance_id} agregada al load balancer {lb_name}")
        
        # Verificar targets del load balancer
        targets = await cloud_integration.load_balancer.get_targets(lb_name)
        print(f"\n📋 Targets del load balancer '{lb_name}': {targets}")
        
        # Simular health checks
        print(f"\n🏥 Simulando health checks:")
        for instance_id in instances:
            is_healthy = await cloud_integration.load_balancer.health_check(instance_id)
            status = "✅ Saludable" if is_healthy else "❌ No saludable"
            print(f"   Instancia {instance_id}: {status}")
        
        # Simular remoción de target
        if len(instances) > 1:
            instance_to_remove = instances[0]
            await cloud_integration.load_balancer.remove_target(lb_name, instance_to_remove)
            print(f"\n🗑️ Instancia {instance_to_remove} removida del load balancer")
            
            remaining_targets = await cloud_integration.load_balancer.get_targets(lb_name)
            print(f"   Targets restantes: {remaining_targets}")
    
    # Cerrar módulo
    await cloud_integration.shutdown()
    print(f"\n✅ Módulo de load balancing cerrado correctamente")

async def kubernetes_integration_example():
    """Example of Kubernetes integration."""
    print("\n☸️ Ejemplo de Integración con Kubernetes")
    print("=" * 50)
    
    # Crear módulo con Kubernetes habilitado
    cloud_integration = await create_cloud_integration_module_with_defaults(
        enabled_providers=[CloudProvider.AWS],
        auto_scaling=True,
        load_balancing=True
    )
    
    print(f"✅ Módulo con integración Kubernetes creado")
    
    # Desplegar aplicación en Kubernetes
    k8s_deployment = {
        "name": "blaze-ai-app",
        "image": "blaze-ai:latest",
        "port": 8080,
        "replicas": 3,
        "namespace": "blaze-ai"
    }
    
    try:
        if cloud_integration.k8s_manager:
            deployment_name = await cloud_integration.k8s_manager.create_deployment(k8s_deployment)
            print(f"☸️ Aplicación desplegada en Kubernetes: {deployment_name}")
            print(f"   Imagen: {k8s_deployment['image']}")
            print(f"   Puerto: {k8s_deployment['port']}")
            print(f"   Réplicas: {k8s_deployment['replicas']}")
            print(f"   Namespace: {k8s_deployment['namespace']}")
        else:
            print(f"⚠️ Manager de Kubernetes no disponible")
    
    except Exception as e:
        print(f"❌ Error desplegando en Kubernetes: {e}")
    
    # Cerrar módulo
    await cloud_integration.shutdown()
    print(f"\n✅ Módulo de Kubernetes cerrado correctamente")

async def cost_optimization_example():
    """Example of cost optimization strategies."""
    print("\n💰 Ejemplo de Optimización de Costos")
    print("=" * 50)
    
    # Crear módulo con configuración de optimización de costos
    cloud_integration = await create_cloud_integration_module_with_defaults(
        enabled_providers=[CloudProvider.AWS, CloudProvider.AZURE],
        auto_scaling=True,
        min_instances=1,  # Mínimo para ahorrar costos
        max_instances=8,  # Límite para controlar gastos
        target_cpu_utilization=80.0,  # Mayor utilización antes de escalar
        target_memory_utilization=85.0
    )
    
    print(f"✅ Módulo de optimización de costos creado")
    print(f"   Instancias mínimas: {cloud_integration.config.min_instances}")
    print(f"   Instancias máximas: {cloud_integration.config.max_instances}")
    print(f"   CPU objetivo: {cloud_integration.config.target_cpu_utilization}%")
    print(f"   Memoria objetivo: {cloud_integration.config.target_memory_utilization}%")
    
    # Desplegar aplicación optimizada para costos
    cost_optimized_app = {
        "name": "cost-optimized-app",
        "provider": "aws",
        "region": "us-east-1",
        "instance_type": "t3.small",  # Instancia económica
        "image_id": "ami-0c55b159cbfafe1f0",
        "min_instances": 1,
        "max_instances": 8,
        "scaling_policy": "cpu_based",
        "load_balancer": True,
        "environment_variables": {
            "COST_OPTIMIZATION": "enabled",
            "AUTO_SHUTDOWN": "true",
            "RESOURCE_MONITORING": "enabled"
        }
    }
    
    try:
        deployment_id = await cloud_integration.deploy_to_cloud(cost_optimized_app)
        print(f"💰 Aplicación optimizada para costos desplegada: {deployment_id}")
        
        # Simular monitoreo de costos
        print(f"\n📊 Simulando monitoreo de costos:")
        for i in range(3):
            await asyncio.sleep(5)
            
            status = await cloud_integration.get_deployment_status(deployment_id)
            if status:
                estimated_cost = status['current_instances'] * 0.05  # $0.05 por instancia por hora
                print(f"   Verificación {i+1}:")
                print(f"     Instancias: {status['current_instances']}")
                print(f"     Costo estimado por hora: ${estimated_cost:.2f}")
                print(f"     Costo estimado por día: ${estimated_cost * 24:.2f}")
                print(f"     CPU: {status['cpu_utilization']:.1f}%")
                
                # Simular escalado basado en costos
                if i == 1 and status['cpu_utilization'] < 30:
                    print(f"     💰 CPU baja, escalando hacia abajo para ahorrar...")
                    await cloud_integration.scale_deployment(deployment_id, 1)
        
    except Exception as e:
        print(f"❌ Error en ejemplo de optimización de costos: {e}")
    
    # Cerrar módulo
    await cloud_integration.shutdown()
    print(f"\n✅ Módulo de optimización de costos cerrado correctamente")

async def main():
    """Main function to run all examples."""
    print("☁️ Blaze AI Cloud Integration Module - Ejemplos Completos")
    print("=" * 70)
    
    try:
        # Ejecutar todos los ejemplos
        await basic_cloud_integration_example()
        await multi_cloud_deployment_example()
        await auto_scaling_example()
        await load_balancing_example()
        await kubernetes_integration_example()
        await cost_optimization_example()
        
        print(f"\n🎉 Todos los ejemplos completados exitosamente!")
        print(f"📚 Revisa la documentación para más detalles sobre cada funcionalidad")
        
    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())

