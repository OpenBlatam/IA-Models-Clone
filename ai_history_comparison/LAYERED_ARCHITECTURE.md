# 🏗️ Arquitectura por Capas - AI History Comparison System

## 📋 **Estructura por Capas Completa**

```
ai_history_comparison/
├── 📁 presentation_layer/             # Capa de Presentación
│   ├── __init__.py
│   ├── api/                          # API REST
│   │   ├── __init__.py
│   │   ├── v1/                       # Versión 1
│   │   │   ├── __init__.py
│   │   │   ├── content_controller.py
│   │   │   ├── analysis_controller.py
│   │   │   ├── comparison_controller.py
│   │   │   ├── report_controller.py
│   │   │   └── system_controller.py
│   │   ├── v2/                       # Versión 2
│   │   │   ├── __init__.py
│   │   │   ├── advanced_controller.py
│   │   │   └── ml_controller.py
│   │   ├── websocket/                # WebSocket
│   │   │   ├── __init__.py
│   │   │   ├── realtime_controller.py
│   │   │   └── streaming_controller.py
│   │   ├── middleware/               # Middleware de presentación
│   │   │   ├── __init__.py
│   │   │   ├── auth_middleware.py
│   │   │   ├── cors_middleware.py
│   │   │   ├── rate_limit_middleware.py
│   │   │   └── logging_middleware.py
│   │   └── serializers/              # Serializadores
│   │       ├── __init__.py
│   │       ├── content_serializer.py
│   │       ├── analysis_serializer.py
│   │       ├── comparison_serializer.py
│   │       └── report_serializer.py
│   ├── cli/                          # Interfaz de línea de comandos
│   │   ├── __init__.py
│   │   ├── commands.py
│   │   ├── content_commands.py
│   │   ├── analysis_commands.py
│   │   └── system_commands.py
│   ├── web/                          # Interfaz web
│   │   ├── __init__.py
│   │   ├── templates/
│   │   ├── static/
│   │   ├── routes.py
│   │   └── views.py
│   └── grpc/                         # gRPC (opcional)
│       ├── __init__.py
│       ├── content_service.py
│       ├── analysis_service.py
│       └── comparison_service.py
│
├── 📁 application_layer/              # Capa de Aplicación
│   ├── __init__.py
│   ├── services/                     # Servicios de aplicación
│   │   ├── __init__.py
│   │   ├── content_application_service.py
│   │   ├── analysis_application_service.py
│   │   ├── comparison_application_service.py
│   │   ├── report_application_service.py
│   │   └── system_application_service.py
│   ├── use_cases/                    # Casos de uso
│   │   ├── __init__.py
│   │   ├── content_use_cases.py
│   │   ├── analysis_use_cases.py
│   │   ├── comparison_use_cases.py
│   │   ├── report_use_cases.py
│   │   └── system_use_cases.py
│   ├── dto/                          # Data Transfer Objects
│   │   ├── __init__.py
│   │   ├── request_dto.py
│   │   ├── response_dto.py
│   │   ├── content_dto.py
│   │   ├── analysis_dto.py
│   │   ├── comparison_dto.py
│   │   └── report_dto.py
│   ├── validators/                   # Validadores de aplicación
│   │   ├── __init__.py
│   │   ├── content_validator.py
│   │   ├── analysis_validator.py
│   │   ├── comparison_validator.py
│   │   └── report_validator.py
│   ├── mappers/                      # Mapeadores
│   │   ├── __init__.py
│   │   ├── content_mapper.py
│   │   ├── analysis_mapper.py
│   │   ├── comparison_mapper.py
│   │   └── report_mapper.py
│   ├── events/                       # Eventos de aplicación
│   │   ├── __init__.py
│   │   ├── content_events.py
│   │   ├── analysis_events.py
│   │   ├── comparison_events.py
│   │   └── report_events.py
│   └── handlers/                     # Manejadores de eventos
│       ├── __init__.py
│       ├── content_event_handler.py
│       ├── analysis_event_handler.py
│       ├── comparison_event_handler.py
│       └── report_event_handler.py
│
├── 📁 domain_layer/                   # Capa de Dominio
│   ├── __init__.py
│   ├── entities/                     # Entidades de dominio
│   │   ├── __init__.py
│   │   ├── base_entity.py
│   │   ├── content_entity.py
│   │   ├── analysis_entity.py
│   │   ├── comparison_entity.py
│   │   ├── report_entity.py
│   │   ├── trend_entity.py
│   │   └── user_entity.py
│   ├── value_objects/                # Objetos de valor
│   │   ├── __init__.py
│   │   ├── content_id.py
│   │   ├── analysis_id.py
│   │   ├── comparison_id.py
│   │   ├── report_id.py
│   │   ├── content_hash.py
│   │   ├── analysis_score.py
│   │   ├── comparison_metrics.py
│   │   └── timestamp.py
│   ├── aggregates/                   # Agregados
│   │   ├── __init__.py
│   │   ├── content_aggregate.py
│   │   ├── analysis_aggregate.py
│   │   ├── comparison_aggregate.py
│   │   └── report_aggregate.py
│   ├── services/                     # Servicios de dominio
│   │   ├── __init__.py
│   │   ├── content_domain_service.py
│   │   ├── analysis_domain_service.py
│   │   ├── comparison_domain_service.py
│   │   ├── report_domain_service.py
│   │   └── validation_domain_service.py
│   ├── repositories/                 # Interfaces de repositorio
│   │   ├── __init__.py
│   │   ├── content_repository_interface.py
│   │   ├── analysis_repository_interface.py
│   │   ├── comparison_repository_interface.py
│   │   ├── report_repository_interface.py
│   │   └── user_repository_interface.py
│   ├── events/                       # Eventos de dominio
│   │   ├── __init__.py
│   │   ├── domain_event.py
│   │   ├── content_domain_events.py
│   │   ├── analysis_domain_events.py
│   │   ├── comparison_domain_events.py
│   │   └── report_domain_events.py
│   ├── specifications/               # Especificaciones
│   │   ├── __init__.py
│   │   ├── content_specifications.py
│   │   ├── analysis_specifications.py
│   │   ├── comparison_specifications.py
│   │   └── report_specifications.py
│   └── policies/                     # Políticas de dominio
│       ├── __init__.py
│       ├── content_policies.py
│       ├── analysis_policies.py
│       ├── comparison_policies.py
│       └── report_policies.py
│
├── 📁 infrastructure_layer/           # Capa de Infraestructura
│   ├── __init__.py
│   ├── database/                     # Base de datos
│   │   ├── __init__.py
│   │   ├── connection/               # Conexiones
│   │   │   ├── __init__.py
│   │   │   ├── database_connection.py
│   │   │   ├── postgres_connection.py
│   │   │   ├── mysql_connection.py
│   │   │   └── sqlite_connection.py
│   │   ├── models/                   # Modelos de base de datos
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py
│   │   │   ├── content_model.py
│   │   │   ├── analysis_model.py
│   │   │   ├── comparison_model.py
│   │   │   ├── report_model.py
│   │   │   └── user_model.py
│   │   ├── repositories/             # Implementaciones de repositorio
│   │   │   ├── __init__.py
│   │   │   ├── base_repository.py
│   │   │   ├── content_repository_impl.py
│   │   │   ├── analysis_repository_impl.py
│   │   │   ├── comparison_repository_impl.py
│   │   │   ├── report_repository_impl.py
│   │   │   └── user_repository_impl.py
│   │   ├── migrations/               # Migraciones
│   │   │   ├── __init__.py
│   │   │   ├── alembic.ini
│   │   │   ├── env.py
│   │   │   └── versions/
│   │   └── seeders/                  # Datos de prueba
│   │       ├── __init__.py
│   │       ├── content_seeder.py
│   │       ├── analysis_seeder.py
│   │       └── user_seeder.py
│   ├── cache/                        # Sistema de caché
│   │   ├── __init__.py
│   │   ├── cache_interface.py
│   │   ├── redis_cache.py
│   │   ├── memory_cache.py
│   │   ├── file_cache.py
│   │   └── cache_manager.py
│   ├── external_services/            # Servicios externos
│   │   ├── __init__.py
│   │   ├── llm/                      # Servicios LLM
│   │   │   ├── __init__.py
│   │   │   ├── llm_service_interface.py
│   │   │   ├── openai_service.py
│   │   │   ├── anthropic_service.py
│   │   │   ├── google_service.py
│   │   │   ├── huggingface_service.py
│   │   │   └── llm_service_factory.py
│   │   ├── storage/                  # Almacenamiento
│   │   │   ├── __init__.py
│   │   │   ├── storage_interface.py
│   │   │   ├── s3_storage.py
│   │   │   ├── gcs_storage.py
│   │   │   ├── azure_storage.py
│   │   │   ├── local_storage.py
│   │   │   └── storage_factory.py
│   │   ├── monitoring/               # Monitoreo
│   │   │   ├── __init__.py
│   │   │   ├── metrics_collector.py
│   │   │   ├── prometheus_metrics.py
│   │   │   ├── sentry_monitoring.py
│   │   │   ├── health_checker.py
│   │   │   └── performance_monitor.py
│   │   └── messaging/                # Mensajería
│   │       ├── __init__.py
│   │       ├── message_bus.py
│   │       ├── event_bus.py
│   │       ├── queue_service.py
│   │       ├── notification_service.py
│   │       └── webhook_service.py
│   ├── security/                     # Seguridad
│   │   ├── __init__.py
│   │   ├── authentication/           # Autenticación
│   │   │   ├── __init__.py
│   │   │   ├── auth_service.py
│   │   │   ├── jwt_service.py
│   │   │   ├── oauth_service.py
│   │   │   └── session_service.py
│   │   ├── authorization/            # Autorización
│   │   │   ├── __init__.py
│   │   │   ├── permission_service.py
│   │   │   ├── role_service.py
│   │   │   └── policy_service.py
│   │   └── encryption/               # Cifrado
│   │       ├── __init__.py
│   │       ├── encryption_service.py
│   │       ├── hash_service.py
│   │       └── key_management.py
│   └── logging/                      # Logging
│       ├── __init__.py
│       ├── logger_config.py
│       ├── structured_logger.py
│       ├── audit_logger.py
│       └── performance_logger.py
│
├── 📁 shared_layer/                   # Capa Compartida
│   ├── __init__.py
│   ├── constants/                    # Constantes
│   │   ├── __init__.py
│   │   ├── app_constants.py
│   │   ├── error_codes.py
│   │   ├── status_codes.py
│   │   └── message_constants.py
│   ├── exceptions/                   # Excepciones compartidas
│   │   ├── __init__.py
│   │   ├── base_exception.py
│   │   ├── validation_exception.py
│   │   ├── business_exception.py
│   │   ├── infrastructure_exception.py
│   │   └── external_service_exception.py
│   ├── utils/                        # Utilidades compartidas
│   │   ├── __init__.py
│   │   ├── date_utils.py
│   │   ├── string_utils.py
│   │   ├── validation_utils.py
│   │   ├── encryption_utils.py
│   │   ├── file_utils.py
│   │   └── network_utils.py
│   ├── types/                        # Tipos personalizados
│   │   ├── __init__.py
│   │   ├── result.py
│   │   ├── option.py
│   │   ├── either.py
│   │   └── response.py
│   ├── decorators/                   # Decoradores
│   │   ├── __init__.py
│   │   ├── retry_decorator.py
│   │   ├── cache_decorator.py
│   │   ├── rate_limit_decorator.py
│   │   ├── timing_decorator.py
│   │   └── validation_decorator.py
│   └── middleware/                   # Middleware compartido
│       ├── __init__.py
│       ├── error_handler.py
│       ├── request_id_middleware.py
│       ├── correlation_id_middleware.py
│       └── performance_middleware.py
│
├── 📁 cross_cutting_concerns/         # Aspectos Transversales
│   ├── __init__.py
│   ├── configuration/                # Configuración
│   │   ├── __init__.py
│   │   ├── app_config.py
│   │   ├── database_config.py
│   │   ├── cache_config.py
│   │   ├── llm_config.py
│   │   ├── security_config.py
│   │   └── monitoring_config.py
│   ├── dependency_injection/         # Inyección de dependencias
│   │   ├── __init__.py
│   │   ├── container.py
│   │   ├── service_registry.py
│   │   ├── factory_registry.py
│   │   └── dependency_resolver.py
│   ├── interceptors/                 # Interceptores
│   │   ├── __init__.py
│   │   ├── logging_interceptor.py
│   │   ├── metrics_interceptor.py
│   │   ├── security_interceptor.py
│   │   ├── performance_interceptor.py
│   │   └── error_interceptor.py
│   └── aspects/                      # Aspectos
│       ├── __init__.py
│       ├── caching_aspect.py
│       ├── logging_aspect.py
│       ├── security_aspect.py
│       ├── performance_aspect.py
│       └── transaction_aspect.py
│
├── 📁 tests/                         # Tests por capas
│   ├── __init__.py
│   ├── presentation_tests/           # Tests de presentación
│   │   ├── __init__.py
│   │   ├── api_tests/
│   │   ├── cli_tests/
│   │   └── web_tests/
│   ├── application_tests/            # Tests de aplicación
│   │   ├── __init__.py
│   │   ├── service_tests/
│   │   ├── use_case_tests/
│   │   └── validator_tests/
│   ├── domain_tests/                 # Tests de dominio
│   │   ├── __init__.py
│   │   ├── entity_tests/
│   │   ├── service_tests/
│   │   └── specification_tests/
│   ├── infrastructure_tests/         # Tests de infraestructura
│   │   ├── __init__.py
│   │   ├── database_tests/
│   │   ├── cache_tests/
│   │   ├── external_service_tests/
│   │   └── security_tests/
│   ├── integration_tests/            # Tests de integración
│   │   ├── __init__.py
│   │   ├── api_integration_tests/
│   │   ├── database_integration_tests/
│   │   └── external_service_integration_tests/
│   └── e2e_tests/                    # Tests end-to-end
│       ├── __init__.py
│       └── scenarios/
│
├── 📁 scripts/                       # Scripts por capas
│   ├── __init__.py
│   ├── setup/                        # Scripts de configuración
│   │   ├── __init__.py
│   │   ├── setup_layered_structure.py
│   │   ├── setup_database.py
│   │   ├── setup_cache.py
│   │   └── setup_external_services.py
│   ├── migration/                    # Scripts de migración
│   │   ├── __init__.py
│   │   ├── migrate_database.py
│   │   ├── migrate_cache.py
│   │   └── migrate_external_services.py
│   ├── deployment/                   # Scripts de despliegue
│   │   ├── __init__.py
│   │   ├── deploy_application.py
│   │   ├── deploy_database.py
│   │   └── deploy_infrastructure.py
│   └── maintenance/                  # Scripts de mantenimiento
│       ├── __init__.py
│       ├── backup_database.py
│       ├── cleanup_cache.py
│       └── health_check.py
│
├── 📁 docs/                          # Documentación por capas
│   ├── __init__.py
│   ├── presentation_docs/            # Documentación de presentación
│   │   ├── api_documentation.md
│   │   ├── cli_documentation.md
│   │   └── web_documentation.md
│   ├── application_docs/             # Documentación de aplicación
│   │   ├── service_documentation.md
│   │   ├── use_case_documentation.md
│   │   └── dto_documentation.md
│   ├── domain_docs/                  # Documentación de dominio
│   │   ├── entity_documentation.md
│   │   ├── service_documentation.md
│   │   └── repository_documentation.md
│   ├── infrastructure_docs/          # Documentación de infraestructura
│   │   ├── database_documentation.md
│   │   ├── cache_documentation.md
│   │   ├── external_service_documentation.md
│   │   └── security_documentation.md
│   └── architecture_docs/            # Documentación de arquitectura
│       ├── layered_architecture.md
│       ├── design_patterns.md
│       └── best_practices.md
│
├── 📁 config/                        # Configuración por capas
│   ├── __init__.py
│   ├── presentation_config/          # Configuración de presentación
│   │   ├── __init__.py
│   │   ├── api_config.yaml
│   │   ├── cli_config.yaml
│   │   └── web_config.yaml
│   ├── application_config/           # Configuración de aplicación
│   │   ├── __init__.py
│   │   ├── service_config.yaml
│   │   ├── use_case_config.yaml
│   │   └── validator_config.yaml
│   ├── domain_config/                # Configuración de dominio
│   │   ├── __init__.py
│   │   ├── entity_config.yaml
│   │   ├── service_config.yaml
│   │   └── repository_config.yaml
│   ├── infrastructure_config/        # Configuración de infraestructura
│   │   ├── __init__.py
│   │   ├── database_config.yaml
│   │   ├── cache_config.yaml
│   │   ├── external_service_config.yaml
│   │   └── security_config.yaml
│   └── shared_config/                # Configuración compartida
│       ├── __init__.py
│       ├── constants_config.yaml
│       ├── exception_config.yaml
│       └── utils_config.yaml
│
├── main.py                           # Punto de entrada principal
├── requirements.txt                  # Dependencias principales
├── requirements-dev.txt              # Dependencias de desarrollo
├── requirements-test.txt             # Dependencias de testing
├── pyproject.toml                    # Configuración del proyecto
├── .env.example                      # Variables de entorno
├── .gitignore
├── README.md
└── CHANGELOG.md
```

## 🏗️ **Principios de Arquitectura por Capas**

### **1. Separación Estricta de Responsabilidades**
- **Presentation Layer**: Solo manejo de requests/responses
- **Application Layer**: Orquestación y casos de uso
- **Domain Layer**: Lógica de negocio pura
- **Infrastructure Layer**: Implementaciones técnicas
- **Shared Layer**: Componentes reutilizables
- **Cross-Cutting Concerns**: Aspectos transversales

### **2. Inversión de Dependencias**
- Las capas superiores dependen de abstracciones
- Las implementaciones están en capas inferiores
- Interfaces bien definidas entre capas

### **3. Aislamiento de Capas**
- Cada capa tiene responsabilidades específicas
- No hay dependencias circulares
- Comunicación solo a través de interfaces

### **4. Testabilidad por Capas**
- Tests unitarios por capa
- Tests de integración entre capas
- Mocks y stubs fáciles de implementar

## 🔧 **Beneficios de la Arquitectura por Capas**

### **✅ Mantenibilidad**
- Código organizado por responsabilidades
- Cambios localizados en capas específicas
- Fácil debugging y troubleshooting

### **✅ Escalabilidad**
- Capas independientes escalables
- Fácil agregar nuevas funcionalidades
- Deployment granular por capa

### **✅ Testabilidad**
- Tests unitarios por capa
- Tests de integración entre capas
- Cobertura de testing granular

### **✅ Reutilización**
- Capas reutilizables en otros proyectos
- Interfaces estándar
- Componentes intercambiables

### **✅ Colaboración**
- Equipos pueden trabajar en capas independientes
- Conflictos de merge reducidos
- Especialización por capa

## 📊 **Flujo de Datos por Capas**

```
Request → Presentation Layer → Application Layer → Domain Layer → Infrastructure Layer
   ↓              ↓                    ↓                ↓                ↓
Response ← Presentation Layer ← Application Layer ← Domain Layer ← Infrastructure Layer
```

### **1. Presentation Layer**
- Recibe requests HTTP/gRPC/CLI
- Valida formato de entrada
- Serializa/deserializa datos
- Maneja errores de presentación

### **2. Application Layer**
- Orquesta casos de uso
- Valida reglas de negocio
- Coordina servicios de dominio
- Maneja transacciones

### **3. Domain Layer**
- Contiene lógica de negocio pura
- Define entidades y agregados
- Implementa reglas de dominio
- Es independiente de frameworks

### **4. Infrastructure Layer**
- Implementa interfaces de dominio
- Maneja persistencia de datos
- Integra servicios externos
- Implementa aspectos técnicos

### **5. Shared Layer**
- Proporciona utilidades comunes
- Define tipos y constantes
- Maneja excepciones compartidas
- Implementa decoradores

### **6. Cross-Cutting Concerns**
- Maneja configuración
- Implementa inyección de dependencias
- Proporciona interceptores
- Maneja aspectos transversales

## 🚀 **Implementación por Capas**

### **Fase 1: Shared Layer (30 minutos)**
- Constantes y tipos
- Excepciones base
- Utilidades comunes
- Decoradores

### **Fase 2: Domain Layer (60 minutos)**
- Entidades y agregados
- Servicios de dominio
- Interfaces de repositorio
- Eventos de dominio

### **Fase 3: Infrastructure Layer (90 minutos)**
- Implementaciones de repositorio
- Servicios externos
- Sistema de caché
- Seguridad

### **Fase 4: Application Layer (75 minutos)**
- Servicios de aplicación
- Casos de uso
- DTOs y validadores
- Manejadores de eventos

### **Fase 5: Presentation Layer (60 minutos)**
- Controladores de API
- Serializadores
- Middleware
- CLI y Web

### **Fase 6: Cross-Cutting Concerns (45 minutos)**
- Configuración
- Inyección de dependencias
- Interceptores
- Aspectos

## 📈 **Métricas de Arquitectura por Capas**

| Métrica | Valor | Beneficio |
|---------|-------|-----------|
| **Separación de Responsabilidades** | Alta | Cambios localizados |
| **Inversión de Dependencias** | Completa | Fácil testing |
| **Aislamiento de Capas** | Total | Sin acoplamiento |
| **Testabilidad** | Alta | Tests por capa |
| **Reutilización** | Alta | Capas intercambiables |
| **Mantenibilidad** | Alta | Código organizado |

## 🎯 **Patrones de Diseño por Capa**

### **Presentation Layer**
- **Controller Pattern**: Manejo de requests
- **Serializer Pattern**: Transformación de datos
- **Middleware Pattern**: Aspectos transversales

### **Application Layer**
- **Service Pattern**: Orquestación de casos de uso
- **Use Case Pattern**: Casos de uso específicos
- **DTO Pattern**: Transferencia de datos

### **Domain Layer**
- **Entity Pattern**: Entidades de dominio
- **Aggregate Pattern**: Agregados de dominio
- **Repository Pattern**: Interfaces de persistencia

### **Infrastructure Layer**
- **Repository Implementation**: Implementaciones de repositorio
- **Factory Pattern**: Creación de objetos
- **Adapter Pattern**: Adaptación de servicios externos

### **Shared Layer**
- **Utility Pattern**: Funciones auxiliares
- **Decorator Pattern**: Funcionalidad adicional
- **Exception Pattern**: Manejo de errores

### **Cross-Cutting Concerns**
- **Dependency Injection**: Inyección de dependencias
- **Aspect-Oriented Programming**: Aspectos transversales
- **Configuration Pattern**: Configuración centralizada







