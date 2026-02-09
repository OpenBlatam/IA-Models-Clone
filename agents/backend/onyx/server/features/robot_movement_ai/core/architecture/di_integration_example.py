"""
Dependency Injection Integration Examples
=========================================

Ejemplos de cómo integrar el sistema de DI con diferentes frameworks.
"""

# ============================================================================
# FastAPI Integration
# ============================================================================

"""
Ejemplo de integración con FastAPI:

from fastapi import FastAPI, Depends
from core.architecture.di_setup import (
    setup_dependency_injection,
    get_move_robot_use_case,
    create_dependency
)
from core.architecture.application_layer import (
    MoveRobotUseCase,
    MoveRobotCommand
)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Configurar DI al iniciar la aplicación
    await setup_dependency_injection({
        'repository_type': 'in_memory',
        'enable_event_bus': True
    })

@app.post("/api/v1/robots/{robot_id}/move")
async def move_robot(
    robot_id: str,
    target_x: float,
    target_y: float,
    target_z: float,
    use_case: MoveRobotUseCase = Depends(create_dependency(MoveRobotUseCase))
):
    command = MoveRobotCommand(
        robot_id=robot_id,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z
    )
    result = await use_case.execute(command)
    return result
"""

# ============================================================================
# Standalone Application
# ============================================================================

"""
Ejemplo de uso en aplicación standalone:

import asyncio
from core.architecture.di_setup import setup_dependency_injection
from core.architecture.application_layer import (
    MoveRobotCommand,
    GetRobotStatusQuery
)

async def main():
    # Setup DI
    container = await setup_dependency_injection({
        'repository_type': 'in_memory'
    })
    
    # Resolver use cases
    move_use_case = await container.resolve_async(MoveRobotUseCase)
    status_use_case = await container.resolve_async(GetRobotStatusUseCase)
    
    # Usar use cases
    command = MoveRobotCommand(
        robot_id="robot-1",
        target_x=0.5,
        target_y=0.3,
        target_z=0.2
    )
    result = await move_use_case.execute(command)
    print(f"Movimiento creado: {result.movement_id}")
    
    # Obtener estado
    query = GetRobotStatusQuery(robot_id="robot-1")
    status = await status_use_case.execute(query)
    print(f"Estado del robot: {status.is_connected}")

if __name__ == "__main__":
    asyncio.run(main())
"""

# ============================================================================
# Testing Integration
# ============================================================================

"""
Ejemplo de uso en tests:

import pytest
from core.architecture.di_setup import DISetup
from core.architecture.infrastructure_repositories import (
    InMemoryRobotRepository,
    InMemoryMovementRepository
)

@pytest.fixture
async def di_setup():
    # Crear setup con repositorios en memoria
    setup = DISetup({
        'repository_type': 'in_memory',
        'enable_event_bus': False
    })
    await setup.setup()
    yield setup
    # Cleanup si es necesario

@pytest.mark.asyncio
async def test_move_robot(di_setup):
    use_case = await di_setup.resolve(MoveRobotUseCase)
    
    command = MoveRobotCommand(
        robot_id="test-1",
        target_x=0.5,
        target_y=0.3,
        target_z=0.2
    )
    
    result = await use_case.execute(command)
    assert result.movement_id is not None
    assert result.status == "executing"
"""

# ============================================================================
# Scoped Dependencies (per request)
# ============================================================================

"""
Ejemplo de uso con scopes (por request):

from core.architecture.di_setup import get_di_setup

async def handle_request(request_id: str):
    di_setup = get_di_setup()
    container = di_setup.get_container()
    
    # Crear scope para este request
    scope_id = container.create_scope()
    container.enter_scope(scope_id)
    
    try:
        # Resolver servicios scoped
        use_case = await container.resolve_async(MoveRobotUseCase)
        # ... usar use case
    finally:
        # Salir del scope y limpiar
        container.exit_scope()
"""




