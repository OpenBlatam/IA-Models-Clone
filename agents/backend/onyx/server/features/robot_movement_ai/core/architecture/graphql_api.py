"""
GraphQL API para Robot Movement AI v2.0
API GraphQL completa con queries, mutations y subscriptions
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from strawberry import Schema, Query, Mutation, Subscription, Field, type as strawberry_type
    from strawberry.fastapi import GraphQLRouter
    import strawberry
    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False
    Query = None
    Mutation = None
    Subscription = None


if STRAWBERRY_AVAILABLE:
    @strawberry.type
    class Robot:
        """Tipo GraphQL para Robot"""
        id: str
        name: str
        status: str
        position_x: float
        position_y: float
        position_z: float
    
    @strawberry.type
    class Movement:
        """Tipo GraphQL para Movement"""
        id: str
        robot_id: str
        start_x: float
        start_y: float
        start_z: float
        end_x: float
        end_y: float
        end_z: float
        status: str
        duration: Optional[float] = None
    
    @strawberry.input
    class MoveRobotInput:
        """Input para mover robot"""
        robot_id: str
        target_x: float
        target_y: float
        target_z: float
        speed: Optional[float] = None
    
    class RobotQuery:
        """Queries GraphQL para robots"""
        
        @strawberry.field
        async def robot(self, id: str) -> Optional[Robot]:
            """Obtener robot por ID"""
            from core.architecture.di_setup import resolve_service
            from core.architecture.infrastructure_repositories import IRobotRepository
            
            repo = resolve_service(IRobotRepository)
            robot = await repo.get_by_id(id)
            
            if robot:
                return Robot(
                    id=robot.id,
                    name=robot.name,
                    status=robot.status.value,
                    position_x=robot.position.x,
                    position_y=robot.position.y,
                    position_z=robot.position.z
                )
            return None
        
        @strawberry.field
        async def robots(self) -> List[Robot]:
            """Listar todos los robots"""
            from core.architecture.di_setup import resolve_service
            from core.architecture.infrastructure_repositories import IRobotRepository
            
            repo = resolve_service(IRobotRepository)
            robots = await repo.find_all()
            
            return [
                Robot(
                    id=r.id,
                    name=r.name,
                    status=r.status.value,
                    position_x=r.position.x,
                    position_y=r.position.y,
                    position_z=r.position.z
                )
                for r in robots
            ]
    
    class RobotMutation:
        """Mutations GraphQL para robots"""
        
        @strawberry.mutation
        async def move_robot(self, input: MoveRobotInput) -> Movement:
            """Mover robot"""
            from core.architecture.di_setup import resolve_service
            from core.architecture.application_layer import MoveRobotCommand, MoveRobotUseCase
            
            use_case = resolve_service(MoveRobotUseCase)
            command = MoveRobotCommand(
                robot_id=input.robot_id,
                target_x=input.target_x,
                target_y=input.target_y,
                target_z=input.target_z,
                speed=input.speed
            )
            
            result = await use_case.execute(command)
            
            return Movement(
                id=result.movement_id,
                robot_id=input.robot_id,
                start_x=0.0,  # Obtener de estado actual
                start_y=0.0,
                start_z=0.0,
                end_x=input.target_x,
                end_y=input.target_y,
                end_z=input.target_z,
                status=result.status,
                duration=result.duration
            )
    
    class RobotSubscription:
        """Subscriptions GraphQL para robots"""
        
        @strawberry.subscription
        async def robot_status(self, robot_id: str) -> Robot:
            """Suscribirse a cambios de estado de robot"""
            import asyncio
            
            while True:
                from core.architecture.di_setup import resolve_service
                from core.architecture.infrastructure_repositories import IRobotRepository
                
                repo = resolve_service(IRobotRepository)
                robot = await repo.get_by_id(robot_id)
                
                if robot:
                    yield Robot(
                        id=robot.id,
                        name=robot.name,
                        status=robot.status.value,
                        position_x=robot.position.x,
                        position_y=robot.position.y,
                        position_z=robot.position.z
                    )
                
                await asyncio.sleep(1)
    
    # Crear schema
    schema = Schema(
        query=RobotQuery,
        mutation=RobotMutation,
        subscription=RobotSubscription
    )
    
    def create_graphql_router(path: str = "/graphql"):
        """Crear router GraphQL para FastAPI"""
        if not STRAWBERRY_AVAILABLE:
            return None
        
        return GraphQLRouter(schema, path=path)
else:
    def create_graphql_router(path: str = "/graphql"):
        """Crear router GraphQL (strawberry no disponible)"""
        print("Warning: strawberry not installed. GraphQL API disabled.")
        print("Install with: pip install strawberry-graphql")
        return None




