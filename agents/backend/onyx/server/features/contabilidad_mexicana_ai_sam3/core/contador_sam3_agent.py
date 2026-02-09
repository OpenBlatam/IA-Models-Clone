"""
Contador SAM3 Agent - Main class for Mexican accounting with SAM3 architecture
=============================================================================

Refactored with:
- Integration of refactored components (TaskManager, ServiceHandler, etc.)
- Improved startup/shutdown lifecycle
- Better error handling
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient
from ..config.contador_sam3_config import ContadorSAM3Config
from .system_prompts_builder import SystemPromptsBuilder
from .task_manager import TaskManager, FileTaskRepository
from .parallel_executor import ParallelExecutor
from .service_handler import ServiceHandler, ServiceType
from .task_creator import TaskCreator
from .helpers import create_output_directories

logger = logging.getLogger(__name__)


class ContadorSAM3Agent:
    """
    Autonomous agent for Mexican accounting based on SAM3 architecture.
    
    Features:
    - Continuous 24/7 operation
    - Parallel task execution
    - OpenRouter LLM integration
    - TruthGPT optimization
    - Automatic task management
    - Mexican accounting services
    """
    
    def __init__(
        self,
        config: Optional[ContadorSAM3Config] = None,
        max_parallel_tasks: int = 10,
        output_dir: str = "contador_sam3_output",
        debug: bool = False,
    ):
        self.config = config or ContadorSAM3Config()
        self.config.validate()
        
        self.openrouter_client = OpenRouterClient(api_key=self.config.openrouter.api_key)
        self.truthgpt_client = TruthGPTClient(config=self.config.truthgpt.to_dict() if self.config.truthgpt else {})
        
        self.output_dir = Path(output_dir)
        self.output_dirs = create_output_directories(
            self.output_dir,
            ["results", "tasks", "storage"]
        )
        
        # Initialize TaskManager with FileRepository
        self.task_manager = TaskManager(
            repository=FileTaskRepository(str(self.output_dirs["storage"]))
        )
        
        self.parallel_executor = ParallelExecutor(max_workers=max_parallel_tasks)
        self.debug = debug
        self.running = False
        
        # System prompts
        self.system_prompts = SystemPromptsBuilder.build_all_prompts()
        
        # Service handler
        self.service_handler = ServiceHandler(
            openrouter_client=self.openrouter_client,
            truthgpt_client=self.truthgpt_client,
            config=self.config,
            system_prompts=self.system_prompts
        )
        
        logger.info(f"Initialized ContadorSAM3Agent with {max_parallel_tasks} max parallel tasks")
    
    async def start(self):
        """Start the autonomous agent in continuous operation mode."""
        if self.running:
            logger.warning("Agent is already running")
            return
        
        self.running = True
        logger.info("Starting Contador SAM3 Agent (24/7 mode)")
        
        try:
            # Initialize task manager (load existing tasks)
            await self.task_manager.initialize()
            
            # Start parallel executor
            await self.parallel_executor.start()
            
            # Main event loop for continuous operation
            while self.running:
                try:
                    # Get pending tasks from task manager
                    tasks = await self.task_manager.get_pending_tasks(limit=10)
                    
                    if tasks:
                        # Submit tasks to parallel executor
                        for task in tasks:
                            await self.parallel_executor.submit_task(
                                self._process_task,
                                task
                            )
                    
                    # Wait before next iteration
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(5.0)
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the autonomous agent."""
        logger.info("Stopping Contador SAM3 Agent")
        self.running = False
        await self.parallel_executor.stop()
        await self.openrouter_client.close()
        await self.truthgpt_client.close()
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task."""
        task_id = task.get("id", "unknown")
        service_type_str = task.get("service_type")
        parameters = task.get("parameters", {})
        
        logger.info(f"Processing task {task_id}: {service_type_str}")
        
        try:
            # Update task status
            await self.task_manager.update_task_status(task_id, "processing")
            
            # Convert string service type to Enum
            try:
                service_type = ServiceType(service_type_str)
            except ValueError:
                raise ValueError(f"Unknown service type: {service_type_str}")
            
            # Execute service using handler
            result = await self.service_handler.handle(service_type, parameters)
            
            if result.success:
                # Save result
                await self.task_manager.complete_task(task_id, result.to_dict())
                logger.info(f"Task {task_id} completed successfully")
                return result.to_dict()
            else:
                # Handle service failure
                error_msg = result.error or "Unknown error"
                await self.task_manager.fail_task(task_id, error_msg)
                logger.error(f"Task {task_id} failed: {error_msg}")
                raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            await self.task_manager.fail_task(task_id, str(e))
            raise
    
    # Public API methods
    async def calcular_impuestos(
        self,
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """Submit tax calculation task."""
        return await TaskCreator.create_calcular_impuestos_task(
            self.task_manager, regimen, tipo_impuesto, datos, priority
        )
    
    async def asesoria_fiscal(
        self,
        pregunta: str,
        contexto: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit fiscal advice task."""
        return await TaskCreator.create_asesoria_fiscal_task(
            self.task_manager, pregunta, contexto, priority
        )
    
    async def guia_fiscal(
        self,
        tema: str,
        nivel_detalle: str = "completo",
        priority: int = 0,
    ) -> str:
        """Submit fiscal guide task."""
        return await TaskCreator.create_guia_fiscal_task(
            self.task_manager, tema, nivel_detalle, priority
        )
    
    async def tramite_sat(
        self,
        tipo_tramite: str,
        detalles: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit SAT procedure task."""
        return await TaskCreator.create_tramite_sat_task(
            self.task_manager, tipo_tramite, detalles, priority
        )
    
    async def ayuda_declaracion(
        self,
        tipo_declaracion: str,
        periodo: str,
        datos: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit declaration assistance task."""
        return await TaskCreator.create_ayuda_declaracion_task(
            self.task_manager, tipo_declaracion, periodo, datos, priority
        )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        return await self.task_manager.get_task_status(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task."""
        return await self.task_manager.get_task_result(task_id)
