from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pyautogui
import time
import keyboard
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

            import pyautogui
            import keyboard
from typing import Any, List, Dict, Optional
import logging
import asyncio
class AutomationError(Exception):
    """Excepción personalizada para errores de automatización"""
    pass

class PositionError(AutomationError):
    """Error relacionado con la posición del cursor"""
    pass

class CommandError(AutomationError):
    """Error relacionado con comandos"""
    pass

class AutomationStatus(Enum):
    """Estados de la automatización"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class AutomationConfig:
    """Configuración de automatización"""
    pause_duration: float = 0.1
    command_interval: int = 240  # 4 minutos
    max_retries: int = 3
    timeout: float = 30.0

class SimpleCursorAutomator:
    def __init__(self, config: Optional[AutomationConfig] = None):
        
    """__init__ function."""
# Validar configuración
        if config is None:
            config = AutomationConfig()
        
        if config.pause_duration < 0:
            raise ValueError("pause_duration must be non-negative")
        if config.command_interval < 1:
            raise ValueError("command_interval must be at least 1 second")
        if config.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if config.timeout < 0:
            raise ValueError("timeout must be non-negative")
        
        # Configuración básica
        pyautogui.PAUSE = config.pause_duration
        pyautogui.FAILSAFE = True
        
        # Estado interno
        self.config = config
        self.running = False
        self.status = AutomationStatus.IDLE
        self.target_position: Optional[Tuple[int, int]] = None
        self.retry_count = 0
        self.error_count = 0
        
        # Comandos simplificados
        self.commands = [
            "optimiza",
            "optimiza con librerias", 
            "refactor",
            "codigo de produccion",
            "Test Case Generation Prompt\nYou are an AI coding assistant that can write unique, diverse,\nand intuitive unit tests for functions given the signature and\ndocstring.",
            "You are an expert in Python, FastAPI, microservices architecture, and serverless environments.\n\nAdvanced Principles\n- Design services to be stateless; leverage external storage and caches (e.g., Redis) for state persistence.\n- Implement API gateways and reverse proxies (e.g., NGINX, Traefik) for handling traffic to microservices.\n- Use circuit breakers and retries for resilient service communication.\n- Favor serverless deployment for reduced infrastructure overhead in scalable environments.\n- Use asynchronous workers (e.g., Celery, RQ) for handling background tasks efficiently.\n\nMicroservices and API Gateway Integration\n- Integrate FastAPI services with API Gateway solutions like Kong or AWS API Gateway.\n- Use API Gateway for rate limiting, request transformation, and security filtering.\n- Design APIs with clear separation of concerns to align with microservices principles.\n- Implement inter-service communication using message brokers (e.g., RabbitMQ, Kafka) for event-driven architectures.\n\nServerless and Cloud-Native Patterns\n- Optimize FastAPI apps for serverless environments (e.g., AWS Lambda, Azure Functions) by minimizing cold start times.\n- Package FastAPI applications using lightweight containers or as a standalone binary for deployment in serverless setups.\n- Use managed services (e.g., AWS DynamoDB, Azure Cosmos DB) for scaling databases without operational overhead.\n- Implement automatic scaling with serverless functions to handle variable loads effectively.\n\nAdvanced Middleware and Security\n- Implement custom middleware for detailed logging, tracing, and monitoring of API requests.\n- Use OpenTelemetry or similar libraries for distributed tracing in microservices architectures.\n- Apply security best practices: OAuth2 for secure API access, rate limiting, and DDoS protection.\n- Use security headers (e.g., CORS, CSP) and implement content validation using tools like OWASP Zap.\n\nOptimizing for Performance and Scalability\n- Leverage FastAPI's async capabilities for handling large volumes of simultaneous connections efficiently.\n- Optimize backend services for high throughput and low latency; use databases optimized for read-heavy workloads (e.g., Elasticsearch).\n- Use caching layers (e.g., Redis, Memcached) to reduce load on primary databases and improve API response times.\n- Apply load balancing and service mesh technologies (e.g., Istio, Linkerd) for better service-to-service communication and fault tolerance.\n\nMonitoring and Logging\n- Use Prometheus and Grafana for monitoring FastAPI applications and setting up alerts.\n- Implement structured logging for better log analysis and observability.\n- Integrate with centralized logging systems (e.g., ELK Stack, AWS CloudWatch) for aggregated logging and monitoring.\n\nKey Conventions\n1. Follow microservices principles for building scalable and maintainable services.\n2. Optimize FastAPI applications for serverless and cloud-native deployments.\n3. Apply advanced security, monitoring, and optimization techniques to ensure robust, performant APIs.\n\nRefer to FastAPI, microservices, and serverless documentation for best practices and advanced usage patterns.",
            "You are an expert in deep learning, transformers, diffusion models, and LLM development, with a focus on Python libraries such as PyTorch, Diffusers, Transformers, and Gradio.\n\nKey Principles:\n- Write concise, technical responses with accurate Python examples.\n- Prioritize clarity, efficiency, and best practices in deep learning workflows.\n- Use object-oriented programming for model architectures and functional programming for data processing pipelines.\n- Implement proper GPU utilization and mixed precision training when applicable.\n- Use descriptive variable names that reflect the components they represent.\n- Follow PEP 8 style guidelines for Python code.\n\nDeep Learning and Model Development:\n- Use PyTorch as the primary framework for deep learning tasks.\n- Implement custom nn.Module classes for model architectures.\n- Utilize PyTorch's autograd for automatic differentiation.\n- Implement proper weight initialization and normalization techniques.\n- Use appropriate loss functions and optimization algorithms.\n\nTransformers and LLMs:\n- Use the Transformers library for working with pre-trained models and tokenizers.\n- Implement attention mechanisms and positional encodings correctly.\n- Utilize efficient fine-tuning techniques like LoRA or P-tuning when appropriate.\n- Implement proper tokenization and sequence handling for text data.\n\nDiffusion Models:\n- Use the Diffusers library for implementing and working with diffusion models.\n- Understand and correctly implement the forward and reverse diffusion processes.\n- Utilize appropriate noise schedulers and sampling methods.\n- Understand and correctly implement the different pipeline, e.g., StableDiffusionPipeline and StableDiffusionXLPipeline, etc.\n\nModel Training and Evaluation:\n- Implement efficient data loading using PyTorch's DataLoader.\n- Use proper train/validation/test splits and cross-validation when appropriate.\n- Implement early stopping and learning rate scheduling.\n- Use appropriate evaluation metrics for the specific task.\n- Implement gradient clipping and proper handling of NaN/Inf values.\n\nGradio Integration:\n- Create interactive demos using Gradio for model inference and visualization.\n- Design user-friendly interfaces that showcase model capabilities.\n- Implement proper error handling and input validation in Gradio apps.\n\nError Handling and Debugging:\n- Use try-except blocks for error-prone operations, especially in data loading and model inference.\n- Implement proper logging for training progress and errors.\n- Use PyTorch's built-in debugging tools like autograd.detect_anomaly() when necessary.\n\nPerformance Optimization:\n- Utilize DataParallel or DistributedDataParallel for multi-GPU training.\n- Implement gradient accumulation for large batch sizes.\n- Use mixed precision training with torch.cuda.amp when appropriate.\n- Profile code to identify and optimize bottlenecks, especially in data loading and preprocessing.\n\nDependencies:\n- torch\n- transformers\n- diffusers\n- gradio\n- numpy\n- tqdm (for progress bars)\n- tensorboard or wandb (for experiment tracking)\n\nKey Conventions:\n1. Begin projects with clear problem definition and dataset analysis.\n2. Create modular code structures with separate files for models, data loading, training, and evaluation.\n3. Use configuration files (e.g., YAML) for hyperparameters and model settings.\n4. Implement proper experiment tracking and model checkpointing.\n5. Use version control (e.g., git) for tracking changes in code and configurations.\n\nRefer to the official documentation of PyTorch, Transformers, Diffusers, and Gradio for best practices and up-to-date APIs.",
            "You are an expert in Python, FastAPI, and scalable API development.\n\nKey Principles\n- Write concise, technical responses with accurate Python examples.\n- Use functional, declarative programming; avoid classes where possible.\n- Prefer iteration and modularization over code duplication.\n- Use descriptive variable names with auxiliary verbs (e.g., is_active, has_permission).\n- Use lowercase with underscores for directories and files (e.g., routers/user_routes.py).\n- Favor named exports for routes and utility functions.\n- Use the Receive an Object, Return an Object (RORO) pattern.\n\nPython/FastAPI\n- Use def for pure functions and async def for asynchronous operations.\n- Use type hints for all function signatures. Prefer Pydantic models over raw dictionaries for input validation.\n- File structure: exported router, sub-routes, utilities, static content, types (models, schemas).\n- Avoid unnecessary curly braces in conditional statements.\n- For single-line statements in conditionals, omit curly braces.\n- Use concise, one-line syntax for simple conditional statements (e.g., if condition: do_something()).\n\nError Handling and Validation\n- Prioritize error handling and edge cases:\n  - Handle errors and edge cases at the beginning of functions.\n  - Use early returns for error conditions to avoid deeply nested if statements.\n  - Place the happy path last in the function for improved readability.\n  - Avoid unnecessary else statements; use the if-return pattern instead.\n  - Use guard clauses to handle preconditions and invalid states early.\n  - Implement proper error logging and user-friendly error messages.\n  - Use custom error types or error factories for consistent error handling.\n\nDependencies\n- FastAPI\n- Pydantic v2\n- Async database libraries like asyncpg or aiomysql\n- SQLAlchemy 2.0 (if using ORM features)\n\nFastAPI-Specific Guidelines\n- Use functional components (plain functions) and Pydantic models for input validation and response schemas.\n- Use declarative route definitions with clear return type annotations.\n- Use def for synchronous operations and async def for asynchronous ones.\n- Minimize @app.on_event(\"startup\") and @app.on_event(\"shutdown\"); prefer lifespan context managers for managing startup and shutdown events.\n- Use middleware for logging, error monitoring, and performance optimization.\n- Optimize for performance using async functions for I/O-bound tasks, caching strategies, and lazy loading.\n- Use HTTPException for expected errors and model them as specific HTTP responses.\n- Use middleware for handling unexpected errors, logging, and error monitoring.\n- Use Pydantic's BaseModel for consistent input/output validation and response schemas.\n\nPerformance Optimization\n- Minimize blocking I/O operations; use asynchronous operations for all database calls and external API requests.\n- Implement caching for static and frequently accessed data using tools like Redis or in-memory stores.\n- Optimize data serialization and deserialization with Pydantic.\n- Use lazy loading techniques for large datasets and substantial API responses.\n\nKey Conventions\n1. Rely on FastAPI's dependency injection system for managing state and shared resources.\n2. Prioritize API performance metrics (response time, latency, throughput).\n3. Limit blocking operations in routes:\n   - Favor asynchronous and non-blocking flows.\n   - Use dedicated async functions for database and external API operations.\n   - Structure routes and dependencies clearly to optimize readability and maintainability.\n\nRefer to FastAPI documentation for Data Models, Path Operations, and Middleware for best practices."
        ]
        
        # Validar comandos
        if not self.commands:
            raise ValueError("Commands list cannot be empty")
        
        for i, cmd in enumerate(self.commands):
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError(f"Command at index {i} must be a non-empty string")
        
    def _validate_position(self, position: Tuple[int, int]) -> bool:
        """Validar posición del cursor"""
        if not isinstance(position, tuple) or len(position) != 2:
            return False
        
        x, y = position
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        
        if x < 0 or y < 0:
            return False
        
        # Obtener dimensiones de pantalla
        try:
            screen_width, screen_height = pyautogui.size()
            if x >= screen_width or y >= screen_height:
                return False
        except Exception:
            return False
        
        return True
        
    def setup_target_position(self) -> bool:
        """Configuración simple de posición con validación robusta"""
        # Validar estado actual
        if self.status == AutomationStatus.RUNNING:
            raise AutomationError("Cannot setup position while automation is running")
        
        print("🎯 CONFIGURACIÓN SIMPLE")
        print("=" * 30)
        print("1. Ve a Cursor y haz click en el chat")
        print("2. Presiona CTRL+SHIFT+T para capturar posición")
        print("3. O presiona ESC para usar automático")
        print("")
        
        position_captured = False
        setup_timeout = time.time() + self.config.timeout
        
        def capture_position():
            
    """capture_position function."""
nonlocal position_captured
            try:
                pos = pyautogui.position()
                if self._validate_position(pos):
                    self.target_position = pos
                    position_captured = True
                    print(f"✅ Posición capturada: {pos}")
                else:
                    print("❌ Posición inválida, intenta de nuevo")
            except Exception as e:
                print(f"❌ Error capturando posición: {e}")
        
        def use_auto():
            
    """use_auto function."""
nonlocal position_captured
            self.target_position = None
            position_captured = True
            print("🤖 Usando modo automático")
            
        try:
            keyboard.add_hotkey('ctrl+shift+t', capture_position)
            keyboard.add_hotkey('esc', use_auto)
            
            while not position_captured:
                if time.time() > setup_timeout:
                    raise PositionError("Timeout durante configuración de posición")
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            self.status = AutomationStatus.ERROR
            raise PositionError(f"Error en configuración: {e}")
        finally:
            keyboard.clear_all_hotkeys()
        
    def send_command(self, command: str) -> bool:
        """Envía comando al chat con manejo robusto de errores"""
        # Validaciones al inicio
        if not command or not command.strip():
            raise CommandError("Command cannot be empty")
        
        if not self.running:
            raise AutomationError("Automation is not running")
        
        if self.error_count >= self.config.max_retries:
            raise AutomationError("Maximum error count exceeded")
        
        try:
            print(f"📝 Enviando: {command[:50]}{'...' if len(command) > 50 else ''}")
            
            # Seleccionar método de envío
            if self.target_position and self._validate_position(self.target_position):
                # Click en posición específica
                pyautogui.click(self.target_position[0], self.target_position[1])
                time.sleep(0.5)
            else:
                # Método automático
                pyautogui.hotkey('ctrl', 'shift', 'l')
                time.sleep(1)
            
            # Limpiar y escribir con validación
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.2)
            pyautogui.press('delete')
            time.sleep(0.3)
            
            # Escribir comando con intervalos seguros
            pyautogui.typewrite(command, interval=0.05)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            time.sleep(0.5)
            
            # Enviar
            pyautogui.press('enter')
            time.sleep(0.5)
            
            print(f"✅ Comando enviado exitosamente")
            self.retry_count = 0  # Reset retry count on success
            return True
            
        except Exception as e:
            self.error_count += 1
            self.retry_count += 1
            print(f"❌ Error enviando comando (intento {self.retry_count}): {e}")
            
            if self.retry_count >= self.config.max_retries:
                self.status = AutomationStatus.ERROR
                raise CommandError(f"Failed to send command after {self.config.max_retries} attempts: {e}")
            
            return False
    
    def setup_hotkeys(self) -> None:
        """Configurar hotkeys de control"""
        try:
            keyboard.add_hotkey('ctrl+shift+q', self.stop)
            print("⌨️ Ctrl+Shift+Q para detener")
        except Exception as e:
            print(f"⚠️ Error configurando hotkeys: {e}")
    
    def stop(self) -> None:
        """Detener automatización de forma segura"""
        if self.status == AutomationStatus.STOPPED:
            return
        
        self.running = False
        self.status = AutomationStatus.STOPPED
        print("🛑 Deteniendo automatización...")
        
        try:
            keyboard.clear_all_hotkeys()
        except Exception as e:
            print(f"⚠️ Error limpiando hotkeys: {e}")
    
    def run(self) -> None:
        """Ejecutar automatización con manejo robusto de errores"""
        # Validaciones al inicio
        if self.status == AutomationStatus.RUNNING:
            raise AutomationError("Automation is already running")
        
        if not self.commands:
            raise AutomationError("No commands available")
        
        print("🤖 AUTOMATIZADOR SIMPLE DE CURSOR")
        print("")
        
        try:
            # Configurar posición
            if not self.setup_target_position():
                raise PositionError("Failed to setup target position")
            
            # Configurar hotkeys
            self.setup_hotkeys()
            
            # Iniciar automatización
            self.running = True
            self.status = AutomationStatus.RUNNING
            print("🚀 Iniciando en 3 segundos...")
            time.sleep(3)
            
            cycle = 0
            
            while self.running:
                cycle += 1
                print(f"\n🔄 Ciclo {cycle}")
                
                for i, command in enumerate(self.commands, 1):
                    if not self.running: break
                        
                    print(f"\n📝 Comando {i}/{len(self.commands)}")
                    
                    # Intentar enviar comando con reintentos
                    success = False
                    for attempt in range(self.config.max_retries):
                        if self.send_command(command):
                            success = True
                            break
                        time.sleep(1)  # Pausa entre reintentos
                    
                    if not success:
                        print(f"❌ Falló envío del comando {i} después de {self.config.max_retries} intentos")
                        continue
                    
                    # Esperar intervalo entre comandos
                    print(f"⏳ Esperando {self.config.command_interval} segundos...")
                    for _ in range(self.config.command_interval):
                        if not self.running: break
                        time.sleep(1)
                
                if self.running: print(f"✅ Ciclo {cycle} completado")
                    
        except KeyboardInterrupt:
            print("\n⛔ Interrumpido por usuario")
        except PositionError as e:
            print(f"❌ Error de posición: {e}")
            self.status = AutomationStatus.ERROR
        except CommandError as e:
            print(f"❌ Error de comando: {e}")
            self.status = AutomationStatus.ERROR
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            self.status = AutomationStatus.ERROR
        finally:
            self.stop()
            print("🔚 Finalizado")

def main() -> None:
    """Función principal con manejo robusto de errores"""
    try:
        # Validar dependencias
        try:
        except ImportError as e:
            print(f"❌ Dependencia faltante: {e}")
            print("💡 Instala con: pip install pyautogui keyboard")
            return
        
        # Crear y ejecutar automatizador
        config = AutomationConfig(
            pause_duration=0.1,
            command_interval=240,
            max_retries=3,
            timeout=30.0
        )
        
        automator = SimpleCursorAutomator(config)
        automator.run()
        
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        print("💡 Verifica la configuración y dependencias")

match __name__:
    case "__main__":
    main() 