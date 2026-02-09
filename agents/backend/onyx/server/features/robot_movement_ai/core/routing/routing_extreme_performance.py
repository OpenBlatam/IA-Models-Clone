"""
Routing Extreme Performance Optimizations
=========================================

Optimizaciones extremas de rendimiento para máximo throughput.
Incluye: TensorRT, ONNX Runtime, INT8 Quantization, Dynamic Batching, etc.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import threading
from collections import deque
import time

try:
    import multiprocessing as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available")


class DynamicBatching:
    """Dynamic batching inteligente para máximo throughput."""
    
    def __init__(self, max_batch_size: int = 128, timeout_ms: float = 10.0):
        """
        Inicializar dynamic batching.
        
        Args:
            max_batch_size: Tamaño máximo de batch
            timeout_ms: Timeout en milisegundos para formar batch
        """
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms / 1000.0  # Convertir a segundos
        self.queue = deque()
        self.lock = threading.Lock()
        self.last_batch_time = time.time()
    
    def add_request(self, request: Any, priority: int = 0) -> bool:
        """
        Agregar request al batch.
        
        Args:
            request: Request a agregar
            priority: Prioridad (mayor = más prioritario)
        
        Returns:
            True si se debe procesar batch inmediatamente
        """
        with self.lock:
            self.queue.append((priority, time.time(), request))
            
            # Procesar si batch está lleno
            if len(self.queue) >= self.max_batch_size:
                return True
            
            # Procesar si timeout
            elapsed = time.time() - self.last_batch_time
            if elapsed >= self.timeout_ms:
                return True
            
            return False
    
    def get_batch(self, max_size: Optional[int] = None) -> List[Any]:
        """
        Obtener batch de requests.
        
        Args:
            max_size: Tamaño máximo del batch
        
        Returns:
            Lista de requests
        """
        with self.lock:
            if not self.queue:
                return []
            
            max_size = max_size or self.max_batch_size
            batch_size = min(len(self.queue), max_size)
            
            # Ordenar por prioridad (mayor primero)
            sorted_items = sorted(self.queue, key=lambda x: x[0], reverse=True)
            
            batch = [item[2] for item in sorted_items[:batch_size]]
            
            # Remover items procesados
            for _ in range(batch_size):
                self.queue.popleft()
            
            self.last_batch_time = time.time()
            return batch


class ONNXRuntimeOptimizer:
    """Optimizador usando ONNX Runtime para inferencia ultra-rápida."""
    
    def __init__(self, providers: Optional[List[str]] = None):
        """
        Inicializar ONNX Runtime optimizer.
        
        Args:
            providers: Lista de providers (CUDAExecutionProvider, CPUExecutionProvider)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.lock = threading.Lock()
    
    def export_to_onnx(
        self,
        model: nn.Module,
        model_id: str,
        example_input: torch.Tensor,
        output_path: str
    ) -> str:
        """
        Exportar modelo a ONNX.
        
        Args:
            model: Modelo PyTorch
            model_id: ID del modelo
            example_input: Input de ejemplo
            output_path: Ruta donde guardar
        
        Returns:
            Ruta al archivo ONNX
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        model.eval()
        
        # Exportar a ONNX
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True
        )
        
        logger.info(f"Model {model_id} exported to ONNX: {output_path}")
        return output_path
    
    def load_onnx_model(self, model_path: str, model_id: str) -> ort.InferenceSession:
        """
        Cargar modelo ONNX.
        
        Args:
            model_path: Ruta al archivo ONNX
            model_id: ID del modelo
        
        Returns:
            InferenceSession de ONNX Runtime
        """
        with self.lock:
            if model_id in self.sessions:
                return self.sessions[model_id]
            
            # Crear sesión optimizada
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            
            self.sessions[model_id] = session
            logger.info(f"ONNX model {model_id} loaded with providers: {self.providers}")
            return session
    
    def infer(self, session: ort.InferenceSession, inputs: np.ndarray) -> np.ndarray:
        """
        Inferencia con ONNX Runtime.
        
        Args:
            session: InferenceSession
            inputs: Inputs como numpy array
        
        Returns:
            Outputs como numpy array
        """
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inputs})
        return outputs[0]


class INT8Quantizer:
    """Quantizador INT8 para modelos ultra-rápidos."""
    
    def __init__(self):
        """Inicializar quantizador."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor]
    ) -> nn.Module:
        """
        Quantizar modelo a INT8.
        
        Args:
            model: Modelo PyTorch
            calibration_data: Datos de calibración
        
        Returns:
            Modelo quantizado
        """
        model.eval()
        
        # Preparar modelo para quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrar con datos
        with torch.no_grad():
            for data in calibration_data:
                model(data)
        
        # Convertir a quantized
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        logger.info("Model quantized to INT8")
        return quantized_model


class SharedMemoryCache:
    """Cache usando memoria compartida para procesos múltiples."""
    
    def __init__(self, max_size: int = 10000):
        """
        Inicializar cache de memoria compartida.
        
        Args:
            max_size: Tamaño máximo del cache
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        # Usar threading.Lock ya que mp puede no estar disponible
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Guardar valor en cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # LRU eviction
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()


class ProcessPoolExecutor:
    """Executor con pool de procesos para máximo paralelismo."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Inicializar pool de procesos.
        
        Args:
            max_workers: Número máximo de workers (None = CPU count)
        """
        try:
            import multiprocessing as mp_module
            self.max_workers = max_workers or mp_module.cpu_count()
            self.pool = mp_module.Pool(processes=self.max_workers)
            self.use_multiprocessing = True
        except Exception:
            # Fallback a threading
            import concurrent.futures
            self.max_workers = max_workers or 4
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
            self.use_multiprocessing = False
            self.pool = None
    
    def submit(self, func: Callable, *args, **kwargs):
        """Enviar tarea al pool."""
        if self.use_multiprocessing and self.pool:
            return self.pool.apply_async(func, args, kwargs)
        else:
            return self.executor.submit(func, *args, **kwargs)


class TorchScriptOptimizer:
    """Optimizador TorchScript con optimizaciones agresivas."""
    
    def __init__(self):
        """Inicializar optimizador TorchScript."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
    
    def optimize_script(
        self,
        scripted_model: torch.jit.ScriptModule,
        optimize_for_inference: bool = True,
        enable_fusion: bool = True
    ) -> torch.jit.ScriptModule:
        """
        Optimizar modelo TorchScript.
        
        Args:
            scripted_model: Modelo TorchScript
            optimize_for_inference: Optimizar para inferencia
            enable_fusion: Habilitar fusion de operaciones
        
        Returns:
            Modelo optimizado
        """
        if optimize_for_inference:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        if enable_fusion:
            # Fusionar operaciones comunes
            torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
        
        logger.info("TorchScript model optimized")
        return scripted_model


class ExtremePerformanceRouter:
    """Router con optimizaciones extremas de rendimiento."""
    
    def __init__(
        self,
        use_onnx: bool = True,
        use_quantization: bool = False,
        use_tensorrt: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        Inicializar router extremo.
        
        Args:
            use_onnx: Usar ONNX Runtime
            use_quantization: Usar INT8 quantization
            use_tensorrt: Usar TensorRT (requiere NVIDIA GPU)
            max_workers: Número de workers para procesamiento paralelo
        """
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.use_quantization = use_quantization and TORCH_AVAILABLE
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE
        
        self.onnx_optimizer = ONNXRuntimeOptimizer() if self.use_onnx else None
        self.quantizer = INT8Quantizer() if self.use_quantization else None
        self.dynamic_batching = DynamicBatching()
        self.shared_cache = SharedMemoryCache()
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.torchscript_optimizer = TorchScriptOptimizer() if TORCH_AVAILABLE else None
        
        self.optimized_models: Dict[str, Any] = {}
    
    def optimize_model_for_inference(
        self,
        model: nn.Module,
        model_id: str,
        example_input: torch.Tensor,
        calibration_data: Optional[List[torch.Tensor]] = None
    ) -> Any:
        """
        Optimizar modelo con todas las técnicas disponibles.
        
        Args:
            model: Modelo PyTorch
            model_id: ID del modelo
            example_input: Input de ejemplo
            calibration_data: Datos para calibración (INT8)
        
        Returns:
            Modelo optimizado
        """
        if model_id in self.optimized_models:
            return self.optimized_models[model_id]
        
        optimized = model
        
        # 1. TorchScript compilation
        if TORCH_AVAILABLE and self.torchscript_optimizer:
            try:
                scripted = torch.jit.trace(model, example_input)
                optimized = self.torchscript_optimizer.optimize_script(scripted)
                logger.info(f"Model {model_id} compiled with TorchScript")
            except Exception as e:
                logger.warning(f"TorchScript compilation failed: {e}")
        
        # 2. INT8 Quantization
        if self.use_quantization and calibration_data:
            try:
                optimized = self.quantizer.quantize_model(optimized, calibration_data)
                logger.info(f"Model {model_id} quantized to INT8")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        
        # 3. ONNX Export (si está habilitado)
        if self.use_onnx:
            try:
                onnx_path = f"/tmp/{model_id}.onnx"
                self.onnx_optimizer.export_to_onnx(model, model_id, example_input, onnx_path)
                onnx_session = self.onnx_optimizer.load_onnx_model(onnx_path, model_id)
                self.optimized_models[model_id] = onnx_session
                logger.info(f"Model {model_id} optimized with ONNX Runtime")
                return onnx_session
            except Exception as e:
                logger.warning(f"ONNX optimization failed: {e}")
        
        self.optimized_models[model_id] = optimized
        return optimized
    
    def batch_infer(
        self,
        model: Any,
        inputs: List[torch.Tensor],
        use_dynamic_batching: bool = True
    ) -> List[torch.Tensor]:
        """
        Inferencia en batch optimizada.
        
        Args:
            model: Modelo optimizado
            inputs: Lista de inputs
            use_dynamic_batching: Usar dynamic batching
        
        Returns:
            Lista de outputs
        """
        if not inputs:
            return []
        
        # Dynamic batching
        if use_dynamic_batching:
            # Agregar requests al batch
            for inp in inputs:
                self.dynamic_batching.add_request(inp)
            
            # Obtener batch
            batch = self.dynamic_batching.get_batch()
            if not batch:
                batch = inputs
        else:
            batch = inputs
        
        # Procesar batch
        if isinstance(model, ort.InferenceSession):
            # ONNX Runtime
            batch_tensor = torch.stack(batch).numpy()
            outputs = self.onnx_optimizer.infer(model, batch_tensor)
            return [torch.from_numpy(outputs[i]) for i in range(len(batch))]
        else:
            # PyTorch
            batch_tensor = torch.stack(batch)
            with torch.no_grad():
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()
                outputs = model(batch_tensor)
            return [outputs[i] for i in range(len(batch))]

