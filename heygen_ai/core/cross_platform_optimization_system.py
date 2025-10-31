"""
Cross-Platform Optimization System for HeyGen AI Enterprise

This module provides hardware-specific optimizations for different platforms:
- GPU optimization (CUDA, ROCm, Metal)
- CPU optimization (Intel, AMD, ARM)
- Edge device optimization (mobile, embedded)
- Platform detection and auto-configuration
- Hardware-specific performance tuning
"""

import logging
import platform
import subprocess
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """Configuration for cross-platform optimization."""
    
    # Platform detection
    auto_detect_platform: bool = True
    force_platform: Optional[str] = None
    
    # GPU optimization
    enable_cuda_optimization: bool = True
    enable_rocm_optimization: bool = True
    enable_metal_optimization: bool = True
    
    # CPU optimization
    enable_intel_optimization: bool = True
    enable_amd_optimization: bool = True
    enable_arm_optimization: bool = True
    
    # Edge optimization
    enable_mobile_optimization: bool = True
    enable_embedded_optimization: bool = True
    
    # Performance tuning
    optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"
    enable_auto_tuning: bool = True


class PlatformDetector:
    """Detects and identifies the current hardware platform."""
    
    def __init__(self):
        self.platform_info = {}
        self.detected_platform = None
        
    def detect_platform(self) -> Dict[str, Any]:
        """Detect current platform and hardware capabilities."""
        try:
            # Basic system info
            self.platform_info = {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()[0]
            }
            
            # Detect GPU capabilities
            gpu_info = self._detect_gpu()
            if gpu_info:
                self.platform_info["gpu"] = gpu_info
            
            # Detect CPU capabilities
            cpu_info = self._detect_cpu()
            if cpu_info:
                self.platform_info["cpu"] = cpu_info
            
            # Detect memory capabilities
            memory_info = self._detect_memory()
            if memory_info:
                self.platform_info["memory"] = memory_info
            
            # Determine platform type
            self.detected_platform = self._determine_platform_type()
            self.platform_info["platform_type"] = self.detected_platform
            
            logger.info(f"Platform detected: {self.detected_platform}")
            return self.platform_info
            
        except Exception as e:
            logger.error(f"Platform detection failed: {e}")
            return {"error": str(e)}
    
    def _detect_gpu(self) -> Optional[Dict[str, Any]]:
        """Detect GPU capabilities."""
        try:
            gpu_info = {}
            
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_info["cuda"] = {
                    "available": True,
                    "version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "compute_capability": torch.cuda.get_device_capability(0)
                }
                
                # Get detailed GPU info
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info[f"gpu_{i}"] = {
                        "name": props.name,
                        "memory_total": props.total_memory,
                        "memory_mb": props.total_memory // (1024 * 1024),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multi_processor_count
                    }
            
            # Check for ROCm (AMD)
            if hasattr(torch, 'hip') and torch.hip.is_available():
                gpu_info["rocm"] = {
                    "available": True,
                    "version": torch.version.hip
                }
            
            # Check for Metal (Apple)
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info["metal"] = {
                    "available": True,
                    "device_name": "Apple Silicon GPU"
                }
            
            return gpu_info if gpu_info else None
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return None
    
    def _detect_cpu(self) -> Optional[Dict[str, Any]]:
        """Detect CPU capabilities."""
        try:
            cpu_info = {
                "count": os.cpu_count(),
                "architecture": platform.architecture()[0]
            }
            
            # Try to get CPU brand
            try:
                if platform.system() == "Windows":
                    cpu_info["brand"] = platform.processor()
                elif platform.system() == "Linux":
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if line.startswith("model name"):
                                cpu_info["brand"] = line.split(":")[1].strip()
                                break
                elif platform.system() == "Darwin":  # macOS
                    result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        cpu_info["brand"] = result.stdout.strip()
            except Exception:
                pass
            
            # Detect CPU vendor
            if "intel" in cpu_info.get("brand", "").lower():
                cpu_info["vendor"] = "intel"
            elif "amd" in cpu_info.get("brand", "").lower():
                cpu_info["vendor"] = "amd"
            elif "arm" in cpu_info.get("brand", "").lower() or "apple" in cpu_info.get("brand", "").lower():
                cpu_info["vendor"] = "arm"
            else:
                cpu_info["vendor"] = "unknown"
            
            return cpu_info
            
        except Exception as e:
            logger.error(f"CPU detection failed: {e}")
            return None
    
    def _detect_memory(self) -> Optional[Dict[str, Any]]:
        """Detect memory capabilities."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total // (1024**3),
                "available_gb": memory.available // (1024**3),
                "percent_used": memory.percent
            }
            
            return memory_info
            
        except ImportError:
            logger.warning("psutil not available for memory detection")
            return None
        except Exception as e:
            logger.error(f"Memory detection failed: {e}")
            return None
    
    def _determine_platform_type(self) -> str:
        """Determine the primary platform type."""
        try:
            if self.platform_info.get("gpu", {}).get("cuda", {}).get("available", False):
                return "cuda_gpu"
            elif self.platform_info.get("gpu", {}).get("rocm", {}).get("available", False):
                return "rocm_gpu"
            elif self.platform_info.get("gpu", {}).get("metal", {}).get("available", False):
                return "metal_gpu"
            elif self.platform_info.get("cpu", {}).get("vendor") == "intel":
                return "intel_cpu"
            elif self.platform_info.get("cpu", {}).get("vendor") == "amd":
                return "amd_cpu"
            elif self.platform_info.get("cpu", {}).get("vendor") == "arm":
                return "arm_cpu"
            else:
                return "generic"
                
        except Exception as e:
            logger.error(f"Platform type determination failed: {e}")
            return "unknown"


class GPUOptimizer:
    """GPU-specific optimization strategies."""
    
    def __init__(self, platform_info: Dict[str, Any]):
        self.platform_info = platform_info
        self.gpu_info = platform_info.get("gpu", {})
        
    def optimize_for_cuda(self) -> Dict[str, Any]:
        """Optimize for CUDA GPUs."""
        try:
            if not self.gpu_info.get("cuda", {}).get("available", False):
                return {"status": "cuda_not_available"}
            
            cuda_info = self.gpu_info["cuda"]
            optimizations = {
                "status": "success",
                "platform": "cuda",
                "device_count": cuda_info["device_count"],
                "compute_capability": cuda_info["compute_capability"],
                "recommendations": []
            }
            
            # Check compute capability for optimization level
            major, minor = cuda_info["compute_capability"]
            if major >= 8:  # Ampere or newer
                optimizations["recommendations"].extend([
                    "Enable Tensor Cores for FP16 operations",
                    "Use CUDA Graphs for repeated operations",
                    "Enable asynchronous memory operations"
                ])
            elif major >= 7:  # Volta or newer
                optimizations["recommendations"].extend([
                    "Enable Tensor Cores for mixed precision",
                    "Use structured sparsity if applicable"
                ])
            
            # Memory optimization recommendations
            for i in range(cuda_info["device_count"]):
                gpu_key = f"gpu_{i}"
                if gpu_key in self.gpu_info:
                    gpu_memory = self.gpu_info[gpu_key]["memory_mb"]
                    if gpu_memory >= 16000:  # 16GB+
                        optimizations["recommendations"].append(f"GPU {i}: Large memory available - consider larger batch sizes")
                    elif gpu_memory <= 8000:  # 8GB or less
                        optimizations["recommendations"].append(f"GPU {i}: Limited memory - consider gradient checkpointing")
            
            return optimizations
            
        except Exception as e:
            logger.error(f"CUDA optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_for_rocm(self) -> Dict[str, Any]:
        """Optimize for ROCm (AMD) GPUs."""
        try:
            if not self.gpu_info.get("rocm", {}).get("available", False):
                return {"status": "rocm_not_available"}
            
            optimizations = {
                "status": "success",
                "platform": "rocm",
                "recommendations": [
                    "Use ROCm-specific kernels when available",
                    "Enable HIP optimizations",
                    "Consider mixed precision training",
                    "Use AMD-specific memory management"
                ]
            }
            
            return optimizations
            
        except Exception as e:
            logger.error(f"ROCm optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_for_metal(self) -> Dict[str, Any]:
        """Optimize for Metal (Apple Silicon) GPUs."""
        try:
            if not self.gpu_info.get("metal", {}).get("available", False):
                return {"status": "metal_not_available"}
            
            optimizations = {
                "status": "success",
                "platform": "metal",
                "recommendations": [
                    "Use MPS backend for optimal performance",
                    "Enable Metal Performance Shaders",
                    "Consider Core ML for deployment",
                    "Use Apple-specific optimizations"
                ]
            }
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Metal optimization failed: {e}")
            return {"status": "error", "error": str(e)}


class CPUOptimizer:
    """CPU-specific optimization strategies."""
    
    def __init__(self, platform_info: Dict[str, Any]):
        self.platform_info = platform_info
        self.cpu_info = platform_info.get("cpu", {})
        
    def optimize_for_intel(self) -> Dict[str, Any]:
        """Optimize for Intel CPUs."""
        try:
            if self.cpu_info.get("vendor") != "intel":
                return {"status": "not_intel_cpu"}
            
            optimizations = {
                "status": "success",
                "platform": "intel_cpu",
                "recommendations": [
                    "Enable Intel MKL optimizations",
                    "Use Intel OpenMP runtime",
                    "Consider Intel oneAPI optimizations",
                    "Enable AVX-512 if available"
                ]
            }
            
            # Check for specific Intel features
            brand = self.cpu_info.get("brand", "").lower()
            if "xeon" in brand:
                optimizations["recommendations"].append("Server CPU detected - optimize for multi-socket configurations")
            elif "i7" in brand or "i9" in brand:
                optimizations["recommendations"].append("High-end desktop CPU - enable turbo boost optimizations")
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Intel optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_for_amd(self) -> Dict[str, Any]:
        """Optimize for AMD CPUs."""
        try:
            if self.cpu_info.get("vendor") != "amd":
                return {"status": "not_amd_cpu"}
            
            optimizations = {
                "status": "success",
                "platform": "amd_cpu",
                "recommendations": [
                    "Enable AMD BLIS optimizations",
                    "Use AMD-specific math libraries",
                    "Consider ROCm CPU optimizations",
                    "Enable AVX2/AVX-512 if available"
                ]
            }
            
            return optimizations
            
        except Exception as e:
            logger.error(f"AMD optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_for_arm(self) -> Dict[str, Any]:
        """Optimize for ARM CPUs."""
        try:
            if self.cpu_info.get("vendor") != "arm":
                return {"status": "not_arm_cpu"}
            
            optimizations = {
                "status": "success",
                "platform": "arm_cpu",
                "recommendations": [
                    "Use ARM-optimized BLAS libraries",
                    "Enable NEON vectorization",
                    "Consider ARM Compute Library",
                    "Optimize for ARM memory hierarchy"
                ]
            }
            
            # Check for Apple Silicon
            if "apple" in self.cpu_info.get("brand", "").lower():
                optimizations["recommendations"].extend([
                    "Apple Silicon detected - use Metal Performance Shaders",
                    "Enable Core ML optimizations",
                    "Use Apple-specific neural engine if available"
                ])
            
            return optimizations
            
        except Exception as e:
            logger.error(f"ARM optimization failed: {e}")
            return {"status": "error", "error": str(e)}


class EdgeOptimizer:
    """Edge device optimization strategies."""
    
    def __init__(self, platform_info: Dict[str, Any]):
        self.platform_info = platform_info
        
    def optimize_for_mobile(self) -> Dict[str, Any]:
        """Optimize for mobile devices."""
        try:
            optimizations = {
                "status": "success",
                "platform": "mobile",
                "recommendations": [
                    "Use model quantization (INT8/FP16)",
                    "Enable model pruning and compression",
                    "Use mobile-specific frameworks (Core ML, TensorFlow Lite)",
                    "Optimize for battery life",
                    "Consider cloud offloading for heavy operations"
                ]
            }
            
            # Check if it's Apple device
            if self.platform_info.get("gpu", {}).get("metal", {}).get("available", False):
                optimizations["recommendations"].extend([
                    "Use Metal Performance Shaders",
                    "Enable Core ML optimizations"
                ])
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Mobile optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_for_embedded(self) -> Dict[str, Any]:
        """Optimize for embedded devices."""
        try:
            optimizations = {
                "status": "success",
                "platform": "embedded",
                "recommendations": [
                    "Use extreme model compression",
                    "Enable INT4 quantization if supported",
                    "Use specialized embedded frameworks",
                    "Optimize for memory constraints",
                    "Consider hardware accelerators (NPU, DSP)"
                ]
            }
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Embedded optimization failed: {e}")
            return {"status": "error", "error": str(e)}


class CrossPlatformOptimizationSystem:
    """Main system for cross-platform optimization."""
    
    def __init__(self, config: Optional[PlatformConfig] = None):
        self.config = config or PlatformConfig()
        self.logger = logging.getLogger(f"{__name__}.system")
        
        # Initialize components
        self.platform_detector = PlatformDetector()
        self.gpu_optimizer = None
        self.cpu_optimizer = None
        self.edge_optimizer = None
        
        # Platform info
        self.platform_info = {}
        self.detected_platform = None
        
        # Auto-detect platform if enabled
        if self.config.auto_detect_platform:
            self.detect_platform()
    
    def detect_platform(self) -> Dict[str, Any]:
        """Detect current platform."""
        try:
            self.platform_info = self.platform_detector.detect_platform()
            self.detected_platform = self.platform_info.get("platform_type", "unknown")
            
            # Initialize optimizers
            self.gpu_optimizer = GPUOptimizer(self.platform_info)
            self.cpu_optimizer = CPUOptimizer(self.platform_info)
            self.edge_optimizer = EdgeOptimizer(self.platform_info)
            
            self.logger.info(f"Platform detection completed: {self.detected_platform}")
            return self.platform_info
            
        except Exception as e:
            self.logger.error(f"Platform detection failed: {e}")
            return {"error": str(e)}
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations for current platform."""
        try:
            if not self.platform_info:
                self.detect_platform()
            
            recommendations = {
                "platform_info": self.platform_info,
                "platform_type": self.detected_platform,
                "optimizations": {}
            }
            
            # GPU optimizations
            if self.gpu_optimizer:
                if self.gpu_optimizer.gpu_info.get("cuda", {}).get("available", False):
                    recommendations["optimizations"]["cuda"] = self.gpu_optimizer.optimize_for_cuda()
                
                if self.gpu_optimizer.gpu_info.get("rocm", {}).get("available", False):
                    recommendations["optimizations"]["rocm"] = self.gpu_optimizer.optimize_for_rocm()
                
                if self.gpu_optimizer.gpu_info.get("metal", {}).get("available", False):
                    recommendations["optimizations"]["metal"] = self.gpu_optimizer.optimize_for_metal()
            
            # CPU optimizations
            if self.cpu_optimizer:
                cpu_vendor = self.cpu_optimizer.cpu_info.get("vendor")
                if cpu_vendor == "intel":
                    recommendations["optimizations"]["intel_cpu"] = self.cpu_optimizer.optimize_for_intel()
                elif cpu_vendor == "amd":
                    recommendations["optimizations"]["amd_cpu"] = self.cpu_optimizer.optimize_for_amd()
                elif cpu_vendor == "arm":
                    recommendations["optimizations"]["arm_cpu"] = self.cpu_optimizer.optimize_for_arm()
            
            # Edge optimizations
            if self.edge_optimizer:
                if self.detected_platform in ["metal_gpu", "arm_cpu"]:
                    recommendations["optimizations"]["mobile"] = self.edge_optimizer.optimize_for_mobile()
                
                if self.platform_info.get("cpu", {}).get("count", 0) <= 4:
                    recommendations["optimizations"]["embedded"] = self.edge_optimizer.optimize_for_embedded()
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Optimization recommendations failed: {e}")
            return {"error": str(e)}
    
    def apply_platform_optimizations(self) -> Dict[str, Any]:
        """Apply platform-specific optimizations."""
        try:
            recommendations = self.get_optimization_recommendations()
            
            if "error" in recommendations:
                return recommendations
            
            applied_optimizations = {
                "status": "success",
                "platform": self.detected_platform,
                "applied_optimizations": [],
                "optimization_level": self.config.optimization_level
            }
            
            # Apply GPU optimizations
            if "cuda" in recommendations["optimizations"]:
                applied_optimizations["applied_optimizations"].append("CUDA optimizations enabled")
                
                # Set PyTorch optimizations
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    
                    if self.config.optimization_level == "aggressive":
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
            
            # Apply CPU optimizations
            if "intel_cpu" in recommendations["optimizations"]:
                applied_optimizations["applied_optimizations"].append("Intel CPU optimizations enabled")
                
                # Set environment variables for Intel optimizations
                os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
                os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
            
            # Apply edge optimizations
            if "mobile" in recommendations["optimizations"]:
                applied_optimizations["applied_optimizations"].append("Mobile optimizations enabled")
                
                # Set PyTorch mobile optimizations
                if hasattr(torch, 'jit'):
                    torch.jit.enable_onednn_fusion(True)
            
            return applied_optimizations
            
        except Exception as e:
            self.logger.error(f"Platform optimization application failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_platform_summary(self) -> Dict[str, Any]:
        """Get comprehensive platform summary."""
        try:
            if not self.platform_info:
                self.detect_platform()
            
            summary = {
                "timestamp": time.time(),
                "platform_info": self.platform_info,
                "detected_platform": self.detected_platform,
                "optimization_recommendations": self.get_optimization_recommendations(),
                "applied_optimizations": self.apply_platform_optimizations()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Platform summary generation failed: {e}")
            return {"error": str(e)}


# Factory functions
def create_cross_platform_optimization_system(config: Optional[PlatformConfig] = None) -> CrossPlatformOptimizationSystem:
    """Create a cross-platform optimization system."""
    if config is None:
        config = PlatformConfig()
    
    return CrossPlatformOptimizationSystem(config)


def create_platform_config_for_performance() -> PlatformConfig:
    """Create platform configuration optimized for performance."""
    return PlatformConfig(
        auto_detect_platform=True,
        enable_cuda_optimization=True,
        enable_intel_optimization=True,
        optimization_level="aggressive",
        enable_auto_tuning=True
    )


def create_platform_config_for_compatibility() -> PlatformConfig:
    """Create platform configuration optimized for compatibility."""
    return PlatformConfig(
        auto_detect_platform=True,
        enable_cuda_optimization=True,
        enable_intel_optimization=True,
        optimization_level="conservative",
        enable_auto_tuning=False
    )


if __name__ == "__main__":
    # Test the cross-platform optimization system
    config = create_platform_config_for_performance()
    system = create_cross_platform_optimization_system(config)
    
    # Get platform summary
    summary = system.get_platform_summary()
    
    if "error" not in summary:
        print(f"Platform detected: {summary['detected_platform']}")
        print(f"Platform info: {summary['platform_info']}")
        
        # Get optimization recommendations
        recommendations = system.get_optimization_recommendations()
        print(f"Optimization recommendations: {recommendations}")
        
        # Apply optimizations
        applied = system.apply_platform_optimizations()
        print(f"Applied optimizations: {applied}")
    else:
        print(f"Platform detection failed: {summary['error']}")
    
    print("Cross-platform optimization system test completed")
