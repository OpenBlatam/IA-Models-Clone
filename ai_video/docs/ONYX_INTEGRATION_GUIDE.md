# Onyx AI Video System - Integration Guide

## Overview

The AI Video System has been completely adapted to leverage Onyx's existing infrastructure, utilities, and capabilities. This integration provides seamless operation within the Onyx ecosystem while maintaining all the advanced features of the AI Video system.

## 🚀 **Onyx Integration Components**

### 1. **Onyx Integration Manager** (`core/onyx_integration.py`)

**Core Integration Layer**
- **Onyx LLM Manager**: Uses Onyx's LLM factory and provider system
- **Onyx Task Manager**: Leverages Onyx's threadpool concurrency utilities
- **Onyx Security Manager**: Integrates with Onyx's security and validation functions
- **Onyx Performance Manager**: Uses Onyx's timing and GPU utilities
- **Onyx File Manager**: Integrates with Onyx's file processing utilities

**Key Features:**
```python
# Get Onyx LLM instance
llm = await onyx_integration.llm_manager.get_default_llm()

# Use Onyx threading for parallel execution
results = await onyx_integration.task_manager.run_parallel(tasks)

# Validate with Onyx security
is_valid = await onyx_integration.security_manager.validate_access(user_id, resource_id)

# Use Onyx performance monitoring
with onyx_integration.performance_manager.time_operation("video_generation"):
    result = await generate_video()
```

### 2. **Onyx Video Workflow** (`onyx_video_workflow.py`)

**Adapted Workflow Engine**
- **Onyx LLM Integration**: Uses Onyx's LLM factory for text generation
- **Vision Capabilities**: Leverages Onyx's vision LLM support
- **Performance Optimization**: Uses Onyx's timing and GPU utilities
- **Error Handling**: Integrates with Onyx's error handling patterns

**Workflow Steps:**
```python
# Content analysis using Onyx LLM
content_analysis = await llm_manager.generate_text(prompt, llm)

# Vision processing with Onyx vision LLM
vision_result = await llm_manager.generate_with_vision(prompt, image_data)

# Performance monitoring with Onyx utilities
with time_function("video_generation"):
    result = await generate_video()
```

### 3. **Onyx Plugin Manager** (`onyx_plugin_manager.py`)

**Enhanced Plugin System**
- **Onyx Threading**: Uses Onyx's threadpool concurrency for plugin execution
- **Onyx File Processing**: Integrates with Onyx's file utilities
- **Onyx Performance**: Uses Onyx's timing and GPU utilities
- **Onyx Logging**: Integrates with Onyx's logging system

**Plugin Development:**
```python
class OnyxContentAnalyzerPlugin(OnyxPluginBase):
    async def _initialize_plugin(self):
        # Use Onyx LLM
        self.llm = await onyx_integration.llm_manager.get_default_llm()
    
    async def process(self, context):
        # Use Onyx text processing
        analysis = await onyx_integration.llm_manager.generate_text(prompt, self.llm)
        keywords = extract_keywords(context.request.input_text)
        return {"analysis": analysis, "keywords": keywords}
```

### 4. **Onyx Main System** (`onyx_main.py`)

**Unified Entry Point**
- **Onyx Integration**: Coordinates all Onyx components
- **Onyx Logging**: Uses Onyx's structured logging
- **Onyx Telemetry**: Integrates with Onyx's telemetry system
- **Onyx Error Handling**: Uses Onyx's error handling patterns

## 🔧 **Onyx Utilities Integration**

### **LLM Integration**
```python
# Use Onyx's LLM factory
from onyx.llm.factory import get_default_llms, get_default_llm_with_vision

# Get default LLM
llm = await onyx_integration.llm_manager.get_default_llm()

# Get vision LLM
vision_llm = await onyx_integration.llm_manager.get_vision_llm()

# Generate text
response = await llm_manager.generate_text(prompt, llm)

# Generate with vision
vision_response = await llm_manager.generate_with_vision(prompt, image_data)
```

### **Threading Integration**
```python
# Use Onyx's threadpool concurrency
from onyx.utils.threadpool_concurrency import run_functions_in_parallel, FunctionCall

# Run tasks in parallel
function_calls = [FunctionCall(task) for task in tasks]
results = run_functions_in_parallel(function_calls, max_workers=10)

# Use Onyx's timeout utilities
from onyx.utils.threadpool_concurrency import run_with_timeout
result = run_with_timeout(30.0, long_running_function, *args)
```

### **Logging Integration**
```python
# Use Onyx's logging system
from onyx.utils.logger import setup_logger

logger = setup_logger("ai_video")

# Structured logging with Onyx context
logger.info("Processing video request", extra={
    "request_id": request_id,
    "user_id": user_id,
    "operation": "video_generation"
})
```

### **Performance Integration**
```python
# Use Onyx's timing utilities
from onyx.utils.timing import time_function

with time_function("video_generation"):
    result = await generate_video()

# Use Onyx's GPU utilities
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available

if is_gpu_available():
    gpu_info = get_gpu_info()
    # Use GPU acceleration
```

### **Security Integration**
```python
# Use Onyx's security utilities
from onyx.core.functions import validate_user_access

# Validate user access
has_access = await validate_user_access(user_id, resource_id)

# Use Onyx's text processing
from onyx.utils.text_processing import clean_text, extract_keywords

cleaned_text = clean_text(input_text)
keywords = extract_keywords(input_text)
```

### **File Processing Integration**
```python
# Use Onyx's file utilities
from onyx.utils.file import get_file_extension, get_file_size

file_info = {
    "extension": get_file_extension(file_path),
    "size": get_file_size(file_path)
}
```

## 📊 **Onyx Monitoring Integration**

### **Telemetry Integration**
```python
# Use Onyx's telemetry
from onyx.utils.telemetry import TelemetryLogger

telemetry = TelemetryLogger()

# Log events
telemetry.log_info("video_generated", {
    "request_id": request_id,
    "duration": duration,
    "gpu_used": gpu_available
})

telemetry.log_error("video_generation_failed", {
    "request_id": request_id,
    "error": str(error)
})
```

### **Performance Monitoring**
```python
# Use Onyx's performance utilities
with time_function("video_generation"):
    result = await generate_video()

# Record metrics
performance_metrics = {
    "operation": "video_generation",
    "duration": duration,
    "success": True
}
```

## 🔄 **Onyx Workflow Integration**

### **Video Generation Workflow**
```python
# 1. Content Analysis (Onyx LLM)
content_analysis = await llm_manager.generate_text(
    f"Analyze this content: {input_text}", llm
)

# 2. Script Generation (Onyx LLM)
script = await llm_manager.generate_text(
    f"Generate script from: {content_analysis}", llm
)

# 3. Storyboard Creation (Onyx LLM)
storyboard = await llm_manager.generate_text(
    f"Create storyboard from: {script}", llm
)

# 4. Visual Style Definition (Onyx LLM)
visual_style = await llm_manager.generate_text(
    f"Define visual style for: {storyboard}", llm
)

# 5. Video Generation (Onyx GPU + Processing)
with time_function("final_video_generation"):
    video = await generate_final_video(storyboard, visual_style)
```

### **Plugin Execution Workflow**
```python
# Execute plugins using Onyx threading
plugin_tasks = [
    FunctionCall(plugin.process, context)
    for plugin in enabled_plugins
]

plugin_results = run_functions_in_parallel(
    plugin_tasks,
    max_workers=onyx_config.max_workers
)
```

## 🛡️ **Onyx Security Integration**

### **Input Validation**
```python
# Use Onyx's text processing for validation
cleaned_text = clean_text(input_text)
keywords = extract_keywords(input_text)

# Validate with Onyx security
is_valid = await validate_user_access(user_id, resource_id)
```

### **Access Control**
```python
# Use Onyx's access control
has_access = await onyx_integration.security_manager.validate_access(
    user_id, resource_id
)

if not has_access:
    raise ValidationError("Access denied")
```

### **Data Protection**
```python
# Use Onyx's encryption
from onyx.utils.encryption import encrypt_data, decrypt_data

encrypted_data = encrypt_data(sensitive_data)
decrypted_data = decrypt_data(encrypted_data)
```

## ⚡ **Onyx Performance Integration**

### **GPU Acceleration**
```python
# Check GPU availability
if is_gpu_available():
    gpu_info = get_gpu_info()
    # Use GPU for video processing
    result = await process_with_gpu(data)
else:
    # Fallback to CPU
    result = await process_with_cpu(data)
```

### **Parallel Processing**
```python
# Use Onyx's parallel processing
tasks = [
    FunctionCall(process_video_segment, segment)
    for segment in video_segments
]

results = run_functions_in_parallel(tasks, max_workers=4)
```

### **Caching**
```python
# Use Onyx's thread-safe caching
from onyx.utils.threadpool_concurrency import ThreadSafeDict

cache = ThreadSafeDict()

# Cache results
cache[f"video_{request_id}"] = {
    "result": video_result,
    "timestamp": time.time()
}
```

## 📈 **Onyx Metrics Integration**

### **System Metrics**
```python
# Get Onyx system metrics
onyx_metrics = await onyx_integration.get_metrics()

# Add AI Video metrics
ai_video_metrics = {
    "request_count": self.request_count,
    "error_count": self.error_count,
    "error_rate": self.error_count / max(self.request_count, 1),
    "uptime": time.time() - self.start_time
}

# Combine metrics
combined_metrics = {**onyx_metrics, "ai_video": ai_video_metrics}
```

### **Performance Tracking**
```python
# Track performance with Onyx utilities
with time_function("video_generation"):
    result = await generate_video()

# Record metrics
self._record_performance_metrics("video_generation", duration, success)
```

## 🔧 **Configuration Integration**

### **Onyx Configuration**
```python
# Use Onyx's configuration system
from onyx.core.config import get_config

config = get_config()

# AI Video specific configuration
ai_video_config = {
    "use_onyx_logging": True,
    "use_onyx_llm": True,
    "use_onyx_telemetry": True,
    "use_onyx_encryption": True,
    "use_onyx_threading": True,
    "max_workers": 10,
    "timeout_seconds": 300
}
```

### **Environment Variables**
```bash
# Onyx environment variables
export ONYX_LOG_LEVEL=INFO
export ONYX_ENVIRONMENT=production
export ONYX_GPU_ENABLED=true

# AI Video specific variables
export AI_VIDEO_USE_ONYX=true
export AI_VIDEO_MAX_WORKERS=10
export AI_VIDEO_TIMEOUT=300
```

## 🚀 **Usage Examples**

### **Basic Video Generation**
```python
from onyx_ai_video import get_system

# Get Onyx AI Video system
system = await get_system()

# Generate video
request = VideoRequest(
    input_text="Create a video about AI",
    user_id="user123"
)

response = await system.generate_video(request)
print(f"Video generated: {response.output_url}")
```

### **Video Generation with Plugins**
```python
# Generate video with plugins
request = VideoRequest(
    input_text="Create a video about AI",
    user_id="user123",
    plugins=["content_analyzer", "visual_enhancer"]
)

response = await system.generate_video(request)
print(f"Plugin results: {response.metadata['plugin_results']}")
```

### **Vision-Based Video Generation**
```python
# Generate video with vision
with open("image.jpg", "rb") as f:
    image_data = f.read()

response = await system.generate_video_with_vision(request, image_data)
print(f"Vision video generated: {response.output_url}")
```

### **System Status and Metrics**
```python
# Get system status
status = await system.get_system_status()
print(f"System status: {status['status']}")

# Get metrics
metrics = await system.get_metrics()
print(f"Request count: {metrics['ai_video']['request_count']}")
```

## 🔄 **Migration from Standalone to Onyx**

### **Step 1: Update Imports**
```python
# Old imports
from ai_video.core import get_system
from ai_video.utils import log_message

# New Onyx imports
from onyx_ai_video import get_system
from onyx.utils.logger import setup_logger
```

### **Step 2: Update Configuration**
```python
# Old configuration
config = {
    "logging": {"level": "INFO"},
    "llm": {"provider": "openai"}
}

# New Onyx configuration
config = {
    "use_onyx_logging": True,
    "use_onyx_llm": True,
    "use_onyx_telemetry": True
}
```

### **Step 3: Update Logging**
```python
# Old logging
log_message("Processing video")

# New Onyx logging
logger = setup_logger("ai_video")
logger.info("Processing video", extra={"request_id": request_id})
```

### **Step 4: Update LLM Usage**
```python
# Old LLM usage
llm = get_llm(provider="openai")

# New Onyx LLM usage
llm = await onyx_integration.llm_manager.get_default_llm()
```

## 📊 **Benefits of Onyx Integration**

### **Performance Benefits**
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Parallel Processing**: Efficient threading with Onyx's utilities
- **Caching**: Thread-safe caching for improved performance
- **Optimization**: Onyx's performance monitoring and optimization

### **Reliability Benefits**
- **Error Handling**: Robust error handling with Onyx patterns
- **Retry Logic**: Automatic retry with exponential backoff
- **Timeout Management**: Proper timeout handling for long operations
- **Resource Management**: Efficient resource cleanup and management

### **Security Benefits**
- **Access Control**: Integration with Onyx's access control system
- **Input Validation**: Robust input validation and sanitization
- **Data Protection**: Encryption and security utilities
- **Audit Trail**: Comprehensive logging and audit capabilities

### **Monitoring Benefits**
- **Telemetry**: Integration with Onyx's telemetry system
- **Metrics**: Comprehensive performance metrics
- **Logging**: Structured logging with context
- **Health Checks**: System health monitoring

### **Scalability Benefits**
- **Horizontal Scaling**: Support for multiple instances
- **Load Balancing**: Integration with Onyx's load balancing
- **Resource Management**: Efficient resource utilization
- **Performance Optimization**: Automatic performance tuning

## 🎯 **Future Enhancements**

### **Planned Onyx Integrations**
- **Database Integration**: Full integration with Onyx's database system
- **Authentication**: Integration with Onyx's authentication system
- **API Gateway**: Integration with Onyx's API gateway
- **Microservices**: Full microservices architecture support

### **Advanced Features**
- **Real-time Processing**: Real-time video processing capabilities
- **Distributed Processing**: Distributed video processing across nodes
- **Advanced Analytics**: Advanced analytics and reporting
- **Machine Learning**: Integration with Onyx's ML capabilities

This Onyx integration provides a comprehensive, production-ready AI Video system that seamlessly operates within the Onyx ecosystem while maintaining all advanced features and capabilities. 