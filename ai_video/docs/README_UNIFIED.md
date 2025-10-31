# AI Video System - Complete Integrated Solution

A comprehensive, production-ready AI video generation system with advanced plugin architecture, robust workflow management, and extensive monitoring capabilities.

## 🚀 Features

### Core System
- **🎬 Video Generation Pipeline**: Complete workflow from URL to video
- **🔌 Plugin Architecture**: Extensible plugin system with dynamic loading
- **📊 Advanced Monitoring**: Real-time metrics, health checks, and performance tracking
- **⚙️ Configuration Management**: Flexible configuration with environment variables
- **🔄 State Management**: Persistent workflow state with recovery capabilities
- **🎯 Event System**: Comprehensive event handling and notifications

### Plugin System
- **🔍 Auto-Discovery**: Automatic plugin discovery and loading
- **✅ Multi-Level Validation**: Basic, standard, strict, and security validation
- **🔄 Lifecycle Management**: Complete plugin lifecycle with state management
- **📈 Performance Monitoring**: Plugin-specific metrics and statistics
- **🔒 Security**: Sandboxing and security validation for plugins
- **🎯 Event Handling**: Plugin events with custom handlers

### Content Processing
- **🌐 Web Extraction**: Multiple extraction methods (newspaper3k, trafilatura, BeautifulSoup)
- **🤖 AI Suggestions**: Intelligent content suggestions for videos
- **🎨 Visual Styles**: AI-powered visual style recommendations
- **🎵 Music Selection**: Automated music and sound effect suggestions
- **🔄 Transitions**: Smart transition recommendations

### Video Generation
- **👤 Avatar Support**: Multiple avatar options and customization
- **🎬 Multiple Formats**: Support for various video formats and resolutions
- **⚡ Performance**: Optimized generation with caching and parallel processing
- **🔧 Customization**: Extensive customization options for video generation

## 📦 Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ai-video-system

# Install dependencies
pip install -r requirements_unified.txt

# Run setup
python plugins/setup.py --all

# Test the system
python main.py --health
```

### Manual Installation

1. **Install Core Dependencies**:
   ```bash
   pip install aiohttp beautifulsoup4 newspaper3k trafilatura pyyaml pydantic
   ```

2. **Install Optional Dependencies**:
   ```bash
   pip install pandas numpy matplotlib opencv-python moviepy
   ```

3. **Setup Configuration**:
   ```bash
   python main.py --create-config ai_video_config.json
   ```

4. **Initialize System**:
   ```bash
   python main.py --init
   ```

## 🎯 Quick Examples

### Basic Video Generation

```python
from ai_video.main import quick_generate

# Generate a video from a URL
video = await quick_generate("https://example.com", avatar="professional_male")
print(f"Generated: {video.title}")
```

### Advanced Usage

```python
from ai_video.main import AIVideoSystem

# Create and initialize system
system = AIVideoSystem("config.json")
await system.initialize()

# Generate video with custom settings
video = await system.generate_video(
    url="https://example.com",
    avatar="professional_female",
    user_edits={
        "duration": 45,
        "style": "modern",
        "music": "upbeat"
    }
)

# Get system statistics
stats = system.get_system_stats()
print(f"System stats: {stats}")

# Shutdown
await system.shutdown()
```

### Batch Processing

```python
from ai_video.main import batch_generate

# Generate multiple videos
urls = [
    "https://example1.com",
    "https://example2.com",
    "https://example3.com"
]

videos = await batch_generate(urls)
print(f"Generated {len(videos)} videos")
```

### Plugin Development

```python
from ai_video.plugins import BasePlugin, PluginMetadata

class MyCustomPlugin(BasePlugin):
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "my_custom_plugin"
        self.version = "1.0.0"
        self.description = "My awesome custom plugin"
        self.author = "Your Name"
        self.category = "processor"
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            category=self.category,
            dependencies={
                "requests": ">=2.25.0",
                "pandas": ">=1.3.0"
            }
        )
    
    async def initialize(self):
        # Initialize your plugin
        pass
    
    async def process_data(self, data):
        # Your plugin logic here
        return {"processed": data, "timestamp": time.time()}
    
    async def cleanup(self):
        # Cleanup resources
        pass
```

## 📋 System Architecture

### Core Components

```
AI Video System
├── Main System (main.py)
├── Integrated Workflow (integrated_workflow.py)
├── Plugin System (plugins/)
│   ├── Plugin Manager (manager.py)
│   ├── Plugin Loader (loader.py)
│   ├── Plugin Validator (validator.py)
│   ├── Plugin Registry (registry.py)
│   └── Plugin Integration (integration.py)
├── Configuration (config.py)
├── Workflow Engine (video_workflow.py)
├── Content Extraction (web_extract.py)
├── AI Suggestions (suggestions.py)
├── Video Generation (video_generator.py)
├── State Management (state_repository.py)
├── Metrics & Monitoring (metrics.py)
└── Models & Types (models.py)
```

### Plugin Architecture

```
Plugin System
├── Base Classes (base.py)
├── Plugin Manager (manager.py)
├── Plugin Loader (loader.py)
├── Plugin Validator (validator.py)
├── Plugin Registry (registry.py)
├── Plugin Integration (integration.py)
├── Configuration (config.py)
├── Setup & Installation (setup.py)
├── Testing (test_system.py)
├── Examples (examples/)
│   └── Web Extractor Plugin (web_extractor_plugin.py)
└── Documentation (README.md)
```

## ⚙️ Configuration

### Configuration Files

**JSON Configuration** (`ai_video_config.json`):
```json
{
  "plugins": {
    "auto_discover": true,
    "auto_load": true,
    "validation_level": "standard",
    "plugin_dirs": ["./plugins", "./ai_video/plugins", "./extensions"]
  },
  "workflow": {
    "max_concurrent_workflows": 5,
    "workflow_timeout": 300,
    "default_duration": 30.0,
    "default_resolution": "1920x1080"
  },
  "ai": {
    "default_model": "gpt-4",
    "max_tokens": 4000,
    "temperature": 0.7
  },
  "storage": {
    "local_storage_path": "./storage",
    "temp_directory": "./temp",
    "output_directory": "./output"
  },
  "security": {
    "enable_auth": false,
    "enable_rate_limiting": true
  },
  "monitoring": {
    "log_level": "INFO",
    "enable_metrics": true
  }
}
```

### Environment Variables

```bash
# Plugin configuration
export AI_VIDEO_PLUGIN_AUTO_DISCOVER=true
export AI_VIDEO_PLUGIN_AUTO_LOAD=true
export AI_VIDEO_PLUGIN_VALIDATION_LEVEL=standard

# Workflow configuration
export AI_VIDEO_MAX_CONCURRENT_WORKFLOWS=5
export AI_VIDEO_WORKFLOW_TIMEOUT=300
export AI_VIDEO_DEFAULT_DURATION=30.0

# AI configuration
export AI_VIDEO_DEFAULT_MODEL=gpt-4
export AI_VIDEO_MAX_TOKENS=4000
export AI_VIDEO_TEMPERATURE=0.7

# Storage configuration
export AI_VIDEO_STORAGE_PATH=./storage
export AI_VIDEO_TEMP_DIR=./temp
export AI_VIDEO_OUTPUT_DIR=./output

# Monitoring configuration
export AI_VIDEO_LOG_LEVEL=INFO
export AI_VIDEO_ENABLE_METRICS=true
```

## 🔧 Development

### Creating Plugins

1. **Create Plugin File**:
   ```python
   # my_plugin.py
   from ai_video.plugins import BasePlugin, PluginMetadata
   
   class MyPlugin(BasePlugin):
       def __init__(self, config=None):
           super().__init__(config)
           self.name = "my_plugin"
           self.version = "1.0.0"
           self.description = "My custom plugin"
           self.author = "Your Name"
           self.category = "processor"
       
       def get_metadata(self) -> PluginMetadata:
           return PluginMetadata(
               name=self.name,
               version=self.version,
               description=self.description,
               author=self.author,
               category=self.category
           )
       
       async def initialize(self):
           # Initialize your plugin
           pass
       
       async def cleanup(self):
           # Cleanup resources
           pass
   ```

2. **Place in Plugin Directory**:
   ```bash
   cp my_plugin.py ./plugins/
   ```

3. **Test Plugin**:
   ```bash
   python plugins/test_system.py --unit
   ```

### Testing

```bash
# Run all tests
python plugins/test_system.py --all

# Run specific test types
python plugins/test_system.py --unit
python plugins/test_system.py --integration
python plugins/test_system.py --performance

# Verbose output
python plugins/test_system.py --all --verbose
```

### CLI Usage

```bash
# Initialize system
python main.py --init

# Generate single video
python main.py --url "https://example.com" --avatar "professional_male"

# Batch generation
python main.py --batch "https://example1.com" "https://example2.com"

# Show system statistics
python main.py --stats

# Show system health
python main.py --health

# Create configuration file
python main.py --create-config my_config.json
```

## 📊 Monitoring

### System Statistics

```python
from ai_video.main import AIVideoSystem

system = AIVideoSystem()
await system.initialize()

# Get comprehensive statistics
stats = system.get_system_stats()
print(f"System stats: {stats}")

# Get health status
health = system.get_health_status()
print(f"Health: {health}")
```

### Plugin Statistics

```python
from ai_video.plugins import quick_start

manager = await quick_start()

# Get plugin statistics
stats = manager.get_stats()
print(f"Plugin stats: {stats}")

# Get health report
health = manager.get_health_report()
print(f"Health: {health}")
```

## 🚨 Error Handling

The system provides comprehensive error handling:

```python
try:
    video = await system.generate_video("https://example.com")
except PluginError as e:
    print(f"Plugin error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except WorkflowError as e:
    print(f"Workflow error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔒 Security

### Security Features

- **Plugin Validation**: All plugins are validated before loading
- **Configuration Validation**: Configuration is validated against schemas
- **Security Checks**: Security validation for potentially dangerous operations
- **Sandboxing**: Plugins run in isolated environments
- **Access Control**: Configurable access permissions
- **Content Filtering**: Optional content filtering and NSFW detection
- **Rate Limiting**: Configurable rate limiting for API endpoints

### Security Configuration

```json
{
  "security": {
    "enable_auth": true,
    "auth_token_expiry": 3600,
    "enable_url_validation": true,
    "allowed_domains": ["example.com", "trusted-site.com"],
    "blocked_domains": ["malicious-site.com"],
    "enable_content_filtering": true,
    "filter_inappropriate_content": true,
    "enable_nsfw_detection": false,
    "enable_rate_limiting": true,
    "max_requests_per_minute": 60,
    "max_requests_per_hour": 1000
  }
}
```

## 📈 Performance

### Optimization Tips

1. **Use Connection Pooling**: Reuse HTTP connections
2. **Enable Caching**: Use built-in caching for repeated operations
3. **Concurrent Processing**: Process multiple workflows concurrently
4. **Lazy Loading**: Load plugins only when needed
5. **Resource Management**: Properly cleanup resources
6. **Monitoring**: Use built-in metrics to identify bottlenecks

### Performance Monitoring

```python
# Get performance metrics
stats = system.get_system_stats()
print(f"Average generation time: {stats['avg_generation_time']:.2f}s")
print(f"Success rate: {stats['success_rate']:.1%}")

# Get plugin-specific metrics
plugin_stats = manager.get_plugin_stats()
print(f"Plugin performance: {plugin_stats}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your plugin or improvements
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd ai-video-system

# Install development dependencies
pip install -r requirements_unified.txt

# Run tests
python plugins/test_system.py --all

# Run linting
flake8 plugins/
black plugins/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: See the examples and API reference
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions
- **Email**: Contact the development team

## 🔄 Changelog

### Version 1.0.0
- Initial release of integrated AI video system
- Complete plugin architecture
- Advanced workflow management
- Comprehensive monitoring and metrics
- Security validation and sandboxing
- Performance optimization
- Extensive documentation and examples

---

**Made with ❤️ by the AI Video Team**

## 🎯 Next Steps

1. **Install the system** using the quick start guide
2. **Explore the examples** in the plugins/examples directory
3. **Create your first plugin** following the development guide
4. **Customize the configuration** for your specific needs
5. **Monitor performance** using the built-in metrics
6. **Contribute** to the project by creating plugins or improvements

The AI Video System is designed to be production-ready, highly extensible, and easy to use. Whether you're generating videos for marketing, education, or entertainment, this system provides all the tools you need to create high-quality AI-generated videos efficiently and reliably. 