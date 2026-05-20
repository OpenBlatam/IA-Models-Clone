# Plugin System - Imagen Video Enhancer AI

## Overview

The plugin system allows you to extend the functionality of the enhancer with custom enhancement types and processing pipelines.

## Creating a Plugin

### Basic Plugin Structure

```python
from imagen_video_enhancer_ai.core.plugin_system import EnhancementPlugin

class MyCustomPlugin(EnhancementPlugin):
    def get_name(self) -> str:
        return "my_custom_plugin"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def process(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Your custom processing logic
        return {
            "success": True,
            "result": "Custom enhancement result"
        }
    
    def validate(self, parameters: Dict[str, Any]) -> bool:
        # Validate parameters
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "custom_param": {"type": "string"}
            }
        }
```

## Registering a Plugin

```python
from imagen_video_enhancer_ai.core.plugin_system import PluginManager

plugin_manager = PluginManager()
plugin = MyCustomPlugin()
plugin_manager.register(plugin)
plugin_manager.enable("my_custom_plugin")
```

## Loading Plugins from Directory

```python
# Load all plugins from a directory
plugin_manager.load_from_directory("/path/to/plugins")
```

## Using Plugins

```python
# Execute a plugin
result = await plugin_manager.execute_plugin(
    plugin_name="my_custom_plugin",
    file_path="/path/to/file.jpg",
    parameters={"custom_param": "value"}
)
```

## Plugin API

### Methods

- `get_name()`: Return plugin name
- `get_version()`: Return plugin version
- `process(file_path, parameters)`: Process enhancement
- `validate(parameters)`: Validate parameters
- `get_config_schema()`: Return configuration schema

### Plugin Manager Methods

- `register(plugin)`: Register a plugin
- `unregister(plugin_name)`: Remove a plugin
- `enable(plugin_name)`: Enable a plugin
- `disable(plugin_name)`: Disable a plugin
- `get_plugin(plugin_name)`: Get plugin instance
- `list_plugins()`: List all plugins
- `execute_plugin(plugin_name, file_path, parameters)`: Execute plugin
- `load_from_directory(directory)`: Load plugins from directory

## Example Plugin

```python
class WatermarkPlugin(EnhancementPlugin):
    def get_name(self) -> str:
        return "watermark"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def process(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        from PIL import Image, ImageDraw, ImageFont
        
        watermark_text = parameters.get("text", "Watermark")
        position = parameters.get("position", "bottom-right")
        
        # Load image
        img = Image.open(file_path)
        
        # Add watermark
        draw = ImageDraw.Draw(img)
        # ... watermark logic ...
        
        # Save result
        output_path = file_path.replace(".jpg", "_watermarked.jpg")
        img.save(output_path)
        
        return {
            "success": True,
            "output_path": output_path
        }
    
    def validate(self, parameters: Dict[str, Any]) -> bool:
        return "text" in parameters
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "position": {
                    "type": "string",
                    "enum": ["top-left", "top-right", "bottom-left", "bottom-right"]
                }
            }
        }
```




