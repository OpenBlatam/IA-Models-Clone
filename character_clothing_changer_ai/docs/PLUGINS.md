# 🔌 Plugin System Documentation

## Overview

The Character Clothing Changer AI includes a powerful plugin system that allows you to extend and customize the application without modifying core code.

## Plugin Structure

A plugin is a JavaScript object with the following structure:

```javascript
const MyPlugin = {
    name: 'my-plugin',
    version: '1.0.0',
    description: 'Plugin description',
    author: 'Your Name',
    enabled: true,
    
    init() {
        // Initialize plugin
    },
    
    cleanup() {
        // Cleanup when plugin is unregistered
    },
    
    hooks: {
        'hook-name': (data, ...args) => {
            // Hook implementation
            return modifiedData;
        }
    }
};
```

## Registering a Plugin

```javascript
// Register plugin
PluginManager.register('my-plugin', MyPlugin);

// Or auto-register (if plugin exports itself)
// Plugin will be registered when script loads
```

## Available Hooks

### Form Hooks

#### `form:before_submit`
Called before form submission. Can modify form data.

**Parameters:**
- `formData` (FormData) - Form data to be submitted

**Returns:** Modified FormData

**Example:**
```javascript
hooks: {
    'form:before_submit': (formData) => {
        // Add custom field
        formData.append('custom_field', 'value');
        return formData;
    }
}
```

#### `form:after_submit`
Called after successful form submission. Can modify result data.

**Parameters:**
- `result` (Object) - Result data from API

**Returns:** Modified result object

**Example:**
```javascript
hooks: {
    'form:after_submit': (result) => {
        // Add custom processing
        result.customProperty = 'value';
        return result;
    }
}
```

### Result Hooks

#### `result:display`
Called before displaying result. Can modify HTML.

**Parameters:**
- `html` (string) - Result HTML
- `data` (Object) - Result data

**Returns:** Modified HTML string

**Example:**
```javascript
hooks: {
    'result:display': (html, data) => {
        // Add custom element
        return html + '<div class="custom">Custom content</div>';
    }
}
```

### API Hooks

#### `api:before_request`
Called before API request. Can modify request options.

**Parameters:**
- `options` (Object) - Fetch options
- `endpoint` (string) - API endpoint
- `method` (string) - HTTP method

**Returns:** Modified options object

**Example:**
```javascript
hooks: {
    'api:before_request': (options, endpoint, method) => {
        // Add custom header
        options.headers = options.headers || {};
        options.headers['X-Custom-Header'] = 'value';
        return options;
    }
}
```

#### `api:after_request`
Called after API response. Can modify response data.

**Parameters:**
- `data` (Object) - Response data
- `endpoint` (string) - API endpoint
- `method` (string) - HTTP method
- `status` (number) - HTTP status code

**Returns:** Modified data object

**Example:**
```javascript
hooks: {
    'api:after_request': (data, endpoint, method, status) => {
        // Add custom processing
        if (status === 200) {
            data.processed = true;
        }
        return data;
    }
}
```

### UI Hooks

#### `ui:tab_switch`
Called when tab is switched.

**Parameters:**
- `tabName` (string) - Name of the tab

**Example:**
```javascript
hooks: {
    'ui:tab_switch': (tabName) => {
        console.log(`Switched to ${tabName} tab`);
    }
}
```

#### `ui:theme_change`
Called when theme is changed.

**Parameters:**
- `theme` (string) - New theme name

**Example:**
```javascript
hooks: {
    'ui:theme_change': (theme) => {
        console.log(`Theme changed to ${theme}`);
    }
}
```

## Plugin API

### PluginManager Methods

#### `register(name, plugin)`
Register a plugin.

```javascript
PluginManager.register('my-plugin', MyPlugin);
```

#### `unregister(name)`
Unregister a plugin.

```javascript
PluginManager.unregister('my-plugin');
```

#### `get(name)`
Get a plugin instance.

```javascript
const plugin = PluginManager.get('my-plugin');
```

#### `has(name)`
Check if plugin is registered.

```javascript
if (PluginManager.has('my-plugin')) {
    // Plugin is registered
}
```

#### `getAll()`
Get all registered plugins.

```javascript
const plugins = PluginManager.getAll();
```

#### `executeHook(hookName, ...args)`
Execute a hook synchronously.

```javascript
const result = PluginManager.executeHook('form:before_submit', formData);
```

#### `executeHookAsync(hookName, ...args)`
Execute a hook asynchronously.

```javascript
const result = await PluginManager.executeHookAsync('form:before_submit', formData);
```

#### `getPluginInfo(name)`
Get plugin information.

```javascript
const info = PluginManager.getPluginInfo('my-plugin');
// Returns: { name, version, description, author, enabled, hooks }
```

#### `setPluginEnabled(name, enabled)`
Enable or disable a plugin.

```javascript
PluginManager.setPluginEnabled('my-plugin', false);
```

## Example Plugins

### Custom Field Plugin

```javascript
const CustomFieldPlugin = {
    name: 'custom-field',
    version: '1.0.0',
    description: 'Adds custom field to form',
    
    init() {
        // Add custom field to form
        const form = document.getElementById('clothingForm');
        // ... add field
    },
    
    hooks: {
        'form:before_submit': (formData) => {
            const customValue = document.getElementById('customField')?.value;
            if (customValue) {
                formData.append('custom_field', customValue);
            }
            return formData;
        }
    }
};

PluginManager.register('custom-field', CustomFieldPlugin);
```

### Analytics Plugin

```javascript
const AnalyticsPlugin = {
    name: 'analytics',
    version: '1.0.0',
    description: 'Tracks form submissions',
    
    hooks: {
        'form:after_submit': (result) => {
            // Track to analytics service
            if (typeof AdvancedAnalytics !== 'undefined') {
                AdvancedAnalytics.trackEvent('form_submitted', {
                    success: result.changed,
                    characterName: result.character_name
                });
            }
            return result;
        }
    }
};

PluginManager.register('analytics', AnalyticsPlugin);
```

## Best Practices

1. **Always return data from hooks** - Hooks should return the modified data
2. **Handle errors gracefully** - Wrap hook logic in try-catch
3. **Use descriptive names** - Plugin names should be clear and unique
4. **Document hooks** - Document what your plugin does
5. **Test plugins** - Test plugins before registering
6. **Clean up resources** - Implement cleanup method if needed

## Plugin Lifecycle

1. **Registration** - Plugin is registered with `PluginManager.register()`
2. **Initialization** - `init()` method is called
3. **Hook Registration** - Plugin hooks are registered
4. **Active** - Plugin is active and hooks are executed
5. **Cleanup** - `cleanup()` method is called when unregistered

## Events

Plugins can listen to events via EventBus:

```javascript
init() {
    if (typeof EventBus !== 'undefined') {
        EventBus.on('form:submitted', (data) => {
            // Handle event
        });
    }
}
```

## Configuration

Plugins can access configuration via ConfigManager:

```javascript
init() {
    if (typeof ConfigManager !== 'undefined') {
        const apiUrl = ConfigManager.get('api.baseUrl');
    }
}
```

