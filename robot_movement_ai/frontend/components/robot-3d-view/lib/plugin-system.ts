/**
 * Plugin System for extensibility
 * @module robot-3d-view/lib/plugin-system
 */

import type { SceneConfig } from '../schemas/validation-schemas';

/**
 * Plugin interface
 */
export interface Plugin {
  id: string;
  name: string;
  version: string;
  description?: string;
  enabled: boolean;
  initialize?: () => void | Promise<void>;
  cleanup?: () => void | Promise<void>;
  onConfigChange?: (config: SceneConfig) => void;
  onFrame?: (delta: number) => void;
  render?: () => React.ReactNode;
}

/**
 * Plugin Manager class
 */
export class PluginManager {
  private plugins: Map<string, Plugin> = new Map();
  private initialized = false;

  /**
   * Registers a plugin
   */
  register(plugin: Plugin): boolean {
    if (this.plugins.has(plugin.id)) {
      console.warn(`Plugin ${plugin.id} is already registered`);
      return false;
    }

    this.plugins.set(plugin.id, plugin);
    return true;
  }

  /**
   * Unregisters a plugin
   */
  unregister(id: string): boolean {
    const plugin = this.plugins.get(id);
    if (plugin?.cleanup) {
      plugin.cleanup();
    }
    return this.plugins.delete(id);
  }

  /**
   * Gets a plugin
   */
  getPlugin(id: string): Plugin | undefined {
    return this.plugins.get(id);
  }

  /**
   * Gets all plugins
   */
  getAllPlugins(): Plugin[] {
    return Array.from(this.plugins.values());
  }

  /**
   * Gets enabled plugins
   */
  getEnabledPlugins(): Plugin[] {
    return this.getAllPlugins().filter((p) => p.enabled);
  }

  /**
   * Enables a plugin
   */
  enablePlugin(id: string): boolean {
    const plugin = this.plugins.get(id);
    if (!plugin) return false;

    plugin.enabled = true;
    if (plugin.initialize && this.initialized) {
      plugin.initialize();
    }
    return true;
  }

  /**
   * Disables a plugin
   */
  disablePlugin(id: string): boolean {
    const plugin = this.plugins.get(id);
    if (!plugin) return false;

    plugin.enabled = false;
    if (plugin.cleanup) {
      plugin.cleanup();
    }
    return true;
  }

  /**
   * Initializes all enabled plugins
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    const enabledPlugins = this.getEnabledPlugins();
    await Promise.all(
      enabledPlugins.map((plugin) => {
        if (plugin.initialize) {
          return Promise.resolve(plugin.initialize());
        }
        return Promise.resolve();
      })
    );

    this.initialized = true;
  }

  /**
   * Calls onConfigChange for all enabled plugins
   */
  notifyConfigChange(config: SceneConfig): void {
    this.getEnabledPlugins().forEach((plugin) => {
      if (plugin.onConfigChange) {
        plugin.onConfigChange(config);
      }
    });
  }

  /**
   * Calls onFrame for all enabled plugins
   */
  notifyFrame(delta: number): void {
    this.getEnabledPlugins().forEach((plugin) => {
      if (plugin.onFrame) {
        plugin.onFrame(delta);
      }
    });
  }

  /**
   * Renders all enabled plugins
   */
  renderPlugins(): React.ReactNode[] {
    return this.getEnabledPlugins()
      .map((plugin) => {
        if (plugin.render) {
          return <div key={plugin.id}>{plugin.render()}</div>;
        }
        return null;
      })
      .filter((node) => node !== null) as React.ReactNode[];
  }

  /**
   * Cleans up all plugins
   */
  async cleanup(): Promise<void> {
    const plugins = this.getAllPlugins();
    await Promise.all(
      plugins.map((plugin) => {
        if (plugin.cleanup) {
          return Promise.resolve(plugin.cleanup());
        }
        return Promise.resolve();
      })
    );

    this.plugins.clear();
    this.initialized = false;
  }
}

/**
 * Global plugin manager instance
 */
export const pluginManager = new PluginManager();



