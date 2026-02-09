/**
 * Hook for plugin management
 * @module robot-3d-view/hooks/use-plugins
 */

import { useState, useEffect, useCallback } from 'react';
import { pluginManager, type Plugin } from '../lib/plugin-system';

/**
 * Hook for managing plugins
 * 
 * @returns Plugin state and actions
 * 
 * @example
 * ```tsx
 * const { plugins, registerPlugin, enablePlugin } = usePlugins();
 * ```
 */
export function usePlugins() {
  const [plugins, setPlugins] = useState<Plugin[]>(() =>
    pluginManager.getAllPlugins()
  );

  useEffect(() => {
    const updatePlugins = () => {
      setPlugins(pluginManager.getAllPlugins());
    };

    const interval = setInterval(updatePlugins, 100);
    return () => clearInterval(interval);
  }, []);

  const registerPlugin = useCallback((plugin: Plugin) => {
    pluginManager.register(plugin);
    setPlugins(pluginManager.getAllPlugins());
  }, []);

  const unregisterPlugin = useCallback((id: string) => {
    pluginManager.unregister(id);
    setPlugins(pluginManager.getAllPlugins());
  }, []);

  const enablePlugin = useCallback((id: string) => {
    pluginManager.enablePlugin(id);
    setPlugins(pluginManager.getAllPlugins());
  }, []);

  const disablePlugin = useCallback((id: string) => {
    pluginManager.disablePlugin(id);
    setPlugins(pluginManager.getAllPlugins());
  }, []);

  const getEnabledPlugins = useCallback(() => {
    return pluginManager.getEnabledPlugins();
  }, []);

  return {
    plugins,
    registerPlugin,
    unregisterPlugin,
    enablePlugin,
    disablePlugin,
    getEnabledPlugins,
  };
}



