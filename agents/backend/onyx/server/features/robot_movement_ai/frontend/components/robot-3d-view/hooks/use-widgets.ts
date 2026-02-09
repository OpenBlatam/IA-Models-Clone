/**
 * Hook for widget management
 * @module robot-3d-view/hooks/use-widgets
 */

import { useState, useEffect, useCallback } from 'react';
import { widgetManager, type WidgetConfig } from '../lib/widget-system';

/**
 * Hook for managing widgets
 * 
 * @returns Widget state and actions
 * 
 * @example
 * ```tsx
 * const { widgets, addWidget, removeWidget, updateWidget } = useWidgets();
 * ```
 */
export function useWidgets() {
  const [widgets, setWidgets] = useState<WidgetConfig[]>(() =>
    widgetManager.getAllWidgets()
  );

  useEffect(() => {
    const updateWidgets = () => {
      setWidgets(widgetManager.getAllWidgets());
    };

    // Update periodically
    const interval = setInterval(updateWidgets, 100);
    return () => clearInterval(interval);
  }, []);

  const addWidget = useCallback((widget: WidgetConfig) => {
    widgetManager.addWidget(widget);
    setWidgets(widgetManager.getAllWidgets());
  }, []);

  const removeWidget = useCallback((id: string) => {
    widgetManager.removeWidget(id);
    setWidgets(widgetManager.getAllWidgets());
  }, []);

  const updateWidget = useCallback((id: string, updates: Partial<WidgetConfig>) => {
    widgetManager.updateWidget(id, updates);
    setWidgets(widgetManager.getAllWidgets());
  }, []);

  const toggleWidget = useCallback((id: string) => {
    widgetManager.toggleWidget(id);
    setWidgets(widgetManager.getAllWidgets());
  }, []);

  const resetToDefaults = useCallback(() => {
    widgetManager.resetToDefaults();
    setWidgets(widgetManager.getAllWidgets());
  }, []);

  const getVisibleWidgets = useCallback(() => {
    return widgetManager.getVisibleWidgets();
  }, []);

  return {
    widgets,
    addWidget,
    removeWidget,
    updateWidget,
    toggleWidget,
    resetToDefaults,
    getVisibleWidgets,
  };
}



