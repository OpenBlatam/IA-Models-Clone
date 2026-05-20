/**
 * Widget System for customizable UI components
 * @module robot-3d-view/lib/widget-system
 */

/**
 * Widget configuration
 */
export interface WidgetConfig {
  id: string;
  type: WidgetType;
  title: string;
  position: WidgetPosition;
  size: WidgetSize;
  visible: boolean;
  pinned: boolean;
  order: number;
  props?: Record<string, unknown>;
}

/**
 * Widget types
 */
export type WidgetType =
  | 'stats'
  | 'controls'
  | 'info'
  | 'history'
  | 'recording'
  | 'presets'
  | 'themes'
  | 'config'
  | 'help'
  | 'custom';

/**
 * Widget position
 */
export interface WidgetPosition {
  x: number;
  y: number;
  anchor: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
}

/**
 * Widget size
 */
export interface WidgetSize {
  width: number;
  height: number;
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  resizable?: boolean;
}

/**
 * Widget Manager class
 */
export class WidgetManager {
  private widgets: Map<string, WidgetConfig> = new Map();
  private defaultWidgets: WidgetConfig[] = [];

  /**
   * Registers a default widget
   */
  registerDefault(widget: WidgetConfig): void {
    this.defaultWidgets.push(widget);
    this.widgets.set(widget.id, widget);
  }

  /**
   * Adds a widget
   */
  addWidget(widget: WidgetConfig): void {
    this.widgets.set(widget.id, widget);
  }

  /**
   * Removes a widget
   */
  removeWidget(id: string): boolean {
    return this.widgets.delete(id);
  }

  /**
   * Gets a widget
   */
  getWidget(id: string): WidgetConfig | undefined {
    return this.widgets.get(id);
  }

  /**
   * Gets all widgets
   */
  getAllWidgets(): WidgetConfig[] {
    return Array.from(this.widgets.values()).sort((a, b) => a.order - b.order);
  }

  /**
   * Gets visible widgets
   */
  getVisibleWidgets(): WidgetConfig[] {
    return this.getAllWidgets().filter((w) => w.visible);
  }

  /**
   * Updates a widget
   */
  updateWidget(id: string, updates: Partial<WidgetConfig>): boolean {
    const widget = this.widgets.get(id);
    if (!widget) return false;

    this.widgets.set(id, { ...widget, ...updates });
    return true;
  }

  /**
   * Toggles widget visibility
   */
  toggleWidget(id: string): boolean {
    const widget = this.widgets.get(id);
    if (!widget) return false;

    widget.visible = !widget.visible;
    return true;
  }

  /**
   * Resets to default widgets
   */
  resetToDefaults(): void {
    this.widgets.clear();
    this.defaultWidgets.forEach((widget) => {
      this.widgets.set(widget.id, { ...widget });
    });
  }

  /**
   * Exports widget configuration
   */
  export(): WidgetConfig[] {
    return this.getAllWidgets();
  }

  /**
   * Imports widget configuration
   */
  import(configs: WidgetConfig[]): void {
    this.widgets.clear();
    configs.forEach((config) => {
      this.widgets.set(config.id, config);
    });
  }
}

/**
 * Default widget configurations
 */
export const DEFAULT_WIDGETS: WidgetConfig[] = [
  {
    id: 'stats',
    type: 'stats',
    title: 'Statistics',
    position: { x: 10, y: 10, anchor: 'top-left' },
    size: { width: 200, height: 150, resizable: true },
    visible: false,
    pinned: false,
    order: 1,
  },
  {
    id: 'controls',
    type: 'controls',
    title: 'Controls',
    position: { x: 10, y: 10, anchor: 'top-right' },
    size: { width: 250, height: 300, resizable: true },
    visible: true,
    pinned: true,
    order: 2,
  },
  {
    id: 'presets',
    type: 'presets',
    title: 'Presets',
    position: { x: 10, y: 10, anchor: 'top-right' },
    size: { width: 200, height: 150, resizable: false },
    visible: true,
    pinned: false,
    order: 3,
  },
  {
    id: 'themes',
    type: 'themes',
    title: 'Themes',
    position: { x: 10, y: 170, anchor: 'top-right' },
    size: { width: 200, height: 150, resizable: false },
    visible: true,
    pinned: false,
    order: 4,
  },
];

/**
 * Global widget manager instance
 */
export const widgetManager = new WidgetManager();

// Initialize with defaults
DEFAULT_WIDGETS.forEach((widget) => {
  widgetManager.registerDefault(widget);
});



