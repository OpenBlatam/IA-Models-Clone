/**
 * Keyboard shortcuts system
 * @module robot-3d-view/utils/shortcuts
 */

/**
 * Shortcut action types
 */
export type ShortcutAction =
  | 'toggle-stats'
  | 'toggle-gizmo'
  | 'toggle-grid'
  | 'toggle-objects'
  | 'toggle-auto-rotate'
  | 'toggle-stars'
  | 'toggle-waypoints'
  | 'screenshot'
  | 'camera-front'
  | 'camera-top'
  | 'camera-side'
  | 'camera-iso'
  | 'reset-camera'
  | 'toggle-fullscreen'
  | 'undo'
  | 'redo'
  | 'start-recording'
  | 'stop-recording';

/**
 * Shortcut configuration
 */
export interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  action: ShortcutAction;
  description: string;
}

/**
 * Default keyboard shortcuts
 */
export const SHORTCUTS: ShortcutConfig[] = [
  { key: 's', action: 'toggle-stats', description: 'Toggle statistics' },
  { key: 'g', action: 'toggle-gizmo', description: 'Toggle gizmo' },
  { key: 'h', action: 'toggle-grid', description: 'Toggle grid' },
  { key: 'o', action: 'toggle-objects', description: 'Toggle objects' },
  { key: 'r', action: 'toggle-auto-rotate', description: 'Toggle auto-rotate' },
  { key: 'k', action: 'toggle-stars', description: 'Toggle stars' },
  { key: 'w', action: 'toggle-waypoints', description: 'Toggle waypoints' },
  { key: 'p', action: 'screenshot', description: 'Take screenshot' },
  { key: '1', action: 'camera-front', description: 'Front camera' },
  { key: '2', action: 'camera-top', description: 'Top camera' },
  { key: '3', action: 'camera-side', description: 'Side camera' },
  { key: '4', action: 'camera-iso', description: 'Isometric camera' },
  { key: '0', action: 'reset-camera', description: 'Reset camera' },
  { key: 'f', action: 'toggle-fullscreen', description: 'Toggle fullscreen' },
  { key: 'z', ctrl: true, action: 'undo', description: 'Undo' },
  { key: 'z', ctrl: true, shift: true, action: 'redo', description: 'Redo' },
  { key: 'r', ctrl: true, action: 'start-recording', description: 'Start recording' },
  { key: 's', ctrl: true, action: 'stop-recording', description: 'Stop recording' },
];

/**
 * Checks if a keyboard event matches a shortcut
 * 
 * @param event - Keyboard event
 * @param shortcut - Shortcut configuration
 * @returns true if event matches shortcut
 */
export function matchesShortcut(
  event: KeyboardEvent,
  shortcut: ShortcutConfig
): boolean {
  const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
  const ctrlMatches = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey;
  const shiftMatches = shortcut.shift ? event.shiftKey : !event.shiftKey;
  const altMatches = shortcut.alt ? event.altKey : !event.altKey;
  const metaMatches = shortcut.meta ? event.metaKey : !event.metaKey;

  return (
    keyMatches && ctrlMatches && shiftMatches && altMatches && metaMatches
  );
}

/**
 * Gets shortcut description for display
 * 
 * @param shortcut - Shortcut configuration
 * @returns Human-readable shortcut string
 */
export function getShortcutDisplay(shortcut: ShortcutConfig): string {
  const parts: string[] = [];
  if (shortcut.ctrl) parts.push('Ctrl');
  if (shortcut.shift) parts.push('Shift');
  if (shortcut.alt) parts.push('Alt');
  if (shortcut.meta) parts.push('Meta');
  parts.push(shortcut.key.toUpperCase());
  return parts.join(' + ');
}

/**
 * Finds shortcut by action
 * 
 * @param action - Shortcut action
 * @returns Shortcut configuration or undefined
 */
export function findShortcutByAction(
  action: ShortcutAction
): ShortcutConfig | undefined {
  return SHORTCUTS.find((s) => s.action === action);
}

