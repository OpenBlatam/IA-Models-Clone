/**
 * Hook for keyboard shortcuts
 * @module robot-3d-view/hooks/use-shortcuts
 */

import { useEffect, useCallback } from 'react';
import { matchesShortcut, type ShortcutAction } from '../utils/shortcuts';

/**
 * Shortcut handler function type
 */
export type ShortcutHandler = (action: ShortcutAction) => void;

/**
 * Options for useShortcuts hook
 */
interface UseShortcutsOptions {
  /** Enable shortcuts */
  enabled?: boolean;
  /** Handler for shortcut actions */
  onShortcut?: ShortcutHandler;
  /** Ignore shortcuts when typing in inputs */
  ignoreInInputs?: boolean;
}

/**
 * Hook for managing keyboard shortcuts
 * 
 * @param options - Shortcut options
 * @returns Shortcut management functions
 * 
 * @example
 * ```tsx
 * const { handleShortcut } = useShortcuts({
 *   enabled: true,
 *   onShortcut: (action) => {
 *     if (action === 'toggle-stats') toggleStats();
 *   },
 * });
 * ```
 */
export function useShortcuts(options: UseShortcutsOptions = {}) {
  const { enabled = true, onShortcut, ignoreInInputs = true } = options;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled || !onShortcut) return;

      // Ignore if typing in input
      if (
        ignoreInInputs &&
        (event.target instanceof HTMLInputElement ||
          event.target instanceof HTMLTextAreaElement ||
          (event.target as HTMLElement)?.isContentEditable)
      ) {
        return;
      }

      // Import shortcuts dynamically to avoid circular dependencies
      import('../utils/shortcuts').then(({ SHORTCUTS }) => {
        const shortcut = SHORTCUTS.find((s) => matchesShortcut(event, s));
        if (shortcut) {
          event.preventDefault();
          onShortcut(shortcut.action);
        }
      });
    },
    [enabled, onShortcut, ignoreInInputs]
  );

  useEffect(() => {
    if (!enabled) return;

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [enabled, handleKeyDown]);

  return {
    enabled,
  };
}



