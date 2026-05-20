/**
 * Custom hook for keyboard shortcuts.
 * Provides keyboard navigation and shortcuts functionality.
 */

import { useEffect, useCallback } from 'react';

/**
 * Keyboard shortcut configuration.
 */
export interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
  handler: () => void;
  description?: string;
}

/**
 * Options for keyboard shortcuts hook.
 */
export interface UseKeyboardShortcutsOptions {
  shortcuts: KeyboardShortcut[];
  enabled?: boolean;
  preventDefault?: boolean;
}

/**
 * Custom hook for handling keyboard shortcuts.
 * @param options - Shortcut configuration
 */
export function useKeyboardShortcuts({
  shortcuts,
  enabled = true,
  preventDefault = true,
}: UseKeyboardShortcutsOptions): void {
  /**
   * Handles keydown events.
   */
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      const matchingShortcut = shortcuts.find((shortcut) => {
        const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatches = shortcut.ctrlKey === undefined || event.ctrlKey === shortcut.ctrlKey;
        const shiftMatches = shortcut.shiftKey === undefined || event.shiftKey === shortcut.shiftKey;
        const altMatches = shortcut.altKey === undefined || event.altKey === shortcut.altKey;
        const metaMatches = shortcut.metaKey === undefined || event.metaKey === shortcut.metaKey;

        return (
          keyMatches &&
          ctrlMatches &&
          shiftMatches &&
          altMatches &&
          metaMatches
        );
      });

      if (matchingShortcut) {
        if (preventDefault) {
          event.preventDefault();
        }
        matchingShortcut.handler();
      }
    },
    [shortcuts, enabled, preventDefault]
  );

  useEffect(() => {
    if (!enabled) return;

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown, enabled]);
}

/**
 * Common keyboard shortcuts for music player.
 */
export const MUSIC_SHORTCUTS = {
  PLAY_PAUSE: ' ',
  NEXT: 'ArrowRight',
  PREVIOUS: 'ArrowLeft',
  VOLUME_UP: 'ArrowUp',
  VOLUME_DOWN: 'ArrowDown',
  MUTE: 'm',
  SEARCH: 'k',
  ANALYZE: 'a',
  COMPARE: 'c',
} as const;

