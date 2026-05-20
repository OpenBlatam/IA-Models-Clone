import { useEffect, useCallback, useRef } from 'react';

export type KeyboardShortcut = {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  action: () => void;
  description?: string;
};

type ShortcutHandler = {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  handler: () => void;
};

export function useKeyboardShortcuts() {
  const shortcutsRef = useRef<ShortcutHandler[]>([]);

  const registerShortcut = useCallback((keyCombo: string, handler: () => void) => {
    // Parse key combination (e.g., "ctrl+k", "escape", "ctrl+/")
    const parts = keyCombo.toLowerCase().split('+').map(s => s.trim());
    const key = parts[parts.length - 1];
    const ctrl = parts.includes('ctrl') || parts.includes('cmd');
    const shift = parts.includes('shift');
    const alt = parts.includes('alt');

    shortcutsRef.current.push({
      key,
      ctrl,
      shift,
      alt,
      handler,
    });

    // Return cleanup function
    return () => {
      shortcutsRef.current = shortcutsRef.current.filter(
        (s) => s.key !== key || s.ctrl !== ctrl || s.shift !== shift || s.alt !== alt || s.handler !== handler
      );
    };
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      shortcutsRef.current.forEach((shortcut) => {
        // Normalize key names for matching
        const eventKey = event.key.toLowerCase();
        const shortcutKey = shortcut.key.toLowerCase();
        
        // Handle special keys
        let keyMatch = false;
        if (shortcutKey === 'escape') {
          keyMatch = eventKey === 'escape' || event.key === 'Escape';
        } else if (shortcutKey === '/') {
          keyMatch = eventKey === '/' || event.key === '/';
        } else {
          keyMatch = eventKey === shortcutKey;
        }
        
        const ctrlMatch = shortcut.ctrl ? (event.ctrlKey || event.metaKey) : (!event.ctrlKey && !event.metaKey);
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          event.preventDefault();
          shortcut.handler();
        }
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return { registerShortcut };
}

// Legacy function for backward compatibility
export function useKeyboardShortcutsLegacy(shortcuts: KeyboardShortcut[]) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      shortcuts.forEach((shortcut) => {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey || event.metaKey : !event.ctrlKey && !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          event.preventDefault();
          shortcut.action();
        }
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
}

export const defaultShortcuts: KeyboardShortcut[] = [
  {
    key: 'h',
    ctrl: true,
    action: () => {
      // Home action - will be set by component
    },
    description: 'Ir a posición home',
  },
  {
    key: 's',
    ctrl: true,
    action: () => {
      // Stop action - will be set by component
    },
    description: 'Detener robot',
  },
  {
    key: 'r',
    ctrl: true,
    action: () => {
      // Record action - will be set by component
    },
    description: 'Iniciar/Detener grabación',
  },
];

