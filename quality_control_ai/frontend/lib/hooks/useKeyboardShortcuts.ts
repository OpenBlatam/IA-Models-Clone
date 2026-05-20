import { useEffect } from 'react';

interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  handler: () => void;
}

export const useKeyboardShortcuts = (shortcuts: KeyboardShortcut[]): void => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent): void => {
      shortcuts.forEach((shortcut) => {
        const keyMatch = e.key === shortcut.key || e.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrlKey === undefined || e.ctrlKey === shortcut.ctrlKey;
        const shiftMatch = shortcut.shiftKey === undefined || e.shiftKey === shortcut.shiftKey;
        const altMatch = shortcut.altKey === undefined || e.altKey === shortcut.altKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          e.preventDefault();
          shortcut.handler();
        }
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
};

