import { useEffect } from 'react';

interface UseKeyboardShortcutOptions {
  key: string;
  handler: () => void;
  enabled?: boolean;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
}

export const useKeyboardShortcut = ({
  key,
  handler,
  enabled = true,
  ctrlKey = false,
  shiftKey = false,
  altKey = false,
}: UseKeyboardShortcutOptions): void => {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent): void => {
      if (
        e.key === key &&
        e.ctrlKey === ctrlKey &&
        e.shiftKey === shiftKey &&
        e.altKey === altKey
      ) {
        e.preventDefault();
        handler();
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [key, handler, enabled, ctrlKey, shiftKey, altKey]);
};

