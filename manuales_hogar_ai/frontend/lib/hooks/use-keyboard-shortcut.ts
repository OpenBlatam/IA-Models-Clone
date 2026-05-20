import { useEffect } from 'react';

interface UseKeyboardShortcutOptions {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  onPress: () => void;
  enabled?: boolean;
}

export const useKeyboardShortcut = ({
  key,
  ctrl = false,
  shift = false,
  alt = false,
  meta = false,
  onPress,
  enabled = true,
}: UseKeyboardShortcutOptions): void => {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent): void => {
      if (
        event.key === key &&
        event.ctrlKey === ctrl &&
        event.shiftKey === shift &&
        event.altKey === alt &&
        event.metaKey === meta
      ) {
        event.preventDefault();
        onPress();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [key, ctrl, shift, alt, meta, onPress, enabled]);
};

