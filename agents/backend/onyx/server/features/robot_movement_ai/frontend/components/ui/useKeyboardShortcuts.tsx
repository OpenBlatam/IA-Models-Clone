'use client';

import { useHotkeys } from 'react-hotkeys-hook';

interface ShortcutConfig {
  keys: string;
  callback: () => void;
  description?: string;
  enabled?: boolean;
  preventDefault?: boolean;
}

export function useKeyboardShortcuts(shortcuts: ShortcutConfig[]) {
  shortcuts.forEach(({ keys, callback, enabled = true, preventDefault = true }) => {
    useHotkeys(
      keys,
      (event) => {
        if (preventDefault) {
          event.preventDefault();
        }
        callback();
      },
      { enabled }
    );
  });
}



