import { useEffect } from 'react';

type HotkeyCallback = (event: KeyboardEvent) => void;

interface HotkeyConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  callback: HotkeyCallback;
}

export const useHotkeys = (hotkeys: HotkeyConfig[]): void => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      hotkeys.forEach(({ key, ctrl, shift, alt, meta, callback }) => {
        const keyMatch = event.key.toLowerCase() === key.toLowerCase();
        const ctrlMatch = ctrl === undefined || event.ctrlKey === ctrl;
        const shiftMatch = shift === undefined || event.shiftKey === shift;
        const altMatch = alt === undefined || event.altKey === alt;
        const metaMatch = meta === undefined || event.metaKey === meta;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch) {
          event.preventDefault();
          callback(event);
        }
      });
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [hotkeys]);
};

