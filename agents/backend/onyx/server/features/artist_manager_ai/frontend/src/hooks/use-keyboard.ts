import { useEffect } from 'react';

type KeyHandler = (event: KeyboardEvent) => void;

interface UseKeyboardOptions {
  target?: 'window' | 'document';
  preventDefault?: boolean;
}

export const useKeyboard = (
  key: string | string[],
  handler: KeyHandler,
  options: UseKeyboardOptions = {}
) => {
  const { target = 'window', preventDefault = false } = options;

  useEffect(() => {
    const keys = Array.isArray(key) ? key : [key];
    const targetElement = target === 'window' ? window : document;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (keys.includes(event.key) || keys.includes(event.code)) {
        if (preventDefault) {
          event.preventDefault();
        }
        handler(event);
      }
    };

    targetElement.addEventListener('keydown', handleKeyDown);

    return () => {
      targetElement.removeEventListener('keydown', handleKeyDown);
    };
  }, [key, handler, target, preventDefault]);
};

export const useEscape = (handler: () => void) => {
  useKeyboard('Escape', handler, { preventDefault: true });
};

export const useEnter = (handler: () => void) => {
  useKeyboard('Enter', handler, { preventDefault: true });
};

