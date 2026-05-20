import { useEffect, useState } from 'react';

export const useKeyPress = (targetKey: string | string[]): boolean => {
  const [keyPressed, setKeyPressed] = useState(false);
  const targetKeys = Array.isArray(targetKey) ? targetKey : [targetKey];

  useEffect(() => {
    const downHandler = (event: KeyboardEvent): void => {
      if (targetKeys.includes(event.key)) {
        setKeyPressed(true);
      }
    };

    const upHandler = (event: KeyboardEvent): void => {
      if (targetKeys.includes(event.key)) {
        setKeyPressed(false);
      }
    };

    window.addEventListener('keydown', downHandler);
    window.addEventListener('keyup', upHandler);

    return () => {
      window.removeEventListener('keydown', downHandler);
      window.removeEventListener('keyup', upHandler);
    };
  }, [targetKey]);

  return keyPressed;
};

