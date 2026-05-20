'use client';

import { useEffect, useState } from 'react';

export function useKeyPress(targetKey: string | string[]): boolean {
  const [keyPressed, setKeyPressed] = useState(false);
  const keys = Array.isArray(targetKey) ? targetKey : [targetKey];

  useEffect(() => {
    const downHandler = (event: KeyboardEvent) => {
      if (keys.includes(event.key)) {
        setKeyPressed(true);
      }
    };

    const upHandler = (event: KeyboardEvent) => {
      if (keys.includes(event.key)) {
        setKeyPressed(false);
      }
    };

    window.addEventListener('keydown', downHandler);
    window.addEventListener('keyup', upHandler);

    return () => {
      window.removeEventListener('keydown', downHandler);
      window.removeEventListener('keyup', upHandler);
    };
  }, [keys.join(',')]);

  return keyPressed;
}

