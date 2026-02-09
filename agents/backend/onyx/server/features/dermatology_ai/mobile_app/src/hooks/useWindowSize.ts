import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

interface WindowSize {
  width: number;
  height: number;
}

export const useWindowSize = (): WindowSize => {
  const [windowSize, setWindowSize] = useState<WindowSize>(() => {
    const { width, height } = Dimensions.get('window');
    return { width, height };
  });

  useEffect(() => {
    const subscription = Dimensions.addEventListener(
      'change',
      ({ window }: { window: ScaledSize }) => {
        setWindowSize({
          width: window.width,
          height: window.height,
        });
      }
    );

    return () => {
      subscription?.remove();
    };
  }, []);

  return windowSize;
};

