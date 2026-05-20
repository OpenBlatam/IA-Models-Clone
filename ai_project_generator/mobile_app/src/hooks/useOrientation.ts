import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

type Orientation = 'portrait' | 'landscape';

export const useOrientation = () => {
  const [orientation, setOrientation] = useState<Orientation>('portrait');
  const [dimensions, setDimensions] = useState(Dimensions.get('window'));

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }: { window: ScaledSize }) => {
      setDimensions(window);
      setOrientation(window.height > window.width ? 'portrait' : 'landscape');
    });

    const initialDimensions = Dimensions.get('window');
    setOrientation(initialDimensions.height > initialDimensions.width ? 'portrait' : 'landscape');

    return () => {
      subscription?.remove();
    };
  }, []);

  return {
    orientation,
    isPortrait: orientation === 'portrait',
    isLandscape: orientation === 'landscape',
    dimensions,
    width: dimensions.width,
    height: dimensions.height,
  };
};

