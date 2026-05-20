import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

type Orientation = 'portrait' | 'landscape';

interface OrientationState {
  orientation: Orientation;
  isPortrait: boolean;
  isLandscape: boolean;
  dimensions: ScaledSize;
}

/**
 * Hook to monitor device orientation
 * Provides orientation state and dimensions
 */
export function useOrientation(): OrientationState {
  const [dimensions, setDimensions] = useState<ScaledSize>(
    Dimensions.get('window')
  );

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDimensions(window);
    });

    return () => subscription?.remove();
  }, []);

  const isPortrait = dimensions.height >= dimensions.width;
  const isLandscape = dimensions.width > dimensions.height;
  const orientation: Orientation = isPortrait ? 'portrait' : 'landscape';

  return {
    orientation,
    isPortrait,
    isLandscape,
    dimensions,
  };
}

