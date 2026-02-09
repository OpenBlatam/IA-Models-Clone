import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

export function useWindowDimensions(): ScaledSize {
  const [dimensions, setDimensions] = useState(Dimensions.get('window'));

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDimensions(window);
    });

    return () => {
      subscription?.remove();
    };
  }, []);

  return dimensions;
}

