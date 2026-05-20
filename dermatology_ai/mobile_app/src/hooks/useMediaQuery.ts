import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

interface Breakpoints {
  sm?: number;
  md?: number;
  lg?: number;
  xl?: number;
}

const defaultBreakpoints: Breakpoints = {
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
};

export const useMediaQuery = (
  query: string,
  breakpoints: Breakpoints = defaultBreakpoints
): boolean => {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const updateMatches = () => {
      const { width } = Dimensions.get('window');
      const bp = { ...defaultBreakpoints, ...breakpoints };

      // Parse query like "(min-width: 768px)"
      const match = query.match(/\((\w+)-width:\s*(\d+)px\)/);
      if (!match) {
        setMatches(false);
        return;
      }

      const [, direction, value] = match;
      const breakpointValue = parseInt(value, 10);

      if (direction === 'min') {
        setMatches(width >= breakpointValue);
      } else if (direction === 'max') {
        setMatches(width <= breakpointValue);
      }
    };

    updateMatches();

    const subscription = Dimensions.addEventListener('change', updateMatches);

    return () => {
      subscription?.remove();
    };
  }, [query, breakpoints]);

  return matches;
};

