import { useWindowDimensions } from 'react-native';

export function useResponsiveDimensions() {
  const { width, height } = useWindowDimensions();

  const isSmallScreen = width < 375;
  const isMediumScreen = width >= 375 && width < 768;
  const isLargeScreen = width >= 768;
  const isTablet = width >= 768;
  const isLandscape = width > height;

  return {
    width,
    height,
    isSmallScreen,
    isMediumScreen,
    isLargeScreen,
    isTablet,
    isLandscape,
  };
}

