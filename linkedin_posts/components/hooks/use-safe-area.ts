import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useWindowDimensions } from 'react-native';

interface SafeAreaMetrics {
  top: number;
  bottom: number;
  left: number;
  right: number;
  width: number;
  height: number;
}

interface SafeAreaUtils {
  insets: SafeAreaMetrics;
  isTablet: boolean;
  isLandscape: boolean;
  getSafeAreaStyle: (edges?: ('top' | 'bottom' | 'left' | 'right')[]) => any;
  getContentHeight: () => number;
}

export function useSafeArea(): SafeAreaUtils {
  const insets = useSafeAreaInsets();
  const { width, height } = useWindowDimensions();

  const isTablet = width > 768;
  const isLandscape = width > height;

  function getSafeAreaStyle(edges: ('top' | 'bottom' | 'left' | 'right')[] = ['top', 'bottom', 'left', 'right']) {
    return {
      paddingTop: edges.includes('top') ? insets.top : 0,
      paddingBottom: edges.includes('bottom') ? insets.bottom : 0,
      paddingLeft: edges.includes('left') ? insets.left : 0,
      paddingRight: edges.includes('right') ? insets.right : 0,
    };
  }

  function getContentHeight(): number {
    return height - insets.top - insets.bottom;
  }

  return {
    insets: {
      top: insets.top,
      bottom: insets.bottom,
      left: insets.left,
      right: insets.right,
      width,
      height,
    },
    isTablet,
    isLandscape,
    getSafeAreaStyle,
    getContentHeight,
  };
}

// Export the hook
export default useSafeArea; 