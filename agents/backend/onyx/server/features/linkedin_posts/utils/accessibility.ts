import React from 'react';
import {
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  AccessibilityInfo,
  Platform,
  Dimensions,
  PixelRatio,
  useWindowDimensions,
} from 'react-native';
import { useColorScheme } from 'react-native';

// Accessibility Manager for text scaling and font adjustments
export class AccessibilityManager {
  private static instance: AccessibilityManager;
  private fontScale: number = 1.0;
  private isBoldTextEnabled: boolean = false;
  private isReduceMotionEnabled: boolean = false;
  private isScreenReaderEnabled: boolean = false;
  private colorScheme: 'light' | 'dark' = 'light';

  static getInstance(): AccessibilityManager {
    if (!AccessibilityManager.instance) {
      AccessibilityManager.instance = new AccessibilityManager();
    }
    return AccessibilityManager.instance;
  }

  async initializeAccessibility(): Promise<void> {
    try {
      // Get system font scale
      this.fontScale = await AccessibilityInfo.getRecommendedFontSize();
      
      // Check accessibility features
      this.isBoldTextEnabled = await AccessibilityInfo.isBoldTextEnabled();
      this.isReduceMotionEnabled = await AccessibilityInfo.isReduceMotionEnabled();
      this.isScreenReaderEnabled = await AccessibilityInfo.isScreenReaderEnabled();
      
      // Set up listeners
      this.setupAccessibilityListeners();
    } catch (error) {
      console.warn('Accessibility initialization failed:', error);
    }
  }

  private setupAccessibilityListeners(): void {
    AccessibilityInfo.addEventListener('boldTextChanged', (isBoldTextEnabled) => {
      this.isBoldTextEnabled = isBoldTextEnabled;
    });

    AccessibilityInfo.addEventListener('reduceMotionChanged', (isReduceMotionEnabled) => {
      this.isReduceMotionEnabled = isReduceMotionEnabled;
    });

    AccessibilityInfo.addEventListener('screenReaderChanged', (isScreenReaderEnabled) => {
      this.isScreenReaderEnabled = isScreenReaderEnabled;
    });
  }

  getScaledFontSize(baseSize: number): number {
    const scaledSize = baseSize * this.fontScale;
    const minSize = 8;
    const maxSize = 72;
    return Math.max(minSize, Math.min(maxSize, scaledSize));
  }

  getFontWeight(): 'normal' | 'bold' {
    return this.isBoldTextEnabled ? 'bold' : 'normal';
  }

  shouldReduceMotion(): boolean {
    return this.isReduceMotionEnabled;
  }

  isScreenReaderActive(): boolean {
    return this.isScreenReaderEnabled;
  }

  getColorScheme(): 'light' | 'dark' {
    return this.colorScheme;
  }

  setColorScheme(scheme: 'light' | 'dark'): void {
    this.colorScheme = scheme;
  }
}

// React Hook for accessibility
export function useAccessibility() {
  const [accessibilityManager] = React.useState(() => AccessibilityManager.getInstance());
  const [fontScale, setFontScale] = React.useState(1.0);
  const [isBoldTextEnabled, setIsBoldTextEnabled] = React.useState(false);
  const [isReduceMotionEnabled, setIsReduceMotionEnabled] = React.useState(false);
  const [isScreenReaderEnabled, setIsScreenReaderEnabled] = React.useState(false);

  React.useEffect(() => {
    const initializeAccessibility = async () => {
      await accessibilityManager.initializeAccessibility();
      
      // Get initial values
      const boldText = await AccessibilityInfo.isBoldTextEnabled();
      const reduceMotion = await AccessibilityInfo.isReduceMotionEnabled();
      const screenReader = await AccessibilityInfo.isScreenReaderEnabled();
      
      setIsBoldTextEnabled(boldText);
      setIsReduceMotionEnabled(reduceMotion);
      setIsScreenReaderEnabled(screenReader);
    };

    initializeAccessibility();
  }, [accessibilityManager]);

  return {
    fontScale,
    isBoldTextEnabled,
    isReduceMotionEnabled,
    isScreenReaderEnabled,
    getScaledFontSize: (baseSize: number) => accessibilityManager.getScaledFontSize(baseSize),
    getFontWeight: () => accessibilityManager.getFontWeight(),
    shouldReduceMotion: () => accessibilityManager.shouldReduceMotion(),
    isScreenReaderActive: () => accessibilityManager.isScreenReaderActive(),
  };
}

// Hook for scaled font sizes
export function useScaledFontSize(baseSize: number): number {
  const { getScaledFontSize } = useAccessibility();
  return getScaledFontSize(baseSize);
}

// Accessibility utilities
export const AccessibilityUtils = {
  // Calculate contrast ratio for WCAG compliance
  getContrastRatio(color1: string, color2: string): number {
    const getLuminance = (color: string): number => {
      const hex = color.replace('#', '');
      const r = parseInt(hex.substr(0, 2), 16) / 255;
      const g = parseInt(hex.substr(2, 2), 16) / 255;
      const b = parseInt(hex.substr(4, 2), 16) / 255;
      
      const [rs, gs, bs] = [r, g, b].map(c => {
        if (c <= 0.03928) return c / 12.92;
        return Math.pow((c + 0.055) / 1.055, 2.4);
      });
      
      return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    };

    const luminance1 = getLuminance(color1);
    const luminance2 = getLuminance(color2);
    
    const brightest = Math.max(luminance1, luminance2);
    const darkest = Math.min(luminance1, luminance2);
    
    return (brightest + 0.05) / (darkest + 0.05);
  },

  // Check if contrast meets WCAG AA standards
  isHighContrast(color1: string, color2: string): boolean {
    const ratio = this.getContrastRatio(color1, color2);
    return ratio >= 4.5; // WCAG AA standard for normal text
  },

  // Get accessible colors based on background
  getAccessibleTextColor(backgroundColor: string): string {
    const white = '#FFFFFF';
    const black = '#000000';
    
    const whiteContrast = this.getContrastRatio(backgroundColor, white);
    const blackContrast = this.getContrastRatio(backgroundColor, black);
    
    return whiteContrast > blackContrast ? white : black;
  },

  // Get minimum touch target size
  getMinTouchTargetSize(): number {
    return Platform.OS === 'ios' ? 44 : 48; // iOS and Android guidelines
  },
};

// Accessibility styles
export const accessibilityStyles = {
  // Touch target styles
  touchTarget: {
    minHeight: AccessibilityUtils.getMinTouchTargetSize(),
    minWidth: AccessibilityUtils.getMinTouchTargetSize(),
    justifyContent: 'center' as const,
    alignItems: 'center' as const,
  },

  // High contrast colors
  colors: {
    primary: '#007AFF',
    secondary: '#5856D6',
    success: '#34C759',
    warning: '#FF9500',
    error: '#FF3B30',
    background: '#FFFFFF',
    surface: '#F2F2F7',
    text: '#000000',
    textSecondary: '#8E8E93',
  },

  // Typography with accessibility
  typography: {
    h1: {
      fontSize: 32,
      fontWeight: 'bold' as const,
      lineHeight: 40,
    },
    h2: {
      fontSize: 28,
      fontWeight: 'bold' as const,
      lineHeight: 36,
    },
    h3: {
      fontSize: 24,
      fontWeight: '600' as const,
      lineHeight: 32,
    },
    body: {
      fontSize: 16,
      fontWeight: 'normal' as const,
      lineHeight: 24,
    },
    caption: {
      fontSize: 14,
      fontWeight: 'normal' as const,
      lineHeight: 20,
    },
    button: {
      fontSize: 16,
      fontWeight: '600' as const,
      lineHeight: 20,
    },
  },
};

// Accessible Text Component
export function AccessibleText({
  children,
  style,
  accessibilityLabel,
  accessibilityHint,
  accessibilityRole = 'text',
  allowFontScaling = true,
  maxFontSizeMultiplier = 2.0,
  ...props
}: {
  children: React.ReactNode;
  style?: any;
  accessibilityLabel?: string;
  accessibilityHint?: string;
  accessibilityRole?: string;
  allowFontScaling?: boolean;
  maxFontSizeMultiplier?: number;
  [key: string]: any;
}) {
  const { getScaledFontSize, getFontWeight } = useAccessibility();
  
  const scaledStyle = React.useMemo(() => {
    if (!style) return {};
    
    const baseFontSize = style.fontSize || 16;
    const scaledFontSize = getScaledFontSize(baseFontSize);
    const fontWeight = getFontWeight();
    
    return {
      ...style,
      fontSize: scaledFontSize,
      fontWeight,
    };
  }, [style, getScaledFontSize, getFontWeight]);

  return (
    <Text
      style={scaledStyle}
      accessibilityLabel={accessibilityLabel}
      accessibilityHint={accessibilityHint}
      accessibilityRole={accessibilityRole}
      allowFontScaling={allowFontScaling}
      maxFontSizeMultiplier={maxFontSizeMultiplier}
      {...props}
    >
      {children}
    </Text>
  );
}

// Accessible Button Component
export function AccessibleButton({
  children,
  onPress,
  style,
  accessibilityLabel,
  accessibilityHint,
  accessibilityRole = 'button',
  disabled = false,
  ...props
}: {
  children: React.ReactNode;
  onPress: () => void;
  style?: any;
  accessibilityLabel?: string;
  accessibilityHint?: string;
  accessibilityRole?: string;
  disabled?: boolean;
  [key: string]: any;
}) {
  const { getScaledFontSize } = useAccessibility();
  
  const buttonStyle = React.useMemo(() => {
    const minSize = AccessibilityUtils.getMinTouchTargetSize();
    const scaledFontSize = getScaledFontSize(16);
    
    return {
      ...accessibilityStyles.touchTarget,
      ...style,
      fontSize: scaledFontSize,
    };
  }, [style, getScaledFontSize]);

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={disabled}
      accessibilityLabel={accessibilityLabel}
      accessibilityHint={accessibilityHint}
      accessibilityRole={accessibilityRole}
      accessibilityState={{ disabled }}
      {...props}
    >
      {children}
    </TouchableOpacity>
  );
}

// Accessible View Component
export function AccessibleView({
  children,
  style,
  accessibilityLabel,
  accessibilityRole = 'none',
  ...props
}: {
  children: React.ReactNode;
  style?: any;
  accessibilityLabel?: string;
  accessibilityRole?: string;
  [key: string]: any;
}) {
  return (
    <View
      style={style}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole={accessibilityRole}
      {...props}
    >
      {children}
    </View>
  );
}

// Responsive accessibility hook
export function useResponsiveAccessibility() {
  const { width, height } = useWindowDimensions();
  const { getScaledFontSize } = useAccessibility();
  
  const isTablet = width > 768;
  const isLandscape = width > height;
  
  const getResponsiveFontSize = React.useCallback((baseSize: number) => {
    let adjustedSize = baseSize;
    
    if (isTablet) {
      adjustedSize *= 1.2; // Larger fonts on tablets
    }
    
    if (isLandscape) {
      adjustedSize *= 0.9; // Slightly smaller in landscape
    }
    
    return getScaledFontSize(adjustedSize);
  }, [isTablet, isLandscape, getScaledFontSize]);

  return {
    isTablet,
    isLandscape,
    getResponsiveFontSize,
    screenWidth: width,
    screenHeight: height,
  };
}

// Export all utilities and components
export {
  AccessibilityManager,
  useAccessibility,
  useScaledFontSize,
  AccessibilityUtils,
  accessibilityStyles,
  AccessibleText,
  AccessibleButton,
  AccessibleView,
  useResponsiveAccessibility,
}; 