import React from 'react';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { Platform, ViewStyle } from 'react-native';

// Types for safe area configuration
interface SafeAreaConfig {
  readonly edges?: ('top' | 'bottom' | 'left' | 'right')[];
  readonly backgroundColor?: string;
  readonly style?: ViewStyle;
}

interface SafeAreaWrapperProps {
  readonly children: React.ReactNode;
  readonly config?: SafeAreaConfig;
  readonly statusBarStyle?: 'light' | 'dark' | 'auto';
  readonly statusBarHidden?: boolean;
}

// Main SafeAreaProvider component for global safe area management
export function LinkedInSafeAreaProvider({ children }: { children: React.ReactNode }) {
  return (
    <SafeAreaProvider>
      <StatusBar style="auto" />
      {children}
    </SafeAreaProvider>
  );
}

// SafeAreaView wrapper component with configuration
export function SafeAreaWrapper({
  children,
  config = {},
  statusBarStyle = 'auto',
  statusBarHidden = false,
}: SafeAreaWrapperProps) {
  const {
    edges = ['top', 'bottom', 'left', 'right'],
    backgroundColor = '#ffffff',
    style = {},
  } = config;

  return (
    <SafeAreaView
      edges={edges}
      style={[
        {
          flex: 1,
          backgroundColor,
        },
        style,
      ]}
    >
      <StatusBar
        style={statusBarStyle}
        hidden={statusBarHidden}
        backgroundColor={backgroundColor}
        translucent={Platform.OS === 'android'}
      />
      {children}
    </SafeAreaView>
  );
}

// Specialized safe area components for different use cases
export function TopSafeArea({ children, backgroundColor = '#ffffff' }: {
  children: React.ReactNode;
  backgroundColor?: string;
}) {
  return (
    <SafeAreaWrapper
      config={{
        edges: ['top'],
        backgroundColor,
      }}
    >
      {children}
    </SafeAreaWrapper>
  );
}

export function BottomSafeArea({ children, backgroundColor = '#ffffff' }: {
  children: React.ReactNode;
  backgroundColor?: string;
}) {
  return (
    <SafeAreaWrapper
      config={{
        edges: ['bottom'],
        backgroundColor,
      }}
    >
      {children}
    </SafeAreaWrapper>
  );
}

export function FullSafeArea({ children, backgroundColor = '#ffffff' }: {
  children: React.ReactNode;
  backgroundColor?: string;
}) {
  return (
    <SafeAreaWrapper
      config={{
        edges: ['top', 'bottom', 'left', 'right'],
        backgroundColor,
      }}
    >
      {children}
    </SafeAreaWrapper>
  );
}

// Safe area hook for accessing safe area insets
export function useSafeAreaInsets() {
  const { top, bottom, left, right } = useSafeAreaInsets();
  
  return {
    top,
    bottom,
    left,
    right,
    horizontal: left + right,
    vertical: top + bottom,
  };
}

// Utility function to get safe area styles
export function getSafeAreaStyles(backgroundColor = '#ffffff') {
  return {
    container: {
      flex: 1,
      backgroundColor,
    },
    content: {
      flex: 1,
      paddingHorizontal: 20,
    },
    header: {
      paddingTop: Platform.OS === 'ios' ? 44 : 20,
      paddingBottom: 20,
    },
    footer: {
      paddingBottom: Platform.OS === 'ios' ? 34 : 20,
    },
  };
}

// Export all components and utilities
export {
  SafeAreaProvider,
  SafeAreaView,
} from 'react-native-safe-area-context';

export type { SafeAreaConfig, SafeAreaWrapperProps }; 