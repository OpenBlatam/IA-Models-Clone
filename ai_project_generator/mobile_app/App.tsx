import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { QueryProvider } from './src/providers/QueryProvider';
import { ToastProvider } from './src/components/Toast';
import { ThemeProvider, useTheme } from './src/contexts/ThemeContext';
import { ErrorBoundary } from './src/components/ErrorBoundary';
import { NetworkStatusBar } from './src/components/NetworkStatusBar';
import { AppNavigator } from './src/navigation/AppNavigator';
import { useDeepLinking } from './src/hooks/useDeepLinking';
import { useAppShortcuts } from './src/components/KeyboardShortcuts';

const AppContent = () => {
  const { isDark } = useTheme();
  useDeepLinking();
  useAppShortcuts();
  
  return (
    <>
      <NetworkStatusBar />
      <AppNavigator />
      <StatusBar style={isDark ? 'light' : 'dark'} />
    </>
  );
};

export default function App() {
  return (
    <ErrorBoundary>
      <QueryProvider>
        <ThemeProvider>
          <ToastProvider>
            <SafeAreaProvider>
              <AppContent />
            </SafeAreaProvider>
          </ToastProvider>
        </ThemeProvider>
      </QueryProvider>
    </ErrorBoundary>
  );
}
