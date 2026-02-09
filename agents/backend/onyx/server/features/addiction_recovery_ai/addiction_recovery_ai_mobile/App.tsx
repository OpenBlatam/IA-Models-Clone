import React, { useEffect, useCallback } from 'react';
import { StatusBar } from 'expo-status-bar';
import { QueryClientProvider } from '@tanstack/react-query';
import { createOptimizedQueryClient } from '@/utils/react-query-optimization';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import * as SplashScreen from 'expo-splash-screen';
import '@/i18n'; // Initialize i18n
import { AppNavigator } from '@/navigation/AppNavigator';
import { useAuthStore } from '@/store/auth-store';
import {
  ErrorBoundary,
  ToastContainer,
  FlashMessageContainer,
} from '@/components';
import { ThemeProvider } from '@/context/theme-context';
import { StyleSheet } from 'react-native';

// Keep the splash screen visible while we fetch resources
SplashScreen.preventAutoHideAsync();

const queryClient = createOptimizedQueryClient();

function AppContent(): JSX.Element {
  const { checkAuth } = useAuthStore();
  const [appIsReady, setAppIsReady] = React.useState(false);

  useEffect(() => {
    async function prepare(): Promise<void> {
      try {
        // Pre-load fonts, make any API calls you need to do here
        await checkAuth();
      } catch (e) {
        console.warn(e);
      } finally {
        setAppIsReady(true);
      }
    }

    prepare();
  }, [checkAuth]);

  const onLayoutRootView = useCallback(async () => {
    if (appIsReady) {
      await SplashScreen.hideAsync();
    }
  }, [appIsReady]);

  if (!appIsReady) {
    return <></>;
  }

  return (
    <GestureHandlerRootView style={styles.container} onLayout={onLayoutRootView}>
      <AppNavigator />
    </GestureHandlerRootView>
  );
}

export default function App(): JSX.Element {
  return (
    <ErrorBoundary>
      <SafeAreaProvider>
        <ThemeProvider>
          <QueryClientProvider client={queryClient}>
            <StatusBar style="auto" />
            <AppContent />
            <ToastContainer />
            <FlashMessageContainer />
          </QueryClientProvider>
        </ThemeProvider>
      </SafeAreaProvider>
    </ErrorBoundary>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

