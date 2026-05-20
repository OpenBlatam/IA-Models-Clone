import { Stack } from 'expo-router';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ToastProvider } from '../contexts/toast-context';
import { SnackbarProvider } from '../contexts/snackbar-context';
import { ThemeProvider } from '../contexts/theme-context';
import { ErrorBoundary } from '../components/common/error-boundary';
import { FlashMessageProvider } from '../components/common/flash-message';
import { NetworkStatusBar } from '../components/common/network-status';
import { StatusBar } from 'expo-status-bar';
import { useColorScheme } from 'react-native';
import { useEffect } from 'react';
import * as SplashScreen from 'expo-splash-screen';
import { COLORS } from '../constants/config';
import '../i18n/config';
import { initializeErrorTracking } from '../utils/error-handler';

SplashScreen.preventAutoHideAsync();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000,
    },
  },
});

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  useEffect(() => {
    initializeErrorTracking();
    
    async function prepare() {
      try {
        await SplashScreen.hideAsync();
      } catch (e) {
        console.warn(e);
      }
    }
    
    prepare();
  }, []);

  return (
    <ErrorBoundary>
      <SafeAreaProvider>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider>
              <ToastProvider>
                <SnackbarProvider>
                  <FlashMessageProvider />
                  <NetworkStatusBar />
                  <StatusBar style={isDark ? 'light' : 'dark'} />
            <Stack
              screenOptions={{
                headerStyle: {
                  backgroundColor: COLORS.surface,
                },
                headerTintColor: COLORS.text,
                headerTitleStyle: {
                  fontWeight: '600',
                },
                contentStyle: {
                  backgroundColor: COLORS.background,
                },
                animation: 'slide_from_right',
              }}
            >
              <Stack.Screen
                name="(tabs)"
                options={{
                  headerShown: false,
                }}
              />
              <Stack.Screen
                name="analysis"
                options={{
                  title: 'Analysis',
                  presentation: 'card',
                }}
              />
              <Stack.Screen
                name="recommendations"
                options={{
                  title: 'Recommendations',
                  presentation: 'card',
                }}
              />
              <Stack.Screen
                name="compare"
                options={{
                  title: 'Compare Tracks',
                  presentation: 'card',
                }}
              />
              <Stack.Screen
                name="history"
                options={{
                  title: 'History',
                  presentation: 'card',
                }}
              />
              <Stack.Screen
                name="comparison-results"
                options={{
                  title: 'Comparison Results',
                  presentation: 'card',
                }}
              />
            </Stack>
                </SnackbarProvider>
            </ToastProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </SafeAreaProvider>
    </ErrorBoundary>
  );
}

