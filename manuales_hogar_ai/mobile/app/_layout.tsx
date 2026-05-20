import { Stack } from 'expo-router';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { useEffect } from 'react';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { StripeProvider } from '@stripe/stripe-react-native';
import { ErrorBoundary } from '@/components/error-boundary';
import { AppProvider } from '@/lib/context/app-context';
import { AuthProvider } from '@/lib/context/auth-context';
import { ToastProvider } from '@/components/ui/toast-provider';
import { NetworkStatus } from '@/components/ui/network-status';
import { initializeStripe } from '@/lib/stripe/config';
import '@/lib/i18n/config';

// Keep splash screen visible while loading
SplashScreen.preventAutoHideAsync();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
    },
  },
});

export default function RootLayout() {
  useEffect(() => {
    async function prepare() {
      try {
        // Initialize Stripe
        await initializeStripe();
        // Pre-load fonts, make any API calls you need to do here
        await new Promise((resolve) => setTimeout(resolve, 1000));
      } catch (e) {
        console.warn(e);
      } finally {
        await SplashScreen.hideAsync();
      }
    }

    prepare();
  }, []);

  return (
    <ErrorBoundary>
      <GestureHandlerRootView style={{ flex: 1 }}>
        <SafeAreaProvider>
          <QueryClientProvider client={queryClient}>
            <StripeProvider publishableKey={process.env.EXPO_PUBLIC_STRIPE_PUBLISHABLE_KEY || ''}>
              <AppProvider>
                <AuthProvider>
                  <ToastProvider>
                    <StatusBar style="auto" />
                    <NetworkStatus />
                    <Stack
                    screenOptions={{
                      headerShown: false,
                      contentStyle: { backgroundColor: '#fff' },
                    }}
                  >
                    <Stack.Screen name="(auth)" options={{ headerShown: false }} />
                    <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
                    <Stack.Screen
                      name="manual/[id]"
                      options={{
                        presentation: 'card',
                        headerShown: true,
                        title: 'Manual',
                      }}
                    />
                    <Stack.Screen
                      name="generate"
                      options={{
                        presentation: 'modal',
                        headerShown: true,
                        title: 'Generar Manual',
                      }}
                    />
                    <Stack.Screen
                      name="subscription"
                      options={{
                        presentation: 'card',
                        headerShown: true,
                        title: 'Subscription',
                      }}
                    />
                  </Stack>
                  </ToastProvider>
                </AuthProvider>
              </AppProvider>
            </StripeProvider>
          </QueryClientProvider>
        </SafeAreaProvider>
      </GestureHandlerRootView>
    </ErrorBoundary>
  );
}

