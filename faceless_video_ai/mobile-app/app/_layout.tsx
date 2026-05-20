import { Stack } from 'expo-router';
import { AppProvider } from '@/providers/app-provider';
import { OfflineBanner } from '@/components/ui/offline-banner';
import * as SplashScreen from 'expo-splash-screen';
import { useEffect } from 'react';

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  useEffect(() => {
    async function prepare() {
      try {
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
    <AppProvider>
      <OfflineBanner />
      <Stack
        screenOptions={{
          headerShown: false,
        }}
      >
            <Stack.Screen name="(auth)" options={{ headerShown: false }} />
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen
              name="video-detail"
              options={{
                presentation: 'modal',
                headerShown: true,
                title: 'Video Details',
              }}
            />
            <Stack.Screen
              name="video-generation"
              options={{
                presentation: 'modal',
                headerShown: true,
                title: 'Generate Video',
              }}
            />
            <Stack.Screen
              name="video-settings"
              options={{
                presentation: 'modal',
                headerShown: true,
                title: 'Video Settings',
              }}
            />
            <Stack.Screen
              name="template-detail"
              options={{
                presentation: 'modal',
                headerShown: true,
                title: 'Template',
              }}
            />
            <Stack.Screen
              name="batch-generation"
              options={{
                presentation: 'modal',
                headerShown: true,
                title: 'Batch Generation',
              }}
            />
            <Stack.Screen
              name="analytics"
              options={{
                headerShown: true,
                title: 'Analytics',
              }}
            />
            <Stack.Screen
              name="search"
              options={{
                headerShown: true,
                title: 'Search',
              }}
            />
            <Stack.Screen
              name="profile"
              options={{
                headerShown: true,
                title: 'Profile',
              }}
            />
            <Stack.Screen
              name="settings"
              options={{
                headerShown: true,
                title: 'Settings',
              }}
            />
      </Stack>
    </AppProvider>
  );
}

