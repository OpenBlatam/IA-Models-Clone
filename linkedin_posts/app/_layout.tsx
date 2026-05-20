import React from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: '#fff',
          },
          headerTintColor: '#007AFF',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
          headerShadowVisible: false,
        }}
      >
        <Stack.Screen
          name="index"
          options={{
            title: 'LinkedIn Posts',
            headerLargeTitle: true,
            headerLargeTitleStyle: {
              fontSize: 28,
              fontWeight: 'bold',
            },
          }}
        />
        <Stack.Screen
          name="post-generator"
          options={{
            title: 'Create Post',
            presentation: 'modal',
          }}
        />
        <Stack.Screen
          name="post-history"
          options={{
            title: 'Post History',
          }}
        />
        <Stack.Screen
          name="settings"
          options={{
            title: 'Settings',
          }}
        />
      </Stack>
    </SafeAreaProvider>
  );
} 