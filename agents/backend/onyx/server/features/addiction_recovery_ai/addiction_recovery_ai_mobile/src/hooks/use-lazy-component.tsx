import React, { Suspense, ComponentType, LazyExoticComponent } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { useColors } from '@/theme/colors';

interface LazyComponentWrapperProps {
  children: React.ReactNode;
}

function LoadingFallback(): JSX.Element {
  const colors = useColors();
  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <ActivityIndicator size="large" color={colors.primary} />
    </View>
  );
}

export function LazyComponentWrapper({ children }: LazyComponentWrapperProps): JSX.Element {
  return <Suspense fallback={<LoadingFallback />}>{children}</Suspense>;
}

export function createLazyComponent<P extends object>(
  importFn: () => Promise<{ default: ComponentType<P> }>
): LazyExoticComponent<ComponentType<P>> {
  return React.lazy(importFn);
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

