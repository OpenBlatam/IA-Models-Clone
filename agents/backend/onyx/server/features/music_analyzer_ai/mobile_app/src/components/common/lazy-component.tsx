import React, { Suspense, ComponentType, LazyExoticComponent } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { COLORS, SPACING } from '../../constants/config';

interface LazyComponentProps {
  fallback?: React.ReactNode;
}

/**
 * Higher-order component for lazy loading components
 * Implements code splitting for better performance
 */
export function withLazyLoading<P extends object>(
  Component: LazyExoticComponent<ComponentType<P>>
) {
  return function LazyComponent(props: P & LazyComponentProps) {
    const { fallback, ...componentProps } = props;

    return (
      <Suspense
        fallback={
          fallback || (
            <View style={styles.container}>
              <ActivityIndicator size="large" color={COLORS.primary} />
            </View>
          )
        }
      >
        <Component {...(componentProps as P)} />
      </Suspense>
    );
  };
}

/**
 * Default loading fallback component
 */
export function LoadingFallback() {
  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color={COLORS.primary} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: SPACING.lg,
  },
});

