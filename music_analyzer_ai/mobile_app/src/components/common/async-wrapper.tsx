import React, { ReactNode } from 'react';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import { ErrorState } from './error-state';
import { SkeletonScreen } from './skeleton-screen';
import { COLORS } from '../../constants/config';

interface AsyncWrapperProps<T> {
  data: T | undefined;
  isLoading: boolean;
  error: Error | null;
  loadingComponent?: ReactNode;
  errorComponent?: (error: Error) => ReactNode;
  onRetry?: () => void;
  children: (data: T) => ReactNode;
  showSkeleton?: boolean;
  skeletonProps?: {
    itemCount?: number;
    showHeader?: boolean;
    showImage?: boolean;
    showFooter?: boolean;
  };
}

export function AsyncWrapper<T>({
  data,
  isLoading,
  error,
  loadingComponent,
  errorComponent,
  onRetry,
  children,
  showSkeleton = true,
  skeletonProps,
}: AsyncWrapperProps<T>): React.ReactElement {
  if (isLoading) {
    if (loadingComponent) {
      return <>{loadingComponent}</>;
    }
    if (showSkeleton) {
      return <SkeletonScreen {...skeletonProps} />;
    }
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  if (error) {
    if (errorComponent) {
      return <>{errorComponent(error)}</>;
    }
    return (
      <ErrorState
        message={error.message || 'An error occurred'}
        onRetry={onRetry}
        showDetails={__DEV__}
        details={error.stack}
      />
    );
  }

  if (!data) {
    return (
      <ErrorState
        message="No data available"
        onRetry={onRetry}
      />
    );
  }

  return <>{children(data)}</>;
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.background,
  },
});

