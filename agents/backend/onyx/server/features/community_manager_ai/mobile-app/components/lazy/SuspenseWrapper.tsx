import { Suspense, ReactNode } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';

interface SuspenseWrapperProps {
  children: ReactNode;
  fallback?: ReactNode;
}

function DefaultFallback() {
  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#0ea5e9" />
    </View>
  );
}

export function SuspenseWrapper({ children, fallback = <DefaultFallback /> }: SuspenseWrapperProps) {
  return <Suspense fallback={fallback}>{children}</Suspense>;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
});


