import React from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { useNetwork } from '@/contexts/network-context';
import { useTheme } from '@/contexts/theme-context';

export function OfflineBanner() {
  const { isOffline, showOfflineBanner, setShowOfflineBanner } = useNetwork();
  const { colors } = useTheme();
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    if (showOfflineBanner && isOffline) {
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }).start();
    } else {
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }).start(() => {
        if (!isOffline) {
          setShowOfflineBanner(false);
        }
      });
    }
  }, [showOfflineBanner, isOffline]);

  if (!showOfflineBanner || !isOffline) {
    return null;
  }

  return (
    <Animated.View
      style={[
        styles.banner,
        {
          backgroundColor: colors.error,
          opacity: fadeAnim,
        },
      ]}
    >
      <Text style={styles.text}>No internet connection</Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  banner: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
});


