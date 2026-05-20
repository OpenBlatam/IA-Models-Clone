import React from 'react';
import { View, ActivityIndicator, Text, StyleSheet } from 'react-native';
import { useColorScheme } from 'react-native';

interface LoadingProps {
  message?: string;
  size?: 'small' | 'large';
}

export function Loading({ message, size = 'large' }: LoadingProps) {
  const colorScheme = useColorScheme();

  return (
    <View style={styles.container}>
      <ActivityIndicator size={size} color={colorScheme === 'dark' ? '#FFFFFF' : '#007AFF'} />
      {message && (
        <Text style={[styles.message, colorScheme === 'dark' && styles.messageDark]}>
          {message}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  message: {
    marginTop: 16,
    fontSize: 16,
    color: '#000000',
  },
  messageDark: {
    color: '#FFFFFF',
  },
});


