import React from 'react';
import Toast from 'react-native-toast-message';
import { View, Text, StyleSheet } from 'react-native';
import { useColors } from '@/theme/colors';

interface ToastConfig {
  type: 'success' | 'error' | 'info';
  text1: string;
  text2?: string;
  position?: 'top' | 'bottom';
  visibilityTime?: number;
}

export function showToast(config: ToastConfig): void {
  Toast.show({
    type: config.type,
    text1: config.text1,
    text2: config.text2,
    position: config.position || 'top',
    visibilityTime: config.visibilityTime || 3000,
  });
}

export function ToastContainer(): JSX.Element {
  return <Toast />;
}

