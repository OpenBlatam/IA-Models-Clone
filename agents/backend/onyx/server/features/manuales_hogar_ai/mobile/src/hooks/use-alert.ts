/**
 * useAlert Hook
 * =============
 * Hook for showing alert dialogs
 */

import { useState, useCallback } from 'react';
import { Alert as RNAlert } from 'react-native';

export function useAlert() {
  const showAlert = useCallback(
    (
      title: string,
      message: string,
      buttons?: Array<{
        text: string;
        onPress?: () => void;
        style?: 'default' | 'cancel' | 'destructive';
      }>
    ) => {
      RNAlert.alert(title, message, buttons);
    },
    []
  );

  const showConfirm = useCallback(
    (
      title: string,
      message: string,
      onConfirm: () => void,
      onCancel?: () => void
    ) => {
      RNAlert.alert(
        title,
        message,
        [
          {
            text: 'Cancel',
            style: 'cancel',
            onPress: onCancel,
          },
          {
            text: 'Confirm',
            onPress: onConfirm,
            style: 'default',
          },
        ],
        { cancelable: true }
      );
    },
    []
  );

  return {
    showAlert,
    showConfirm,
  };
}



