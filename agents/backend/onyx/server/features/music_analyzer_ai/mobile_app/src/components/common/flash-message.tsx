import React, { useEffect } from 'react';
import FlashMessage, { showMessage } from 'react-native-flash-message';
import { COLORS } from '../../constants/config';

export function FlashMessageProvider() {
  return (
    <FlashMessage
      position="top"
      floating={true}
      style={{
        paddingTop: 50,
      }}
    />
  );
}

export function showSuccessMessage(message: string) {
  showMessage({
    message,
    type: 'success',
    backgroundColor: COLORS.success,
    color: COLORS.text,
  });
}

export function showErrorMessage(message: string) {
  showMessage({
    message,
    type: 'danger',
    backgroundColor: COLORS.error,
    color: COLORS.text,
  });
}

export function showInfoMessage(message: string) {
  showMessage({
    message,
    type: 'info',
    backgroundColor: COLORS.info,
    color: COLORS.text,
  });
}

export function showWarningMessage(message: string) {
  showMessage({
    message,
    type: 'warning',
    backgroundColor: COLORS.warning,
    color: COLORS.text,
  });
}

