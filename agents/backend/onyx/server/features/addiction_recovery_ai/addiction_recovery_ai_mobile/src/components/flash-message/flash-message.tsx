import React, { useEffect } from 'react';
import FlashMessage, { showMessage, MessageOptions } from 'react-native-flash-message';
import { useColors } from '@/theme/colors';

export function showFlashMessage(options: MessageOptions): void {
  showMessage(options);
}

export function FlashMessageContainer(): JSX.Element {
  const colors = useColors();

  return (
    <FlashMessage
      position="top"
      style={{
        paddingTop: 20,
      }}
    />
  );
}

