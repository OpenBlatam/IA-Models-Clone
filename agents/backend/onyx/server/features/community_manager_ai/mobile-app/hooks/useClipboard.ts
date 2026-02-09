import { useState } from 'react';
import * as Clipboard from 'expo-clipboard';
import Toast from 'react-native-toast-message';

export function useClipboard() {
  const [copiedText, setCopiedText] = useState<string>('');

  const copyToClipboard = async (text: string, showNotification: boolean = true) => {
    try {
      await Clipboard.setStringAsync(text);
      setCopiedText(text);
      
      if (showNotification) {
        Toast.show({
          type: 'success',
          text1: 'Copied to clipboard',
        });
      }
      
      return true;
    } catch (error) {
      if (showNotification) {
        Toast.show({
          type: 'error',
          text1: 'Failed to copy',
        });
      }
      return false;
    }
  };

  const getFromClipboard = async (): Promise<string> => {
    try {
      const text = await Clipboard.getStringAsync();
      setCopiedText(text);
      return text;
    } catch (error) {
      return '';
    }
  };

  return {
    copiedText,
    copyToClipboard,
    getFromClipboard,
  };
}
