import { useState, useCallback } from 'react';
import * as Clipboard from 'expo-clipboard';

export function useClipboard() {
  const [clipboardContent, setClipboardContent] = useState<string>('');

  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await Clipboard.setStringAsync(text);
      setClipboardContent(text);
      return true;
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      return false;
    }
  }, []);

  const getClipboardContent = useCallback(async () => {
    try {
      const text = await Clipboard.getStringAsync();
      setClipboardContent(text);
      return text;
    } catch (error) {
      console.error('Failed to get clipboard content:', error);
      return '';
    }
  }, []);

  return {
    clipboardContent,
    copyToClipboard,
    getClipboardContent,
  };
}


