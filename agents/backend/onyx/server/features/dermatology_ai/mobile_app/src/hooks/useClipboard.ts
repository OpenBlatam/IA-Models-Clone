import { useState } from 'react';
import * as Clipboard from 'expo-clipboard';

export const useClipboard = () => {
  const [copiedText, setCopiedText] = useState<string>('');

  const copyToClipboard = async (text: string): Promise<boolean> => {
    try {
      await Clipboard.setStringAsync(text);
      setCopiedText(text);
      return true;
    } catch (error) {
      console.error('Error copying to clipboard:', error);
      return false;
    }
  };

  const getFromClipboard = async (): Promise<string> => {
    try {
      const text = await Clipboard.getStringAsync();
      setCopiedText(text);
      return text;
    } catch (error) {
      console.error('Error getting from clipboard:', error);
      return '';
    }
  };

  return {
    copiedText,
    copyToClipboard,
    getFromClipboard,
  };
};

