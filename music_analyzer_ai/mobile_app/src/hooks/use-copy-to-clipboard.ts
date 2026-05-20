import { useState, useCallback } from 'react';
import * as Clipboard from 'expo-clipboard';

/**
 * Hook for copying text to clipboard
 * Returns copy function and success state
 */
export function useCopyToClipboard() {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async (text: string) => {
    try {
      await Clipboard.setStringAsync(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      return true;
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      return false;
    }
  }, []);

  return {
    copy,
    copied,
  };
}

