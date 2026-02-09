import { useState, useCallback } from 'react';
import * as Clipboard from 'expo-clipboard';

/**
 * Hook for clipboard operations
 * Provides copy and paste functionality
 */
export function useClipboard() {
  const [clipboardContent, setClipboardContent] = useState<string>('');

  const copyToClipboard = useCallback(async (text: string): Promise<boolean> => {
    try {
      await Clipboard.setStringAsync(text);
      setClipboardContent(text);
      return true;
    } catch (error) {
      console.error('Copy to clipboard error:', error);
      return false;
    }
  }, []);

  const getClipboardContent = useCallback(async (): Promise<string | null> => {
    try {
      const text = await Clipboard.getStringAsync();
      setClipboardContent(text);
      return text;
    } catch (error) {
      console.error('Get clipboard error:', error);
      return null;
    }
  }, []);

  const clearClipboard = useCallback(async (): Promise<void> => {
    try {
      await Clipboard.setStringAsync('');
      setClipboardContent('');
    } catch (error) {
      console.error('Clear clipboard error:', error);
    }
  }, []);

  return {
    clipboardContent,
    copyToClipboard,
    getClipboardContent,
    clearClipboard,
  };
}

