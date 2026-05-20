import { useState, useCallback } from 'react';
import { copyToClipboard, readFromClipboard } from '@/lib/utils/clipboard';

export interface UseClipboardReturn {
  value: string | null;
  copy: (text: string) => Promise<boolean>;
  read: () => Promise<string | null>;
  clear: () => void;
}

/**
 * Hook for clipboard operations
 */
export function useClipboard(): UseClipboardReturn {
  const [value, setValue] = useState<string | null>(null);

  const copy = useCallback(async (text: string): Promise<boolean> => {
    try {
      const success = await copyToClipboard(text);
      if (success) {
        setValue(text);
      }
      return success;
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      return false;
    }
  }, []);

  const read = useCallback(async (): Promise<string | null> => {
    try {
      const text = await readFromClipboard();
      setValue(text);
      return text;
    } catch (error) {
      console.error('Failed to read from clipboard:', error);
      return null;
    }
  }, []);

  const clear = useCallback(() => {
    setValue(null);
  }, []);

  return {
    value,
    copy,
    read,
    clear,
  };
}



