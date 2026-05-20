import { useState, useCallback } from 'react';
import { copyToClipboard, readFromClipboard } from '@/lib/utils';

interface UseClipboardOptions {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

export const useClipboard = (options: UseClipboardOptions = {}) => {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const copy = useCallback(
    async (text: string): Promise<boolean> => {
      try {
        const success = await copyToClipboard(text);
        if (success) {
          setCopied(true);
          setError(null);
          options.onSuccess?.();
          setTimeout(() => setCopied(false), 2000);
        } else {
          throw new Error('Failed to copy to clipboard');
        }
        return success;
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Unknown error');
        setError(error);
        options.onError?.(error);
        return false;
      }
    },
    [options]
  );

  const read = useCallback(async (): Promise<string | null> => {
    try {
      const text = await readFromClipboard();
      setError(null);
      return text;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      options.onError?.(error);
      return null;
    }
  }, [options]);

  return { copy, read, copied, error };
};

