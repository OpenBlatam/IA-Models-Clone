/**
 * Custom hook for copying text to clipboard.
 * Provides copy functionality with success/error feedback.
 */

import { useState, useCallback } from 'react';

/**
 * Return type for useCopyToClipboard hook.
 */
export interface UseCopyToClipboardReturn {
  copy: (text: string) => Promise<boolean>;
  isCopied: boolean;
  error: Error | null;
}

/**
 * Custom hook for copying text to clipboard.
 * Provides copy functionality with state management.
 *
 * @returns Copy function and state
 */
export function useCopyToClipboard(): UseCopyToClipboardReturn {
  const [isCopied, setIsCopied] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const copy = useCallback(async (text: string): Promise<boolean> => {
    try {
      if (typeof navigator === 'undefined' || !navigator.clipboard) {
        throw new Error('Clipboard API not available');
      }

      await navigator.clipboard.writeText(text);
      setIsCopied(true);
      setError(null);

      // Reset copied state after 2 seconds
      setTimeout(() => {
        setIsCopied(false);
      }, 2000);

      return true;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to copy');
      setError(error);
      setIsCopied(false);
      return false;
    }
  }, []);

  return {
    copy,
    isCopied,
    error,
  };
}

