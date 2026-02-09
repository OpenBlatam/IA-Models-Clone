/**
 * useCopyToClipboard Hook
 * Hook for copying text to clipboard with feedback
 */

'use client';

import { useState, useCallback, useRef } from 'react';

interface UseCopyToClipboardOptions {
  timeout?: number;
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

/**
 * Hook to copy text to clipboard
 * @param options - Configuration options
 * @returns Tuple with [copied, copy, reset]
 */
export const useCopyToClipboard = (
  options: UseCopyToClipboardOptions = {}
): [boolean, (text: string) => Promise<void>, () => void] => {
  const { timeout = 2000, onSuccess, onError } = options;
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();

  const copy = useCallback(async (text: string) => {
    try {
      // Fallback for older browsers
      if (!navigator.clipboard) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.opacity = '0';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
      } else {
        await navigator.clipboard.writeText(text);
      }

      setCopied(true);
      onSuccess?.();

      // Clear existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // Set new timeout
      timeoutRef.current = setTimeout(() => {
        setCopied(false);
      }, timeout);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to copy text');
      console.error('Failed to copy text:', error);
      setCopied(false);
      onError?.(error);
    }
  }, [timeout, onSuccess, onError]);

  const reset = useCallback(() => {
    setCopied(false);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  }, []);

  return [copied, copy, reset];
};


