import { useState, useCallback } from 'react';

interface UseClipboardOptions {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

export const useClipboard = (options: UseClipboardOptions = {}) => {
  const { onSuccess, onError } = options;
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const copy = useCallback(
    async (text: string) => {
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
        } else {
          // Fallback for older browsers
          const textArea = document.createElement('textarea');
          textArea.value = text;
          textArea.style.position = 'fixed';
          textArea.style.opacity = '0';
          document.body.appendChild(textArea);
          textArea.select();
          document.execCommand('copy');
          document.body.removeChild(textArea);
        }

        setCopied(true);
        setError(null);
        if (onSuccess) {
          onSuccess();
        }

        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Failed to copy');
        setError(error);
        setCopied(false);
        if (onError) {
          onError(error);
        }
      }
    },
    [onSuccess, onError]
  );

  return { copy, copied, error };
};



