import { useState, useCallback } from 'react';
import { useToast } from './useToast';

export const useCopyToClipboard = () => {
  const [copied, setCopied] = useState(false);
  const toast = useToast();

  const copy = useCallback(
    async (text: string): Promise<void> => {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        toast.success('Copied to clipboard');
        setTimeout(() => setCopied(false), 2000);
      } catch (error) {
        toast.error('Failed to copy to clipboard');
      }
    },
    [toast]
  );

  return { copy, copied };
};

