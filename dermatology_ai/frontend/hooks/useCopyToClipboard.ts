import { useState } from 'react';

interface UseCopyToClipboardReturn {
  copy: (text: string) => Promise<void>;
  copied: boolean;
  error: Error | null;
}

export function useCopyToClipboard(): UseCopyToClipboardReturn {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setError(null);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to copy'));
      setCopied(false);
    }
  };

  return { copy, copied, error };
}


