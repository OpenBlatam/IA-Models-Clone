import { useCallback, useState } from 'react';

interface ShareData {
  title?: string;
  text?: string;
  url?: string;
  files?: File[];
}

export const useShare = () => {
  const [isSupported, setIsSupported] = useState(
    typeof navigator !== 'undefined' && 'share' in navigator
  );
  const [error, setError] = useState<Error | null>(null);

  const share = useCallback(async (data: ShareData): Promise<boolean> => {
    if (!isSupported) {
      setError(new Error('Web Share API is not supported'));
      return false;
    }

    try {
      await navigator.share(data);
      setError(null);
      return true;
    } catch (err) {
      const shareError = err instanceof Error ? err : new Error('Share failed');
      if (shareError.name !== 'AbortError') {
        setError(shareError);
      }
      return false;
    }
  }, [isSupported]);

  return { share, isSupported, error };
};

