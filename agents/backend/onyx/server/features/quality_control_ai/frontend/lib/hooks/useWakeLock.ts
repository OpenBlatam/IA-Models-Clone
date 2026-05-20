import { useEffect, useState, useCallback } from 'react';

export const useWakeLock = () => {
  const [isSupported, setIsSupported] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const wakeLockRef = useState<WakeLockSentinel | null>(null)[0];

  useEffect(() => {
    if (typeof navigator !== 'undefined' && 'wakeLock' in navigator) {
      setIsSupported(true);
    }
  }, []);

  const request = useCallback(async (): Promise<boolean> => {
    if (!isSupported) {
      setError(new Error('Wake Lock API is not supported'));
      return false;
    }

    try {
      const wakeLock = await (navigator as Navigator & { wakeLock: WakeLock }).wakeLock.request(
        'screen'
      );
      setIsActive(true);
      setError(null);

      wakeLock.addEventListener('release', () => {
        setIsActive(false);
      });

      return true;
    } catch (err) {
      const wakeLockError = err instanceof Error ? err : new Error('Wake Lock request failed');
      setError(wakeLockError);
      setIsActive(false);
      return false;
    }
  }, [isSupported]);

  const release = useCallback(async (): Promise<void> => {
    if (wakeLockRef) {
      await wakeLockRef.release();
      setIsActive(false);
    }
  }, [wakeLockRef]);

  useEffect(() => {
    return () => {
      if (wakeLockRef) {
        wakeLockRef.release();
      }
    };
  }, [wakeLockRef]);

  return { request, release, isSupported, isActive, error };
};

