import { useEffect, useState, useCallback } from 'react';

interface UseServiceWorkerOptions {
  onUpdateAvailable?: () => void;
  onUpdateInstalled?: () => void;
}

export const useServiceWorker = (options: UseServiceWorkerOptions = {}) => {
  const { onUpdateAvailable, onUpdateInstalled } = options;
  const [registration, setRegistration] = useState<ServiceWorkerRegistration | null>(null);
  const [updateAvailable, setUpdateAvailable] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
      return;
    }

    const register = async (): Promise<void> => {
      try {
        const reg = await navigator.serviceWorker.register('/sw.js');
        setRegistration(reg);

        reg.addEventListener('updatefound', () => {
          const newWorker = reg.installing;
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                setUpdateAvailable(true);
                if (onUpdateAvailable) {
                  onUpdateAvailable();
                }
              }
            });
          }
        });

        if (reg.waiting) {
          setUpdateAvailable(true);
          if (onUpdateAvailable) {
            onUpdateAvailable();
          }
        }
      } catch (error) {
        console.error('Service Worker registration failed:', error);
      }
    };

    register();

    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (onUpdateInstalled) {
        onUpdateInstalled();
      }
    });
  }, [onUpdateAvailable, onUpdateInstalled]);

  const update = useCallback(async () => {
    if (registration?.waiting) {
      registration.waiting.postMessage({ type: 'SKIP_WAITING' });
    }
  }, [registration]);

  const unregister = useCallback(async () => {
    if (registration) {
      const success = await registration.unregister();
      if (success) {
        setRegistration(null);
      }
    }
  }, [registration]);

  return {
    registration,
    updateAvailable,
    update,
    unregister,
    supported: typeof window !== 'undefined' && 'serviceWorker' in navigator,
  };
};



