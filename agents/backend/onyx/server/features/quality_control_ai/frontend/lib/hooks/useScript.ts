import { useEffect, useState } from 'react';

interface UseScriptOptions {
  src: string | null;
  async?: boolean;
  defer?: boolean;
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

export const useScript = (options: UseScriptOptions) => {
  const { src, async = true, defer = false, onLoad, onError } = options;
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');

  useEffect(() => {
    if (!src) {
      setState('ready');
      return;
    }

    let script: HTMLScriptElement | null = document.querySelector(`script[src="${src}"]`);

    if (!script) {
      script = document.createElement('script');
      script.src = src;
      script.async = async;
      script.defer = defer;

      const handleLoad = (): void => {
        setState('ready');
        onLoad?.();
      };

      const handleError = (): void => {
        setState('error');
        onError?.(new Error(`Failed to load script: ${src}`));
      };

      script.addEventListener('load', handleLoad);
      script.addEventListener('error', handleError);

      document.body.appendChild(script);
    } else {
      if (script.getAttribute('data-status') === 'ready') {
        setState('ready');
      }
    }

    return () => {
      // Don't remove script on cleanup to allow reuse
    };
  }, [src, async, defer, onLoad, onError]);

  return state;
};

