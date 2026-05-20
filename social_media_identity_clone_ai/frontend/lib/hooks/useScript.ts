import { useState, useEffect } from 'react';

interface UseScriptOptions {
  async?: boolean;
  defer?: boolean;
  id?: string;
  onLoad?: () => void;
  onError?: (error: Event) => void;
}

export const useScript = (src: string, options: UseScriptOptions = {}): [boolean, boolean] => {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }

    const existingScript = options.id ? document.getElementById(options.id) : null;
    if (existingScript) {
      setLoaded(true);
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = options.async ?? true;
    script.defer = options.defer ?? false;

    if (options.id) {
      script.id = options.id;
    }

    script.onload = () => {
      setLoaded(true);
      if (options.onLoad) {
        options.onLoad();
      }
    };

    script.onerror = (e) => {
      setError(true);
      if (options.onError) {
        options.onError(e);
      }
    };

    document.body.appendChild(script);

    return () => {
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
    };
  }, [src, options.id, options.async, options.defer, options.onLoad, options.onError]);

  return [loaded, error];
};



