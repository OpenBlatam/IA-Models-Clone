import { useState, useEffect } from 'react';

interface UseImageOptions {
  crossOrigin?: string;
  onLoad?: (img: HTMLImageElement) => void;
  onError?: (error: Event) => void;
}

export const useImage = (src: string, options: UseImageOptions = {}): {
  image: HTMLImageElement | null;
  loading: boolean;
  error: Event | null;
} => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Event | null>(null);

  useEffect(() => {
    if (!src) {
      return;
    }

    const img = new Image();
    setLoading(true);
    setError(null);

    if (options.crossOrigin) {
      img.crossOrigin = options.crossOrigin;
    }

    img.onload = () => {
      setImage(img);
      setLoading(false);
      if (options.onLoad) {
        options.onLoad(img);
      }
    };

    img.onerror = (e) => {
      setError(e);
      setLoading(false);
      if (options.onError) {
        options.onError(e);
      }
    };

    img.src = src;

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [src, options.crossOrigin, options.onLoad, options.onError]);

  return { image, loading, error };
};



