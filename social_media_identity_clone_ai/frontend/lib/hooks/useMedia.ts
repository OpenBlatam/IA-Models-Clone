import { useState, useEffect } from 'react';

interface UseMediaOptions {
  matches?: string;
  media?: MediaQueryList;
}

export const useMedia = (query: string | MediaQueryList, options: UseMediaOptions = {}): boolean => {
  const [matches, setMatches] = useState<boolean>(() => {
    if (typeof window === 'undefined') {
      return false;
    }

    if (options.media) {
      return options.media.matches;
    }

    if (typeof query === 'string') {
      return window.matchMedia(query).matches;
    }

    return query.matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const mediaQuery = options.media || (typeof query === 'string' ? window.matchMedia(query) : query);

    const handleChange = (event: MediaQueryListEvent): void => {
      setMatches(event.matches);
    };

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
    } else {
      (mediaQuery as any).addListener(handleChange);
    }

    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', handleChange);
      } else {
        (mediaQuery as any).removeListener(handleChange);
      }
    };
  }, [query, options.media]);

  return matches;
};



