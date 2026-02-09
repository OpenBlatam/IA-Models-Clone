'use client';

import { useEffect } from 'react';
import { useAppStore } from '@/lib/store';

export const useTheme = () => {
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);
  const toggleTheme = useAppStore((state) => state.toggleTheme);
  const getEffectiveTheme = useAppStore((state) => state.getEffectiveTheme);

  useEffect(() => {
    const root = window.document.documentElement;
    const effectiveTheme = getEffectiveTheme();

    root.classList.remove('light', 'dark');
    root.classList.add(effectiveTheme);

    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleChange = () => {
        const newTheme = mediaQuery.matches ? 'dark' : 'light';
        root.classList.remove('light', 'dark');
        root.classList.add(newTheme);
      };

      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [theme, getEffectiveTheme]);

  return {
    theme,
    setTheme,
    toggleTheme,
    effectiveTheme: getEffectiveTheme(),
  };
};



