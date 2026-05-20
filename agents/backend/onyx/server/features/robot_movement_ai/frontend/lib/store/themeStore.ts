import { create } from 'zustand';

type Theme = 'dark' | 'light' | 'system';

interface ThemeState {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

// Simple localStorage persistence
const getStoredTheme = (): Theme => {
  if (typeof window === 'undefined') return 'dark';
  const stored = localStorage.getItem('robot-theme');
  return (stored as Theme) || 'dark';
};

const setStoredTheme = (theme: Theme) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem('robot-theme', theme);
  }
};

export const useThemeStore = create<ThemeState>((set) => ({
  theme: getStoredTheme(),
  toggleTheme: () =>
    set((state) => {
      const newTheme = state.theme === 'dark' ? 'light' : state.theme === 'light' ? 'system' : 'dark';
      setStoredTheme(newTheme);
      return { theme: newTheme };
    }),
  setTheme: (theme) => {
    setStoredTheme(theme);
    set({ theme });
  },
}));

