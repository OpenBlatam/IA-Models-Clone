import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

type Theme = 'light' | 'dark' | 'system';

interface AppState {
  theme: Theme;
  sidebarOpen: boolean;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  getEffectiveTheme: () => 'light' | 'dark';
}

const getSystemTheme = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        theme: 'system',
        sidebarOpen: true,
        setTheme: (theme) => set({ theme }),
        toggleTheme: () => {
          const current = get().theme;
          if (current === 'light') {
            set({ theme: 'dark' });
          } else if (current === 'dark') {
            set({ theme: 'system' });
          } else {
            set({ theme: 'light' });
          }
        },
        toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
        setSidebarOpen: (open) => set({ sidebarOpen: open }),
        getEffectiveTheme: () => {
          const theme = get().theme;
          if (theme === 'system') {
            return getSystemTheme();
          }
          return theme;
        },
      }),
      {
        name: 'app-storage',
      }
    )
  )
);

