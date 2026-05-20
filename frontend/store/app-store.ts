import { create } from 'zustand';
import type { TaskStatus, StatsResponse, HealthResponse } from '@/types/api';

interface AppState {
  // Health & Stats
  health: HealthResponse | null;
  stats: StatsResponse | null;
  isConnected: boolean;
  
  // Current task
  currentTask: TaskStatus | null;
  currentTaskId: string | null;
  
  // UI State
  activeView: 'dashboard' | 'generate' | 'tasks' | 'documents' | 'stats' | 'favorites';
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setHealth: (health: HealthResponse) => void;
  setStats: (stats: StatsResponse) => void;
  setConnected: (connected: boolean) => void;
  setCurrentTask: (task: TaskStatus | null) => void;
  setCurrentTaskId: (taskId: string | null) => void;
  setActiveView: (view: 'dashboard' | 'generate' | 'tasks' | 'documents' | 'stats' | 'favorites') => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  health: null,
  stats: null,
  isConnected: false,
  currentTask: null,
  currentTaskId: null,
  activeView: 'dashboard' as const,
  isLoading: false,
  error: null,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,
  
  setHealth: (health) => set({ health }),
  setStats: (stats) => set({ stats }),
  setConnected: (connected) => set({ isConnected: connected }),
  setCurrentTask: (task) => set({ currentTask: task }),
  setCurrentTaskId: (taskId) => set({ currentTaskId: taskId }),
  setActiveView: (view) => set({ activeView: view }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  reset: () => set(initialState),
}));

