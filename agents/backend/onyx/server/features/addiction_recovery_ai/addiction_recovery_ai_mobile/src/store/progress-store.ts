import { create } from 'zustand';
import { apiService } from '@/services/api';
import type { ProgressResponse, StatsResponse, LogEntryRequest, LogEntryResponse } from '@/types';

interface ProgressState {
  progress: ProgressResponse | null;
  stats: StatsResponse | null;
  isLoading: boolean;
  error: string | null;
}

interface ProgressActions {
  fetchProgress: (userId: string) => Promise<void>;
  fetchStats: (userId: string) => Promise<void>;
  logEntry: (data: LogEntryRequest) => Promise<LogEntryResponse>;
  clearError: () => void;
}

type ProgressStore = ProgressState & ProgressActions;

const initialState: ProgressState = {
  progress: null,
  stats: null,
  isLoading: false,
  error: null,
};

export const useProgressStore = create<ProgressStore>((set) => ({
  ...initialState,

  fetchProgress: async (userId: string) => {
    try {
      set({ isLoading: true, error: null });
      const progress = await apiService.getProgress(userId);
      set({ progress, isLoading: false });
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Error al cargar progreso';
      set({
        error: errorMessage,
        isLoading: false,
      });
    }
  },

  fetchStats: async (userId: string) => {
    try {
      set({ isLoading: true, error: null });
      const stats = await apiService.getStats(userId);
      set({ stats, isLoading: false });
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Error al cargar estadísticas';
      set({
        error: errorMessage,
        isLoading: false,
      });
    }
  },

  logEntry: async (data: LogEntryRequest) => {
    try {
      set({ isLoading: true, error: null });
      const entry = await apiService.logEntry(data);
      // Refresh progress after logging
      if (data.user_id) {
        await apiService.getProgress(data.user_id).then((progress) => {
          set({ progress });
        });
      }
      set({ isLoading: false });
      return entry;
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Error al registrar entrada';
      set({
        error: errorMessage,
        isLoading: false,
      });
      throw error;
    }
  },

  clearError: () => set({ error: null }),
}));

