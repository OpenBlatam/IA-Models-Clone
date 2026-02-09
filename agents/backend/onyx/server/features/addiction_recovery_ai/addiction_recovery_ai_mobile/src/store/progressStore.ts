import { create } from 'zustand';
import { apiService } from '@/services/api';
import { ProgressResponse, StatsResponse, LogEntryRequest, LogEntryResponse } from '@/types';

interface ProgressState {
  progress: ProgressResponse | null;
  stats: StatsResponse | null;
  isLoading: boolean;
  error: string | null;
  fetchProgress: (userId: string) => Promise<void>;
  fetchStats: (userId: string) => Promise<void>;
  logEntry: (data: LogEntryRequest) => Promise<LogEntryResponse>;
  clearError: () => void;
}

export const useProgressStore = create<ProgressState>((set) => ({
  progress: null,
  stats: null,
  isLoading: false,
  error: null,

  fetchProgress: async (userId: string) => {
    try {
      set({ isLoading: true, error: null });
      const progress = await apiService.getProgress(userId);
      set({ progress, isLoading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Error al cargar progreso',
        isLoading: false,
      });
    }
  },

  fetchStats: async (userId: string) => {
    try {
      set({ isLoading: true, error: null });
      const stats = await apiService.getStats(userId);
      set({ stats, isLoading: false });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Error al cargar estadísticas',
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
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Error al registrar entrada',
        isLoading: false,
      });
      throw error;
    }
  },

  clearError: () => set({ error: null }),
}));

