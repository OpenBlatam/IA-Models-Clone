import type { User } from './api';

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface VideoState {
  currentVideo: string | null;
  videos: Record<string, unknown>;
  isLoading: boolean;
}

export interface AppState {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  apiBaseUrl: string;
}


