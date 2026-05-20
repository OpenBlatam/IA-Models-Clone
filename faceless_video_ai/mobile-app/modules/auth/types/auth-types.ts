/**
 * Auth module specific types
 */

import type { User } from '@/types/api';

export interface LoginFormData {
  email: string;
  password: string;
}

export interface RegisterFormData {
  email: string;
  password: string;
  confirmPassword: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface AuthGuardProps {
  children: React.ReactNode;
  requireAuth?: boolean;
}

