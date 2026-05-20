import React, { createContext, useContext, useEffect } from 'react';
import { useAuthStore } from '@/store/auth-store';
import type { User } from '@/types/api';

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const {
    user,
    isAuthenticated,
    isLoading,
    login: storeLogin,
    register: storeRegister,
    logout: storeLogout,
    refreshUser: storeRefreshUser,
  } = useAuthStore();

  const login = async (email: string, password: string) => {
    await storeLogin({ email, password });
  };

  const register = async (email: string, password: string) => {
    await storeRegister({ email, password });
  };

  const logout = async () => {
    await storeLogout();
  };

  const refreshUser = async () => {
    await storeRefreshUser();
  };

  // Auto-refresh user on mount if authenticated
  useEffect(() => {
    if (isAuthenticated && user) {
      refreshUser();
    }
  }, [isAuthenticated]);

  const value: AuthContextValue = {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}


