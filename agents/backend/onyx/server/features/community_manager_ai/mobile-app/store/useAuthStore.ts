import { create } from 'zustand';
import * as SecureStore from 'expo-secure-store';

interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  setToken: (token: string | null) => Promise<void>;
  logout: () => Promise<void>;
  initialize: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  isAuthenticated: false,
  token: null,
  
  setToken: async (token: string | null) => {
    if (token) {
      await SecureStore.setItemAsync('auth_token', token);
      set({ token, isAuthenticated: true });
    } else {
      await SecureStore.deleteItemAsync('auth_token');
      set({ token: null, isAuthenticated: false });
    }
  },
  
  logout: async () => {
    await SecureStore.deleteItemAsync('auth_token');
    set({ token: null, isAuthenticated: false });
  },
  
  initialize: async () => {
    const token = await SecureStore.getItemAsync('auth_token');
    set({ token, isAuthenticated: !!token });
  },
}));


