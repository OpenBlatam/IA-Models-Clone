import { create } from 'zustand'
import type { User } from '@/types'
import { authService } from '@/services/auth.service'

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, username: string) => Promise<void>
  logout: () => void
  fetchUser: () => Promise<void>
}

export const useAuthStore = create<AuthState>((set) => ({
  user: typeof window !== 'undefined' ? JSON.parse(localStorage.getItem('user') || 'null') : null,
  isAuthenticated: typeof window !== 'undefined' ? !!localStorage.getItem('auth_token') : false,
  isLoading: false,
  login: async (email: string, password: string) => {
    set({ isLoading: true })
    try {
      const response = await authService.login({ email, password })
      if (typeof window !== 'undefined') {
        localStorage.setItem('user', JSON.stringify(response.user))
      }
      set({
        user: response.user,
        isAuthenticated: true,
        isLoading: false,
      })
    } catch (error) {
      set({ isLoading: false })
      throw error
    }
  },
  register: async (email: string, password: string, username: string) => {
    set({ isLoading: true })
    try {
      const response = await authService.register({ email, password, username })
      if (typeof window !== 'undefined') {
        localStorage.setItem('user', JSON.stringify(response.user))
      }
      set({
        user: response.user,
        isAuthenticated: true,
        isLoading: false,
      })
    } catch (error) {
      set({ isLoading: false })
      throw error
    }
  },
  logout: () => {
    authService.logout()
    if (typeof window !== 'undefined') {
      localStorage.removeItem('user')
    }
    set({
      user: null,
      isAuthenticated: false,
    })
  },
  fetchUser: async () => {
    set({ isLoading: true })
    try {
      const user = await authService.getCurrentUser()
      if (typeof window !== 'undefined') {
        localStorage.setItem('user', JSON.stringify(user))
      }
      set({
        user,
        isAuthenticated: true,
        isLoading: false,
      })
    } catch (error) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('user')
      }
      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
      })
    }
  },
}))

