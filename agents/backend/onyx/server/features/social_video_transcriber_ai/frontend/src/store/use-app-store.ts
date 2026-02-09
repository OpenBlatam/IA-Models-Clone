import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { apiClient } from '@/lib/api-client'

interface AppState {
  apiKey: string | null
  setApiKey: (key: string | null) => void
  isAuthenticated: boolean
  currentJobId: string | null
  setCurrentJobId: (jobId: string | null) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      apiKey: null,
      setApiKey: (key) => {
        apiClient.setApiKey(key)
        set({ apiKey: key, isAuthenticated: !!key })
      },
      isAuthenticated: false,
      currentJobId: null,
      setCurrentJobId: (jobId) => set({ currentJobId: jobId }),
    }),
    {
      name: 'app-storage',
      partialize: (state) => ({ apiKey: state.apiKey }),
    }
  )
)




