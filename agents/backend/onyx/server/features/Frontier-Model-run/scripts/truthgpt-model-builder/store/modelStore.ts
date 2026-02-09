import { create } from 'zustand'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface Model {
  id: string
  name: string
  description: string
  status: 'creating' | 'completed' | 'failed'
  githubUrl: string | null
  createdAt: Date
  progress?: number
  currentStep?: string
  spec?: {
    type: string
    architecture: string
  }
}

interface ModelStore {
  messages: Message[]
  currentModel: Model | null
  addMessage: (message: Message) => void
  setCurrentModel: (model: Model | null) => void
  clearMessages: () => void
}

export const useModelStore = create<ModelStore>((set) => ({
  messages: [],
  currentModel: null,
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  setCurrentModel: (model) => set({ currentModel: model }),
  clearMessages: () => set({ messages: [], currentModel: null }),
}))

