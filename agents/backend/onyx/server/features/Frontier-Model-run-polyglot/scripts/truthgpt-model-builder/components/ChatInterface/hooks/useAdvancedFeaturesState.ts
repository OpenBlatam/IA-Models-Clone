import { useState, useCallback } from 'react'

export interface AdvancedFeaturesState {
  aiInsights: boolean
  readingProgress: number
  syncProvider: 'none' | 'localStorage' | 'indexedDB'
  achievements: Set<string>
  activePlugins: string[]
  collaborators: string[]
  focusTimer: number
  focusGoal: number
  smartSuggestions: any[]
}

export interface AdvancedFeaturesActions {
  setAiInsights: (enabled: boolean) => void
  setReadingProgress: (progress: number) => void
  setSyncProvider: (provider: 'none' | 'localStorage' | 'indexedDB') => void
  addAchievement: (achievement: string) => void
  removeAchievement: (achievement: string) => void
  setActivePlugins: (plugins: string[]) => void
  addPlugin: (plugin: string) => void
  removePlugin: (plugin: string) => void
  setCollaborators: (collaborators: string[]) => void
  addCollaborator: (collaborator: string) => void
  removeCollaborator: (collaborator: string) => void
  setFocusTimer: (timer: number) => void
  setFocusGoal: (goal: number) => void
  setSmartSuggestions: (suggestions: any[]) => void
}

export function useAdvancedFeaturesState() {
  const [aiInsights, setAiInsights] = useState(false)
  const [readingProgress, setReadingProgress] = useState(0)
  const [syncProvider, setSyncProvider] = useState<'none' | 'localStorage' | 'indexedDB'>('localStorage')
  const [achievements, setAchievements] = useState<Set<string>>(new Set())
  const [activePlugins, setActivePlugins] = useState<string[]>([])
  const [collaborators, setCollaborators] = useState<string[]>([])
  const [focusTimer, setFocusTimer] = useState(0)
  const [focusGoal, setFocusGoal] = useState(25) // Pomodoro default
  const [smartSuggestions, setSmartSuggestions] = useState<any[]>([])

  const state: AdvancedFeaturesState = {
    aiInsights,
    readingProgress,
    syncProvider,
    achievements,
    activePlugins,
    collaborators,
    focusTimer,
    focusGoal,
    smartSuggestions
  }

  const actions: AdvancedFeaturesActions = {
    setAiInsights: useCallback((enabled: boolean) => setAiInsights(enabled), []),
    setReadingProgress: useCallback((progress: number) => setReadingProgress(progress), []),
    setSyncProvider: useCallback((provider: 'none' | 'localStorage' | 'indexedDB') => setSyncProvider(provider), []),
    addAchievement: useCallback((achievement: string) => {
      setAchievements(prev => new Set([...prev, achievement]))
    }, []),
    removeAchievement: useCallback((achievement: string) => {
      setAchievements(prev => {
        const newSet = new Set(prev)
        newSet.delete(achievement)
        return newSet
      })
    }, []),
    setActivePlugins: useCallback((plugins: string[]) => setActivePlugins(plugins), []),
    addPlugin: useCallback((plugin: string) => {
      setActivePlugins(prev => prev.includes(plugin) ? prev : [...prev, plugin])
    }, []),
    removePlugin: useCallback((plugin: string) => {
      setActivePlugins(prev => prev.filter(p => p !== plugin))
    }, []),
    setCollaborators: useCallback((collaborators: string[]) => setCollaborators(collaborators), []),
    addCollaborator: useCallback((collaborator: string) => {
      setCollaborators(prev => prev.includes(collaborator) ? prev : [...prev, collaborator])
    }, []),
    removeCollaborator: useCallback((collaborator: string) => {
      setCollaborators(prev => prev.filter(c => c !== collaborator))
    }, []),
    setFocusTimer: useCallback((timer: number) => setFocusTimer(timer), []),
    setFocusGoal: useCallback((goal: number) => setFocusGoal(goal), []),
    setSmartSuggestions: useCallback((suggestions: any[]) => setSmartSuggestions(suggestions), [])
  }

  return { state, actions }
}



