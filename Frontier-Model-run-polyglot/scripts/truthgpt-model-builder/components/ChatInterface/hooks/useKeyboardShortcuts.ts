/**
 * Custom hook for keyboard shortcuts
 * Handles keyboard shortcuts and command palette
 */

import { useEffect, useCallback, useState } from 'react'

export interface KeyboardShortcut {
  key: string
  ctrl?: boolean
  shift?: boolean
  alt?: boolean
  meta?: boolean
  handler: () => void
  description?: string
}

export interface KeyboardShortcutsState {
  shortcuts: Map<string, KeyboardShortcut>
  showCommandPalette: boolean
  commandHistory: string[]
}

export interface KeyboardShortcutsActions {
  registerShortcut: (id: string, shortcut: KeyboardShortcut) => void
  unregisterShortcut: (id: string) => void
  setShowCommandPalette: (show: boolean) => void
  addToCommandHistory: (command: string) => void
  clearCommandHistory: () => void
}

export function useKeyboardShortcuts(
  enabled: boolean = true
): KeyboardShortcutsState & KeyboardShortcutsActions {
  const [shortcuts, setShortcuts] = useState<Map<string, KeyboardShortcut>>(new Map())
  const [showCommandPalette, setShowCommandPalette] = useState(false)
  const [commandHistory, setCommandHistory] = useState<string[]>([])

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabled) return

    // Check for command palette (Ctrl/Cmd + K)
    if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
      event.preventDefault()
      setShowCommandPalette(prev => !prev)
      return
    }

    // Check registered shortcuts
    for (const [id, shortcut] of shortcuts.entries()) {
      const keyMatch = shortcut.key.toLowerCase() === event.key.toLowerCase()
      const ctrlMatch = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey
      const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey
      const altMatch = shortcut.alt ? event.altKey : !event.altKey
      const metaMatch = shortcut.meta ? event.metaKey : !event.metaKey

      if (keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch) {
        event.preventDefault()
        shortcut.handler()
        break
      }
    }
  }, [enabled, shortcuts])

  useEffect(() => {
    if (!enabled) return

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [enabled, handleKeyDown])

  const registerShortcut = useCallback((id: string, shortcut: KeyboardShortcut) => {
    setShortcuts(prev => {
      const next = new Map(prev)
      next.set(id, shortcut)
      return next
    })
  }, [])

  const unregisterShortcut = useCallback((id: string) => {
    setShortcuts(prev => {
      const next = new Map(prev)
      next.delete(id)
      return next
    })
  }, [])

  const addToCommandHistory = useCallback((command: string) => {
    setCommandHistory(prev => {
      const newHistory = [command, ...prev.filter(c => c !== command)]
      return newHistory.slice(0, 50) // Keep last 50 commands
    })
  }, [])

  const clearCommandHistory = useCallback(() => {
    setCommandHistory([])
  }, [])

  return {
    shortcuts,
    showCommandPalette,
    commandHistory,
    registerShortcut,
    unregisterShortcut,
    setShowCommandPalette,
    addToCommandHistory,
    clearCommandHistory,
  }
}




