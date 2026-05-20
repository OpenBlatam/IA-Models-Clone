import { useState, useCallback } from 'react'

export interface OrganizationState {
  groupingMode: 'none' | 'time' | 'topic' | 'role'
  threadParent: string | null
  availableTags: string[]
  editingNote: string | null
  readingSpeed: 'slow' | 'normal' | 'fast'
  editingMessage: string | null
}

export interface OrganizationActions {
  setGroupingMode: (mode: 'none' | 'time' | 'topic' | 'role') => void
  setThreadParent: (parent: string | null) => void
  setAvailableTags: (tags: string[]) => void
  addTag: (tag: string) => void
  removeTag: (tag: string) => void
  setEditingNote: (note: string | null) => void
  setReadingSpeed: (speed: 'slow' | 'normal' | 'fast') => void
  setEditingMessage: (messageId: string | null) => void
}

export function useOrganizationState() {
  const [groupingMode, setGroupingMode] = useState<'none' | 'time' | 'topic' | 'role'>('none')
  const [threadParent, setThreadParent] = useState<string | null>(null)
  const [availableTags, setAvailableTags] = useState<string[]>(['importante', 'pregunta', 'respuesta', 'código', 'documentación'])
  const [editingNote, setEditingNote] = useState<string | null>(null)
  const [readingSpeed, setReadingSpeed] = useState<'slow' | 'normal' | 'fast'>('normal')
  const [editingMessage, setEditingMessage] = useState<string | null>(null)

  const state: OrganizationState = {
    groupingMode,
    threadParent,
    availableTags,
    editingNote,
    readingSpeed,
    editingMessage
  }

  const actions: OrganizationActions = {
    setGroupingMode: useCallback((mode: 'none' | 'time' | 'topic' | 'role') => setGroupingMode(mode), []),
    setThreadParent: useCallback((parent: string | null) => setThreadParent(parent), []),
    setAvailableTags: useCallback((tags: string[]) => setAvailableTags(tags), []),
    addTag: useCallback((tag: string) => {
      setAvailableTags(prev => prev.includes(tag) ? prev : [...prev, tag])
    }, []),
    removeTag: useCallback((tag: string) => {
      setAvailableTags(prev => prev.filter(t => t !== tag))
    }, []),
    setEditingNote: useCallback((note: string | null) => setEditingNote(note), []),
    setReadingSpeed: useCallback((speed: 'slow' | 'normal' | 'fast') => setReadingSpeed(speed), []),
    setEditingMessage: useCallback((messageId: string | null) => setEditingMessage(messageId), [])
  }

  return { state, actions }
}



