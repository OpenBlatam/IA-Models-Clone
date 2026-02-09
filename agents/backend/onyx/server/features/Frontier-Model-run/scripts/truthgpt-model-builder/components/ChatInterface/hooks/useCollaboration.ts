/**
 * Custom hook for collaboration features
 * Handles collaboration mode, collaborators, sharing, and real-time features
 */

import { useState, useCallback, useEffect } from 'react'

export interface CollaborationState {
  collaborationMode: boolean
  collaborators: string[]
  messageSharing: Map<string, string[]>
  showShareMenu: boolean
  shareTarget: string | null
  messageComments: Map<string, { author: string, content: string, timestamp: number }[]>
  showComments: boolean
  messageVotes: Map<string, { up: number, down: number }>
  messageRatings: Map<string, number>
  realTimeSync: boolean
}

export interface CollaborationActions {
  setCollaborationMode: (enabled: boolean) => void
  addCollaborator: (collaboratorId: string) => void
  removeCollaborator: (collaboratorId: string) => void
  shareMessage: (messageId: string, collaboratorIds: string[]) => void
  unshareMessage: (messageId: string, collaboratorId: string) => void
  setShowShareMenu: (show: boolean) => void
  setShareTarget: (target: string | null) => void
  addComment: (messageId: string, author: string, content: string) => void
  removeComment: (messageId: string, commentIndex: number) => void
  setShowComments: (show: boolean) => void
  voteMessage: (messageId: string, vote: 'up' | 'down') => void
  rateMessage: (messageId: string, rating: number) => void
  setRealTimeSync: (enabled: boolean) => void
  exportCollaborationData: () => string
  importCollaborationData: (dataJson: string) => void
}

const STORAGE_KEY = 'chat-collaboration-settings'

export function useCollaboration(): CollaborationState & CollaborationActions {
  const [collaborationMode, setCollaborationMode] = useState(false)
  const [collaborators, setCollaborators] = useState<string[]>([])
  const [messageSharing, setMessageSharing] = useState<Map<string, string[]>>(new Map())
  const [showShareMenu, setShowShareMenu] = useState(false)
  const [shareTarget, setShareTarget] = useState<string | null>(null)
  const [messageComments, setMessageComments] = useState<Map<string, { author: string, content: string, timestamp: number }[]>>(new Map())
  const [showComments, setShowComments] = useState(false)
  const [messageVotes, setMessageVotes] = useState<Map<string, { up: number, down: number }>>(new Map())
  const [messageRatings, setMessageRatings] = useState<Map<string, number>>(new Map())
  const [realTimeSync, setRealTimeSync] = useState(false)

  // Load from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        setCollaborationMode(parsed.collaborationMode || false)
        setCollaborators(parsed.collaborators || [])
        setMessageSharing(new Map(parsed.messageSharing || []))
        setMessageComments(new Map(parsed.messageComments || []))
        setMessageVotes(new Map(parsed.messageVotes || []))
        setMessageRatings(new Map(parsed.messageRatings || []))
        setRealTimeSync(parsed.realTimeSync || false)
      }
    } catch (error) {
      console.error('Error loading collaboration settings:', error)
    }
  }, [])

  // Save to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        collaborationMode,
        collaborators,
        messageSharing: Array.from(messageSharing.entries()),
        messageComments: Array.from(messageComments.entries()),
        messageVotes: Array.from(messageVotes.entries()),
        messageRatings: Array.from(messageRatings.entries()),
        realTimeSync,
      }))
    } catch (error) {
      console.error('Error saving collaboration settings:', error)
    }
  }, [collaborationMode, collaborators, messageSharing, messageComments, messageVotes, messageRatings, realTimeSync])

  const addCollaborator = useCallback((collaboratorId: string) => {
    setCollaborators(prev => {
      if (!prev.includes(collaboratorId)) {
        return [...prev, collaboratorId]
      }
      return prev
    })
  }, [])

  const removeCollaborator = useCallback((collaboratorId: string) => {
    setCollaborators(prev => prev.filter(id => id !== collaboratorId))
  }, [])

  const shareMessage = useCallback((messageId: string, collaboratorIds: string[]) => {
    setMessageSharing(prev => {
      const next = new Map(prev)
      next.set(messageId, collaboratorIds)
      return next
    })
  }, [])

  const unshareMessage = useCallback((messageId: string, collaboratorId: string) => {
    setMessageSharing(prev => {
      const next = new Map(prev)
      const shared = next.get(messageId) || []
      next.set(messageId, shared.filter(id => id !== collaboratorId))
      return next
    })
  }, [])

  const addComment = useCallback((messageId: string, author: string, content: string) => {
    setMessageComments(prev => {
      const next = new Map(prev)
      const comments = next.get(messageId) || []
      next.set(messageId, [
        ...comments,
        {
          author,
          content,
          timestamp: Date.now(),
        },
      ])
      return next
    })
  }, [])

  const removeComment = useCallback((messageId: string, commentIndex: number) => {
    setMessageComments(prev => {
      const next = new Map(prev)
      const comments = next.get(messageId) || []
      if (comments.length > commentIndex) {
        next.set(messageId, comments.filter((_, i) => i !== commentIndex))
      }
      return next
    })
  }, [])

  const voteMessage = useCallback((messageId: string, vote: 'up' | 'down') => {
    setMessageVotes(prev => {
      const next = new Map(prev)
      const current = next.get(messageId) || { up: 0, down: 0 }
      next.set(messageId, {
        ...current,
        [vote]: current[vote] + 1,
      })
      return next
    })
  }, [])

  const rateMessage = useCallback((messageId: string, rating: number) => {
    if (rating < 1 || rating > 5) return

    setMessageRatings(prev => {
      const next = new Map(prev)
      next.set(messageId, rating)
      return next
    })
  }, [])

  const exportCollaborationData = useCallback((): string => {
    return JSON.stringify({
      collaborators,
      messageSharing: Array.from(messageSharing.entries()),
      messageComments: Array.from(messageComments.entries()),
      messageVotes: Array.from(messageVotes.entries()),
      messageRatings: Array.from(messageRatings.entries()),
    }, null, 2)
  }, [collaborators, messageSharing, messageComments, messageVotes, messageRatings])

  const importCollaborationData = useCallback((dataJson: string) => {
    try {
      const parsed = JSON.parse(dataJson)
      if (parsed.collaborators) setCollaborators(parsed.collaborators)
      if (parsed.messageSharing) setMessageSharing(new Map(parsed.messageSharing))
      if (parsed.messageComments) setMessageComments(new Map(parsed.messageComments))
      if (parsed.messageVotes) setMessageVotes(new Map(parsed.messageVotes))
      if (parsed.messageRatings) setMessageRatings(new Map(parsed.messageRatings))
    } catch (error) {
      console.error('Error importing collaboration data:', error)
      throw new Error('Invalid collaboration data format')
    }
  }, [])

  return {
    // State
    collaborationMode,
    collaborators,
    messageSharing,
    showShareMenu,
    shareTarget,
    messageComments,
    showComments,
    messageVotes,
    messageRatings,
    realTimeSync,
    // Actions
    setCollaborationMode,
    addCollaborator,
    removeCollaborator,
    shareMessage,
    unshareMessage,
    setShowShareMenu,
    setShareTarget,
    addComment,
    removeComment,
    setShowComments,
    voteMessage,
    rateMessage,
    setRealTimeSync,
    exportCollaborationData,
    importCollaborationData,
  }
}




