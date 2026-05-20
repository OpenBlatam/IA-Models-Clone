/**
 * Custom hook for message management
 * Handles favorites, pins, archives, tags, notes, reactions, and bookmarks
 */

import { useState, useCallback, useEffect } from 'react'

export interface MessageManagementState {
  favoriteMessages: Set<string>
  pinnedMessages: Set<string>
  archivedMessages: Set<string>
  selectedMessages: Set<string>
  messageTags: Map<string, string[]>
  messageNotes: Map<string, string>
  messageReactions: Map<string, string[]>
  messageBookmarks: Map<string, { name: string, category: string, tags: string[] }>
  editingNote: string | null
  availableTags: string[]
  bookmarkCategories: string[]
}

export interface MessageManagementActions {
  toggleFavorite: (messageId: string) => void
  togglePin: (messageId: string) => void
  toggleArchive: (messageId: string) => void
  toggleSelection: (messageId: string) => void
  selectAll: () => void
  deselectAll: () => void
  addTag: (messageId: string, tag: string) => void
  removeTag: (messageId: string, tag: string) => void
  setTags: (messageId: string, tags: string[]) => void
  addNote: (messageId: string, note: string) => void
  removeNote: (messageId: string) => void
  startEditingNote: (messageId: string) => void
  stopEditingNote: () => void
  addReaction: (messageId: string, reaction: string) => void
  removeReaction: (messageId: string, reaction: string) => void
  addBookmark: (messageId: string, name: string, category?: string, tags?: string[]) => void
  removeBookmark: (messageId: string) => void
  addAvailableTag: (tag: string) => void
  addBookmarkCategory: (category: string) => void
}

const STORAGE_KEYS = {
  favorites: 'bulk-chat-favorites',
  pins: 'bulk-chat-pins',
  archives: 'bulk-chat-archives',
  tags: 'bulk-chat-tags',
  notes: 'bulk-chat-notes',
  bookmarks: 'bulk-chat-bookmarks',
  availableTags: 'bulk-chat-available-tags',
  bookmarkCategories: 'bulk-chat-bookmark-categories',
}

export function useMessageManagement(
  messageIds: string[] = []
): MessageManagementState & MessageManagementActions {
  // State
  const [favoriteMessages, setFavoriteMessages] = useState<Set<string>>(new Set())
  const [pinnedMessages, setPinnedMessages] = useState<Set<string>>(new Set())
  const [archivedMessages, setArchivedMessages] = useState<Set<string>>(new Set())
  const [selectedMessages, setSelectedMessages] = useState<Set<string>>(new Set())
  const [messageTags, setMessageTags] = useState<Map<string, string[]>>(new Map())
  const [messageNotes, setMessageNotes] = useState<Map<string, string>>(new Map())
  const [messageReactions, setMessageReactions] = useState<Map<string, string[]>>(new Map())
  const [messageBookmarks, setMessageBookmarks] = useState<Map<string, { name: string, category: string, tags: string[] }>>(new Map())
  const [editingNote, setEditingNote] = useState<string | null>(null)
  const [availableTags, setAvailableTags] = useState<string[]>(['importante', 'pregunta', 'respuesta', 'código', 'documentación'])
  const [bookmarkCategories, setBookmarkCategories] = useState<string[]>(['favoritos', 'importante', 'referencia'])

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const savedFavorites = localStorage.getItem(STORAGE_KEYS.favorites)
      if (savedFavorites) {
        setFavoriteMessages(new Set(JSON.parse(savedFavorites)))
      }

      const savedPins = localStorage.getItem(STORAGE_KEYS.pins)
      if (savedPins) {
        setPinnedMessages(new Set(JSON.parse(savedPins)))
      }

      const savedArchives = localStorage.getItem(STORAGE_KEYS.archives)
      if (savedArchives) {
        setArchivedMessages(new Set(JSON.parse(savedArchives)))
      }

      const savedTags = localStorage.getItem(STORAGE_KEYS.tags)
      if (savedTags) {
        setMessageTags(new Map(JSON.parse(savedTags)))
      }

      const savedNotes = localStorage.getItem(STORAGE_KEYS.notes)
      if (savedNotes) {
        setMessageNotes(new Map(JSON.parse(savedNotes)))
      }

      const savedBookmarks = localStorage.getItem(STORAGE_KEYS.bookmarks)
      if (savedBookmarks) {
        setMessageBookmarks(new Map(JSON.parse(savedBookmarks)))
      }

      const savedAvailableTags = localStorage.getItem(STORAGE_KEYS.availableTags)
      if (savedAvailableTags) {
        setAvailableTags(JSON.parse(savedAvailableTags))
      }

      const savedCategories = localStorage.getItem(STORAGE_KEYS.bookmarkCategories)
      if (savedCategories) {
        setBookmarkCategories(JSON.parse(savedCategories))
      }
    } catch (error) {
      console.error('Error loading message management data:', error)
    }
  }, [])

  // Persist favorites
  useEffect(() => {
    try {
      if (favoriteMessages.size > 0) {
        localStorage.setItem(STORAGE_KEYS.favorites, JSON.stringify(Array.from(favoriteMessages)))
      } else {
        localStorage.removeItem(STORAGE_KEYS.favorites)
      }
    } catch (error) {
      console.error('Error saving favorites:', error)
    }
  }, [favoriteMessages])

  // Persist pins
  useEffect(() => {
    try {
      if (pinnedMessages.size > 0) {
        localStorage.setItem(STORAGE_KEYS.pins, JSON.stringify(Array.from(pinnedMessages)))
      } else {
        localStorage.removeItem(STORAGE_KEYS.pins)
      }
    } catch (error) {
      console.error('Error saving pins:', error)
    }
  }, [pinnedMessages])

  // Persist archives
  useEffect(() => {
    try {
      if (archivedMessages.size > 0) {
        localStorage.setItem(STORAGE_KEYS.archives, JSON.stringify(Array.from(archivedMessages)))
      } else {
        localStorage.removeItem(STORAGE_KEYS.archives)
      }
    } catch (error) {
      console.error('Error saving archives:', error)
    }
  }, [archivedMessages])

  // Persist tags
  useEffect(() => {
    try {
      if (messageTags.size > 0) {
        localStorage.setItem(STORAGE_KEYS.tags, JSON.stringify(Array.from(messageTags.entries())))
      } else {
        localStorage.removeItem(STORAGE_KEYS.tags)
      }
    } catch (error) {
      console.error('Error saving tags:', error)
    }
  }, [messageTags])

  // Persist notes
  useEffect(() => {
    try {
      if (messageNotes.size > 0) {
        localStorage.setItem(STORAGE_KEYS.notes, JSON.stringify(Array.from(messageNotes.entries())))
      } else {
        localStorage.removeItem(STORAGE_KEYS.notes)
      }
    } catch (error) {
      console.error('Error saving notes:', error)
    }
  }, [messageNotes])

  // Persist bookmarks
  useEffect(() => {
    try {
      if (messageBookmarks.size > 0) {
        localStorage.setItem(STORAGE_KEYS.bookmarks, JSON.stringify(Array.from(messageBookmarks.entries())))
      } else {
        localStorage.removeItem(STORAGE_KEYS.bookmarks)
      }
    } catch (error) {
      console.error('Error saving bookmarks:', error)
    }
  }, [messageBookmarks])

  // Actions
  const toggleFavorite = useCallback((messageId: string) => {
    setFavoriteMessages(prev => {
      const next = new Set(prev)
      if (next.has(messageId)) {
        next.delete(messageId)
      } else {
        next.add(messageId)
      }
      return next
    })
  }, [])

  const togglePin = useCallback((messageId: string) => {
    setPinnedMessages(prev => {
      const next = new Set(prev)
      if (next.has(messageId)) {
        next.delete(messageId)
      } else {
        next.add(messageId)
      }
      return next
    })
  }, [])

  const toggleArchive = useCallback((messageId: string) => {
    setArchivedMessages(prev => {
      const next = new Set(prev)
      if (next.has(messageId)) {
        next.delete(messageId)
      } else {
        next.add(messageId)
      }
      return next
    })
  }, [])

  const toggleSelection = useCallback((messageId: string) => {
    setSelectedMessages(prev => {
      const next = new Set(prev)
      if (next.has(messageId)) {
        next.delete(messageId)
      } else {
        next.add(messageId)
      }
      return next
    })
  }, [])

  const selectAll = useCallback(() => {
    setSelectedMessages(new Set(messageIds))
  }, [messageIds])

  const deselectAll = useCallback(() => {
    setSelectedMessages(new Set())
  }, [])

  const addTag = useCallback((messageId: string, tag: string) => {
    setMessageTags(prev => {
      const next = new Map(prev)
      const currentTags = next.get(messageId) || []
      if (!currentTags.includes(tag)) {
        next.set(messageId, [...currentTags, tag])
      }
      return next
    })
  }, [])

  const removeTag = useCallback((messageId: string, tag: string) => {
    setMessageTags(prev => {
      const next = new Map(prev)
      const currentTags = next.get(messageId) || []
      next.set(messageId, currentTags.filter(t => t !== tag))
      return next
    })
  }, [])

  const setTags = useCallback((messageId: string, tags: string[]) => {
    setMessageTags(prev => {
      const next = new Map(prev)
      next.set(messageId, tags)
      return next
    })
  }, [])

  const addNote = useCallback((messageId: string, note: string) => {
    setMessageNotes(prev => {
      const next = new Map(prev)
      next.set(messageId, note)
      return next
    })
    setEditingNote(null)
  }, [])

  const removeNote = useCallback((messageId: string) => {
    setMessageNotes(prev => {
      const next = new Map(prev)
      next.delete(messageId)
      return next
    })
  }, [])

  const startEditingNote = useCallback((messageId: string) => {
    setEditingNote(messageId)
  }, [])

  const stopEditingNote = useCallback(() => {
    setEditingNote(null)
  }, [])

  const addReaction = useCallback((messageId: string, reaction: string) => {
    setMessageReactions(prev => {
      const next = new Map(prev)
      const currentReactions = next.get(messageId) || []
      if (!currentReactions.includes(reaction)) {
        next.set(messageId, [...currentReactions, reaction])
      }
      return next
    })
  }, [])

  const removeReaction = useCallback((messageId: string, reaction: string) => {
    setMessageReactions(prev => {
      const next = new Map(prev)
      const currentReactions = next.get(messageId) || []
      next.set(messageId, currentReactions.filter(r => r !== reaction))
      return next
    })
  }, [])

  const addBookmark = useCallback((messageId: string, name: string, category: string = 'favoritos', tags: string[] = []) => {
    setMessageBookmarks(prev => {
      const next = new Map(prev)
      next.set(messageId, { name, category, tags })
      return next
    })
  }, [])

  const removeBookmark = useCallback((messageId: string) => {
    setMessageBookmarks(prev => {
      const next = new Map(prev)
      next.delete(messageId)
      return next
    })
  }, [])

  const addAvailableTag = useCallback((tag: string) => {
    setAvailableTags(prev => {
      if (!prev.includes(tag)) {
        return [...prev, tag]
      }
      return prev
    })
  }, [])

  const addBookmarkCategory = useCallback((category: string) => {
    setBookmarkCategories(prev => {
      if (!prev.includes(category)) {
        return [...prev, category]
      }
      return prev
    })
  }, [])

  return {
    // State
    favoriteMessages,
    pinnedMessages,
    archivedMessages,
    selectedMessages,
    messageTags,
    messageNotes,
    messageReactions,
    messageBookmarks,
    editingNote,
    availableTags,
    bookmarkCategories,
    // Actions
    toggleFavorite,
    togglePin,
    toggleArchive,
    toggleSelection,
    selectAll,
    deselectAll,
    addTag,
    removeTag,
    setTags,
    addNote,
    removeNote,
    startEditingNote,
    stopEditingNote,
    addReaction,
    removeReaction,
    addBookmark,
    removeBookmark,
    addAvailableTag,
    addBookmarkCategory,
  }
}




