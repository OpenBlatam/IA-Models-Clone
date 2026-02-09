/**
 * Hook modular para acciones de mensajes
 * Centraliza todas las funciones relacionadas con manipulación de mensajes
 */

import { useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { MessageService } from '../services/MessageService'
import type { MessageActionsState } from '../types/state.types'

// MessageActionsState ahora se importa de types/state.types

export interface MessageActions {
  addAttachment: (messageId: string, type: string, url: string, name: string) => void
  addLink: (messageId: string, url: string, title: string, description: string) => void
  addNotification: (type: string, title: string, body: string) => void
  addBookmark: (messageId: string, name: string, category: string, tags?: string[]) => void
  highlightMessage: (messageId: string, color: string, note?: string) => void
  addAnnotation: (messageId: string, annotation: string) => void
  removeAttachment: (messageId: string, index: number) => void
  removeLink: (messageId: string, index: number) => void
  removeBookmark: (messageId: string) => void
  removeHighlight: (messageId: string) => void
  removeAnnotation: (messageId: string) => void
}

export function useMessageActions(
  state: MessageActionsState,
  setState: React.Dispatch<React.SetStateAction<MessageActionsState>>
): MessageActions {
  
  const addAttachment = useCallback((messageId: string, type: string, url: string, name: string) => {
    setState(prev => ({
      ...prev,
      messageAttachments: MessageService.addAttachment(
        prev.messageAttachments,
        messageId,
        { type, url, name }
      )
    }))
    toast.success('Adjunto agregado', { icon: '📎' })
  }, [setState])

  const addLink = useCallback((messageId: string, url: string, title: string, description: string) => {
    setState(prev => ({
      ...prev,
      messageLinks: MessageService.addLink(
        prev.messageLinks,
        messageId,
        { url, title, description }
      )
    }))
    toast.success('Enlace agregado', { icon: '🔗' })
  }, [setState])

  const addNotification = useCallback((type: string, title: string, body: string) => {
    setState(prev => ({
      ...prev,
      messageNotifications: MessageService.createNotification(
        prev.messageNotifications,
        { type, title, body, read: false, timestamp: Date.now() }
      )
    }))
    toast.info(title, { icon: '🔔' })
  }, [setState])

  const addBookmark = useCallback((messageId: string, name: string, category: string, tags: string[] = []) => {
    setState(prev => ({
      ...prev,
      messageBookmarks: MessageService.addBookmark(
        prev.messageBookmarks,
        messageId,
        { name, category, tags }
      )
    }))
    toast.success('Marcador agregado', { icon: '🔖' })
  }, [setState])

  const highlightMessage = useCallback((messageId: string, color: string, note?: string) => {
    setState(prev => ({
      ...prev,
      messageHighlights: MessageService.highlightMessage(
        prev.messageHighlights,
        messageId,
        { color, note }
      )
    }))
    toast.success('Mensaje resaltado', { icon: '✨' })
  }, [setState])

  const addAnnotation = useCallback((messageId: string, annotation: string) => {
    setState(prev => ({
      ...prev,
      messageAnnotations: MessageService.addAnnotation(
        prev.messageAnnotations,
        messageId,
        { annotation, timestamp: Date.now() }
      )
    }))
    toast.success('Anotación agregada', { icon: '📝' })
  }, [setState])

  const removeAttachment = useCallback((messageId: string, index: number) => {
    setState(prev => ({
      ...prev,
      messageAttachments: MessageService.removeAttachment(
        prev.messageAttachments,
        messageId,
        index
      )
    }))
    toast.success('Adjunto eliminado', { icon: '🗑️' })
  }, [setState])

  const removeLink = useCallback((messageId: string, index: number) => {
    setState(prev => ({
      ...prev,
      messageLinks: MessageService.removeLink(
        prev.messageLinks,
        messageId,
        index
      )
    }))
    toast.success('Enlace eliminado', { icon: '🗑️' })
  }, [setState])

  const removeBookmark = useCallback((messageId: string) => {
    setState(prev => ({
      ...prev,
      messageBookmarks: MessageService.removeBookmark(prev.messageBookmarks, messageId)
    }))
    toast.success('Marcador eliminado', { icon: '🗑️' })
  }, [setState])

  const removeHighlight = useCallback((messageId: string) => {
    setState(prev => ({
      ...prev,
      messageHighlights: MessageService.removeHighlight(prev.messageHighlights, messageId)
    }))
    toast.success('Resaltado eliminado', { icon: '🗑️' })
  }, [setState])

  const removeAnnotation = useCallback((messageId: string) => {
    setState(prev => ({
      ...prev,
      messageAnnotations: MessageService.removeAnnotation(prev.messageAnnotations, messageId)
    }))
    toast.success('Anotación eliminada', { icon: '🗑️' })
  }, [setState])

  return {
    addAttachment,
    addLink,
    addNotification,
    addBookmark,
    highlightMessage,
    addAnnotation,
    removeAttachment,
    removeLink,
    removeBookmark,
    removeHighlight,
    removeAnnotation
  }
}

