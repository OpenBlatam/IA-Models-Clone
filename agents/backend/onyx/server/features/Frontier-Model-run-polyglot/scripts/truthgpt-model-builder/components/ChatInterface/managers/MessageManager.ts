/**
 * Manager que coordina múltiples servicios de mensajes
 * Facilita operaciones complejas que requieren múltiples servicios
 */

import { AttachmentService } from '../services/attachments/AttachmentService'
import { LinkService } from '../services/links/LinkService'
import { NotificationService } from '../services/notifications/NotificationService'
import { BookmarkService } from '../services/bookmarks/BookmarkService'
import { HighlightService } from '../services/highlights/HighlightService'
import { AnnotationService } from '../services/annotations/AnnotationService'
import type { MessageActionsState } from '../types/state.types'

export class MessageManager {
  /**
   * Obtiene todas las métricas de un mensaje
   */
  static getMessageMetrics(
    state: MessageActionsState,
    messageId: string
  ): {
    attachments: number
    links: number
    hasBookmark: boolean
    hasHighlight: boolean
    hasAnnotation: boolean
    unreadNotifications: number
  } {
    return {
      attachments: AttachmentService.count(state.messageAttachments, messageId),
      links: LinkService.get(state.messageLinks, messageId).length,
      hasBookmark: BookmarkService.get(state.messageBookmarks, messageId) !== undefined,
      hasHighlight: HighlightService.has(state.messageHighlights, messageId),
      hasAnnotation: AnnotationService.get(state.messageAnnotations, messageId) !== undefined,
      unreadNotifications: NotificationService.countUnread(state.messageNotifications)
    }
  }

  /**
   * Limpia todos los datos de un mensaje
   */
  static clearMessage(
    state: MessageActionsState,
    messageId: string
  ): MessageActionsState {
    return {
      ...state,
      messageAttachments: AttachmentService.clear(state.messageAttachments, messageId),
      messageLinks: new Map(state.messageLinks).set(messageId, []),
      messageBookmarks: BookmarkService.remove(state.messageBookmarks, messageId),
      messageHighlights: HighlightService.remove(state.messageHighlights, messageId),
      messageAnnotations: AnnotationService.remove(state.messageAnnotations, messageId)
    }
  }

  /**
   * Duplica todos los datos de un mensaje a otro
   */
  static duplicateMessageData(
    state: MessageActionsState,
    sourceMessageId: string,
    targetMessageId: string
  ): MessageActionsState {
    // Duplicar adjuntos
    const attachments = AttachmentService.get(state.messageAttachments, sourceMessageId)
    let newAttachments = state.messageAttachments
    if (attachments.length > 0) {
      attachments.forEach(attachment => {
        newAttachments = AttachmentService.add(newAttachments, targetMessageId, attachment)
      })
    }

    // Duplicar enlaces
    const links = LinkService.get(state.messageLinks, sourceMessageId)
    let newLinks = state.messageLinks
    if (links.length > 0) {
      links.forEach(link => {
        newLinks = LinkService.add(newLinks, targetMessageId, link)
      })
    }

    // Duplicar marcador
    const bookmark = BookmarkService.get(state.messageBookmarks, sourceMessageId)
    const newBookmarks = bookmark
      ? BookmarkService.add(state.messageBookmarks, targetMessageId, bookmark)
      : state.messageBookmarks

    // Duplicar resaltado
    const highlight = HighlightService.get(state.messageHighlights, sourceMessageId)
    const newHighlights = highlight
      ? HighlightService.add(state.messageHighlights, targetMessageId, highlight)
      : state.messageHighlights

    // Duplicar anotación
    const annotation = AnnotationService.get(state.messageAnnotations, sourceMessageId)
    const newAnnotations = annotation
      ? AnnotationService.add(state.messageAnnotations, targetMessageId, annotation)
      : state.messageAnnotations

    return {
      ...state,
      messageAttachments: newAttachments,
      messageLinks: newLinks,
      messageBookmarks: newBookmarks,
      messageHighlights: newHighlights,
      messageAnnotations: newAnnotations
    }
  }

  /**
   * Obtiene un resumen completo de un mensaje
   */
  static getMessageSummary(
    state: MessageActionsState,
    messageId: string
  ): {
    messageId: string
    attachments: number
    links: number
    bookmark?: { name: string; category: string }
    highlight?: { color: string }
    annotation?: { preview: string }
  } {
    const bookmark = BookmarkService.get(state.messageBookmarks, messageId)
    const highlight = HighlightService.get(state.messageHighlights, messageId)
    const annotation = AnnotationService.get(state.messageAnnotations, messageId)

    return {
      messageId,
      attachments: AttachmentService.count(state.messageAttachments, messageId),
      links: LinkService.get(state.messageLinks, messageId).length,
      bookmark: bookmark ? { name: bookmark.name, category: bookmark.category } : undefined,
      highlight: highlight ? { color: highlight.color } : undefined,
      annotation: annotation ? { preview: annotation.annotation.substring(0, 50) } : undefined
    }
  }
}



