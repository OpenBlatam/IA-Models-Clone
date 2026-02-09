/**
 * Compositor para combinar múltiples servicios
 * Facilita operaciones que requieren coordinación entre servicios
 */

import { AttachmentService } from '../services/attachments/AttachmentService'
import { LinkService } from '../services/links/LinkService'
import { BookmarkService } from '../services/bookmarks/BookmarkService'
import type { MessageActionsState } from '../types/state.types'

export class MessageCompositor {
  /**
   * Crea un mensaje completo con adjuntos, enlaces y marcador
   */
  static createCompleteMessage(
    state: MessageActionsState,
    messageId: string,
    options: {
      attachments?: Array<{ type: string; url: string; name: string }>
      links?: Array<{ url: string; title: string; description: string }>
      bookmark?: { name: string; category: string; tags?: string[] }
    }
  ): MessageActionsState {
    let newState = { ...state }

    // Agregar adjuntos
    if (options.attachments) {
      options.attachments.forEach(attachment => {
        newState.messageAttachments = AttachmentService.add(
          newState.messageAttachments,
          messageId,
          attachment
        )
      })
    }

    // Agregar enlaces
    if (options.links) {
      options.links.forEach(link => {
        newState.messageLinks = LinkService.add(
          newState.messageLinks,
          messageId,
          link
        )
      })
    }

    // Agregar marcador
    if (options.bookmark) {
      newState.messageBookmarks = BookmarkService.add(
        newState.messageBookmarks,
        messageId,
        {
          name: options.bookmark.name,
          category: options.bookmark.category,
          tags: options.bookmark.tags || []
        }
      )
    }

    return newState
  }

  /**
   * Exporta todos los datos de un mensaje
   */
  static exportMessageData(
    state: MessageActionsState,
    messageId: string
  ): {
    attachments: Array<{ type: string; url: string; name: string }>
    links: Array<{ url: string; title: string; description: string }>
    bookmark?: { name: string; category: string; tags: string[] }
  } {
    return {
      attachments: AttachmentService.get(state.messageAttachments, messageId),
      links: LinkService.get(state.messageLinks, messageId),
      bookmark: BookmarkService.get(state.messageBookmarks, messageId)
    }
  }

  /**
   * Importa datos a un mensaje
   */
  static importMessageData(
    state: MessageActionsState,
    messageId: string,
    data: {
      attachments?: Array<{ type: string; url: string; name: string }>
      links?: Array<{ url: string; title: string; description: string }>
      bookmark?: { name: string; category: string; tags: string[] }
    }
  ): MessageActionsState {
    return this.createCompleteMessage(state, messageId, data)
  }
}



