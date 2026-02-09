/**
 * Servicio general para operaciones de mensajes
 * @deprecated Usar servicios específicos en lugar de este (AttachmentService, LinkService, etc.)
 * Mantenido por compatibilidad con código legacy
 */

import type {
  MessageAttachment,
  MessageLink,
  MessageNotification,
  MessageBookmark,
  MessageHighlight,
  MessageAnnotation
} from '../types/message.types'
import {
  addToMapArray,
  removeFromMapArray,
  addToMap,
  removeFromMap
} from '../utils/mapUtils'

export class MessageService {
  /** @deprecated Usar AttachmentService.add */
  static addAttachment(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string,
    attachment: MessageAttachment
  ): Map<string, MessageAttachment[]> {
    return addToMapArray(attachments, messageId, attachment)
  }

  /** @deprecated Usar AttachmentService.remove */
  static removeAttachment(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string,
    index: number
  ): Map<string, MessageAttachment[]> {
    return removeFromMapArray(attachments, messageId, index)
  }

  /** @deprecated Usar LinkService.add */
  static addLink(
    links: Map<string, MessageLink[]>,
    messageId: string,
    link: MessageLink
  ): Map<string, MessageLink[]> {
    return addToMapArray(links, messageId, link)
  }

  /** @deprecated Usar LinkService.remove */
  static removeLink(
    links: Map<string, MessageLink[]>,
    messageId: string,
    index: number
  ): Map<string, MessageLink[]> {
    return removeFromMapArray(links, messageId, index)
  }

  /** @deprecated Usar NotificationService.create */
  static createNotification(
    notifications: Map<string, MessageNotification>,
    notification: MessageNotification,
    key?: string
  ): Map<string, MessageNotification> {
    const notificationKey = key || `notif-${Date.now()}`
    return addToMap(notifications, notificationKey, notification)
  }

  /** @deprecated Usar BookmarkService.add */
  static addBookmark(
    bookmarks: Map<string, MessageBookmark>,
    messageId: string,
    bookmark: MessageBookmark
  ): Map<string, MessageBookmark> {
    return addToMap(bookmarks, messageId, bookmark)
  }

  /** @deprecated Usar BookmarkService.remove */
  static removeBookmark(
    bookmarks: Map<string, MessageBookmark>,
    messageId: string
  ): Map<string, MessageBookmark> {
    return removeFromMap(bookmarks, messageId)
  }

  /** @deprecated Usar HighlightService.add */
  static highlightMessage(
    highlights: Map<string, MessageHighlight>,
    messageId: string,
    highlight: MessageHighlight
  ): Map<string, MessageHighlight> {
    return addToMap(highlights, messageId, highlight)
  }

  /** @deprecated Usar HighlightService.remove */
  static removeHighlight(
    highlights: Map<string, MessageHighlight>,
    messageId: string
  ): Map<string, MessageHighlight> {
    return removeFromMap(highlights, messageId)
  }

  /** @deprecated Usar AnnotationService.add */
  static addAnnotation(
    annotations: Map<string, MessageAnnotation>,
    messageId: string,
    annotation: MessageAnnotation
  ): Map<string, MessageAnnotation> {
    return addToMap(annotations, messageId, annotation)
  }

  /** @deprecated Usar AnnotationService.remove */
  static removeAnnotation(
    annotations: Map<string, MessageAnnotation>,
    messageId: string
  ): Map<string, MessageAnnotation> {
    return removeFromMap(annotations, messageId)
  }
}

