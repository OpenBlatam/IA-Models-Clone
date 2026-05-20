/**
 * Repository Pattern para acceso a datos de mensajes
 * Abstrae el almacenamiento y permite cambiar implementaciones fácilmente
 */

import type {
  MessageAttachment,
  MessageLink,
  MessageNotification,
  MessageBookmark,
  MessageHighlight,
  MessageAnnotation
} from '../types/message.types'

export interface IMessageRepository {
  // Attachments
  getAttachments(messageId: string): MessageAttachment[]
  addAttachment(messageId: string, attachment: MessageAttachment): void
  removeAttachment(messageId: string, index: number): void
  hasAttachments(messageId: string): boolean

  // Links
  getLinks(messageId: string): MessageLink[]
  addLink(messageId: string, link: MessageLink): void
  removeLink(messageId: string, index: number): void
  hasLinks(messageId: string): boolean

  // Notifications
  getNotification(key: string): MessageNotification | undefined
  addNotification(key: string, notification: MessageNotification): void
  removeNotification(key: string): void
  getAllNotifications(): Map<string, MessageNotification>

  // Bookmarks
  getBookmark(messageId: string): MessageBookmark | undefined
  addBookmark(messageId: string, bookmark: MessageBookmark): void
  removeBookmark(messageId: string): void
  getAllBookmarks(): Map<string, MessageBookmark>

  // Highlights
  getHighlight(messageId: string): MessageHighlight | undefined
  addHighlight(messageId: string, highlight: MessageHighlight): void
  removeHighlight(messageId: string): void
  getAllHighlights(): Map<string, MessageHighlight>

  // Annotations
  getAnnotation(messageId: string): MessageAnnotation | undefined
  addAnnotation(messageId: string, annotation: MessageAnnotation): void
  removeAnnotation(messageId: string): void
  getAllAnnotations(): Map<string, MessageAnnotation>
}

/**
 * Implementación en memoria del repositorio
 */
export class InMemoryMessageRepository implements IMessageRepository {
  private attachments = new Map<string, MessageAttachment[]>()
  private links = new Map<string, MessageLink[]>()
  private notifications = new Map<string, MessageNotification>()
  private bookmarks = new Map<string, MessageBookmark>()
  private highlights = new Map<string, MessageHighlight>()
  private annotations = new Map<string, MessageAnnotation>()

  // Attachments
  getAttachments(messageId: string): MessageAttachment[] {
    return this.attachments.get(messageId) || []
  }

  addAttachment(messageId: string, attachment: MessageAttachment): void {
    const existing = this.attachments.get(messageId) || []
    this.attachments.set(messageId, [...existing, attachment])
  }

  removeAttachment(messageId: string, index: number): void {
    const existing = this.attachments.get(messageId) || []
    const filtered = existing.filter((_, i) => i !== index)
    if (filtered.length === 0) {
      this.attachments.delete(messageId)
    } else {
      this.attachments.set(messageId, filtered)
    }
  }

  hasAttachments(messageId: string): boolean {
    return this.attachments.has(messageId) && this.attachments.get(messageId)!.length > 0
  }

  // Links
  getLinks(messageId: string): MessageLink[] {
    return this.links.get(messageId) || []
  }

  addLink(messageId: string, link: MessageLink): void {
    const existing = this.links.get(messageId) || []
    this.links.set(messageId, [...existing, link])
  }

  removeLink(messageId: string, index: number): void {
    const existing = this.links.get(messageId) || []
    const filtered = existing.filter((_, i) => i !== index)
    if (filtered.length === 0) {
      this.links.delete(messageId)
    } else {
      this.links.set(messageId, filtered)
    }
  }

  hasLinks(messageId: string): boolean {
    return this.links.has(messageId) && this.links.get(messageId)!.length > 0
  }

  // Notifications
  getNotification(key: string): MessageNotification | undefined {
    return this.notifications.get(key)
  }

  addNotification(key: string, notification: MessageNotification): void {
    this.notifications.set(key, notification)
  }

  removeNotification(key: string): void {
    this.notifications.delete(key)
  }

  getAllNotifications(): Map<string, MessageNotification> {
    return new Map(this.notifications)
  }

  // Bookmarks
  getBookmark(messageId: string): MessageBookmark | undefined {
    return this.bookmarks.get(messageId)
  }

  addBookmark(messageId: string, bookmark: MessageBookmark): void {
    this.bookmarks.set(messageId, bookmark)
  }

  removeBookmark(messageId: string): void {
    this.bookmarks.delete(messageId)
  }

  getAllBookmarks(): Map<string, MessageBookmark> {
    return new Map(this.bookmarks)
  }

  // Highlights
  getHighlight(messageId: string): MessageHighlight | undefined {
    return this.highlights.get(messageId)
  }

  addHighlight(messageId: string, highlight: MessageHighlight): void {
    this.highlights.set(messageId, highlight)
  }

  removeHighlight(messageId: string): void {
    this.highlights.delete(messageId)
  }

  getAllHighlights(): Map<string, MessageHighlight> {
    return new Map(this.highlights)
  }

  // Annotations
  getAnnotation(messageId: string): MessageAnnotation | undefined {
    return this.annotations.get(messageId)
  }

  addAnnotation(messageId: string, annotation: MessageAnnotation): void {
    this.annotations.set(messageId, annotation)
  }

  removeAnnotation(messageId: string): void {
    this.annotations.delete(messageId)
  }

  getAllAnnotations(): Map<string, MessageAnnotation> {
    return new Map(this.annotations)
  }
}



