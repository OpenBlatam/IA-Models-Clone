/**
 * Builder Pattern para crear objetos de mensaje complejos
 * Permite construcción fluida y validación
 */

import type {
  MessageAttachment,
  MessageLink,
  MessageNotification,
  MessageBookmark
} from '../types/message.types'
import { MessageValidator } from '../validators/MessageValidator'

export class MessageAttachmentBuilder {
  private attachment: Partial<MessageAttachment> = {}

  withType(type: string): this {
    this.attachment.type = type
    return this
  }

  withUrl(url: string): this {
    this.attachment.url = url
    return this
  }

  withName(name: string): this {
    this.attachment.name = name
    return this
  }

  build(): MessageAttachment {
    if (!this.attachment.type || !this.attachment.url || !this.attachment.name) {
      throw new Error('Type, URL and name are required for attachment')
    }

    const attachment: MessageAttachment = {
      type: this.attachment.type,
      url: this.attachment.url,
      name: this.attachment.name
    }

    const validation = MessageValidator.validateAttachment(attachment)
    if (!validation.valid) {
      throw new Error(`Invalid attachment: ${validation.errors.join(', ')}`)
    }

    return attachment
  }
}

export class MessageLinkBuilder {
  private link: Partial<MessageLink> = {}

  withUrl(url: string): this {
    this.link.url = url
    return this
  }

  withTitle(title: string): this {
    this.link.title = title
    return this
  }

  withDescription(description: string): this {
    this.link.description = description
    return this
  }

  build(): MessageLink {
    if (!this.link.url) {
      throw new Error('URL is required for link')
    }

    const link: MessageLink = {
      url: this.link.url,
      title: this.link.title || '',
      description: this.link.description || ''
    }

    const validation = MessageValidator.validateLink(link)
    if (!validation.valid) {
      throw new Error(`Invalid link: ${validation.errors.join(', ')}`)
    }

    return link
  }
}

export class MessageNotificationBuilder {
  private notification: Partial<MessageNotification> = {
    read: false,
    timestamp: Date.now()
  }

  withType(type: string): this {
    this.notification.type = type
    return this
  }

  withTitle(title: string): this {
    this.notification.title = title
    return this
  }

  withBody(body: string): this {
    this.notification.body = body
    return this
  }

  withRead(read: boolean): this {
    this.notification.read = read
    return this
  }

  withTimestamp(timestamp: number): this {
    this.notification.timestamp = timestamp
    return this
  }

  build(): MessageNotification {
    if (!this.notification.type || !this.notification.title || !this.notification.body) {
      throw new Error('Type, title and body are required for notification')
    }

    const notification: MessageNotification = {
      type: this.notification.type,
      title: this.notification.title,
      body: this.notification.body,
      read: this.notification.read || false,
      timestamp: this.notification.timestamp || Date.now()
    }

    const validation = MessageValidator.validateNotification(notification)
    if (!validation.valid) {
      throw new Error(`Invalid notification: ${validation.errors.join(', ')}`)
    }

    return notification
  }
}

export class MessageBookmarkBuilder {
  private bookmark: Partial<MessageBookmark> = {
    tags: []
  }

  withName(name: string): this {
    this.bookmark.name = name
    return this
  }

  withCategory(category: string): this {
    this.bookmark.category = category
    return this
  }

  withTags(tags: string[]): this {
    this.bookmark.tags = tags
    return this
  }

  addTag(tag: string): this {
    if (!this.bookmark.tags) {
      this.bookmark.tags = []
    }
    this.bookmark.tags.push(tag)
    return this
  }

  build(): MessageBookmark {
    if (!this.bookmark.name || !this.bookmark.category) {
      throw new Error('Name and category are required for bookmark')
    }

    const bookmark: MessageBookmark = {
      name: this.bookmark.name,
      category: this.bookmark.category,
      tags: this.bookmark.tags || []
    }

    const validation = MessageValidator.validateBookmark(bookmark)
    if (!validation.valid) {
      throw new Error(`Invalid bookmark: ${validation.errors.join(', ')}`)
    }

    return bookmark
  }
}



