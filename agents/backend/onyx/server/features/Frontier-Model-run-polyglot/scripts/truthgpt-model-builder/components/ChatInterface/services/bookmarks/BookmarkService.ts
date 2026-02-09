/**
 * Servicio específico para marcadores
 */

import type { MessageBookmark } from '../../types/message.types'
import {
  addToMap,
  getFromMap,
  removeFromMap,
  filterMap
} from '../../utils/mapUtils'

export class BookmarkService {
  static add(
    bookmarks: Map<string, MessageBookmark>,
    messageId: string,
    bookmark: MessageBookmark
  ): Map<string, MessageBookmark> {
    return addToMap(bookmarks, messageId, bookmark)
  }

  static get(
    bookmarks: Map<string, MessageBookmark>,
    messageId: string
  ): MessageBookmark | undefined {
    return getFromMap(bookmarks, messageId)
  }

  static remove(
    bookmarks: Map<string, MessageBookmark>,
    messageId: string
  ): Map<string, MessageBookmark> {
    return removeFromMap(bookmarks, messageId)
  }

  static findByCategory(
    bookmarks: Map<string, MessageBookmark>,
    category: string
  ): Map<string, MessageBookmark> {
    return filterMap(bookmarks, (bookmark) => bookmark.category === category)
  }

  static findByTag(
    bookmarks: Map<string, MessageBookmark>,
    tag: string
  ): Map<string, MessageBookmark> {
    return filterMap(bookmarks, (bookmark) => bookmark.tags.includes(tag))
  }

  /**
   * Obtiene todas las categorías únicas
   */
  static getCategories(
    bookmarks: Map<string, MessageBookmark>
  ): string[] {
    const categories = new Set<string>()
    bookmarks.forEach(bookmark => {
      categories.add(bookmark.category)
    })
    return Array.from(categories)
  }

  /**
   * Obtiene todos los tags únicos
   */
  static getTags(
    bookmarks: Map<string, MessageBookmark>
  ): string[] {
    const tags = new Set<string>()
    bookmarks.forEach(bookmark => {
      bookmark.tags.forEach(tag => tags.add(tag))
    })
    return Array.from(tags)
  }
}

