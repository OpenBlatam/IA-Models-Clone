/**
 * Servicio específico para resaltados
 */

import type { MessageHighlight } from '../../types/message.types'
import {
  addToMap,
  getFromMap,
  removeFromMap,
  hasInMap,
  filterMap
} from '../../utils/mapUtils'

export class HighlightService {
  static add(
    highlights: Map<string, MessageHighlight>,
    messageId: string,
    highlight: MessageHighlight
  ): Map<string, MessageHighlight> {
    return addToMap(highlights, messageId, highlight)
  }

  static get(
    highlights: Map<string, MessageHighlight>,
    messageId: string
  ): MessageHighlight | undefined {
    return getFromMap(highlights, messageId)
  }

  static remove(
    highlights: Map<string, MessageHighlight>,
    messageId: string
  ): Map<string, MessageHighlight> {
    return removeFromMap(highlights, messageId)
  }

  static has(
    highlights: Map<string, MessageHighlight>,
    messageId: string
  ): boolean {
    return hasInMap(highlights, messageId)
  }

  static findByColor(
    highlights: Map<string, MessageHighlight>,
    color: string
  ): Map<string, MessageHighlight> {
    return filterMap(highlights, (highlight) => highlight.color === color)
  }

  /**
   * Actualiza el color de un resaltado
   */
  static updateColor(
    highlights: Map<string, MessageHighlight>,
    messageId: string,
    color: string
  ): Map<string, MessageHighlight> {
    const newMap = new Map(highlights)
    const existing = newMap.get(messageId)
    if (existing) {
      newMap.set(messageId, { ...existing, color })
    }
    return newMap
  }
}

