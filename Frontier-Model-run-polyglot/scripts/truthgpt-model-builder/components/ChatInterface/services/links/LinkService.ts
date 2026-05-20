/**
 * Servicio específico para enlaces
 */

import type { MessageLink } from '../../types/message.types'
import {
  addToMapArray,
  removeFromMapArray,
  getFromMapArray,
  hasInMapArray
} from '../../utils/mapUtils'

export class LinkService {
  static add(
    links: Map<string, MessageLink[]>,
    messageId: string,
    link: MessageLink
  ): Map<string, MessageLink[]> {
    return addToMapArray(links, messageId, link)
  }

  static remove(
    links: Map<string, MessageLink[]>,
    messageId: string,
    index: number
  ): Map<string, MessageLink[]> {
    return removeFromMapArray(links, messageId, index)
  }

  static get(
    links: Map<string, MessageLink[]>,
    messageId: string
  ): MessageLink[] {
    return getFromMapArray(links, messageId)
  }

  static has(
    links: Map<string, MessageLink[]>,
    messageId: string
  ): boolean {
    return hasInMapArray(links, messageId)
  }

  /**
   * Busca enlaces por URL
   */
  static findByUrl(
    links: Map<string, MessageLink[]>,
    messageId: string,
    url: string
  ): MessageLink | undefined {
    return this.get(links, messageId).find(link => link.url === url)
  }

  /**
   * Valida que una URL sea única en el mensaje
   */
  static isUrlUnique(
    links: Map<string, MessageLink[]>,
    messageId: string,
    url: string
  ): boolean {
    return !this.findByUrl(links, messageId, url)
  }
}

