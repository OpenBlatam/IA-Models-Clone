/**
 * Servicio específico para adjuntos
 */

import type { MessageAttachment } from '../../types/message.types'
import {
  addToMapArray,
  removeFromMapArray,
  getFromMapArray,
  hasInMapArray,
  countInMapArray,
  clearFromMap
} from '../../utils/mapUtils'

export class AttachmentService {
  static add(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string,
    attachment: MessageAttachment
  ): Map<string, MessageAttachment[]> {
    return addToMapArray(attachments, messageId, attachment)
  }

  static remove(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string,
    index: number
  ): Map<string, MessageAttachment[]> {
    return removeFromMapArray(attachments, messageId, index)
  }

  static get(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string
  ): MessageAttachment[] {
    return getFromMapArray(attachments, messageId)
  }

  static has(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string
  ): boolean {
    return hasInMapArray(attachments, messageId)
  }

  static count(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string
  ): number {
    return countInMapArray(attachments, messageId)
  }

  static filterByType(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string,
    type: string
  ): MessageAttachment[] {
    return this.get(attachments, messageId).filter(att => att.type === type)
  }

  static clear(
    attachments: Map<string, MessageAttachment[]>,
    messageId: string
  ): Map<string, MessageAttachment[]> {
    return clearFromMap(attachments, messageId)
  }
}

