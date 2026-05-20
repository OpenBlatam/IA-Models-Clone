/**
 * Servicio específico para anotaciones
 */

import type { MessageAnnotation } from '../../types/message.types'
import {
  addToMap,
  getFromMap,
  removeFromMap,
  filterMap
} from '../../utils/mapUtils'

export class AnnotationService {
  static add(
    annotations: Map<string, MessageAnnotation>,
    messageId: string,
    annotation: MessageAnnotation
  ): Map<string, MessageAnnotation> {
    return addToMap(annotations, messageId, annotation)
  }

  static get(
    annotations: Map<string, MessageAnnotation>,
    messageId: string
  ): MessageAnnotation | undefined {
    return getFromMap(annotations, messageId)
  }

  static remove(
    annotations: Map<string, MessageAnnotation>,
    messageId: string
  ): Map<string, MessageAnnotation> {
    return removeFromMap(annotations, messageId)
  }

  /**
   * Actualiza una anotación
   */
  static update(
    annotations: Map<string, MessageAnnotation>,
    messageId: string,
    annotation: string
  ): Map<string, MessageAnnotation> {
    const newMap = new Map(annotations)
    newMap.set(messageId, {
      annotation,
      timestamp: Date.now()
    })
    return newMap
  }

  static search(
    annotations: Map<string, MessageAnnotation>,
    query: string
  ): Map<string, MessageAnnotation> {
    const lowerQuery = query.toLowerCase()
    return filterMap(
      annotations,
      (annotation) => annotation.annotation.toLowerCase().includes(lowerQuery)
    )
  }
}

