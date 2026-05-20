/**
 * Servicio para organización de mensajes
 */

export class OrganizationService {
  /**
   * Filtra un mensaje
   */
  static filterMessage(
    filtering: Map<string, boolean>,
    messageId: string,
    visible: boolean
  ): Map<string, boolean> {
    const newMap = new Map(filtering)
    newMap.set(messageId, visible)
    return newMap
  }

  /**
   * Agrupa mensajes
   */
  static groupMessages(
    grouping: Map<string, { groupBy: string, groups: string[] }>,
    groupBy: string,
    groups: string[]
  ): Map<string, { groupBy: string, groups: string[] }> {
    const newMap = new Map(grouping)
    newMap.set(`group-${Date.now()}`, { groupBy, groups })
    return newMap
  }

  /**
   * Establece prioridad de un mensaje
   */
  static setPriority(
    priority: Map<string, number>,
    messageId: string,
    priorityLevel: number
  ): Map<string, number> {
    const newMap = new Map(priority)
    newMap.set(messageId, priorityLevel)
    return newMap
  }

  /**
   * Ordena mensajes
   */
  static sortMessages(
    field: string,
    order: 'asc' | 'desc' = 'asc'
  ): { field: string, order: 'asc' | 'desc' } {
    return { field, order }
  }

  /**
   * Agrupa mensajes por similitud
   */
  static clusterMessages(
    grouping: Map<string, { groupBy: string, groups: string[] }>,
    threshold: number
  ): Map<string, { groupBy: string, groups: string[] }> {
    const newMap = new Map(grouping)
    newMap.set(`cluster-${Date.now()}`, {
      groupBy: 'similarity',
      groups: []
    })
    return newMap
  }
}



