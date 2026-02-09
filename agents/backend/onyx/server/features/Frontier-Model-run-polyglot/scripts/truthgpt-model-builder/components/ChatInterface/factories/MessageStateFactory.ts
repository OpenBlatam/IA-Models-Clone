/**
 * Factory para crear estados iniciales de mensajes
 */

import type {
  MessageActionsState,
  MessageOrganizationState,
  MessageWorkflowState,
  MessagePollsState,
  AllMessageState
} from '../types/state.types'

export class MessageStateFactory {
  /**
   * Crea un estado inicial vacío para acciones de mensajes
   */
  static createActionsState(): MessageActionsState {
    return {
      messageAttachments: new Map(),
      messageLinks: new Map(),
      messageNotifications: new Map(),
      messageBookmarks: new Map(),
      messageHighlights: new Map(),
      messageAnnotations: new Map()
    }
  }

  /**
   * Crea un estado inicial para organización
   */
  static createOrganizationState(): MessageOrganizationState {
    return {
      messageFiltering: new Map(),
      messageGrouping: new Map(),
      messageSorting: { field: 'timestamp', order: 'desc' },
      messagePriority: new Map()
    }
  }

  /**
   * Crea un estado inicial para workflows
   */
  static createWorkflowState(): MessageWorkflowState {
    return {
      messageCalendar: new Map(),
      messageReminders: new Map(),
      messageTasks: new Map(),
      messageWorkflows: new Map()
    }
  }

  /**
   * Crea un estado inicial para encuestas
   */
  static createPollsState(): MessagePollsState {
    return {
      messagePolls: new Map()
    }
  }

  /**
   * Crea un estado completo inicial
   */
  static createCompleteState(): AllMessageState {
    return {
      ...this.createActionsState(),
      ...this.createOrganizationState(),
      ...this.createWorkflowState(),
      ...this.createPollsState()
    }
  }

  /**
   * Crea un estado desde datos existentes
   */
  static createFromData(data: Partial<AllMessageState>): AllMessageState {
    return {
      ...this.createCompleteState(),
      ...data
    }
  }
}



