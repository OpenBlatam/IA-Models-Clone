/**
 * Tipos para estados de mensajes
 */

import type {
  MessageAttachment,
  MessageLink,
  MessageNotification,
  MessageBookmark,
  MessageHighlight,
  MessageAnnotation,
  MessagePoll,
  MessageTask,
  MessageReminder,
  MessageCalendarEvent,
  MessageWorkflow
} from './message.types'

export interface MessageActionsState {
  messageAttachments: Map<string, MessageAttachment[]>
  messageLinks: Map<string, MessageLink[]>
  messageNotifications: Map<string, MessageNotification>
  messageBookmarks: Map<string, MessageBookmark>
  messageHighlights: Map<string, MessageHighlight>
  messageAnnotations: Map<string, MessageAnnotation>
}

export interface MessageOrganizationState {
  messageFiltering: Map<string, boolean>
  messageGrouping: Map<string, { groupBy: string, groups: string[] }>
  messageSorting: { field: string, order: 'asc' | 'desc' }
  messagePriority: Map<string, number>
}

export interface MessageWorkflowState {
  messageCalendar: Map<string, MessageCalendarEvent>
  messageReminders: Map<string, MessageReminder>
  messageTasks: Map<string, MessageTask>
  messageWorkflows: Map<string, MessageWorkflow>
}

export interface MessagePollsState {
  messagePolls: Map<string, MessagePoll>
}

export interface AllMessageState extends
  MessageActionsState,
  MessageOrganizationState,
  MessageWorkflowState,
  MessagePollsState {}



