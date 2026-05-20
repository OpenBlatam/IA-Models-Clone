/**
 * Hook consolidado para todos los estados relacionados con mensajes
 * Agrupa todos los estados de mensajes en un solo lugar
 */

import { useState } from 'react'
import { useMessageActions, type MessageActions } from './useMessageActions'
import { useMessageOrganization, type MessageOrganizationActions } from './useMessageOrganization'
import { useMessageWorkflow, type MessageWorkflowActions } from './useMessageWorkflow'
import { useMessagePolls, type MessagePollsActions } from './useMessagePolls'
import { MessageStateFactory } from '../factories/MessageStateFactory'
import type {
  MessageActionsState,
  MessageOrganizationState,
  MessageWorkflowState,
  MessagePollsState,
  AllMessageState
} from '../types/state.types'

// AllMessageState ahora se importa de types/state.types

export interface AllMessageActions extends
  MessageActions,
  MessageOrganizationActions,
  MessageWorkflowActions,
  MessagePollsActions {}

export function useMessageState(initialState?: Partial<AllMessageState>) {
  // Usar factory para crear estados iniciales
  const defaultState = MessageStateFactory.createCompleteState()
  const initial = initialState ? MessageStateFactory.createFromData(initialState) : defaultState

  // Estados de acciones
  const [messageAttachments, setMessageAttachments] = useState(initial.messageAttachments)
  const [messageLinks, setMessageLinks] = useState(initial.messageLinks)
  const [messageNotifications, setMessageNotifications] = useState(initial.messageNotifications)
  const [messageBookmarks, setMessageBookmarks] = useState(initial.messageBookmarks)
  const [messageHighlights, setMessageHighlights] = useState(initial.messageHighlights)
  const [messageAnnotations, setMessageAnnotations] = useState(initial.messageAnnotations)

  // Estados de organización
  const [messageFiltering, setMessageFiltering] = useState(initial.messageFiltering)
  const [messageGrouping, setMessageGrouping] = useState(initial.messageGrouping)
  const [messageSorting, setMessageSorting] = useState(initial.messageSorting)
  const [messagePriority, setMessagePriority] = useState(initial.messagePriority)

  // Estados de workflow
  const [messageCalendar, setMessageCalendar] = useState(initial.messageCalendar)
  const [messageReminders, setMessageReminders] = useState(initial.messageReminders)
  const [messageTasks, setMessageTasks] = useState(initial.messageTasks)
  const [messageWorkflows, setMessageWorkflows] = useState(initial.messageWorkflows)

  // Estados de encuestas
  const [messagePolls, setMessagePolls] = useState(initial.messagePolls)

  // Consolidar estados
  const messageState: AllMessageState = {
    messageAttachments,
    messageLinks,
    messageNotifications,
    messageBookmarks,
    messageHighlights,
    messageAnnotations,
    messageFiltering,
    messageGrouping,
    messageSorting,
    messagePriority,
    messageCalendar,
    messageReminders,
    messageTasks,
    messageWorkflows,
    messagePolls
  }

  // Usar los hooks de acciones con setters individuales
  const actionsState: MessageActionsState = {
    messageAttachments,
    messageLinks,
    messageNotifications,
    messageBookmarks,
    messageHighlights,
    messageAnnotations
  }

  const setActionsState = (updater: (prev: MessageActionsState) => MessageActionsState) => {
    const newState = updater(actionsState)
    setMessageAttachments(newState.messageAttachments)
    setMessageLinks(newState.messageLinks)
    setMessageNotifications(newState.messageNotifications)
    setMessageBookmarks(newState.messageBookmarks)
    setMessageHighlights(newState.messageHighlights)
    setMessageAnnotations(newState.messageAnnotations)
  }

  const organizationState: MessageOrganizationState = {
    messageFiltering,
    messageGrouping,
    messageSorting,
    messagePriority
  }

  const setOrganizationState = (updater: (prev: MessageOrganizationState) => MessageOrganizationState) => {
    const newState = updater(organizationState)
    setMessageFiltering(newState.messageFiltering)
    setMessageGrouping(newState.messageGrouping)
    setMessageSorting(newState.messageSorting)
    setMessagePriority(newState.messagePriority)
  }

  const workflowState: MessageWorkflowState = {
    messageCalendar,
    messageReminders,
    messageTasks,
    messageWorkflows
  }

  const setWorkflowState = (updater: (prev: MessageWorkflowState) => MessageWorkflowState) => {
    const newState = updater(workflowState)
    setMessageCalendar(newState.messageCalendar)
    setMessageReminders(newState.messageReminders)
    setMessageTasks(newState.messageTasks)
    setMessageWorkflows(newState.messageWorkflows)
  }

  const pollsState: MessagePollsState = {
    messagePolls
  }

  const setPollsState = (updater: (prev: MessagePollsState) => MessagePollsState) => {
    const newState = updater(pollsState)
    setMessagePolls(newState.messagePolls)
  }

  const actions = useMessageActions(actionsState, setActionsState)
  const organization = useMessageOrganization(organizationState, setOrganizationState)
  const workflow = useMessageWorkflow(workflowState, setWorkflowState)
  const polls = useMessagePolls(pollsState, setPollsState)

  return {
    state: messageState,
    actions: {
      ...actions,
      ...organization,
      ...workflow,
      ...polls
    } as AllMessageActions
  }
}

