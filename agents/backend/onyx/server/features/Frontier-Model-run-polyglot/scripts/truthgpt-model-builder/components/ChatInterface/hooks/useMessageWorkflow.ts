/**
 * Hook modular para workflows y automatización de mensajes
 * Centraliza funciones de programación, traducción, importación, etc.
 */

import { useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { WorkflowService } from '../services/WorkflowService'
import type { MessageWorkflowState } from '../types/state.types'

// MessageWorkflowState ahora se importa de types/state.types

export interface MessageWorkflowActions {
  scheduleEvent: (messageId: string, event: string, date: Date, duration?: number) => void
  setReminder: (messageId: string, reminder: string, date: Date, recurring?: string) => void
  createTask: (messageId: string, task: string, dueDate?: Date) => void
  createWorkflow: (messageId: string, trigger: string, actions: string[]) => void
  scheduleMessage: (messageId: string, date: Date) => void
  translateMessage: (messageId: string, targetLanguage: string) => void
  importMessages: (source: string, format: string) => void
  exportMessages: (format: string, template?: string) => void
}

export function useMessageWorkflow(
  state: MessageWorkflowState,
  setState: React.Dispatch<React.SetStateAction<MessageWorkflowState>>
): MessageWorkflowActions {
  
  const scheduleEvent = useCallback((messageId: string, event: string, date: Date, duration?: number) => {
    setState(prev => ({
      ...prev,
      messageCalendar: WorkflowService.scheduleEvent(
        prev.messageCalendar,
        messageId,
        { event, date, duration }
      )
    }))
    toast.success('Evento programado', { icon: '📅' })
  }, [setState])

  const setReminder = useCallback((messageId: string, reminder: string, date: Date, recurring?: string) => {
    setState(prev => ({
      ...prev,
      messageReminders: WorkflowService.setReminder(
        prev.messageReminders,
        messageId,
        { reminder, date, recurring }
      )
    }))
    toast.success('Recordatorio configurado', { icon: '⏰' })
  }, [setState])

  const createTask = useCallback((messageId: string, task: string, dueDate?: Date) => {
    setState(prev => ({
      ...prev,
      messageTasks: WorkflowService.createTask(
        prev.messageTasks,
        messageId,
        { task, completed: false, dueDate }
      )
    }))
    toast.success('Tarea creada', { icon: '✅' })
  }, [setState])

  const createWorkflow = useCallback((messageId: string, trigger: string, actions: string[]) => {
    setState(prev => ({
      ...prev,
      messageWorkflows: WorkflowService.createWorkflow(
        prev.messageWorkflows,
        messageId,
        { trigger, actions, enabled: true }
      )
    }))
    toast.success('Workflow creado', { icon: '⚙️' })
  }, [setState])

  const scheduleMessage = useCallback((messageId: string, date: Date) => {
    setState(prev => ({
      ...prev,
      messageCalendar: WorkflowService.scheduleMessage(prev.messageCalendar, messageId, date)
    }))
    toast.success('Mensaje programado', { icon: '📤' })
  }, [setState])

  const translateMessage = useCallback((messageId: string, targetLanguage: string) => {
    toast.success(`Traduciendo a ${targetLanguage}...`, { icon: '🌐' })
    // La traducción real se haría en el componente principal
  }, [])

  const importMessages = useCallback((source: string, format: string) => {
    toast.success(`Importando desde ${source} (${format})...`, { icon: '📥' })
    // La importación real se haría en el componente principal
  }, [])

  const exportMessages = useCallback((format: string, template?: string) => {
    toast.success(`Exportando a ${format}...`, { icon: '📤' })
    // La exportación real se haría en el componente principal
  }, [])

  return {
    scheduleEvent,
    setReminder,
    createTask,
    createWorkflow,
    scheduleMessage,
    translateMessage,
    importMessages,
    exportMessages
  }
}

