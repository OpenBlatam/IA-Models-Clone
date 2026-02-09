/**
 * Servicio para workflows y automatización
 */

import type {
  MessageCalendarEvent,
  MessageReminder,
  MessageTask,
  MessageWorkflow
} from '../types/message.types'

export class WorkflowService {
  /**
   * Programa un evento
   */
  static scheduleEvent(
    calendar: Map<string, MessageCalendarEvent>,
    messageId: string,
    event: MessageCalendarEvent
  ): Map<string, MessageCalendarEvent> {
    const newMap = new Map(calendar)
    newMap.set(messageId, event)
    return newMap
  }

  /**
   * Configura un recordatorio
   */
  static setReminder(
    reminders: Map<string, MessageReminder>,
    messageId: string,
    reminder: MessageReminder
  ): Map<string, MessageReminder> {
    const newMap = new Map(reminders)
    newMap.set(messageId, reminder)
    return newMap
  }

  /**
   * Crea una tarea
   */
  static createTask(
    tasks: Map<string, MessageTask>,
    messageId: string,
    task: MessageTask
  ): Map<string, MessageTask> {
    const newMap = new Map(tasks)
    newMap.set(messageId, task)
    return newMap
  }

  /**
   * Crea un workflow
   */
  static createWorkflow(
    workflows: Map<string, MessageWorkflow>,
    messageId: string,
    workflow: MessageWorkflow
  ): Map<string, MessageWorkflow> {
    const newMap = new Map(workflows)
    newMap.set(messageId, workflow)
    return newMap
  }

  /**
   * Programa un mensaje
   */
  static scheduleMessage(
    calendar: Map<string, MessageCalendarEvent>,
    messageId: string,
    date: Date
  ): Map<string, MessageCalendarEvent> {
    const newMap = new Map(calendar)
    newMap.set(messageId, {
      event: 'scheduled_message',
      date,
      duration: undefined
    })
    return newMap
  }
}



