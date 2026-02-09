/**
 * Tipos específicos para mensajes
 */

export interface MessageAttachment {
  type: string
  url: string
  name: string
}

export interface MessageLink {
  url: string
  title: string
  description: string
}

export interface MessageNotification {
  type: string
  title: string
  body: string
  read: boolean
  timestamp: number
}

export interface MessageBookmark {
  name: string
  category: string
  tags: string[]
}

export interface MessageHighlight {
  color: string
  note?: string
}

export interface MessageAnnotation {
  annotation: string
  timestamp: number
}

export interface MessagePoll {
  question: string
  options: string[]
  votes: Map<string, number>
}

export interface MessageTask {
  task: string
  completed: boolean
  dueDate?: Date
}

export interface MessageReminder {
  reminder: string
  date: Date
  recurring?: string
}

export interface MessageCalendarEvent {
  event: string
  date: Date
  duration?: number
}

export interface MessageWorkflow {
  trigger: string
  actions: string[]
  enabled: boolean
}



