/**
 * Hook modular para organización de mensajes
 * Centraliza funciones de ordenamiento, agrupación, filtrado, etc.
 */

import { useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { OrganizationService } from '../services/OrganizationService'
import type { MessageOrganizationState } from '../types/state.types'

// MessageOrganizationState ahora se importa de types/state.types

export interface MessageOrganizationActions {
  sortMessages: (field: string, order?: 'asc' | 'desc') => void
  groupMessages: (groupBy: string, groups: string[]) => void
  setMessagePriority: (messageId: string, priority: number) => void
  filterMessage: (messageId: string, visible: boolean) => void
  clusterMessages: (threshold: number) => void
}

export function useMessageOrganization(
  state: MessageOrganizationState,
  setState: React.Dispatch<React.SetStateAction<MessageOrganizationState>>
): MessageOrganizationActions {
  
  const sortMessages = useCallback((field: string, order: 'asc' | 'desc' = 'asc') => {
    setState(prev => ({
      ...prev,
      messageSorting: OrganizationService.sortMessages(field, order)
    }))
    toast.success(`Mensajes ordenados por ${field} (${order})`, { icon: '📊' })
  }, [setState])

  const groupMessages = useCallback((groupBy: string, groups: string[]) => {
    setState(prev => ({
      ...prev,
      messageGrouping: OrganizationService.groupMessages(prev.messageGrouping, groupBy, groups)
    }))
    toast.success('Mensajes agrupados', { icon: '📦' })
  }, [setState])

  const setMessagePriority = useCallback((messageId: string, priority: number) => {
    setState(prev => ({
      ...prev,
      messagePriority: OrganizationService.setPriority(prev.messagePriority, messageId, priority)
    }))
    toast.success(`Prioridad establecida: ${priority}`, { icon: '⭐' })
  }, [setState])

  const filterMessage = useCallback((messageId: string, visible: boolean) => {
    setState(prev => ({
      ...prev,
      messageFiltering: OrganizationService.filterMessage(prev.messageFiltering, messageId, visible)
    }))
    toast.success(visible ? 'Mensaje visible' : 'Mensaje oculto', { icon: '👁️' })
  }, [setState])

  const clusterMessages = useCallback((threshold: number) => {
    setState(prev => ({
      ...prev,
      messageGrouping: OrganizationService.clusterMessages(prev.messageGrouping, threshold)
    }))
    toast.success(`Mensajes agrupados (umbral: ${threshold})`, { icon: '🔗' })
  }, [setState])

  return {
    sortMessages,
    groupMessages,
    setMessagePriority,
    filterMessage,
    clusterMessages
  }
}

